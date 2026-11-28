# Backend cửa hàng: đơn hàng, phiên chat, email xác nhận. Chat chuyển tiếp sang agent service (services/agent) qua
# HTTP; giỏ hàng nằm trong state của agent nên đọc từ agent (GET /threads/{id}), không tin giỏ do trình duyệt gửi.
# Agent tắt: be vẫn chạy - mở phiên, hủy form vẫn được; chat, tính giá, đặt hàng báo 503 (không có giỏ để đọc).
import asyncio
import json
import logging
import re
import secrets
import threading
import time
import uuid
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Literal

import httpx
import psycopg
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Path, Request
from fastapi.responses import StreamingResponse
from phone_common import pii
from phone_common.config import PII_KEY
from pydantic import BaseModel, Field, field_validator

from phone_api import catalog, mailer, orders, policies, promotions
from phone_api.config import ADMIN_API_KEY, AGENT_API_KEY, AGENT_URL, CHAT_RATE_PER_MIN, SMTP_HOST
from phone_api.policies import MAX_CONTENT, MAX_PROMPT, SLUG, policy_hash

log = logging.getLogger('uvicorn.error')

AGENT_DOWN = 'Dạ, trợ lý tư vấn đang tạm nghỉ, anh/chị thử lại sau ít phút giúp em nha.'
GREETING = ('Dạ, em chào anh/chị. Bên em tư vấn iPhone, Samsung Galaxy, phụ kiện đi kèm, giá và tình trạng hàng, '
            'cùng chính sách đổi trả, giao hàng, trả góp và bảo hành. Anh/chị cần gì cứ hỏi em nha.')
# thread_id đi vào đường dẫn khi gọi agent (/threads/{id}) -> chỉ cho chữ, số, "-", "_" (uuid của giao diện)
THREAD_ID = r'^[\w-]+$'


class ChatRequest(BaseModel):
    # Chuẩn hóa chữ, che số điện thoại/email làm ở agent (hệ thống khác gọi thẳng agent cũng được che)
    message: str = Field(min_length=1, max_length=1000)
    thread_id: str | None = Field(default=None, max_length=100, pattern=THREAD_ID)


class OrderRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=100, pattern=THREAD_ID)
    name: str = Field(min_length=2, max_length=100)
    phone: str = Field(pattern=r'^(0|\+84)\d{9}$')
    address: str = Field(min_length=10, max_length=300)
    email: str | None = Field(default=None, max_length=254)  # không bắt buộc
    promo_code: str | None = Field(default=None, max_length=50)  # mã khuyến mãi, không bắt buộc
    # Form hiện chỉ hỗ trợ giao tận nơi, thanh toán khi nhận hàng
    fulfillment_type: Literal['delivery'] = 'delivery'
    payment: Literal['cod'] = 'cod'

    @field_validator('name', 'address', mode='before')
    @classmethod
    def strip(cls, v):
        return v.strip() if isinstance(v, str) else v

    @field_validator('email', mode='before')
    @classmethod
    def check_email(cls, v):
        # Bỏ trống -> None; có nhập thì phải đúng dạng (giống kiểm tra trên form)
        if not isinstance(v, str) or not v.strip():
            return None
        v = v.strip()
        if not re.fullmatch(r'[^@\s]+@[^@\s]+\.[^@\s]+', v):
            raise ValueError('Email chưa đúng định dạng')
        return v

    @field_validator('promo_code', mode='before')
    @classmethod
    def strip_code(cls, v):
        return (v.strip().upper() or None) if isinstance(v, str) else v

    @field_validator('phone', mode='before')
    @classmethod
    def normalize_phone(cls, v):
        # "0912 345.678" -> "0912345678", giống kiểm tra trên form
        return ''.join(c for c in v if c not in ' .-') if isinstance(v, str) else v


class QuoteRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=100, pattern=THREAD_ID)
    promo_code: str | None = Field(default=None, max_length=50)


class QuoteResponse(BaseModel):
    subtotal: int
    discount: int
    total: int
    promo: str | None = None         # mã (hoặc mô tả khuyến mãi tự động) đang áp dụng
    promo_text: str | None = None    # mô tả đầy đủ: mức giảm, phạm vi, điều kiện, hạn


class CancelRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=100, pattern=THREAD_ID)


class CancelResponse(BaseModel):
    thread_id: str
    answer: str


class OrderResponse(BaseModel):
    thread_id: str
    order_id: str
    answer: str          # tóm tắt dạng chữ + lời cảm ơn (cũng là tin nhắn ghi vào lịch sử hội thoại)
    order: dict          # dữ liệu đơn đọc từ database, giao diện hiện thành thẻ xác nhận
    thanks: str          # lời cảm ơn do LLM viết
    customer_token: str | None = None  # mã khách đã ký, trình duyệt giữ để phiên sau nhận ra khách quen (POST /session)


class SessionRequest(BaseModel):
    end_thread_id: str | None = Field(default=None, max_length=100)    # cuộc trò chuyện cũ cần kết thúc
    customer_token: str | None = Field(default=None, max_length=200)   # từ OrderResponse / lần /session trước


class SessionResponse(BaseModel):
    thread_id: str
    greeting: str
    returning: bool                     # khách quen (đã đặt hàng trước đó)
    customer_token: str | None = None   # None: mã sai / khách không còn -> trình duyệt xóa mã đã lưu


class PolicyIn(BaseModel):
    content_md: str = Field(min_length=1, max_length=MAX_CONTENT)  # markdown: # tiêu đề, ## mục, ### mục con
    prompt: str | None = Field(default=None, max_length=MAX_PROMPT)  # hướng dẫn trả lời riêng cho tài liệu này

    @field_validator('content_md')
    @classmethod
    def not_blank(cls, v):
        if not v.strip():
            raise ValueError('Nội dung chính sách trống')
        return v

    @field_validator('prompt', mode='before')
    @classmethod
    def blank_prompt(cls, v):
        return (v.strip() or None) if isinstance(v, str) else v


class PolicyOut(BaseModel):
    slug: str
    content_md: str
    prompt: str | None = None
    version: int | None = None
    updated_at: datetime | None = None
    synced: bool | None = None   # xuất bản/xóa: đã cập nhật sang agent chưa (False: agent tắt, sync sau sẽ đẩy bù)
    hash: str | None = None      # GET /policies/{slug} (cho agent): phiên bản tài liệu, agent lưu và so sánh nguyên văn


class DraftOut(BaseModel):
    slug: str
    content_md: str
    prompt: str | None = None
    updated_at: datetime | None = None
    check_result: dict | None = None  # kết quả kiểm tra gần nhất (None: chưa kiểm hoặc đã sửa sau khi kiểm)


class CheckIn(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    must_contain: list[list[str]] = []   # mỗi nhóm: chỉ cần 1 cách viết xuất hiện, vd [["30 ngày"], ["7", "bảy"]]


class PublishIn(BaseModel):
    by: str | None = Field(default=None, max_length=100)     # người xuất bản (ghi vào lịch sử)
    note: str | None = Field(default=None, max_length=500)   # lý do sửa


class RollbackIn(PublishIn):
    version: int = Field(ge=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    pii.install_log_filter()  # che số điện thoại, email trong log (handler của uvicorn đã có lúc này)
    if not PII_KEY:
        log.error('Chưa đặt PII_KEY trong .env: POST /order sẽ lỗi (không mã hóa được thông tin khách)')
    if not SMTP_HOST:
        log.warning('Chưa đặt SMTP_HOST trong .env: không gửi email xác nhận đơn hàng')
    try:
        # Bảng customers, orders cho form đặt hàng; mã hóa số điện thoại, email còn ở dạng chữ thường
        orders.ensure_tables()
    except psycopg.Error as e:
        log.error('Chưa tạo được bảng đơn hàng (%s): POST /order sẽ lỗi tới khi Postgres sẵn sàng', type(e).__name__)
    except pii.PiiKeyError as e:
        log.error('Chưa mã hóa được thông tin khách cũ: %s', e)
    try:
        if n := policies.ensure_table():
            log.info('Đã nạp %d chính sách mẫu vào bảng policies', n)
    except psycopg.Error as e:
        log.error('Chưa tạo được bảng chính sách (%s)', type(e).__name__)
    if not ADMIN_API_KEY:
        log.warning('Chưa đặt ADMIN_API_KEY: /admin/policies bị tắt')
    headers = {'Authorization': f'Bearer {AGENT_API_KEY}'} if AGENT_API_KEY else {}
    # read 180s: 1 lượt chat có thể chạy vài tool + model dự phòng; connect 5s: agent tắt thì báo lỗi ngay
    app.state.agent_http = httpx.Client(base_url=AGENT_URL, headers=headers, timeout=httpx.Timeout(180, connect=5))
    log.info('Agent: %s', AGENT_URL)
    yield
    app.state.agent_http.close()


app = FastAPI(title='Phone Shop API', version='0.1.0', lifespan=lifespan)
app.include_router(catalog.router)


def _detail(r: httpx.Response):
    try:
        return r.json().get('detail', r.text)
    except ValueError:
        return r.text


def _agent(request: Request, method: str, path: str, **kwargs) -> httpx.Response:
    """Gọi agent service. Không kết nối được -> 503 AGENT_DOWN; agent trả lỗi -> giữ nguyên mã lỗi và detail."""
    try:
        r = request.app.state.agent_http.request(method, path, **kwargs)
    except httpx.TransportError as e:
        log.warning('Không gọi được agent %s %s (%s)', method, path, type(e).__name__)
        raise HTTPException(503, AGENT_DOWN)
    if r.is_error:
        raise HTTPException(r.status_code, _detail(r))
    return r


def _cart(request: Request, thread_id: str) -> list[dict]:
    return _agent(request, 'GET', f'/threads/{thread_id}').json()['cart']


# Chống spam: mỗi IP tối đa CHAT_RATE_PER_MIN lượt /chat, /chat/stream, /session, /order* trong 60 giây gần nhất.
# IP thật lấy qua uvicorn --proxy-headers (X-Forwarded-For do nginx của fe thêm, FORWARDED_ALLOW_IPS trong compose).
# shortcut: đếm trong RAM, đúng với 1 worker (Dockerfile); chạy nhiều worker/nhiều máy thì chuyển sang Redis.
RATE_DETAIL = 'Dạ, anh/chị nhắn hơi nhanh, chờ em khoảng 1 phút rồi nhắn tiếp giúp em nha.'
MAX_IPS = 10_000  # trần số IP nhớ cùng lúc: quá thì bỏ IP im lâu nhất (O(1), không quét cả dict)
_hits: OrderedDict[str, deque] = OrderedDict()  # thứ tự: IP nhắn gần nhất ở cuối
_hits_lock = threading.Lock()


def limit_rate(request: Request) -> None:
    if CHAT_RATE_PER_MIN <= 0:
        return
    ip = request.client.host if request.client else 'unknown'
    now = time.monotonic()
    with _hits_lock:
        q = _hits.setdefault(ip, deque())
        _hits.move_to_end(ip)
        while len(_hits) > MAX_IPS:
            _hits.popitem(last=False)
        while q and now - q[0] >= 60:
            q.popleft()
        if len(q) >= CHAT_RATE_PER_MIN:
            raise HTTPException(429, RATE_DETAIL, headers={'Retry-After': str(int(60 - (now - q[0])) + 1)})
        q.append(now)


rate_limited = [Depends(limit_rate)]


@app.get('/health')
def health(request: Request):
    # be khỏe kể cả khi agent tắt; trường agent cho biết chat có dùng được không
    try:
        agent = 'ok' if request.app.state.agent_http.get('/health', timeout=3).is_success else 'error'
    except httpx.TransportError:
        agent = 'down'
    return {'status': 'ok', 'agent': agent}


@app.post('/chat', dependencies=rate_limited)
def chat(req: ChatRequest, request: Request):
    return _agent(request, 'POST', '/chat', json=req.model_dump(exclude_none=True)).json()


@app.post('/chat/stream', dependencies=rate_limited)
def chat_stream(req: ChatRequest, request: Request):
    """Chuyển tiếp Server-Sent Events của agent (xem phone_agent.service.chat_stream) tới trình duyệt."""
    http = request.app.state.agent_http
    try:
        r = http.send(http.build_request('POST', '/chat/stream', json=req.model_dump(exclude_none=True)), stream=True)
    except httpx.TransportError as e:
        log.warning('Không gọi được agent /chat/stream (%s)', type(e).__name__)
        raise HTTPException(503, AGENT_DOWN)
    if r.is_error:  # lỗi trước khi stream bắt đầu (vd 422 tin nhắn quá dài)
        r.read()
        r.close()
        raise HTTPException(r.status_code, _detail(r))

    def relay():
        try:
            yield from r.iter_raw()
        except httpx.TransportError as e:
            # Agent tắt giữa chừng: báo lỗi theo đúng dạng sự kiện giao diện đang đọc
            log.warning('Mất kết nối agent khi đang stream (%s)', type(e).__name__)
            yield f"data: {json.dumps({'type': 'error', 'status': 503, 'detail': AGENT_DOWN}, ensure_ascii=False)}\n\n"
        finally:
            r.close()

    # X-Accel-Buffering: tắt buffer của nginx (services/fe) để chữ tới trình duyệt ngay
    return StreamingResponse(relay(), media_type='text/event-stream',
                             headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.post('/order', response_model=OrderResponse, dependencies=rate_limited)
def order(req: OrderRequest, request: Request, background: BackgroundTasks):
    """Khách gửi form đặt hàng: tạo đơn từ giỏ hàng của thread trên agent (không tin giỏ do trình duyệt gửi lên),
    trả tóm tắt đơn lấy từ database + 1-2 câu cảm ơn do LLM viết, và ghi vào lịch sử hội thoại.
    Email xác nhận gửi sau khi đã trả kết quả (BackgroundTasks, mailer.send_order_email) nên khách không phải chờ."""
    cart = _cart(request, req.thread_id)
    try:
        order_id = orders.create_order(cart, req.name, req.phone, req.address, req.fulfillment_type, req.payment,
                                       thread_id=req.thread_id, email=req.email, promo_code=req.promo_code)
        saved = orders.get_order(order_id)
    except (orders.OrderError, promotions.PromoError) as e:
        raise HTTPException(409, str(e))
    except psycopg.Error as e:
        log.error('Lỗi khi lưu đơn hàng', exc_info=e)
        raise HTTPException(503, 'Dạ, em chưa lưu được đơn hàng, anh/chị thử lại sau ít phút giúp em nha.')
    except pii.PiiKeyError as e:
        log.error('Không mã hóa được thông tin khách: %s', e)
        raise HTTPException(503, 'Dạ, em chưa lưu được đơn hàng, anh/chị thử lại sau ít phút giúp em nha.')
    background.add_task(mailer.send_order_email, order_id)
    thanks = orders.thank_you(req.name)
    answer = f'{orders.summary(saved)}\n\n{thanks}'
    try:
        # Lượt sau agent biết đơn đã đặt; giỏ hàng làm trống để khách mua tiếp không bị cộng dồn
        _agent(request, 'POST', f'/threads/{req.thread_id}/order-placed', json={'order_id': order_id, 'answer': answer})
    except HTTPException as e:
        # Đơn đã lưu: vẫn trả thành công, chỉ ghi log (giỏ trên agent chưa được làm trống)
        log.error('Đơn %s đã lưu nhưng chưa ghi được vào cuộc trò chuyện: %s', order_id, e.detail)
    return OrderResponse(thread_id=req.thread_id, order_id=order_id, answer=answer, order=orders.view(saved),
                         thanks=thanks,
                         customer_token=pii.customer_token(saved['customer_id']) if saved.get('customer_id') else None)


@app.post('/order/quote', response_model=QuoteResponse, dependencies=rate_limited)
def order_quote(req: QuoteRequest, request: Request):
    """Form đặt hàng hỏi giá trước khi gửi: tạm tính, giảm giá (mã khách nhập hoặc khuyến mãi tự động), tổng tiền.
    Tính từ giỏ hàng của thread trên agent. Mã không dùng được thì 409 kèm lý do."""
    cart = _cart(request, req.thread_id)
    try:
        price = orders.quote(cart, req.promo_code)
    except (orders.OrderError, promotions.PromoError) as e:
        raise HTTPException(409, str(e))
    except psycopg.Error as e:
        log.error('Lỗi khi tính giá đơn hàng', exc_info=e)
        raise HTTPException(503, 'Dạ, em chưa kiểm tra được mã khuyến mãi, anh/chị thử lại sau ít phút giúp em nha.')
    promo = price['promo']
    return QuoteResponse(subtotal=price['subtotal'], discount=price['discount'], total=price['total'],
                         promo=promotions.label(promo), promo_text=promotions.describe(promo) if promo else None)


@app.post('/session', response_model=SessionResponse, dependencies=rate_limited)
def new_session(req: SessionRequest, request: Request):
    """Kết thúc cuộc trò chuyện cũ và mở cuộc trò chuyện mới trên agent.
    Khách đã đặt hàng (customer_token hợp lệ, có đơn) thì chào lại theo tên, ghi khách quen vào trạng thái hội thoại;
    không thì chào như khách mới. Lời chào là template, không gọi LLM. Agent tắt vẫn trả lời chào."""
    thread_id = str(uuid.uuid4())
    customer_id = pii.verify_customer_token(req.customer_token)
    token, profile = None, None
    if customer_id:
        try:
            profile = orders.customer_profile(customer_id)
            token = pii.customer_token(customer_id) if profile else None
        except psycopg.Error as e:
            # Không đọc được database: vẫn mở phiên, chào như khách mới, giữ mã để lần sau thử lại
            log.error('Không đọc được thông tin khách quen', exc_info=e)
            token = req.customer_token
    greeting = orders.welcome_back(profile) if profile else GREETING
    body = {'thread_id': thread_id, 'end_thread_id': req.end_thread_id}
    if profile:
        body |= {'greeting': greeting, 'customer': profile}
    try:
        _agent(request, 'POST', '/threads', json=body)
    except HTTPException as e:
        log.warning('Chưa mở được cuộc trò chuyện trên agent: %s', e.detail)
    return SessionResponse(thread_id=thread_id, greeting=greeting, returning=bool(profile), customer_token=token)


@app.post('/order/cancel', response_model=CancelResponse, dependencies=rate_limited)
def cancel_order(req: CancelRequest, request: Request):
    """Khách bấm "Hủy đặt hàng" trên form: không tạo đơn, ghi vào lịch sử hội thoại câu trả lời cố định
    (không gọi LLM) và làm trống giỏ hàng để agent không mời chốt đơn / bán kèm tiếp cho giỏ đã hủy."""
    try:
        _agent(request, 'POST', f'/threads/{req.thread_id}/order-cancelled', json={'answer': orders.CANCEL_MESSAGE})
    except HTTPException as e:
        log.warning('Chưa ghi được việc hủy đặt hàng vào cuộc trò chuyện: %s', e.detail)
    return CancelResponse(thread_id=req.thread_id, answer=orders.CANCEL_MESSAGE)


# ---------------------------------------------------------------- quản trị chính sách
# Sửa nháp -> kiểm tra (câu hỏi mẫu) -> xuất bản (thêm version vào lịch sử) -> đẩy sang agent. Khách chỉ thấy bản đã
# xuất bản. Quay lại bản cũ: rollback (thành version mới, không kiểm tra lại).

def require_admin(request: Request) -> None:
    if not ADMIN_API_KEY:
        raise HTTPException(503, 'Chưa đặt ADMIN_API_KEY: endpoint quản trị đang tắt')
    given = request.headers.get('authorization', '').removeprefix('Bearer ')
    if not secrets.compare_digest(given.encode(), ADMIN_API_KEY.encode()):
        raise HTTPException(401, 'Sai hoặc thiếu ADMIN_API_KEY')


admin = [Depends(require_admin)]
DB_DOWN = 'Chưa đọc/ghi được database chính sách, thử lại sau.'
SlugPath = Path(pattern=SLUG)


def _db(fn, *args):
    """Gọi phone_api.policies: lỗi database -> 503, PolicyError (sai trạng thái) -> 409."""
    try:
        return fn(*args)
    except policies.PolicyError as e:
        raise HTTPException(409, str(e))
    except psycopg.Error as e:
        log.error('Lỗi database chính sách', exc_info=e)
        raise HTTPException(503, DB_DOWN)


def _found(doc, what: str):
    if not doc:
        raise HTTPException(404, f'Không có {what}')
    return doc


@app.get('/policies', dependencies=[Depends(catalog.require_catalog_key)])
async def policy_manifest() -> dict[str, str]:
    docs = await asyncio.to_thread(_db, policies.list_all)
    return {doc['slug']: policy_hash(doc['content_md'], doc['prompt']) for doc in docs}


@app.get('/policies/{slug}', response_model=PolicyOut, dependencies=[Depends(catalog.require_catalog_key)])
async def published_policy(slug: str = SlugPath):
    doc = _found(await asyncio.to_thread(_db, policies.get, slug), f'chính sách {slug}')
    return {**doc, 'hash': policy_hash(doc['content_md'], doc['prompt'])}


def _notify(request: Request, doc: dict) -> dict:
    try:
        request.app.state.agent_http.post('/webhooks/policy-updated', json={'slug': doc['slug']},
                                          timeout=300).raise_for_status()
        return {**doc, 'synced': True}
    except httpx.HTTPError as e:
        log.warning('Chưa thông báo được chính sách %s cho agent (%s)', doc['slug'], type(e).__name__)
        return {**doc, 'synced': False}


@app.get('/admin/policies', dependencies=admin)
def admin_list_policies():
    """Mọi tài liệu: version đang dùng, có nháp không, nháp đã đạt kiểm tra chưa."""
    return _db(policies.overview)


@app.get('/admin/policies/{slug}', response_model=PolicyOut, dependencies=admin)
def admin_get_policy(slug: str = SlugPath):
    return _found(_db(policies.get, slug), f'chính sách {slug}')


@app.get('/admin/policies/{slug}/draft', response_model=DraftOut, dependencies=admin)
def admin_get_draft(slug: str = SlugPath):
    return _found(_db(policies.get_draft, slug), f'bản nháp {slug}')


@app.put('/policies/{slug}', response_model=DraftOut, dependencies=admin)
@app.put('/admin/policies/{slug}/draft', response_model=DraftOut, dependencies=admin)
def admin_save_draft(req: PolicyIn, slug: str = SlugPath):
    """Lưu bản nháp (thêm tài liệu mới cũng bắt đầu từ đây). Chưa đổi gì với khách; sửa nháp thì phải kiểm lại."""
    return _db(policies.save_draft, slug, req.content_md, req.prompt)


@app.delete('/admin/policies/{slug}/draft', status_code=204, dependencies=admin)
def admin_delete_draft(slug: str = SlugPath):
    _found(_db(policies.delete_draft, slug), f'bản nháp {slug}')


@app.get('/admin/policies/{slug}/checks', response_model=list[CheckIn], dependencies=admin)
def admin_get_checks(slug: str = SlugPath):
    return _db(policies.get_checks, slug)


@app.put('/admin/policies/{slug}/checks', response_model=list[CheckIn], dependencies=admin)
def admin_set_checks(checks: list[CheckIn], slug: str = SlugPath):
    """Thay bộ câu hỏi kiểm tra. Đổi nội dung chính sách (vd 30 ngày -> 15 ngày) thì sửa must_contain ở đây trước."""
    if len(checks) > 100:
        raise HTTPException(422, 'Tối đa 100 câu hỏi kiểm tra mỗi tài liệu')
    return _db(policies.set_checks, slug, [c.model_dump() for c in checks])


@app.post('/admin/policies/{slug}/draft/check', dependencies=admin)
def admin_check_draft(request: Request, slug: str = SlugPath):
    """Agent thử bản nháp với câu hỏi mẫu (không ghi vào Qdrant, không gọi LLM): mỗi câu, search_policy có lấy ra
    tài liệu này không và đoạn lấy ra có đủ must_contain không. {passed, results: [{question, retrieved, missing, passed}]}"""
    try:
        return _db(policies.check_draft, request.app.state.agent_http, slug)
    except httpx.HTTPError as e:
        raise HTTPException(503, f'Agent chưa kiểm tra được bản nháp ({type(e).__name__})')


@app.post('/admin/policies/{slug}/publish', response_model=PolicyOut, dependencies=admin)
def admin_publish(request: Request, req: PublishIn | None = None, slug: str = SlugPath):
    """Xuất bản bản nháp đã đạt kiểm tra: thành version mới, ghi lịch sử, đẩy sang agent."""
    req = req or PublishIn()
    return _notify(request, _db(policies.publish, slug, req.by, req.note))


@app.get('/admin/policies/{slug}/versions', dependencies=admin)
def admin_versions(slug: str = SlugPath):
    """Lịch sử xuất bản (mới nhất trước): version, người, lý do, lúc xuất bản."""
    return _db(policies.versions, slug)


@app.get('/admin/policies/{slug}/versions/{version}', dependencies=admin)
def admin_get_version(version: int, slug: str = SlugPath):
    return _found(_db(policies.get_version, slug, version), f'{slug} version {version}')


@app.post('/admin/policies/{slug}/rollback', response_model=PolicyOut, dependencies=admin)
def admin_rollback(req: RollbackIn, request: Request, slug: str = SlugPath):
    """Quay lại 1 version cũ ngay (thành version mới, không kiểm tra lại): dùng khi bản vừa xuất bản bị lỗi."""
    return _notify(request, _db(policies.rollback, slug, req.version, req.by, req.note))


@app.delete('/admin/policies/{slug}', response_model=PolicyOut, dependencies=admin)
def admin_delete_policy(request: Request, slug: str = SlugPath):
    """Gỡ tài liệu khỏi agent (khách không còn thấy). Lịch sử và câu hỏi kiểm tra giữ lại."""
    doc = _found(_db(policies.get, slug), f'chính sách {slug}')
    _db(policies.delete, slug)
    return _notify(request, doc)


@app.post('/admin/policies/sync', dependencies=admin)
def admin_sync_policies(request: Request):
    """Đồng bộ ngay (không chờ lần đối soát theo giờ): {pushed: [slug], removed: [slug]}."""
    try:
        response = request.app.state.agent_http.post('/webhooks/policy-reconcile', timeout=300)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        raise HTTPException(503, f'Chưa đồng bộ được sang agent ({type(e).__name__})')
