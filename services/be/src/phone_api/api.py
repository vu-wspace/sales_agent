# Backend cửa hàng: đơn hàng, phiên chat, email xác nhận. Chat chuyển tiếp sang agent service (services/agent) qua
# HTTP; giỏ hàng nằm trong state của agent nên đọc từ agent (GET /threads/{id}), không tin giỏ do trình duyệt gửi.
# Agent tắt: be vẫn chạy - mở phiên, hủy form vẫn được; chat, tính giá, đặt hàng báo 503 (không có giỏ để đọc).
import json
import logging
import re
import uuid
from contextlib import asynccontextmanager
from typing import Literal

import httpx
import psycopg
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from phone_api import mailer, orders
from phone_common import pii, promotions
from phone_common.config import AGENT_API_KEY, AGENT_URL, PII_KEY, SMTP_HOST

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
    headers = {'Authorization': f'Bearer {AGENT_API_KEY}'} if AGENT_API_KEY else {}
    # read 180s: 1 lượt chat có thể chạy vài tool + model dự phòng; connect 5s: agent tắt thì báo lỗi ngay
    app.state.agent_http = httpx.Client(base_url=AGENT_URL, headers=headers, timeout=httpx.Timeout(180, connect=5))
    log.info('Agent: %s', AGENT_URL)
    yield
    app.state.agent_http.close()


app = FastAPI(title='Phone Shop API', version='0.1.0', lifespan=lifespan)


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


@app.get('/health')
def health(request: Request):
    # be khỏe kể cả khi agent tắt; trường agent cho biết chat có dùng được không
    try:
        agent = 'ok' if request.app.state.agent_http.get('/health', timeout=3).is_success else 'error'
    except httpx.TransportError:
        agent = 'down'
    return {'status': 'ok', 'agent': agent}


@app.post('/chat')
def chat(req: ChatRequest, request: Request):
    return _agent(request, 'POST', '/chat', json=req.model_dump(exclude_none=True)).json()


@app.post('/chat/stream')
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


@app.post('/order', response_model=OrderResponse)
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


@app.post('/order/quote', response_model=QuoteResponse)
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


@app.post('/session', response_model=SessionResponse)
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


@app.post('/order/cancel', response_model=CancelResponse)
def cancel_order(req: CancelRequest, request: Request):
    """Khách bấm "Hủy đặt hàng" trên form: không tạo đơn, ghi vào lịch sử hội thoại câu trả lời cố định
    (không gọi LLM) và làm trống giỏ hàng để agent không mời chốt đơn / bán kèm tiếp cho giỏ đã hủy."""
    try:
        _agent(request, 'POST', f'/threads/{req.thread_id}/order-cancelled', json={'answer': orders.CANCEL_MESSAGE})
    except HTTPException as e:
        log.warning('Chưa ghi được việc hủy đặt hàng vào cuộc trò chuyện: %s', e.detail)
    return CancelResponse(thread_id=req.thread_id, answer=orders.CANCEL_MESSAGE)
