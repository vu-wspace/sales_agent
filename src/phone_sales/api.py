"""Service be: API công khai cho giao diện (qua service fe). Kiểm tra, chuẩn hóa, che PII tin nhắn rồi chuyển
lượt chat sang service agent (agent_api.py); tự lo phiên khách quen, đơn hàng, mã khuyến mãi, email xác nhận."""
import json
import logging
import re
import uuid
from contextlib import asynccontextmanager
from typing import Literal

import httpx
import psycopg
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from phone_sales import mailer, orders, pii, promotions
from phone_sales.config import AGENT_URL, PII_KEY, SMTP_HOST
from phone_sales.schemas import ChatRequest, ChatResponse

log = logging.getLogger('uvicorn.error')

AGENT_DOWN = 'Dạ, hệ thống tư vấn đang khởi động hoặc gặp sự cố, anh/chị thử lại sau ít phút giúp em nha.'
# Lượt chat có thể chạy vài phút (nhiều tool, LLM dự phòng): đọc chờ lâu, kết nối thì báo lỗi nhanh
http = httpx.Client(base_url=AGENT_URL, timeout=httpx.Timeout(600, connect=5))

def _agent(method: str, path: str, **kwargs) -> dict:
    """Gọi service agent; lỗi của agent (429, 502...) trả lại nguyên mã và lời nhắn, không gọi được thì 503."""
    try:
        r = http.request(method, path, **kwargs)
    except httpx.HTTPError as e:
        log.error('Không gọi được service agent %s %s: %s', method, path, type(e).__name__)
        raise HTTPException(503, AGENT_DOWN)
    if r.status_code >= 400:
        try:
            detail = r.json()['detail']
        except (ValueError, KeyError, TypeError):
            detail = AGENT_DOWN
        raise HTTPException(r.status_code, detail)
    return r.json()

def _cart(thread_id: str) -> list[dict]:
    # Giỏ hàng lấy từ trạng thái hội thoại trong service agent, không tin giỏ do trình duyệt gửi lên
    return _agent('GET', '/threads/cart', params={'thread_id': thread_id})['cart']

class OrderRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=100)
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
    thread_id: str = Field(min_length=1, max_length=100)
    promo_code: str | None = Field(default=None, max_length=50)

class QuoteResponse(BaseModel):
    subtotal: int
    discount: int
    total: int
    promo: str | None = None         # mã (hoặc mô tả khuyến mãi tự động) đang áp dụng
    promo_text: str | None = None    # mô tả đầy đủ: mức giảm, phạm vi, điều kiện, hạn

class CancelRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=100)

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

GREETING = ('Dạ, em chào anh/chị. Bên em tư vấn iPhone, Samsung Galaxy, phụ kiện đi kèm, giá và tình trạng hàng, '
            'cùng chính sách đổi trả, giao hàng, trả góp và bảo hành. Anh/chị cần gì cứ hỏi em nha.')

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
    yield

app = FastAPI(title='Phone Sales Agent', version='0.1.0', lifespan=lifespan)

@app.get('/health')
def health():
    # Trạng thái của service agent (Qdrant, số đoạn chính sách); agent chưa lên thì 503
    return _agent('GET', '/health')

@app.post('/chat', response_model=ChatResponse)
def chat(req: ChatRequest):
    return _agent('POST', '/chat', json=req.model_dump())

def _sse(event: dict) -> str:
    return f'data: {json.dumps(event, ensure_ascii=False)}\n\n'

def _relay_stream(body: dict):
    try:
        with http.stream('POST', '/chat/stream', json=body) as r:
            if r.status_code != 200:
                r.read()
                try:
                    detail = r.json()['detail']
                except (ValueError, KeyError, TypeError):
                    detail = AGENT_DOWN
                yield _sse({'type': 'error', 'status': r.status_code, 'detail': detail})
                return
            for line in r.iter_lines():
                if line:
                    yield f'{line}\n\n'
    except httpx.HTTPError as e:
        log.error('Mất kết nối tới service agent khi stream: %s', type(e).__name__)
        yield _sse({'type': 'error', 'status': 503, 'detail': AGENT_DOWN})

@app.post('/chat/stream')
def chat_stream(req: ChatRequest):
    """Như /chat nhưng trả Server-Sent Events để giao diện hiện chữ ngay khi LLM sinh ra.
    Mỗi sự kiện là 1 dòng `data: {json}`: start {thread_id}; status {text} khi chạy tool; token {text};
    reset (bỏ phần chữ model viết trước khi gọi tool); done {giống response của /chat}; error {status, detail}."""
    # thread_id mới tạo ở đây để start có ngay mã, kể cả khi agent lỗi trước khi gửi gì
    body = {**req.model_dump(), 'thread_id': req.thread_id or str(uuid.uuid4())}
    # X-Accel-Buffering: tắt buffer của nginx (service fe), để chữ tới trình duyệt ngay
    return StreamingResponse(_relay_stream(body), media_type='text/event-stream',
                             headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

@app.post('/order', response_model=OrderResponse)
def order(req: OrderRequest, background: BackgroundTasks):
    """Khách gửi form đặt hàng: tạo đơn từ giỏ hàng của thread (không tin giỏ do trình duyệt gửi lên),
    trả tóm tắt đơn lấy từ database + 1-2 câu cảm ơn do LLM viết, và ghi vào lịch sử hội thoại.
    Email xác nhận gửi sau khi đã trả kết quả (BackgroundTasks, mailer.send_order_email) nên khách không phải chờ."""
    cart = _cart(req.thread_id)
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
    # Lượt sau agent biết đơn đã đặt; giỏ hàng làm trống để khách mua tiếp không bị cộng dồn.
    # Đơn đã lưu: agent lỗi thì chỉ ghi log, vẫn trả thành công để khách không gửi lại form và đặt trùng đơn
    try:
        _agent('POST', '/threads/update', json={
            'thread_id': req.thread_id, 'human': 'Đã gửi thông tin đặt hàng', 'ai': answer,
            'values': {'cart': [], 'order_form': False, 'orders': [order_id], 'cross_sell': None, 'wanted': None}})
    except HTTPException as e:
        log.error('Đơn %s đã lưu nhưng chưa ghi được vào hội thoại (%s)', order_id, e.status_code)
    return OrderResponse(thread_id=req.thread_id, order_id=order_id, answer=answer, order=orders.view(saved),
                         thanks=thanks,
                         customer_token=pii.customer_token(saved['customer_id']) if saved.get('customer_id') else None)

@app.post('/order/quote', response_model=QuoteResponse)
def order_quote(req: QuoteRequest):
    """Form đặt hàng hỏi giá trước khi gửi: tạm tính, giảm giá (mã khách nhập hoặc khuyến mãi tự động), tổng tiền.
    Tính từ giỏ hàng của thread trên server. Mã không dùng được thì 409 kèm lý do."""
    cart = _cart(req.thread_id)
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
def new_session(req: SessionRequest):
    """Kết thúc cuộc trò chuyện cũ (xóa lịch sử, giỏ hàng khỏi bộ nhớ của agent) và mở cuộc trò chuyện mới.
    Khách đã đặt hàng (customer_token hợp lệ, có đơn) thì chào lại theo tên, ghi khách quen vào trạng thái hội thoại;
    không thì chào như khách mới. Lời chào là template, không gọi LLM."""
    if req.end_thread_id:
        try:
            _agent('POST', '/threads/end', json={'thread_id': req.end_thread_id})
        except HTTPException:
            pass  # thread cũ vẫn tự hết hạn sau 15 phút; không chặn khách mở phiên mới
    thread_id = str(uuid.uuid4())
    customer_id = pii.verify_customer_token(req.customer_token)
    profile = None
    if customer_id:
        try:
            profile = orders.customer_profile(customer_id)
        except psycopg.Error as e:
            # Không đọc được database: vẫn mở phiên, chào như khách mới, giữ mã để lần sau thử lại
            log.error('Không đọc được thông tin khách quen', exc_info=e)
            return SessionResponse(thread_id=thread_id, greeting=GREETING, returning=False,
                                   customer_token=req.customer_token)
    if not profile:
        return SessionResponse(thread_id=thread_id, greeting=GREETING, returning=False)
    greeting = orders.welcome_back(profile)
    _agent('POST', '/threads/update', json={'thread_id': thread_id, 'ai': greeting, 'values': {'customer': profile}})
    return SessionResponse(thread_id=thread_id, greeting=greeting, returning=True,
                           customer_token=pii.customer_token(customer_id))

@app.post('/order/cancel', response_model=CancelResponse)
def cancel_order(req: CancelRequest):
    """Khách bấm "Hủy đặt hàng" trên form: không tạo đơn, ghi vào lịch sử hội thoại câu trả lời cố định
    (không gọi LLM) và làm trống giỏ hàng để agent không mời chốt đơn / bán kèm tiếp cho giỏ đã hủy."""
    _agent('POST', '/threads/update', json={
        'thread_id': req.thread_id, 'human': 'Hủy đặt hàng', 'ai': orders.CANCEL_MESSAGE,
        'values': {'cart': [], 'order_form': False, 'cross_sell': None, 'wanted': None}})
    return CancelResponse(thread_id=req.thread_id, answer=orders.CANCEL_MESSAGE)
