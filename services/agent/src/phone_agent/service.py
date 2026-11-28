# HTTP API của agent: be (services/be) gọi vào đây, hệ thống khác (web khác, Zalo, Messenger...) cũng tích hợp được.
# Agent giữ lịch sử hội thoại + giỏ hàng theo thread_id (Postgres riêng của agent, xem store.py). Bên ngoài chỉ đọc state và ghi qua vài
# sự kiện cố định (đặt hàng, hủy, mở phiên): không có endpoint ghi state tùy ý vì giá trong giỏ được dùng để tạo đơn.
import asyncio
import json
import logging
import re
import secrets
import threading
import time
import uuid
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass

import groq
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Path, Request
from fastapi.responses import StreamingResponse
from langchain_core.exceptions import ModelError
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphRecursionError
from openrouter import errors as openrouter_errors
from phone_common import pii
from phone_common.llm import is_rate_limited
from phone_common.textnorm import normalize_user_text
from pydantic import BaseModel, Field, field_validator

from phone_agent import policy_sync, store
from phone_agent.agent import build_agent
from phone_agent.config import (
    AGENT_API_KEY,
    AGENT_CLIENT_KEYS,
    AGENT_DATABASE_URL,
    AGENT_IMPL,
    CLIENT_RATE_PER_MIN,
)
from phone_agent.graph import build_graph
from phone_agent.vector_db import check_policy, count_documents, get_vector_store

log = logging.getLogger('uvicorn.error')

# Lỗi của LLM/agent được trả thành 429 (quá tải) hoặc 502; lỗi khác là lỗi code -> 500
LLM_ERRORS = (groq.APIError, openrouter_errors.OpenRouterError, openrouter_errors.NoResponseError,
              ModelError, GraphRecursionError)
RATE_LIMIT_DETAIL = 'Dạ, hệ thống đang quá tải, anh/chị thử lại sau ít phút giúp em nha.'

# Dòng trạng thái hiện trên giao diện khi agent đang chạy tool (/chat/stream)
TOOL_STATUS = {
    'search_products': 'Đang tìm sản phẩm phù hợp…',
    'get_product_details': 'Đang xem giá và tồn kho…',
    'find_accessories': 'Đang tìm phụ kiện tương thích…',
    'search_policy': 'Đang tra chính sách cửa hàng…',
    'web_search': 'Đang tìm thông tin trên web…',
    'add_to_cart': 'Đang cập nhật giỏ hàng…',
    'remove_from_cart': 'Đang cập nhật giỏ hàng…',
    'show_order_form': 'Đang mở form đặt hàng…',
    'find_cheaper_alternatives': 'Đang tìm máy giá mềm hơn…',
    'get_promotions': 'Đang xem khuyến mãi…',
}
THINKING_STATUS = 'Em đang cân nhắc…'
# Giới hạn của API chính sách phía agent (hợp đồng với be, xem README): be giữ giới hạn riêng không vượt quá các mức này
SLUG = r'^[a-z0-9][a-z0-9-]*\.md$'
MAX_CONTENT = 100_000
# Khách không nhắn 15 phút: xóa cuộc trò chuyện (giao diện cũng tự mở phiên mới sau 15 phút, index.html)
SESSION_TTL_S = 15 * 60
# Không có AGENT_DATABASE_URL (store.pool là None): giữ _last_seen, _busy trong RAM, chỉ đúng khi chạy 1 process
_last_seen: dict[str, float] = {}  # thread_id -> lúc khách tương tác gần nhất (time.monotonic)
# Ghi lịch sử như lượt trả lời của agent; agent.py (create_agent) không có node track_reply
AS_NODE = 'track_reply' if AGENT_IMPL == 'graph' else None


# Mỗi thread chỉ chạy 1 lượt một lúc: 2 lượt song song cùng đọc rồi cùng ghi state làm lệch giỏ hàng, lịch sử
# (khách bấm gửi liên tục, script gọi thẳng API). Có database thì giữ bằng 1 câu SQL atomic (đúng cả khi nhiều replica);
# không có thì Lock trong process (endpoint sync chạy trong threadpool).
# Quá TURN_TIMEOUT_S coi như lượt cũ đã chết (lỡ không nhả được) để thread không bị khóa mãi.
TURN_TIMEOUT_S = 300
BUSY_DETAIL = 'Dạ, em đang trả lời tin trước, anh/chị chờ em xíu rồi nhắn tiếp nha.'
_busy: dict[str, float] = {}  # thread_id -> lúc bắt đầu lượt đang chạy
_busy_lock = threading.Lock()


def _claim(thread_id: str) -> None:
    if store.pool is not None:
        if not store.claim(thread_id, TURN_TIMEOUT_S):
            raise HTTPException(409, BUSY_DETAIL)
        return
    now = time.monotonic()
    with _busy_lock:
        if now - _busy.get(thread_id, -TURN_TIMEOUT_S) < TURN_TIMEOUT_S:
            raise HTTPException(409, BUSY_DETAIL)
        _busy[thread_id] = now


def _release(thread_id: str) -> None:
    if store.pool is not None:
        store.release(thread_id)
        return
    with _busy_lock:
        _busy.pop(thread_id, None)


def _touch(thread_id: str) -> None:
    if store.pool is not None:
        store.touch(thread_id)
    else:
        _last_seen[thread_id] = time.monotonic()


def _delete_thread(agent, thread_id: str) -> None:
    if store.pool is not None:
        store.forget(thread_id)
    else:
        _last_seen.pop(thread_id, None)
    checkpointer = getattr(agent, 'checkpointer', None)
    if checkpointer is not None:
        checkpointer.delete_thread(thread_id)


def expire_threads(agent, now: float | None = None) -> list[str]:
    """Xóa các cuộc trò chuyện quá SESSION_TTL_S không có tương tác (checkpointer không tự xóa)."""
    if store.pool is not None:
        expired = store.expired(SESSION_TTL_S, TURN_TIMEOUT_S)
    else:
        cutoff = (time.monotonic() if now is None else now) - SESSION_TTL_S
        expired = [t for t, seen in list(_last_seen.items()) if seen < cutoff]
        for thread_id in expired:
            _last_seen.pop(thread_id, None)
    checkpointer = getattr(agent, 'checkpointer', None)
    if checkpointer is not None:
        for thread_id in expired:
            checkpointer.delete_thread(thread_id)
    return expired


async def _expire_loop(app: FastAPI):
    while True:
        await asyncio.sleep(60)
        try:
            if n := len(expire_threads(app.state.agent)):
                log.info('Đã xóa %d cuộc trò chuyện quá %d phút không hoạt động', n, SESSION_TTL_S // 60)
            _prune_rate()
        except Exception as e:
            log.error('Lỗi khi xóa cuộc trò chuyện quá hạn', exc_info=e)


def prepare_vector_db(retries: int = 30, delay: float = 2.0) -> int:
    # Qdrant có thể khởi động chậm hơn agent khi chạy bằng docker compose -> thử lại vài lần
    for attempt in range(1, retries + 1):
        try:
            get_vector_store()  # tạo collection nếu chưa có
            return count_documents()
        except Exception as e:
            if attempt == retries:
                raise
            log.warning('Chưa kết nối được Qdrant (%s), thử lại sau %ss (%d/%d)',
                        type(e).__name__, delay, attempt, retries)
            time.sleep(delay)


async def _reconcile_loop():
    while True:
        try:
            await asyncio.to_thread(policy_sync.reconcile)
        except Exception:
            log.exception('Chưa đối soát được chính sách từ be')
        await asyncio.sleep(policy_sync.RECONCILE_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    pii.install_log_filter()  # che số điện thoại, email trong log (handler của uvicorn đã có lúc này)
    clients = _clients()   # sai cú pháp AGENT_CLIENT_KEYS thì dừng ngay lúc khởi động
    if clients:
        log.info('Nguồn được gọi agent: %s', ', '.join(f'{c.name} ({c.rate_per_min or "không giới hạn"}/phút)'
                                                       for c in clients))
    else:
        log.warning('Chưa đặt AGENT_API_KEY / AGENT_CLIENT_KEYS: ai gọi tới được cổng agent đều dùng được, '
                    'chỉ để agent trong mạng nội bộ')
    n = await asyncio.to_thread(prepare_vector_db)
    log.info('Qdrant sẵn sàng: %d tài liệu', n)
    # Có AGENT_DATABASE_URL: lịch sử + giỏ hàng trong Postgres riêng của agent (restart không mất, chạy nhiều replica được).
    # Không có: InMemorySaver trong RAM -> mất khi restart, chỉ chạy 1 worker.
    # graph: có trạng thái hội thoại + giỏ hàng (graph.py); agent: create_agent không có state riêng (agent.py)
    checkpointer = await asyncio.to_thread(store.open_checkpointer, AGENT_DATABASE_URL)
    if store.pool is None:
        log.warning('Chưa đặt AGENT_DATABASE_URL: hội thoại nằm trong RAM, mất khi restart, chỉ chạy được 1 worker')
    build = build_graph if AGENT_IMPL == 'graph' else build_agent
    app.state.agent = build(checkpointer=checkpointer)
    log.info('Dùng agent: %s', AGENT_IMPL)
    expiring = asyncio.create_task(_expire_loop(app))
    reconciling = asyncio.create_task(_reconcile_loop())
    try:
        yield
    finally:
        reconciling.cancel()
        expiring.cancel()
        with suppress(asyncio.CancelledError):
            await reconciling
        with suppress(asyncio.CancelledError):
            await expiring
        store.close()


@dataclass(frozen=True)
class Client:
    name: str          # nguồn gọi: tiền tố thread_id, ghi trong log
    key: str
    rate_per_min: int  # 0: không giới hạn


def parse_clients(be_key: str, spec: str, default_rate: int) -> list[Client]:
    """be (AGENT_API_KEY, không giới hạn) + các hệ thống tích hợp (AGENT_CLIENT_KEYS: "tên:key[:giới hạn/phút]", ...)."""
    clients = [Client('be', be_key, 0)] if be_key else []
    for item in filter(None, (s.strip() for s in spec.split(','))):
        name, key, *rate = item.split(':')
        if (not re.fullmatch(r'[a-z0-9_-]{1,30}', name) or not key or len(rate) > 1
                or (rate and not rate[0].isdigit()) or any(c.name == name or c.key == key for c in clients)):
            raise ValueError(f'AGENT_CLIENT_KEYS sai ở mục "{name}": cần "tên:key[:giới hạn/phút]", '
                             'tên chữ thường, không trùng tên hoặc key với bên khác')
        clients.append(Client(name, key, int(rate[0]) if rate else default_rate))
    return clients


def _clients() -> list[Client]:
    return parse_clients(AGENT_API_KEY, AGENT_CLIENT_KEYS, CLIENT_RATE_PER_MIN)


# Không có database: đếm request trong RAM (chỉ đúng khi 1 process). (client, phút) -> số request
_hits: dict[tuple[str, int], int] = {}
_hits_lock = threading.Lock()


def _over_limit(client: Client) -> bool:
    if not client.rate_per_min:
        return False
    if store.pool is not None:
        n = store.hit(client.name)
    else:
        key = (client.name, int(time.time() // 60))
        with _hits_lock:
            n = _hits[key] = _hits.get(key, 0) + 1
    return n > client.rate_per_min


def _prune_rate() -> None:
    if store.pool is not None:
        store.prune_rate()
        return
    minute = int(time.time() // 60)
    with _hits_lock:
        for key in [k for k in _hits if k[1] < minute - 1]:
            del _hits[key]


HEALTH_PATHS = ('/health', '/v1/health')


def check_api_key(request: Request) -> None:
    """Xác định nguồn gọi theo key (request.state.client), ghi log, giới hạn tốc độ theo key.
    Chưa đặt key nào: client None (chỉ dùng trong mạng nội bộ)."""
    request.state.client = None
    clients = _clients()
    if not clients or request.url.path in HEALTH_PATHS:
        return
    given = request.headers.get('authorization', '').removeprefix('Bearer ')
    client = next((c for c in clients if secrets.compare_digest(given.encode(), c.key.encode())), None)
    if client is None:
        raise HTTPException(401, 'Sai hoặc thiếu API key của agent')
    request.state.client = client
    log.info('%s %s client=%s', request.method, request.url.path, client.name)
    if _over_limit(client):
        raise HTTPException(429, f'Quá {client.rate_per_min} request/phút cho key "{client.name}"',
                            headers={'Retry-After': '60'})


def be_only(request: Request) -> None:
    # Đồng bộ, kiểm tra chính sách: chỉ be gọi (hệ thống tích hợp khác không được kích hoạt embed lại)
    client = request.state.client
    if client is not None and client.name != 'be':
        raise HTTPException(403, 'Endpoint này chỉ dành cho be')


def _tid(request: Request, thread_id: str) -> str:
    """thread_id lưu trong agent = '<nguồn>:<thread_id>' khi có key: mỗi hệ thống chỉ thấy, ghi, xóa hội thoại của
    mình, và xóa được toàn bộ dữ liệu theo nguồn. Không có key (mạng nội bộ): giữ nguyên."""
    client = request.state.client
    return f'{client.name}:{thread_id}' if client else thread_id


app = FastAPI(title='Phone Sales Agent', version='1.0.0', lifespan=lifespan, dependencies=[Depends(check_api_key)])
# Mọi endpoint khai báo trên router, gắn ở cuối file dưới /v1 (và đường dẫn cũ không có /v1 để tương thích)
router = APIRouter()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    thread_id: str | None = Field(default=None, max_length=100)

    @field_validator('message')
    @classmethod
    def normalize(cls, v):
        # NFC, dấu kiểu cũ, viết tắt ("ko", "dc", "20tr"...): luật từ khóa, model và lịch sử cùng đọc 1 dạng.
        # Chạy sau kiểm tra độ dài trên chuỗi gốc ("x" * 1001 rút gọn ký tự lặp sẽ thành "x").
        # Che số điện thoại, email: không tới LLM, LangSmith, lịch sử hội thoại (thông tin nhận hàng nhập ở form)
        v = pii.redact(normalize_user_text(v))
        if not v:
            raise ValueError('Tin nhắn trống')
        return v


class ChatResponse(BaseModel):
    thread_id: str
    answer: str
    tools: list[str]
    # Trạng thái hội thoại (chỉ có khi AGENT_IMPL=graph): bước bán hàng, giỏ hàng, có cần chuyển nhân viên
    stage: str | None = None
    cart: list[dict] = []
    handoff: bool = False
    order_form: bool = False  # hiện form đặt hàng dưới câu trả lời (agent gọi show_order_form)


class ThreadState(BaseModel):
    thread_id: str
    stage: str | None = None
    cart: list[dict] = []
    order_form: bool = False
    orders: list[str] = []


class NewThread(BaseModel):
    thread_id: str | None = Field(default=None, max_length=100)
    greeting: str | None = Field(default=None, max_length=2000)  # lời chào ghi vào lịch sử như agent đã nói
    customer: dict | None = None                                  # hồ sơ khách quen (chỉ dùng khi AGENT_IMPL=graph)
    end_thread_id: str | None = Field(default=None, max_length=100)  # cuộc trò chuyện cũ cần xóa


class PolicyUpdated(BaseModel):
    slug: str = Field(pattern=SLUG, max_length=255)


class CheckQuestion(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    must_contain: list[list[str]] = []   # mỗi nhóm: chỉ cần 1 cách viết xuất hiện, vd [["30 ngày"], ["7", "bảy"]]


class PolicyCheck(BaseModel):
    content_md: str = Field(min_length=1, max_length=MAX_CONTENT)
    questions: list[CheckQuestion] = Field(max_length=100)


class OrderPlaced(BaseModel):
    order_id: str = Field(min_length=1, max_length=50)
    answer: str = Field(min_length=1, max_length=5000)  # tóm tắt đơn + lời cảm ơn, ghi vào lịch sử


class OrderCancelled(BaseModel):
    answer: str = Field(min_length=1, max_length=2000)


def _config(thread_id: str) -> dict:
    return {'configurable': {'thread_id': thread_id}}


@router.get('/health')
def health():
    try:
        return {'status': 'ok', 'documents': count_documents()}
    except Exception as e:
        raise HTTPException(503, f'Qdrant không sẵn sàng: {type(e).__name__}')


def _llm_failure(e: Exception) -> tuple[int, str]:
    if is_rate_limited(e):
        # Ghi lại loại giới hạn (token/phút, token/ngày...) để biết chờ hay phải nâng gói
        log.warning('LLM rate limit: %s', e)
        return 429, RATE_LIMIT_DETAIL
    log.error('Lỗi khi chạy agent', exc_info=e)
    return 502, f'Không tạo được câu trả lời ({type(e).__name__}).'


# Khách gửi đi gửi lại đúng 1 câu ("cho tui xem giá" x10) là spam/lạc đề: trả lời sẵn, không tốn lượt LLM.
# Lần 1-2: trả lời bình thường (gửi lại 1 lần vì lượt trước lỗi, chưa ưng là chuyện thường).
# Lần 3: nhắc nhẹ khách nói rõ hơn. Lần 4 trở đi: chuyển nhân viên (bật handoff như khi khách đòi gặp người thật).
# So sau khi bỏ hoa thường, dấu câu, khoảng trắng.
NUDGE_AT, HANDOFF_AT = 3, 4
MIN_WORDS = 3
NUDGE = ('Dạ, câu này anh/chị vừa hỏi, em đã trả lời ở trên rồi ạ. Anh/chị nói rõ hơn giúp em '
         '(vd tên máy, dung lượng, tầm giá) để em tư vấn đúng ý nha.')
SPAM_HANDOFF = ('Dạ, có vẻ em chưa giúp được đúng ý anh/chị. Em đã chuyển cho nhân viên hỗ trợ, '
                'anh/chị chờ chút nhân viên sẽ liên hệ ạ.')


def _key(text: str) -> str:
    # Chỉ có emoji/dấu câu ("👍", "???"): không có chữ để so -> so nguyên văn
    return ' '.join(re.findall(r'\w+', text.casefold())) or text.strip()


def _times_sent(messages: list, message: str) -> int:
    """Tin này là lần thứ mấy liên tiếp khách gửi đúng câu đó (tính cả tin này).
    Câu dưới MIN_WORDS từ ("ok", "có", "vâng", "👍") không tính: khách đáp lại các câu hỏi khác nhau của bot
    ("lấy màu đen không?", "thêm ốp không?") là bình thường, không phải spam."""
    key = _key(message)
    if len(key.split()) < MIN_WORDS:
        return 1
    n = 1
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            if _key(m.text) != key:
                break
            n += 1
    return n


def _answer_spam(agent, config: dict, message: str) -> dict | None:
    """Khách lặp lại 1 câu: ghi tin của khách + câu trả lời sẵn vào lịch sử, trả state mới. Không lặp thì None."""
    n = _times_sent(agent.get_state(config).values.get('messages', []), message)
    if n < NUDGE_AT:
        return None
    update = {'messages': [HumanMessage(message), AIMessage(NUDGE if n < HANDOFF_AT else SPAM_HANDOFF)]}
    if n >= HANDOFF_AT and AGENT_IMPL == 'graph':
        update |= {'handoff': True, 'stage': 'handoff'}
    log.info('Thread %s gửi lặp 1 câu %d lần: %s', config['configurable']['thread_id'], n,
             'chuyển nhân viên' if n >= HANDOFF_AT else 'nhắc nhẹ')
    agent.update_state(config, update, as_node=AS_NODE)
    return agent.get_state(config).values


def _start_turn(req: ChatRequest, request: Request):
    # thread_id: id bên gọi thấy; key: id lưu trong agent (có tiền tố nguồn)
    agent = request.app.state.agent
    thread_id = req.thread_id or str(uuid.uuid4())
    key = _tid(request, thread_id)
    _touch(key)
    return agent, thread_id, key, _config(key)


def _n_messages(agent, config: dict) -> int:
    # Agent trả về cả lịch sử của thread -> nhớ số message trước lượt này để chỉ lấy message mới
    return len(agent.get_state(config).values.get('messages', []))


def _response(thread_id: str, values: dict, n_before: int) -> ChatResponse:
    new_messages = values['messages'][n_before:]
    tools = [tc['name'] for m in new_messages for tc in (getattr(m, 'tool_calls', None) or [])]
    return ChatResponse(thread_id=thread_id, answer=values['messages'][-1].text, tools=tools,
                        stage=values.get('stage'), cart=values.get('cart') or [], handoff=bool(values.get('handoff')),
                        order_form=bool(values.get('order_form')))


@router.post('/chat', response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    agent, thread_id, key, config = _start_turn(req, request)
    _claim(key)
    try:  # mọi bước sau khi giữ thread nằm trong try: lỗi gì cũng nhả
        n_before = _n_messages(agent, config)
        res = _answer_spam(agent, config, req.message) or \
            agent.invoke({'messages': [{'role': 'user', 'content': req.message}]}, config)
    except LLM_ERRORS as e:
        raise HTTPException(*_llm_failure(e))
    finally:
        _release(key)
    return _response(thread_id, res, n_before)


def _sse(event: dict) -> str:
    return f'data: {json.dumps(event, ensure_ascii=False)}\n\n'


def _stream_events(agent, message: str, thread_id: str, key: str, config: dict):
    # Giữ thread ngay trong generator: generator bị bỏ trước khi chạy (khách ngắt kết nối sớm) thì cũng chưa giữ gì,
    # không bị khóa 5 phút. Đang có lượt khác -> sự kiện error status 409 (giao diện hiện detail)
    try:
        _claim(key)
    except HTTPException as e:
        yield _sse({'type': 'error', 'status': e.status_code, 'detail': e.detail})
        return
    try:
        yield from _turn_events(agent, message, thread_id, config)
    finally:
        _release(key)  # cả khi khách đóng tab giữa chừng (generator bị close)


def _turn_events(agent, message: str, thread_id: str, config: dict):
    yield _sse({'type': 'start', 'thread_id': thread_id})
    status = None  # dòng trạng thái đã gửi gần nhất, tránh gửi trùng
    try:
        n_before = _n_messages(agent, config)
        if (values := _answer_spam(agent, config, message)) is not None:
            yield _sse({'type': 'done', **_response(thread_id, values, n_before).model_dump()})
            return
        stream = agent.stream({'messages': [{'role': 'user', 'content': message}]}, config,
                              stream_mode=['messages', 'updates'])
        for mode, data in stream:
            if mode == 'messages':
                chunk, meta = data
                if meta.get('langgraph_node') != 'model' or not isinstance(chunk, AIMessage):
                    continue  # kết quả tool, node khác: không gửi cho khách
                if chunk.text:
                    yield _sse({'type': 'token', 'text': chunk.text})
                elif chunk.additional_kwargs.get('reasoning_content') and status != THINKING_STATUS:
                    # Model có thinking (NVIDIA) suy nghĩ vài giây trước khi viết: chỉ báo trạng thái,
                    # không gửi nội dung suy nghĩ (ghi chú nội bộ, có thể lộ chi tiết prompt)
                    status = THINKING_STATUS
                    yield _sse({'type': 'status', 'text': status})
                continue
            for update in (data or {}).values():
                messages = (update.get('messages') or []) if isinstance(update, dict) else []
                for m in messages:
                    if isinstance(m, AIMessage) and m.tool_calls:
                        # Model quyết định gọi tool: phần chữ vừa stream (nếu có) chỉ là câu dẫn, bỏ đi
                        yield _sse({'type': 'reset'})
                        for tc in m.tool_calls:
                            status = TOOL_STATUS.get(tc['name'], 'Đang tra cứu…')
                            yield _sse({'type': 'status', 'text': status})
    except LLM_ERRORS as e:
        status, detail = _llm_failure(e)
        yield _sse({'type': 'error', 'status': status, 'detail': detail})
        return
    except Exception as e:
        log.error('Lỗi khi stream câu trả lời', exc_info=e)
        yield _sse({'type': 'error', 'status': 500, 'detail': 'Dạ, hệ thống đang có lỗi, anh/chị thử lại giúp em nha.'})
        return
    values = agent.get_state(config).values
    yield _sse({'type': 'done', **_response(thread_id, values, n_before).model_dump()})


@router.post('/chat/stream')
def chat_stream(req: ChatRequest, request: Request):
    """Như /chat nhưng trả Server-Sent Events để giao diện hiện chữ ngay khi LLM sinh ra.
    Mỗi sự kiện là 1 dòng `data: {json}`: start {thread_id}; status {text} khi chạy tool; token {text};
    reset (bỏ phần chữ model viết trước khi gọi tool); done {giống response của /chat}; error {status, detail}."""
    agent, thread_id, key, config = _start_turn(req, request)
    # X-Accel-Buffering: tắt buffer nếu đặt sau nginx, để chữ tới trình duyệt ngay
    return StreamingResponse(_stream_events(agent, req.message, thread_id, key, config),
                             media_type='text/event-stream',
                             headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@router.post('/threads', response_model=ThreadState)
def new_thread(req: NewThread, request: Request):
    """Mở cuộc trò chuyện mới (xóa cuộc cũ nếu có end_thread_id). greeting/customer: ghi sẵn lời chào và hồ sơ
    khách quen vào state để agent biết khách đã mua gì."""
    agent = request.app.state.agent
    if req.end_thread_id:
        _delete_thread(agent, _tid(request, req.end_thread_id))
    thread_id = req.thread_id or str(uuid.uuid4())
    key = _tid(request, thread_id)
    _touch(key)
    values = {'messages': [AIMessage(req.greeting)]} if req.greeting else {}
    if req.customer and AGENT_IMPL == 'graph':
        values['customer'] = req.customer
    if values:
        agent.update_state(_config(key), values, as_node=AS_NODE)
    return ThreadState(thread_id=thread_id)


@router.get('/threads/{thread_id}', response_model=ThreadState)
def get_thread(thread_id: str, request: Request):
    """Giỏ hàng, bước bán hàng của cuộc trò chuyện (be đọc giỏ ở đây để tính giá, tạo đơn)."""
    key = _tid(request, thread_id)
    _touch(key)
    values = request.app.state.agent.get_state(_config(key)).values
    return ThreadState(thread_id=thread_id, stage=values.get('stage'), cart=values.get('cart') or [],
                       order_form=bool(values.get('order_form')), orders=values.get('orders') or [])


@router.delete('/threads/{thread_id}', status_code=204)
def delete_thread(thread_id: str, request: Request):
    _delete_thread(request.app.state.agent, _tid(request, thread_id))


@router.delete('/threads')
def delete_source_threads(request: Request):
    """Xóa mọi hội thoại của nguồn đang gọi (theo key): ngừng tích hợp, hoặc xóa dữ liệu theo yêu cầu. {deleted: số thread}"""
    prefix = _tid(request, '')
    if not prefix:
        raise HTTPException(400, 'Chỉ dùng được khi agent bật API key (mỗi nguồn 1 key)')
    ids = store.threads_with_prefix(prefix) if store.pool is not None else \
        [t for t in list(_last_seen) if t.startswith(prefix)]
    for thread_id in ids:
        _delete_thread(request.app.state.agent, thread_id)
    return {'deleted': len(ids)}


@router.post('/threads/{thread_id}/order-placed', status_code=204)
def order_placed(thread_id: str, req: OrderPlaced, request: Request):
    """Đơn đã lưu: ghi vào lịch sử để lượt sau agent biết, làm trống giỏ để khách mua tiếp không bị cộng dồn."""
    key = _tid(request, thread_id)
    _touch(key)
    request.app.state.agent.update_state(
        _config(key),
        {'messages': [HumanMessage('Đã gửi thông tin đặt hàng'), AIMessage(req.answer)],
         'cart': [], 'order_form': False, 'orders': [req.order_id], 'cross_sell': None, 'wanted': None},
        as_node=AS_NODE)


@router.post('/threads/{thread_id}/order-cancelled', status_code=204)
def order_cancelled(thread_id: str, req: OrderCancelled, request: Request):
    """Khách hủy form đặt hàng: ghi câu trả lời cố định, làm trống giỏ để agent không mời chốt đơn / bán kèm tiếp."""
    key = _tid(request, thread_id)
    _touch(key)
    request.app.state.agent.update_state(
        _config(key),
        {'messages': [HumanMessage('Hủy đặt hàng'), AIMessage(req.answer)],
         'cart': [], 'order_form': False, 'cross_sell': None, 'wanted': None},
        as_node=AS_NODE)


@router.post('/webhooks/policy-updated', status_code=204, dependencies=[Depends(be_only)])
async def policy_updated(event: PolicyUpdated):
    try:
        await asyncio.to_thread(policy_sync.refresh, event.slug)
    except Exception as exc:
        log.exception('Chưa cập nhật được chính sách %s', event.slug)
        raise HTTPException(503, 'Policy refresh failed') from exc


@router.post('/webhooks/policy-reconcile', dependencies=[Depends(be_only)])
async def reconcile_policies():
    try:
        return await asyncio.to_thread(policy_sync.reconcile)
    except Exception as exc:
        log.exception('Chưa đối soát được chính sách từ be')
        raise HTTPException(503, 'Policy reconciliation failed') from exc


@router.post('/policies/{slug}/check', dependencies=[Depends(be_only)])
def check_policy_draft(req: PolicyCheck, slug: str = Path(pattern=SLUG)):
    """Thử bản nháp với các câu hỏi mẫu, không ghi vào Qdrant: {passed, results: [{question, retrieved, missing, passed}]}."""
    try:
        return check_policy(slug, req.content_md, [q.model_dump() for q in req.questions])
    except ValueError as e:
        raise HTTPException(422, str(e))


# /v1: API có phiên bản cho hệ thống tích hợp (quy tắc thay đổi: README, mục "API của agent").
# Đường dẫn cũ không có /v1 vẫn chạy (không hiện trong /docs) để bên đang dùng chuyển dần, sẽ bỏ ở bản sau.
app.include_router(router, prefix='/v1')
app.include_router(router, include_in_schema=False)
