# HTTP API của agent: be (services/be) gọi vào đây, hệ thống khác (web khác, Zalo, Messenger...) cũng tích hợp được.
# Agent giữ lịch sử hội thoại + giỏ hàng theo thread_id (InMemorySaver). Bên ngoài chỉ đọc state và ghi qua vài
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

import groq
from fastapi import Depends, FastAPI, HTTPException, Path, Request
from fastapi.responses import StreamingResponse
from langchain_core.exceptions import ModelError
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from openrouter import errors as openrouter_errors
from phone_common import pii
from phone_common.config import AGENT_API_KEY, AGENT_IMPL
from phone_common.llm import is_rate_limited
from phone_common.policies import MAX_CONTENT, SLUG
from phone_common.textnorm import normalize_user_text
from pydantic import BaseModel, Field, field_validator

from phone_agent import policy_sync
from phone_agent.agent import build_agent
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
# Khách không nhắn 15 phút: xóa cuộc trò chuyện khỏi RAM (giao diện cũng tự mở phiên mới sau 15 phút, index.html)
SESSION_TTL_S = 15 * 60
_last_seen: dict[str, float] = {}  # thread_id -> lúc khách tương tác gần nhất (time.monotonic)
# Ghi lịch sử như lượt trả lời của agent; agent.py (create_agent) không có node track_reply
AS_NODE = 'track_reply' if AGENT_IMPL == 'graph' else None


# Mỗi thread chỉ chạy 1 lượt một lúc: 2 lượt song song cùng đọc rồi cùng ghi state làm lệch giỏ hàng, lịch sử
# (khách bấm gửi liên tục, script gọi thẳng API). Endpoint sync chạy trong threadpool nên cần Lock.
# Quá TURN_TIMEOUT_S coi như lượt cũ đã chết (lỡ không nhả được) để thread không bị khóa mãi.
TURN_TIMEOUT_S = 300
BUSY_DETAIL = 'Dạ, em đang trả lời tin trước, anh/chị chờ em xíu rồi nhắn tiếp nha.'
_busy: dict[str, float] = {}  # thread_id -> lúc bắt đầu lượt đang chạy
_busy_lock = threading.Lock()


def _claim(thread_id: str) -> None:
    now = time.monotonic()
    with _busy_lock:
        if now - _busy.get(thread_id, -TURN_TIMEOUT_S) < TURN_TIMEOUT_S:
            raise HTTPException(409, BUSY_DETAIL)
        _busy[thread_id] = now


def _release(thread_id: str) -> None:
    with _busy_lock:
        _busy.pop(thread_id, None)


def _touch(thread_id: str) -> None:
    _last_seen[thread_id] = time.monotonic()


def _delete_thread(agent, thread_id: str) -> None:
    _last_seen.pop(thread_id, None)
    checkpointer = getattr(agent, 'checkpointer', None)
    if checkpointer is not None:
        checkpointer.delete_thread(thread_id)


def expire_threads(agent, now: float | None = None) -> list[str]:
    """Xóa các cuộc trò chuyện quá SESSION_TTL_S không có tương tác (InMemorySaver không tự xóa)."""
    cutoff = (time.monotonic() if now is None else now) - SESSION_TTL_S
    expired = [t for t, seen in list(_last_seen.items()) if seen < cutoff]
    for thread_id in expired:
        _delete_thread(agent, thread_id)
    return expired


async def _expire_loop(app: FastAPI):
    while True:
        await asyncio.sleep(60)
        try:
            if n := len(expire_threads(app.state.agent)):
                log.info('Đã xóa %d cuộc trò chuyện quá %d phút không hoạt động', n, SESSION_TTL_S // 60)
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
    # Thành công: chờ RECONCILE_SECONDS. Lỗi (be chưa lên lúc deploy, tài liệu lỗi): 30s, 60s... tới RECONCILE_SECONDS,
    # để Qdrant không trống cả giờ chỉ vì agent khởi động trước be
    retry = policy_sync.RETRY_SECONDS
    while True:
        try:
            ok = not (await asyncio.to_thread(policy_sync.reconcile))['failed']
        except Exception:
            log.exception('Chưa đối soát được chính sách từ be')
            ok = False
        await asyncio.sleep(policy_sync.RECONCILE_SECONDS if ok else retry)
        retry = policy_sync.RETRY_SECONDS if ok else min(retry * 2, policy_sync.RECONCILE_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    pii.install_log_filter()  # che số điện thoại, email trong log (handler của uvicorn đã có lúc này)
    if not AGENT_API_KEY:
        log.warning('Chưa đặt AGENT_API_KEY: ai gọi tới được cổng agent đều dùng được, chỉ để agent trong mạng nội bộ')
    n = await asyncio.to_thread(prepare_vector_db)
    log.info('Qdrant sẵn sàng: %d tài liệu', n)
    # InMemorySaver: lịch sử hội thoại nằm trong RAM của process -> mất khi restart, chỉ chạy 1 worker
    # graph: có trạng thái hội thoại + giỏ hàng (graph.py); agent: create_agent không có state riêng (agent.py)
    build = build_graph if AGENT_IMPL == 'graph' else build_agent
    app.state.agent = build(checkpointer=InMemorySaver())
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


def check_api_key(request: Request) -> None:
    if not AGENT_API_KEY or request.url.path == '/health':
        return
    given = request.headers.get('authorization', '').removeprefix('Bearer ')
    if not secrets.compare_digest(given.encode(), AGENT_API_KEY.encode()):
        raise HTTPException(401, 'Sai hoặc thiếu AGENT_API_KEY')


app = FastAPI(title='Phone Sales Agent', version='0.1.0', lifespan=lifespan, dependencies=[Depends(check_api_key)])


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


@app.get('/health')
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
    agent = request.app.state.agent
    thread_id = req.thread_id or str(uuid.uuid4())
    _touch(thread_id)
    return agent, thread_id, _config(thread_id)


def _n_messages(agent, config: dict) -> int:
    # Agent trả về cả lịch sử của thread -> nhớ số message trước lượt này để chỉ lấy message mới
    return len(agent.get_state(config).values.get('messages', []))


def _response(thread_id: str, values: dict, n_before: int) -> ChatResponse:
    new_messages = values['messages'][n_before:]
    tools = [tc['name'] for m in new_messages for tc in (getattr(m, 'tool_calls', None) or [])]
    return ChatResponse(thread_id=thread_id, answer=values['messages'][-1].text, tools=tools,
                        stage=values.get('stage'), cart=values.get('cart') or [], handoff=bool(values.get('handoff')),
                        order_form=bool(values.get('order_form')))


@app.post('/chat', response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    agent, thread_id, config = _start_turn(req, request)
    _claim(thread_id)
    try:  # mọi bước sau khi giữ thread nằm trong try: lỗi gì cũng nhả
        n_before = _n_messages(agent, config)
        res = _answer_spam(agent, config, req.message) or \
            agent.invoke({'messages': [{'role': 'user', 'content': req.message}]}, config)
    except LLM_ERRORS as e:
        raise HTTPException(*_llm_failure(e))
    finally:
        _release(thread_id)
    return _response(thread_id, res, n_before)


def _sse(event: dict) -> str:
    return f'data: {json.dumps(event, ensure_ascii=False)}\n\n'


def _stream_events(agent, message: str, thread_id: str, config: dict):
    # Giữ thread ngay trong generator: generator bị bỏ trước khi chạy (khách ngắt kết nối sớm) thì cũng chưa giữ gì,
    # không bị khóa 5 phút. Đang có lượt khác -> sự kiện error status 409 (giao diện hiện detail)
    try:
        _claim(thread_id)
    except HTTPException as e:
        yield _sse({'type': 'error', 'status': e.status_code, 'detail': e.detail})
        return
    try:
        yield from _turn_events(agent, message, thread_id, config)
    finally:
        _release(thread_id)  # cả khi khách đóng tab giữa chừng (generator bị close)


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


@app.post('/chat/stream')
def chat_stream(req: ChatRequest, request: Request):
    """Như /chat nhưng trả Server-Sent Events để giao diện hiện chữ ngay khi LLM sinh ra.
    Mỗi sự kiện là 1 dòng `data: {json}`: start {thread_id}; status {text} khi chạy tool; token {text};
    reset (bỏ phần chữ model viết trước khi gọi tool); done {giống response của /chat}; error {status, detail}."""
    agent, thread_id, config = _start_turn(req, request)
    # X-Accel-Buffering: tắt buffer nếu đặt sau nginx, để chữ tới trình duyệt ngay
    return StreamingResponse(_stream_events(agent, req.message, thread_id, config),
                             media_type='text/event-stream',
                             headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.post('/threads', response_model=ThreadState)
def new_thread(req: NewThread, request: Request):
    """Mở cuộc trò chuyện mới (xóa cuộc cũ nếu có end_thread_id). greeting/customer: ghi sẵn lời chào và hồ sơ
    khách quen vào state để agent biết khách đã mua gì."""
    agent = request.app.state.agent
    if req.end_thread_id:
        _delete_thread(agent, req.end_thread_id)
    thread_id = req.thread_id or str(uuid.uuid4())
    _touch(thread_id)
    values = {'messages': [AIMessage(req.greeting)]} if req.greeting else {}
    if req.customer and AGENT_IMPL == 'graph':
        values['customer'] = req.customer
    if values:
        agent.update_state(_config(thread_id), values, as_node=AS_NODE)
    return ThreadState(thread_id=thread_id)


@app.get('/threads/{thread_id}', response_model=ThreadState)
def get_thread(thread_id: str, request: Request):
    """Giỏ hàng, bước bán hàng của cuộc trò chuyện (be đọc giỏ ở đây để tính giá, tạo đơn)."""
    _touch(thread_id)
    values = request.app.state.agent.get_state(_config(thread_id)).values
    return ThreadState(thread_id=thread_id, stage=values.get('stage'), cart=values.get('cart') or [],
                       order_form=bool(values.get('order_form')), orders=values.get('orders') or [])


@app.delete('/threads/{thread_id}', status_code=204)
def delete_thread(thread_id: str, request: Request):
    _delete_thread(request.app.state.agent, thread_id)


@app.post('/threads/{thread_id}/order-placed', status_code=204)
def order_placed(thread_id: str, req: OrderPlaced, request: Request):
    """Đơn đã lưu: ghi vào lịch sử để lượt sau agent biết, làm trống giỏ để khách mua tiếp không bị cộng dồn."""
    _touch(thread_id)
    request.app.state.agent.update_state(
        _config(thread_id),
        {'messages': [HumanMessage('Đã gửi thông tin đặt hàng'), AIMessage(req.answer)],
         'cart': [], 'order_form': False, 'orders': [req.order_id], 'cross_sell': None, 'wanted': None},
        as_node=AS_NODE)


@app.post('/threads/{thread_id}/order-cancelled', status_code=204)
def order_cancelled(thread_id: str, req: OrderCancelled, request: Request):
    """Khách hủy form đặt hàng: ghi câu trả lời cố định, làm trống giỏ để agent không mời chốt đơn / bán kèm tiếp."""
    _touch(thread_id)
    request.app.state.agent.update_state(
        _config(thread_id),
        {'messages': [HumanMessage('Hủy đặt hàng'), AIMessage(req.answer)],
         'cart': [], 'order_form': False, 'cross_sell': None, 'wanted': None},
        as_node=AS_NODE)


@app.post('/webhooks/policy-updated', status_code=204)
async def policy_updated(event: PolicyUpdated):
    try:
        await asyncio.to_thread(policy_sync.refresh, event.slug)
    except Exception as exc:
        log.exception('Chưa cập nhật được chính sách %s', event.slug)
        raise HTTPException(503, 'Policy refresh failed') from exc


@app.post('/webhooks/policy-reconcile')
async def reconcile_policies():
    try:
        return await asyncio.to_thread(policy_sync.reconcile)
    except Exception as exc:
        log.exception('Chưa đối soát được chính sách từ be')
        raise HTTPException(503, 'Policy reconciliation failed') from exc


@app.post('/policies/{slug}/check')
def check_policy_draft(req: PolicyCheck, slug: str = Path(pattern=SLUG)):
    """Thử bản nháp với các câu hỏi mẫu, không ghi vào Qdrant: {passed, results: [{question, retrieved, missing, passed}]}."""
    try:
        return check_policy(slug, req.content_md, [q.model_dump() for q in req.questions])
    except ValueError as e:
        raise HTTPException(422, str(e))
