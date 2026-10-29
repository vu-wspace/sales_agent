import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import groq
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from langchain_core.exceptions import ModelError
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from openrouter import errors as openrouter_errors
from pydantic import BaseModel, Field

from phone_sales.agent import build_agent
from phone_sales.config import AGENT_IMPL, AUTO_INGEST
from phone_sales.graph import build_graph
from phone_sales.llm import is_rate_limited
from phone_sales.vector_db import count_documents, get_vector_store, ingest, is_up_to_date

log = logging.getLogger('uvicorn.error')
STATIC = Path(__file__).parent / 'static'

# Lỗi của LLM/agent được trả cho khách thành 429 (quá tải) hoặc 502; lỗi khác là lỗi code -> 500
LLM_ERRORS = (groq.APIError, openrouter_errors.OpenRouterError, openrouter_errors.NoResponseError,
              ModelError, GraphRecursionError)
RATE_LIMIT_DETAIL = 'Hệ thống đang quá tải, anh/chị vui lòng thử lại sau ít phút.'

# Dòng trạng thái hiện trên giao diện khi agent đang chạy tool (/chat/stream)
TOOL_STATUS = {
    'search_products': 'Đang tìm sản phẩm phù hợp…',
    'get_product_details': 'Đang xem giá và tồn kho…',
    'find_accessories': 'Đang tìm phụ kiện tương thích…',
    'search_policy': 'Đang tra chính sách cửa hàng…',
    'web_search': 'Đang tìm thông tin trên web…',
    'add_to_cart': 'Đang cập nhật giỏ hàng…',
    'remove_from_cart': 'Đang cập nhật giỏ hàng…',
}
THINKING_STATUS = 'Em đang cân nhắc…'

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    thread_id: str | None = Field(default=None, max_length=100)

class ChatResponse(BaseModel):
    thread_id: str
    answer: str
    tools: list[str]
    # Trạng thái hội thoại (chỉ có khi AGENT_IMPL=graph): bước bán hàng, giỏ hàng, có cần chuyển nhân viên
    stage: str | None = None
    cart: list[dict] = []
    handoff: bool = False

def prepare_vector_db(retries: int = 30, delay: float = 2.0) -> int:
    # Qdrant có thể khởi động chậm hơn API khi chạy bằng docker compose -> thử lại vài lần
    for attempt in range(1, retries + 1):
        try:
            db = get_vector_store()  # tạo collection nếu chưa có
            n = count_documents()
            # Nạp lại cả khi file chính sách .md đã sửa (id chunk không còn khớp), không chỉ khi collection rỗng
            if AUTO_INGEST and (n == 0 or not is_up_to_date()):
                log.info('Chính sách trong Qdrant rỗng hoặc đã cũ, đang nạp lại...')
                n = ingest(db)
            return n
        except Exception as e:
            if attempt == retries:
                raise
            log.warning('Chưa kết nối được Qdrant (%s), thử lại sau %ss (%d/%d)',
                        type(e).__name__, delay, attempt, retries)
            time.sleep(delay)

@asynccontextmanager
async def lifespan(app: FastAPI):
    n = await asyncio.to_thread(prepare_vector_db)
    log.info('Qdrant sẵn sàng: %d tài liệu', n)
    # InMemorySaver: lịch sử hội thoại nằm trong RAM của process -> mất khi restart, chỉ chạy 1 worker
    # graph: có trạng thái hội thoại + giỏ hàng (graph.py); agent: create_agent không có state riêng (agent.py)
    build = build_graph if AGENT_IMPL == 'graph' else build_agent
    app.state.agent = build(checkpointer=InMemorySaver())
    log.info('Dùng agent: %s', AGENT_IMPL)
    yield

app = FastAPI(title='Phone Sales Agent', version='0.1.0', lifespan=lifespan)

@app.get('/', include_in_schema=False)
def index():
    return FileResponse(STATIC / 'index.html')

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

def _start_turn(req: ChatRequest, request: Request):
    agent = request.app.state.agent
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {'configurable': {'thread_id': thread_id}}
    # Agent trả về cả lịch sử của thread -> nhớ số message trước lượt này để chỉ lấy message mới
    n_before = len(agent.get_state(config).values.get('messages', []))
    return agent, thread_id, config, n_before

def _response(thread_id: str, values: dict, n_before: int) -> ChatResponse:
    new_messages = values['messages'][n_before:]
    tools = [tc['name'] for m in new_messages for tc in (getattr(m, 'tool_calls', None) or [])]
    return ChatResponse(thread_id=thread_id, answer=values['messages'][-1].text, tools=tools,
                        stage=values.get('stage'), cart=values.get('cart') or [], handoff=bool(values.get('handoff')))

@app.post('/chat', response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    agent, thread_id, config, n_before = _start_turn(req, request)
    try:
        res = agent.invoke({'messages': [{'role': 'user', 'content': req.message}]}, config)
    except LLM_ERRORS as e:
        raise HTTPException(*_llm_failure(e))
    return _response(thread_id, res, n_before)

def _sse(event: dict) -> str:
    return f'data: {json.dumps(event, ensure_ascii=False)}\n\n'

def _stream_events(agent, message: str, thread_id: str, config: dict, n_before: int):
    yield _sse({'type': 'start', 'thread_id': thread_id})
    status = None  # dòng trạng thái đã gửi gần nhất, tránh gửi trùng
    try:
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
        yield _sse({'type': 'error', 'status': 500, 'detail': 'Có lỗi hệ thống, anh/chị vui lòng thử lại.'})
        return
    values = agent.get_state(config).values
    yield _sse({'type': 'done', **_response(thread_id, values, n_before).model_dump()})

@app.post('/chat/stream')
def chat_stream(req: ChatRequest, request: Request):
    """Như /chat nhưng trả Server-Sent Events để giao diện hiện chữ ngay khi LLM sinh ra.
    Mỗi sự kiện là 1 dòng `data: {json}`: start {thread_id}; status {text} khi chạy tool; token {text};
    reset (bỏ phần chữ model viết trước khi gọi tool); done {giống response của /chat}; error {status, detail}."""
    agent, thread_id, config, n_before = _start_turn(req, request)
    # X-Accel-Buffering: tắt buffer nếu đặt sau nginx, để chữ tới trình duyệt ngay
    return StreamingResponse(_stream_events(agent, req.message, thread_id, config, n_before),
                             media_type='text/event-stream',
                             headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})
