import json
import uuid
from types import SimpleNamespace

import groq
import httpx
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from phone_agent import service as svc
from phone_api import api as be


class FakeAgent:
    """Giả lập agent có checkpointer: lưu lịch sử theo thread_id, lượt nào cũng gọi 1 tool."""

    def __init__(self, error: Exception | None = None, fail_after_first_token: bool = False):
        self.threads: dict[str, list] = {}
        self.extra: dict[str, dict] = {}  # state ngoài messages (handoff, stage...) ghi qua update_state
        self.error = error
        self.fail_after_first_token = fail_after_first_token

    def get_state(self, config):
        thread_id = config['configurable']['thread_id']
        return SimpleNamespace(values={**self.extra.get(thread_id, {}), 'messages': list(self.threads.get(thread_id, []))})

    def update_state(self, config, values, as_node=None):
        thread_id = config['configurable']['thread_id']
        self.threads.setdefault(thread_id, []).extend(values.get('messages', []))
        self.extra.setdefault(thread_id, {}).update({k: v for k, v in values.items() if k != 'messages'})

    def _turn(self, config, text):
        history = self.threads.setdefault(config['configurable']['thread_id'], [])
        call_id = f'call-{len(history)}'
        return history, [
            HumanMessage(text),
            AIMessage('Để em xem', tool_calls=[{'name': 'get_product_details', 'args': {}, 'id': call_id}]),
            ToolMessage('...', tool_call_id=call_id),
            AIMessage(f'Trả lời: {text}'),
        ]

    def invoke(self, input, config):
        if self.error:
            raise self.error
        history, turn = self._turn(config, input['messages'][0]['content'])
        history += turn
        return {'messages': list(history)}

    def stream(self, input, config, stream_mode):
        # Giống LangGraph với stream_mode=['messages', 'updates']: chữ từng đoạn + kết quả từng node
        assert stream_mode == ['messages', 'updates']
        if self.error and not self.fail_after_first_token:
            raise self.error
        history, (human, tool_ai, tool_msg, answer) = self._turn(config, input['messages'][0]['content'])
        yield 'messages', (AIMessageChunk(content='Để em xem'), {'langgraph_node': 'model'})
        if self.error:
            raise self.error
        yield 'updates', {'model': {'messages': [tool_ai]}}
        yield 'messages', (tool_msg, {'langgraph_node': 'tools'})          # kết quả tool: không được stream ra
        yield 'updates', {'tools': {'messages': [tool_msg]}}
        for part in ('Trả lời', ': ', human.content):
            yield 'messages', (AIMessageChunk(content=part), {'langgraph_node': 'model'})
        yield 'updates', {'model': {'messages': [answer]}, 'track_reply': None}
        history += [human, tool_ai, tool_msg, answer]


def sse_events(response) -> list[dict]:
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]


@pytest.fixture(autouse=True)
def wire_services():
    # be gọi agent service ngay trong process: TestClient cũng là httpx.Client.
    # Không dùng `with TestClient(...)` để bỏ qua lifespan (không cần Qdrant/model/Postgres thật)
    svc.app.state.agent = FakeAgent()
    be.app.state.agent_http = TestClient(svc.app)
    be._hits.clear()   # giới hạn theo IP: mọi test dùng chung IP 'testclient'
    svc._busy.clear()


@pytest.fixture
def client():
    return TestClient(be.app)


def test_fe_serves_chat_page():
    fe = Path(__file__).resolve().parents[1] / 'services' / 'fe'
    page = (fe / 'static' / 'index.html').read_text(encoding='utf-8')
    assert 'Tư vấn điện thoại' in page
    assert "fetch('/chat/stream'" in page  # giao diện dùng stream
    conf = (fe / 'nginx.conf.template').read_text(encoding='utf-8')
    assert 'no-cache' in conf             # sửa giao diện xong trình duyệt không dùng bản cũ
    assert 'proxy_buffering off' in conf  # chữ stream tới trình duyệt ngay
    for path in ('chat', 'order', 'session'):
        assert path in conf               # giao diện gọi đường dẫn tương đối, nginx chuyển sang be


def test_chat_new_thread(client):
    r = client.post('/chat', json={'message': 'giá iphone 17'})
    assert r.status_code == 200
    data = r.json()
    uuid.UUID(data['thread_id'])
    assert data['answer'] == 'Trả lời: giá iphone 17'
    assert data['tools'] == ['get_product_details']


def test_chat_normalizes_message(client):
    # Viết tắt, dấu kiểu cũ được chuẩn hóa trước khi tới agent (FakeAgent trả lời lại đúng tin nhận được)
    r = client.post('/chat', json={'message': '  mik mún huỷ, giá bn ko '})
    assert r.json()['answer'] == 'Trả lời: mình muốn hủy, giá bao nhiêu không'


def test_chat_stream_events(client):
    r = client.post('/chat/stream', json={'message': 'giá iphone 17'})
    assert r.status_code == 200 and r.headers['content-type'].startswith('text/event-stream')
    events = sse_events(r)
    assert [e['type'] for e in events] == ['start', 'token', 'reset', 'status', 'token', 'token', 'token', 'done']
    assert events[3]['text'] == 'Đang xem giá và tồn kho…'
    # Chữ stream sau reset ghép lại đúng câu trả lời; kết quả tool không bị stream ra
    assert ''.join(e['text'] for e in events[4:7]) == 'Trả lời: giá iphone 17'
    done = events[-1]
    assert done['thread_id'] == events[0]['thread_id']
    assert done['answer'] == 'Trả lời: giá iphone 17' and done['tools'] == ['get_product_details']


def test_chat_stream_keeps_thread(client):
    tid = sse_events(client.post('/chat/stream', json={'message': 'câu 1'}))[0]['thread_id']
    done = sse_events(client.post('/chat/stream', json={'message': 'câu 2', 'thread_id': tid}))[-1]
    assert done['thread_id'] == tid and done['answer'] == 'Trả lời: câu 2' and done['tools'] == ['get_product_details']
    assert len(svc.app.state.agent.threads[tid]) == 8


@pytest.mark.parametrize('fail_after_first_token', [False, True])
def test_chat_stream_rate_limit_sends_error_event(client, fail_after_first_token):
    resp = httpx.Response(429, request=httpx.Request('POST', 'https://api.groq.com'))
    svc.app.state.agent = FakeAgent(error=groq.RateLimitError('rate limit', response=resp, body=None),
                                    fail_after_first_token=fail_after_first_token)
    events = sse_events(client.post('/chat/stream', json={'message': 'hi'}))
    assert events[-1] == {'type': 'error', 'status': 429, 'detail': svc.RATE_LIMIT_DETAIL}
    assert 'done' not in [e['type'] for e in events]


def test_chat_stream_unexpected_error_sends_500_event(client):
    svc.app.state.agent = FakeAgent(error=KeyError('bug'))
    assert sse_events(client.post('/chat/stream', json={'message': 'hi'}))[-1]['status'] == 500


def test_chat_stream_thinking_shows_status_not_content(client):
    # Model có thinking (NVIDIA): nội dung suy nghĩ không được gửi cho khách, chỉ báo trạng thái 1 lần
    class ThinkingAgent(FakeAgent):
        def stream(self, input, config, stream_mode):
            for _ in range(3):
                yield 'messages', (AIMessageChunk(content='', additional_kwargs={'reasoning_content': 'bí mật'}),
                                   {'langgraph_node': 'model'})
            yield 'messages', (AIMessageChunk(content='Dạ chào anh/chị'), {'langgraph_node': 'model'})
            self.threads.setdefault(config['configurable']['thread_id'], []).append(AIMessage('Dạ chào anh/chị'))
    svc.app.state.agent = ThinkingAgent()
    r = client.post('/chat/stream', json={'message': 'chào em'})
    events = sse_events(r)
    assert [e['type'] for e in events] == ['start', 'status', 'token', 'done']
    assert events[1]['text'] == svc.THINKING_STATUS
    assert 'bí mật' not in r.text


def test_chat_stream_validation(client):
    assert client.post('/chat/stream', json={'message': ''}).status_code == 422


def test_chat_same_thread_only_reports_tools_of_current_turn(client):
    tid = client.post('/chat', json={'message': 'câu 1'}).json()['thread_id']
    r = client.post('/chat', json={'message': 'câu 2', 'thread_id': tid})
    data = r.json()
    assert data['thread_id'] == tid
    assert data['answer'] == 'Trả lời: câu 2'
    assert data['tools'] == ['get_product_details']  # không lặp lại tool của lượt trước
    assert len(svc.app.state.agent.threads[tid]) == 8


@pytest.mark.parametrize('body', [{'message': ''}, {'message': ' \u200b '}, {'message': 'x' * 1001}, {}, {'message': 'hi', 'thread_id': 'x' * 101}])
def test_chat_validation(client, body):
    assert client.post('/chat', json=body).status_code == 422


def test_chat_rate_limit_returns_429(client):
    resp = httpx.Response(429, request=httpx.Request('POST', 'https://api.groq.com'))
    svc.app.state.agent = FakeAgent(error=groq.RateLimitError('rate limit', response=resp, body=None))
    r = client.post('/chat', json={'message': 'hi'})
    assert r.status_code == 429
    assert 'quá tải' in r.json()['detail']


def test_chat_request_too_large_rate_limit_returns_429(client):
    # Groq trả 413 khi vượt giới hạn token đầu vào/phút (lỗi người dùng gặp thực tế)
    resp = httpx.Response(413, request=httpx.Request('POST', 'https://api.groq.com'))
    body = {'error': {'message': 'Request too large ... (ITPM): Limit 7000, Requested 7142',
                      'type': 'tokens', 'code': 'rate_limit_exceeded'}}
    svc.app.state.agent = FakeAgent(error=groq.APIStatusError('413', response=resp, body=body))
    r = client.post('/chat', json={'message': 'hi'})
    assert r.status_code == 429


def test_chat_gemini_rate_limit_returns_429(client):
    from langchain_google_genai.chat_models import GoogleRateLimitError
    svc.app.state.agent = FakeAgent(error=GoogleRateLimitError("Error calling model 'gemini-3.5-flash' (RESOURCE_EXHAUSTED)"))
    assert client.post('/chat', json={'message': 'hi'}).status_code == 429


def test_chat_gemini_other_error_returns_502(client):
    from langchain_google_genai.chat_models import GoogleInvalidRequestError
    svc.app.state.agent = FakeAgent(error=GoogleInvalidRequestError('bad request'))
    assert client.post('/chat', json={'message': 'hi'}).status_code == 502


def test_chat_openrouter_rate_limit_returns_429(client):
    import openrouter.errors as E
    resp = httpx.Response(429, request=httpx.Request('POST', 'https://openrouter.ai'))
    data = E.TooManyRequestsResponseErrorData(error={'code': 429, 'message': 'Rate limit exceeded: free-models-per-min'})
    svc.app.state.agent = FakeAgent(error=E.TooManyRequestsResponseError(data, resp))
    assert client.post('/chat', json={'message': 'hi'}).status_code == 429


@pytest.mark.parametrize('make_error', [
    lambda E: E.InternalServerResponseError(E.InternalServerResponseErrorData(error={'code': 500, 'message': 'boom'}),
                                            httpx.Response(500, request=httpx.Request('POST', 'https://openrouter.ai'))),
    lambda E: E.NoResponseError(),  # mất kết nối, không nhận được phản hồi
])
def test_chat_openrouter_other_error_returns_502(client, make_error):
    # Lỗi OpenRouter không kế thừa ModelError: không bắt riêng thì API trả 500 kèm traceback
    import openrouter.errors as E
    svc.app.state.agent = FakeAgent(error=make_error(E))
    assert client.post('/chat', json={'message': 'hi'}).status_code == 502


def test_chat_upstream_error_returns_502(client):
    resp = httpx.Response(500, request=httpx.Request('POST', 'https://api.groq.com'))
    svc.app.state.agent = FakeAgent(error=groq.InternalServerError('boom', response=resp, body=None))
    r = client.post('/chat', json={'message': 'hi'})
    assert r.status_code == 502


def test_health_ok(client, monkeypatch):
    monkeypatch.setattr(svc, 'count_documents', lambda: 34)
    assert TestClient(svc.app).get('/health').json() == {'status': 'ok', 'documents': 34}
    assert client.get('/health').json() == {'status': 'ok', 'agent': 'ok'}


def test_health_qdrant_down(client, monkeypatch):
    def boom():
        raise ConnectionError()
    monkeypatch.setattr(svc, 'count_documents', boom)
    assert TestClient(svc.app).get('/health').status_code == 503
    assert client.get('/health').json()['agent'] == 'error'   # be vẫn khỏe, chỉ báo agent lỗi


def test_qdrant_client_is_reused():
    # /health được gọi mỗi 30s: tạo client mới mỗi lần sẽ tích tụ kết nối
    from phone_agent.vector_db import get_client
    assert get_client() is get_client()


def test_prepare_vector_db_counts_indexed_documents(monkeypatch):
    monkeypatch.setattr(svc, 'get_vector_store', lambda: 'db')
    monkeypatch.setattr(svc, 'count_documents', lambda: 60)
    assert svc.prepare_vector_db() == 60


def test_prepare_vector_db_retries_then_raises(monkeypatch):
    calls = []
    def boom():
        calls.append(1)
        raise ConnectionError()
    monkeypatch.setattr(svc, 'get_vector_store', boom)
    with pytest.raises(ConnectionError):
        svc.prepare_vector_db(retries=3, delay=0)
    assert len(calls) == 3


# ---------------------------------------------------------------- form đặt hàng

CART = [{'variant_id': 'S26-256-DEN', 'name': 'Galaxy S26', 'storage': '256GB', 'color': 'Đen', 'price': 1, 'quantity': 1}]
ORDER_FORM = {'thread_id': 't1', 'name': ' Nguyễn Văn A ', 'phone': '0912 345.678', 'address': '12 Lê Lợi, Quy Nhơn'}


class CartAgent(FakeAgent):
    """Agent có giỏ hàng trong state (như graph.py) và ghi lại update_state."""

    def __init__(self, cart):
        super().__init__()
        self.cart = cart
        self.updates = []

    def get_state(self, config):
        return SimpleNamespace(values={'messages': [], 'cart': self.cart, 'order_form': True})

    def invoke(self, input, config):
        return {**super().invoke(input, config), 'cart': self.cart, 'order_form': True}

    def update_state(self, config, values, as_node=None):
        self.updates.append((config['configurable']['thread_id'], values, as_node))


@pytest.fixture
def fake_orders(monkeypatch):
    class Saved(list):
        pass
    saved = Saved()
    def create_order(cart, *args, thread_id=None, email=None, promo_code=None):
        if not cart:
            raise be.orders.OrderError('Giỏ hàng đang trống')
        saved.append((cart, args, thread_id) + ((email,) if email else ()))
        return 'DH0000000C'
    monkeypatch.setattr(be.orders, 'create_order', create_order)
    monkeypatch.setattr(be.orders, 'get_order', lambda order_id: {'order_id': order_id})
    monkeypatch.setattr(be.orders, 'summary', lambda order: f"Đơn #{order['order_id']}")
    monkeypatch.setattr(be.orders, 'view', lambda order: {'order_id': order['order_id'], 'total': 1})
    monkeypatch.setattr(be.orders, 'thank_you', lambda name: f'Cảm ơn {name}!')
    # Email xác nhận chạy nền sau khi trả kết quả: ghi lại mã đơn thay vì gửi thật
    saved.emails = []
    async def send_order_email(order_id):
        saved.emails.append(order_id)
    monkeypatch.setattr(be.mailer, 'send_order_email', send_order_email)
    return saved


def test_chat_response_has_order_form_flag():
    svc.app.state.agent = CartAgent(CART)
    data = TestClient(be.app).post('/chat', json={'message': 'chốt đơn', 'thread_id': 't1'}).json()
    assert data['order_form'] is True and data['cart'] == CART


def test_order_creates_from_server_cart_and_records_in_thread(fake_orders):
    agent = svc.app.state.agent = CartAgent(CART)
    r = TestClient(be.app).post('/order', json={**ORDER_FORM, 'cart': [{'price': 0}]})  # giỏ client gửi bị bỏ qua
    assert r.status_code == 200
    assert r.json() == {'thread_id': 't1', 'order_id': 'DH0000000C', 'answer': 'Đơn #DH0000000C\n\nCảm ơn Nguyễn Văn A!',
                        'order': {'order_id': 'DH0000000C', 'total': 1}, 'thanks': 'Cảm ơn Nguyễn Văn A!',
                        'customer_token': None}
    assert fake_orders == [(CART, ('Nguyễn Văn A', '0912345678', '12 Lê Lợi, Quy Nhơn', 'delivery', 'cod'), 't1')]
    thread, values, node = agent.updates[0]
    assert thread == 't1' and node == 'track_reply'
    assert values['cart'] == [] and values['cross_sell'] is None and values['orders'] == ['DH0000000C'] and values['messages'][1].text == r.json()['answer']
    assert fake_orders.emails == ['DH0000000C']   # email xác nhận được lên lịch gửi nền


def test_order_error_does_not_send_email(fake_orders):
    svc.app.state.agent = CartAgent([])
    assert TestClient(be.app).post('/order', json=ORDER_FORM).status_code == 409
    assert fake_orders.emails == []


def test_order_empty_cart_returns_409(fake_orders):
    svc.app.state.agent = CartAgent([])
    r = TestClient(be.app).post('/order', json=ORDER_FORM)
    assert r.status_code == 409 and r.json()['detail'] == 'Giỏ hàng đang trống'


def test_order_email_optional_and_validated(fake_orders):
    svc.app.state.agent = CartAgent(CART)
    client = TestClient(be.app)
    assert client.post('/order', json={**ORDER_FORM, 'email': '  '}).status_code == 200      # để trống = không có
    assert client.post('/order', json={**ORDER_FORM, 'email': ' a@b.vn '}).status_code == 200
    assert [s[3:] for s in fake_orders] == [(), ('a@b.vn',)]
    r = client.post('/order', json={**ORDER_FORM, 'email': 'abc'})
    assert r.status_code == 422 and 'Email chưa đúng định dạng' in r.text


@pytest.mark.parametrize('change', [{'phone': '12345'}, {'name': 'A'}, {'address': 'ngắn'}, {'payment': 'installment'}])
def test_order_validation(fake_orders, change):
    svc.app.state.agent = CartAgent(CART)
    assert TestClient(be.app).post('/order', json={**ORDER_FORM, **change}).status_code == 422
    assert fake_orders == []


def test_order_database_error_returns_503(monkeypatch):
    def boom(*a, **k):
        raise be.psycopg.OperationalError('down')
    monkeypatch.setattr(be.orders, 'create_order', boom)
    svc.app.state.agent = CartAgent(CART)
    r = TestClient(be.app).post('/order', json=ORDER_FORM)
    assert r.status_code == 503 and 'chưa lưu được đơn hàng' in r.json()['detail']


def test_order_cancel_records_in_thread_without_creating_order(fake_orders):
    agent = svc.app.state.agent = CartAgent(CART)
    r = TestClient(be.app).post('/order/cancel', json={'thread_id': 't1'})
    assert r.status_code == 200
    assert r.json() == {'thread_id': 't1', 'answer': be.orders.CANCEL_MESSAGE}
    assert r.json()['answer'].startswith('Dạ, thật tiếc') and 'hỗ trợ hay tư vấn gì thêm' in r.json()['answer']
    assert fake_orders == []                                              # không tạo đơn
    thread, values, node = agent.updates[0]
    assert thread == 't1' and node == 'track_reply' and values['cart'] == [] and values['order_form'] is False and values['cross_sell'] is None
    assert [m.text for m in values['messages']] == ['Hủy đặt hàng', be.orders.CANCEL_MESSAGE]


def test_order_cancel_with_create_agent(fake_orders, monkeypatch):
    # AGENT_IMPL=agent: create_agent không có node track_reply (trước đây lỗi InvalidUpdateError -> 500)
    from langgraph.checkpoint.memory import InMemorySaver
    monkeypatch.setattr(svc, 'AS_NODE', None)
    agent = svc.app.state.agent = svc.build_agent(checkpointer=InMemorySaver(), fallback=False)
    r = TestClient(be.app).post('/order/cancel', json={'thread_id': 'a1'})
    assert r.status_code == 200 and r.json()['answer'] == be.orders.CANCEL_MESSAGE
    messages = agent.get_state({'configurable': {'thread_id': 'a1'}}).values['messages']
    assert [m.text for m in messages] == ['Hủy đặt hàng', be.orders.CANCEL_MESSAGE]
    assert fake_orders == []


def test_order_cancel_needs_thread(fake_orders):
    svc.app.state.agent = CartAgent(CART)
    assert TestClient(be.app).post('/order/cancel', json={}).status_code == 422


def test_chat_redacts_phone_and_email_before_agent(client):
    # FakeAgent trả lời lại đúng tin nhận được: agent (LLM, lịch sử) chỉ thấy bản đã che
    r = client.post('/chat', json={'message': 'sđt mình 0912 345 678, mail a.b@gmail.com'})
    assert r.json()['answer'] == 'Trả lời: sđt mình [SỐ ĐIỆN THOẠI], mail [EMAIL]'


def test_order_without_pii_key_returns_503(monkeypatch):
    def no_key(*a, **k):
        raise be.pii.PiiKeyError('Chưa đặt PII_KEY')
    monkeypatch.setattr(be.orders, 'create_order', no_key)
    svc.app.state.agent = CartAgent(CART)
    r = TestClient(be.app).post('/order', json=ORDER_FORM)
    assert r.status_code == 503 and 'chưa lưu được đơn hàng' in r.json()['detail']


# ---------------------------------------------------------------- phiên chat mới, khách quen

PROFILE = {'customer_id': 'KH00000001', 'name': 'Nguyễn Văn A', 'order_count': 2,
           'last_order': {'order_id': 'DH0000000C', 'date': '02/11/2026', 'items': ['iPhone 17 512GB Đen', 'Ốp lưng']}}


class SessionAgent(CartAgent):
    def __init__(self):
        super().__init__([])
        self.checkpointer = SimpleNamespace(deleted=[], delete_thread=lambda t: self.checkpointer.deleted.append(t))


@pytest.fixture
def session_agent(monkeypatch):
    agent = svc.app.state.agent = SessionAgent()
    monkeypatch.setattr(be.orders, 'customer_profile', lambda cid: PROFILE if cid == 'KH00000001' else None)
    return agent


def test_idle_threads_expire_after_ttl(session_agent, monkeypatch):
    monkeypatch.setattr(svc, '_last_seen', {})
    client = TestClient(be.app)
    client.post('/chat', json={'message': 'chào em', 'thread_id': 'idle'})
    client.post('/chat', json={'message': 'chào em', 'thread_id': 'active'})
    svc._last_seen['idle'] -= svc.SESSION_TTL_S + 1
    assert svc.expire_threads(session_agent) == ['idle']
    assert session_agent.checkpointer.deleted == ['idle'] and list(svc._last_seen) == ['active']
    # Khách tự mở phiên mới: thread cũ thôi theo dõi, thread mới bắt đầu tính giờ
    new = client.post('/session', json={'end_thread_id': 'active'}).json()['thread_id']
    assert list(svc._last_seen) == [new]


def test_new_session_for_new_customer_greets_normally_and_ends_old_thread(session_agent):
    data = TestClient(be.app).post('/session', json={'end_thread_id': 'old'}).json()
    assert data['greeting'] == be.GREETING and data['returning'] is False and data['customer_token'] is None
    assert data['thread_id'] and data['thread_id'] != 'old'
    assert session_agent.checkpointer.deleted == ['old'] and session_agent.updates == []


def test_new_session_keeps_returning_customer(session_agent):
    token = be.pii.customer_token('KH00000001')
    data = TestClient(be.app).post('/session', json={'customer_token': token}).json()
    assert data['returning'] is True and data['customer_token'] == token
    assert 'Nguyễn Văn A' in data['greeting'] and 'iPhone 17 512GB Đen' in data['greeting']
    thread, values, node = session_agent.updates[0]
    assert thread == data['thread_id'] and node == 'track_reply'
    assert values['customer'] == PROFILE and values['messages'][0].text == data['greeting']


@pytest.mark.parametrize('token', ['KH00000001.' + '0' * 32,             # chữ ký sai: không xem được khách khác
                                   'KH00000001', ''])
def test_new_session_rejects_forged_token(session_agent, token):
    data = TestClient(be.app).post('/session', json={'customer_token': token}).json()
    assert data['returning'] is False and data['greeting'] == be.GREETING and data['customer_token'] is None


def test_new_session_customer_without_orders_is_new(session_agent):
    data = TestClient(be.app).post('/session', json={'customer_token': be.pii.customer_token('KH00000099')}).json()
    assert data['returning'] is False and data['customer_token'] is None


def test_order_returns_customer_token(fake_orders, monkeypatch):
    svc.app.state.agent = CartAgent(CART)
    monkeypatch.setattr(be.orders, 'get_order', lambda order_id: {'order_id': order_id, 'customer_id': 'KH00000001'})
    data = TestClient(be.app).post('/order', json=ORDER_FORM).json()
    assert be.pii.verify_customer_token(data['customer_token']) == 'KH00000001'


# ---------------------------------------------------------------- tách service: be chạy khi agent tắt, API key

@pytest.fixture
def agent_down(fake_orders):
    def refuse(request):
        raise httpx.ConnectError('agent tắt', request=request)
    be.app.state.agent_http = httpx.Client(transport=httpx.MockTransport(refuse), base_url='http://agent')
    return fake_orders


def test_be_works_when_agent_down(agent_down):
    client = TestClient(be.app)
    assert client.get('/health').json() == {'status': 'ok', 'agent': 'down'}
    session = client.post('/session', json={}).json()                     # vẫn mở phiên, chào như khách mới
    assert session['greeting'] == be.GREETING and session['thread_id']
    assert client.post('/order/cancel', json={'thread_id': 't1'}).json()['answer'] == be.orders.CANCEL_MESSAGE
    for path, body in [('/chat', {'message': 'hi'}), ('/chat/stream', {'message': 'hi'}),
                       ('/order/quote', {'thread_id': 't1'}), ('/order', ORDER_FORM)]:
        r = client.post(path, json=body)
        assert r.status_code == 503 and r.json()['detail'] == be.AGENT_DOWN, path
    assert agent_down == []                                               # không có giỏ thì không tạo đơn


def test_order_saved_even_if_agent_fails_after(fake_orders, monkeypatch):
    # Đơn đã lưu vào database thì vẫn trả thành công dù agent không ghi được lịch sử
    svc.app.state.agent = CartAgent(CART)
    monkeypatch.setattr(svc.app.state.agent, 'update_state', lambda *a, **k: (_ for _ in ()).throw(RuntimeError()))
    be.app.state.agent_http = TestClient(svc.app, raise_server_exceptions=False)
    r = TestClient(be.app).post('/order', json=ORDER_FORM)
    assert r.status_code == 200 and r.json()['order_id'] == 'DH0000000C'


@pytest.mark.parametrize('thread_id', ['../chat', 'a/b', 'x y'])
def test_be_rejects_thread_id_outside_path_charset(client, thread_id):
    # thread_id nằm trong đường dẫn khi gọi agent (/threads/{id}/...)
    assert client.post('/order/cancel', json={'thread_id': thread_id}).status_code == 422
    assert client.post('/chat', json={'message': 'hi', 'thread_id': thread_id}).status_code == 422


def test_agent_api_key(monkeypatch):
    monkeypatch.setattr(svc, 'AGENT_API_KEY', 'bi-mat')
    agent = TestClient(svc.app)
    assert agent.post('/chat', json={'message': 'hi'}).status_code == 401
    assert agent.post('/chat', json={'message': 'hi'}, headers={'Authorization': 'Bearer sai'}).status_code == 401
    assert agent.post('/chat', json={'message': 'hi'}, headers={'Authorization': 'Bearer bi-mat'}).status_code == 200
    monkeypatch.setattr(svc, 'count_documents', lambda: 1)
    assert agent.get('/health').status_code == 200                        # health không cần key


def test_agent_thread_state_for_integrators():
    # Hệ thống khác đọc giỏ hàng / bước bán hàng của 1 cuộc trò chuyện
    svc.app.state.agent = CartAgent(CART)
    assert TestClient(svc.app).get('/threads/t1').json() == {'thread_id': 't1', 'stage': None, 'cart': CART,
                                                             'order_form': True, 'orders': []}


# ---------------------------------------------------------------- chống spam

def test_thread_runs_one_turn_at_a_time(client):
    import time
    svc._busy['t1'] = time.monotonic()           # lượt trước của t1 chưa xong
    r = client.post('/chat', json={'message': 'còn hàng không', 'thread_id': 't1'})
    assert r.status_code == 409 and r.json()['detail'] == svc.BUSY_DETAIL
    # Stream: giữ thread trong generator nên báo bằng sự kiện error (giao diện hiện detail)
    events = sse_events(client.post('/chat/stream', json={'message': 'còn hàng không', 'thread_id': 't1'}))
    assert events == [{'type': 'error', 'status': 409, 'detail': svc.BUSY_DETAIL}]
    assert client.post('/chat', json={'message': 'hi', 'thread_id': 't2'}).status_code == 200   # thread khác vẫn chạy
    svc._busy['t1'] = time.monotonic() - svc.TURN_TIMEOUT_S - 1  # lượt cũ chết, không nhả được: hết hạn thì mở lại
    assert client.post('/chat', json={'message': 'hi', 'thread_id': 't1'}).status_code == 200


def test_thread_released_after_turn_and_errors(client):
    client.post('/chat', json={'message': 'hi', 'thread_id': 't1'})
    sse_events(client.post('/chat/stream', json={'message': 'hi', 'thread_id': 't1'}))
    svc.app.state.agent = FakeAgent(error=KeyError('bug'))
    sse_events(client.post('/chat/stream', json={'message': 'hi', 'thread_id': 't1'}))
    assert svc._busy == {}


def test_rate_limit_per_ip(client, monkeypatch):
    monkeypatch.setattr(be, 'CHAT_RATE_PER_MIN', 3)
    for i in range(3):
        assert client.post('/chat', json={'message': 'hi', 'thread_id': f't{i}'}).status_code == 200
    r = client.post('/chat/stream', json={'message': 'hi', 'thread_id': 't9'})
    assert r.status_code == 429 and r.json()['detail'] == be.RATE_DETAIL and 0 < int(r.headers['retry-after']) <= 61
    assert client.post('/session', json={}).status_code == 429
    be._hits['testclient'] = type(be._hits['testclient'])(t - 61 for t in be._hits['testclient'])  # 1 phút sau
    assert client.post('/chat', json={'message': 'hi', 'thread_id': 't1'}).status_code == 200


def test_same_message_over_and_over_nudges_then_hands_off(client, monkeypatch):
    calls = []
    agent = svc.app.state.agent
    monkeypatch.setattr(agent, 'invoke', lambda input, config, real=agent.invoke: calls.append(1) or real(input, config))
    send = lambda text: client.post('/chat', json={'message': text, 'thread_id': 't1'}).json()
    send('cho tui xem giá')
    assert send('Cho tui xem giá!!')['answer'] == 'Trả lời: Cho tui xem giá!!'   # gửi lại 1 lần: trả lời bình thường
    r = send('cho tui  xem giá ?')                                               # lần 3: nhắc nhẹ, không gọi LLM
    assert r['answer'] == svc.NUDGE and not r['handoff'] and len(calls) == 2
    events = sse_events(client.post('/chat/stream', json={'message': 'cho tui xem giá', 'thread_id': 't1'}))
    done = events[-1]                                                            # lần 4: chuyển nhân viên
    assert done['type'] == 'done' and done['answer'] == svc.SPAM_HANDOFF and done['handoff'] and done['stage'] == 'handoff'
    assert len(calls) == 2 and svc._busy == {}
    # Lịch sử có đủ tin của khách và câu trả lời sẵn (lượt sau model đọc được)
    texts = [m.text for m in agent.threads['t1']]
    assert texts[-4:] == ['cho tui xem giá ?', svc.NUDGE, 'cho tui xem giá', svc.SPAM_HANDOFF]
    assert send('cho tui xem giá iphone 16')['answer'] == 'Trả lời: cho tui xem giá iphone 16'   # nói rõ: bình thường
    assert send('cho tui xem giá')['answer'] == 'Trả lời: cho tui xem giá'
    assert send('cho tui xem giá')['answer'].startswith('Trả lời')
    assert client.post('/chat', json={'message': 'cho tui xem giá', 'thread_id': 't2'}).json()['answer'].startswith('Trả lời')


def test_lock_released_when_reading_state_fails(client):
    class Broken(FakeAgent):
        def get_state(self, config):
            raise RuntimeError('checkpointer lỗi')
    svc.app.state.agent = Broken()
    with pytest.raises(RuntimeError):
        client.post('/chat', json={'message': 'hi', 'thread_id': 't1'})
    assert svc._busy == {}
    events = sse_events(client.post('/chat/stream', json={'message': 'hi', 'thread_id': 't1'}))
    assert events[-1]['type'] == 'error' and events[-1]['status'] == 500 and svc._busy == {}


def test_stream_dropped_before_start_does_not_lock(client):
    # Khách ngắt kết nối trước khi stream chạy: generator bị close khi chưa chạy -> không được giữ thread
    gen = svc._stream_events(svc.app.state.agent, 'hi', 't1', svc._config('t1'))
    gen.close()
    assert svc._busy == {}
    assert client.post('/chat', json={'message': 'hi', 'thread_id': 't1'}).status_code == 200


def test_short_replies_to_different_questions_are_not_spam(client):
    for text in ('ok', 'ok', 'Ok!', 'ok', 'có', 'có', 'có', '👍', '👍', '👍', '👍'):
        r = client.post('/chat', json={'message': text, 'thread_id': 't1'}).json()
        assert r['answer'] == f'Trả lời: {text}' and not r['handoff']


def test_repeat_key_ignores_case_and_punctuation_only():
    assert svc._key('Cho  tui xem GIÁ!!') == svc._key('cho tui xem giá')
    assert svc._key('👍') != svc._key('😡')


def test_two_real_concurrent_turns_on_one_thread(client):
    import threading
    entered, go = threading.Event(), threading.Event()

    class Slow(FakeAgent):
        def invoke(self, input, config):
            entered.set()
            assert go.wait(10)
            return super().invoke(input, config)
    svc.app.state.agent = Slow()
    first = {}
    t = threading.Thread(target=lambda: first.update(
        r=TestClient(be.app).post('/chat', json={'message': 'iphone 16 còn không', 'thread_id': 't1'})))
    t.start()
    assert entered.wait(10)                                 # lượt 1 đang chạy LLM
    r = client.post('/chat', json={'message': 'alo', 'thread_id': 't1'})
    assert r.status_code == 409 and r.json()['detail'] == svc.BUSY_DETAIL   # be chuyển nguyên 409 của agent
    go.set()
    t.join(10)
    assert first['r'].status_code == 200
    assert [m.text for m in svc.app.state.agent.threads['t1'] if m.type == 'human'] == ['iphone 16 còn không']
    assert svc._busy == {}


def test_rate_limit_counts_each_ip_and_can_be_disabled(monkeypatch):
    monkeypatch.setattr(be, 'CHAT_RATE_PER_MIN', 1)
    a, b = TestClient(be.app, client=('1.1.1.1', 1)), TestClient(be.app, client=('2.2.2.2', 1))
    assert a.post('/session', json={}).status_code == 200
    assert a.post('/session', json={}).status_code == 429
    assert b.post('/session', json={}).status_code == 200      # IP khác: đếm riêng
    assert a.post('/order/quote', json={'thread_id': 't1'}).status_code == 429   # đặt hàng cũng bị giới hạn
    monkeypatch.setattr(be, 'CHAT_RATE_PER_MIN', 0)
    assert a.post('/session', json={}).status_code == 200      # 0: tắt


def test_rate_limit_memory_is_capped(monkeypatch):
    monkeypatch.setattr(be, 'MAX_IPS', 3)
    for ip in ('1.0.0.1', '1.0.0.2', '1.0.0.3', '1.0.0.4'):
        TestClient(be.app, client=(ip, 1)).post('/session', json={})
    assert list(be._hits) == ['1.0.0.2', '1.0.0.3', '1.0.0.4']   # bỏ IP im lâu nhất
    TestClient(be.app, client=('1.0.0.2', 1)).post('/session', json={})
    TestClient(be.app, client=('1.0.0.5', 1)).post('/session', json={})
    assert list(be._hits) == ['1.0.0.4', '1.0.0.2', '1.0.0.5']   # IP vừa nhắn lại được giữ
