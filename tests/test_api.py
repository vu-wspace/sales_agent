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
        self.error = error
        self.fail_after_first_token = fail_after_first_token

    def get_state(self, config):
        return SimpleNamespace(values={'messages': list(self.threads.get(config['configurable']['thread_id'], []))})

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


def test_prepare_vector_db_ingests_when_empty(monkeypatch):
    monkeypatch.setattr(svc, 'get_vector_store', lambda: 'db')
    monkeypatch.setattr(svc, 'count_documents', lambda: 0)
    monkeypatch.setattr(svc, 'ingest', lambda db: 34)
    assert svc.prepare_vector_db() == 34


@pytest.mark.parametrize('up_to_date, expected', [(False, 65), (True, 60)])
def test_prepare_vector_db_reingests_only_when_policies_changed(monkeypatch, up_to_date, expected):
    monkeypatch.setattr(svc, 'get_vector_store', lambda: 'db')
    monkeypatch.setattr(svc, 'count_documents', lambda: 60)
    monkeypatch.setattr(svc, 'is_up_to_date', lambda: up_to_date)
    monkeypatch.setattr(svc, 'ingest', lambda db: 65)
    assert svc.prepare_vector_db() == expected


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
