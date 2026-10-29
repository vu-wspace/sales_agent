import json
import uuid
from types import SimpleNamespace

import groq
import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from phone_sales import api


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


@pytest.fixture
def client():
    # Không dùng `with TestClient(...)` để bỏ qua lifespan (không cần Qdrant/model thật)
    api.app.state.agent = FakeAgent()
    return TestClient(api.app)


def test_index_serves_chat_page(client):
    r = client.get('/')
    assert r.status_code == 200
    assert 'text/html' in r.headers['content-type']
    assert 'Tư vấn điện thoại' in r.text
    assert "fetch('/chat/stream'" in r.text  # giao diện dùng stream


def test_chat_new_thread(client):
    r = client.post('/chat', json={'message': 'giá iphone 17'})
    assert r.status_code == 200
    data = r.json()
    uuid.UUID(data['thread_id'])
    assert data['answer'] == 'Trả lời: giá iphone 17'
    assert data['tools'] == ['get_product_details']


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
    assert len(api.app.state.agent.threads[tid]) == 8


@pytest.mark.parametrize('fail_after_first_token', [False, True])
def test_chat_stream_rate_limit_sends_error_event(client, fail_after_first_token):
    resp = httpx.Response(429, request=httpx.Request('POST', 'https://api.groq.com'))
    api.app.state.agent = FakeAgent(error=groq.RateLimitError('rate limit', response=resp, body=None),
                                    fail_after_first_token=fail_after_first_token)
    events = sse_events(client.post('/chat/stream', json={'message': 'hi'}))
    assert events[-1] == {'type': 'error', 'status': 429, 'detail': api.RATE_LIMIT_DETAIL}
    assert 'done' not in [e['type'] for e in events]


def test_chat_stream_unexpected_error_sends_500_event(client):
    api.app.state.agent = FakeAgent(error=KeyError('bug'))
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
    api.app.state.agent = ThinkingAgent()
    r = client.post('/chat/stream', json={'message': 'chào em'})
    events = sse_events(r)
    assert [e['type'] for e in events] == ['start', 'status', 'token', 'done']
    assert events[1]['text'] == api.THINKING_STATUS
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
    assert len(api.app.state.agent.threads[tid]) == 8


@pytest.mark.parametrize('body', [{'message': ''}, {'message': 'x' * 1001}, {}, {'message': 'hi', 'thread_id': 'x' * 101}])
def test_chat_validation(client, body):
    assert client.post('/chat', json=body).status_code == 422


def test_chat_rate_limit_returns_429(client):
    resp = httpx.Response(429, request=httpx.Request('POST', 'https://api.groq.com'))
    api.app.state.agent = FakeAgent(error=groq.RateLimitError('rate limit', response=resp, body=None))
    r = client.post('/chat', json={'message': 'hi'})
    assert r.status_code == 429
    assert 'quá tải' in r.json()['detail']


def test_chat_request_too_large_rate_limit_returns_429(client):
    # Groq trả 413 khi vượt giới hạn token đầu vào/phút (lỗi người dùng gặp thực tế)
    resp = httpx.Response(413, request=httpx.Request('POST', 'https://api.groq.com'))
    body = {'error': {'message': 'Request too large ... (ITPM): Limit 7000, Requested 7142',
                      'type': 'tokens', 'code': 'rate_limit_exceeded'}}
    api.app.state.agent = FakeAgent(error=groq.APIStatusError('413', response=resp, body=body))
    r = client.post('/chat', json={'message': 'hi'})
    assert r.status_code == 429


def test_chat_gemini_rate_limit_returns_429(client):
    from langchain_google_genai.chat_models import GoogleRateLimitError
    api.app.state.agent = FakeAgent(error=GoogleRateLimitError("Error calling model 'gemini-3.5-flash' (RESOURCE_EXHAUSTED)"))
    assert client.post('/chat', json={'message': 'hi'}).status_code == 429


def test_chat_gemini_other_error_returns_502(client):
    from langchain_google_genai.chat_models import GoogleInvalidRequestError
    api.app.state.agent = FakeAgent(error=GoogleInvalidRequestError('bad request'))
    assert client.post('/chat', json={'message': 'hi'}).status_code == 502


def test_chat_openrouter_rate_limit_returns_429(client):
    import openrouter.errors as E
    resp = httpx.Response(429, request=httpx.Request('POST', 'https://openrouter.ai'))
    data = E.TooManyRequestsResponseErrorData(error={'code': 429, 'message': 'Rate limit exceeded: free-models-per-min'})
    api.app.state.agent = FakeAgent(error=E.TooManyRequestsResponseError(data, resp))
    assert client.post('/chat', json={'message': 'hi'}).status_code == 429


@pytest.mark.parametrize('make_error', [
    lambda E: E.InternalServerResponseError(E.InternalServerResponseErrorData(error={'code': 500, 'message': 'boom'}),
                                            httpx.Response(500, request=httpx.Request('POST', 'https://openrouter.ai'))),
    lambda E: E.NoResponseError(),  # mất kết nối, không nhận được phản hồi
])
def test_chat_openrouter_other_error_returns_502(client, make_error):
    # Lỗi OpenRouter không kế thừa ModelError: không bắt riêng thì API trả 500 kèm traceback
    import openrouter.errors as E
    api.app.state.agent = FakeAgent(error=make_error(E))
    assert client.post('/chat', json={'message': 'hi'}).status_code == 502


def test_chat_upstream_error_returns_502(client):
    resp = httpx.Response(500, request=httpx.Request('POST', 'https://api.groq.com'))
    api.app.state.agent = FakeAgent(error=groq.InternalServerError('boom', response=resp, body=None))
    r = client.post('/chat', json={'message': 'hi'})
    assert r.status_code == 502


def test_health_ok(client, monkeypatch):
    monkeypatch.setattr(api, 'count_documents', lambda: 34)
    assert client.get('/health').json() == {'status': 'ok', 'documents': 34}


def test_health_qdrant_down(client, monkeypatch):
    def boom():
        raise ConnectionError()
    monkeypatch.setattr(api, 'count_documents', boom)
    assert client.get('/health').status_code == 503


def test_qdrant_client_is_reused():
    # /health được gọi mỗi 30s: tạo client mới mỗi lần sẽ tích tụ kết nối
    from phone_sales.vector_db import get_client
    assert get_client() is get_client()


def test_prepare_vector_db_ingests_when_empty(monkeypatch):
    monkeypatch.setattr(api, 'get_vector_store', lambda: 'db')
    monkeypatch.setattr(api, 'count_documents', lambda: 0)
    monkeypatch.setattr(api, 'ingest', lambda db: 34)
    assert api.prepare_vector_db() == 34


@pytest.mark.parametrize('up_to_date, expected', [(False, 65), (True, 60)])
def test_prepare_vector_db_reingests_only_when_policies_changed(monkeypatch, up_to_date, expected):
    monkeypatch.setattr(api, 'get_vector_store', lambda: 'db')
    monkeypatch.setattr(api, 'count_documents', lambda: 60)
    monkeypatch.setattr(api, 'is_up_to_date', lambda: up_to_date)
    monkeypatch.setattr(api, 'ingest', lambda db: 65)
    assert api.prepare_vector_db() == expected


def test_prepare_vector_db_retries_then_raises(monkeypatch):
    calls = []
    def boom():
        calls.append(1)
        raise ConnectionError()
    monkeypatch.setattr(api, 'get_vector_store', boom)
    with pytest.raises(ConnectionError):
        api.prepare_vector_db(retries=3, delay=0)
    assert len(calls) == 3
