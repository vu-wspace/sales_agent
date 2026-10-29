import httpx
import groq
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from bank_faqs import agent as agent_module
from bank_faqs import llm
from bank_faqs.fallback import DAILY_COOLDOWN_S, MINUTE_COOLDOWN_S, FallbackMiddleware


class FakeModel(GenericFakeChatModel):
    model: str = 'fake'  # model_name() đọc trường này, giống model thật

    def bind_tools(self, tools, **kwargs):
        return self


class FailingModel(FakeModel):
    error: Exception = RuntimeError('boom')
    calls: int = 0

    def _generate(self, *args, **kwargs):
        self.calls += 1
        raise self.error


def rate_limit(message: str) -> groq.RateLimitError:
    resp = httpx.Response(429, request=httpx.Request('POST', 'https://api.groq.com'))
    return groq.RateLimitError(message, response=resp, body=None)


def reply(name: str, text: str) -> FakeModel:
    return FakeModel(model=name, messages=iter([AIMessage(text)]))


def ask(agent) -> str:
    return agent.invoke({'messages': [{'role': 'user', 'content': 'chào'}]})['messages'][-1].text


@pytest.fixture
def chain(monkeypatch):
    # Model chính và 2 model dự phòng đầu đều hết quota, chỉ model cuối trả lời được
    primary = FailingModel(model='openrouter', messages=iter([]), error=rate_limit('Rate limit exceeded: free-models-per-day'))
    gemini = FailingModel(model='gemini-a', messages=iter([]), error=rate_limit('RESOURCE_EXHAUSTED: GenerateRequestsPerDay'))
    gemini_alt = FailingModel(model='gemini-b', messages=iter([]), error=rate_limit('tokens per minute'))
    groq_model = FakeModel(model='groq', messages=iter([AIMessage('Dạ, em là model dự phòng cuối ạ.')] * 3))
    monkeypatch.setattr(llm, 'open_llm', primary)
    monkeypatch.setattr(llm, 'FALLBACK_LLMS', [gemini, gemini_alt, groq_model])
    return primary, gemini, gemini_alt, groq_model


def test_walks_whole_chain_until_a_model_answers(chain):
    assert ask(agent_module.build_agent()) == 'Dạ, em là model dự phòng cuối ạ.'
    assert [m.calls for m in chain[:3]] == [1, 1, 1]


def test_no_fallback_raises_primary_error(chain):
    # Eval dùng fallback=False: lỗi của model chính phải lộ ra, không bị model khác che
    with pytest.raises(groq.RateLimitError, match='free-models-per-day'):
        ask(agent_module.build_agent(fallback=False))


def test_rate_limited_models_are_skipped_until_cooldown_ends():
    now = [0.0]
    primary = FailingModel(model='openrouter', messages=iter([]), error=rate_limit('free-models-per-day'))
    minute = FailingModel(model='gemini-a', messages=iter([]), error=rate_limit('per minute'))
    ok = FakeModel(model='groq', messages=iter([]))
    mw = FallbackMiddleware(minute, ok, clock=lambda: now[0])

    def invoke(m):
        return m.invoke('x')

    ok.messages = iter([AIMessage('1'), AIMessage('2'), AIMessage('3')])
    assert mw.call(primary, invoke).text == '1'
    assert mw.skip_until == {'openrouter': DAILY_COOLDOWN_S, 'gemini-a': MINUTE_COOLDOWN_S}

    assert mw.call(primary, invoke).text == '2'       # cả 2 model lỗi đang bị bỏ qua: không gọi lại
    assert (primary.calls, minute.calls) == (1, 1)

    now[0] = MINUTE_COOLDOWN_S + 1                    # hết 1 phút: thử lại model bị giới hạn theo phút
    assert mw.call(primary, invoke).text == '3'
    assert (primary.calls, minute.calls) == (1, 2)


def test_other_errors_are_not_skipped_later():
    # Lỗi tạm thời (mất mạng...) không phải giới hạn quota: câu sau vẫn thử model đó trước
    primary = FailingModel(model='openrouter', messages=iter([]), error=ConnectionError('down'))
    ok = FakeModel(model='groq', messages=iter([AIMessage('a'), AIMessage('b')]))
    mw = FallbackMiddleware(ok)
    mw.call(primary, lambda m: m.invoke('x'))
    mw.call(primary, lambda m: m.invoke('x'))
    assert primary.calls == 2 and mw.skip_until == {}


def test_ssl_certificate_errors_are_skipped_for_a_while():
    # Mạng có SSL inspection: lỗi chứng chỉ lặp lại ở mọi lần gọi -> bỏ qua để câu sau không phải chờ model đó lỗi
    from bank_faqs.fallback import SSL_COOLDOWN_S
    error = ConnectionError('[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self-signed certificate')
    gemini = FailingModel(model='gemini-a', messages=iter([]), error=error)
    ok = FakeModel(model='groq', messages=iter([AIMessage('a'), AIMessage('b')]))
    mw = FallbackMiddleware(ok, clock=lambda: 0.0)
    mw.call(gemini, lambda m: m.invoke('x'))
    mw.call(gemini, lambda m: m.invoke('x'))
    assert gemini.calls == 1 and mw.skip_until == {'gemini-a': SSL_COOLDOWN_S}


def test_all_models_cooling_down_still_tries_everything():
    primary = FailingModel(model='openrouter', messages=iter([]), error=rate_limit('free-models-per-day'))
    mw = FallbackMiddleware(clock=lambda: 0.0)
    mw.skip_until['openrouter'] = 10**9
    with pytest.raises(groq.RateLimitError):
        mw.call(primary, lambda m: m.invoke('x'))
    assert primary.calls == 1


def test_graph_uses_same_chain(chain):
    from bank_faqs.graph import build_graph
    res = build_graph().invoke({'messages': [{'role': 'user', 'content': 'chào'}]})
    assert res['messages'][-1].text == 'Dạ, em là model dự phòng cuối ạ.'
