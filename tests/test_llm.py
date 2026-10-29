import httpx
import groq
from langchain_core.exceptions import ModelNotFoundError

from phone_sales.llm import (FALLBACK_LLMS, RETRIES_IN_CHAIN, RETRIES_LAST, google_llm, google_llm_alt, groq_llm,
                           is_permanent_error, is_rate_limited, model_name, nvidia_llm, nvidia_status, open_llm)


def test_only_last_model_in_chain_retries_more():
    # Model đứng trước chỉ thử lại 1 lần rồi nhường model sau; model cuối không còn ai đỡ nên thử lại nhiều hơn
    # (ChatNVIDIA không có max_retries: SDK của nó không tự thử lại, tức 0 lần)
    chain = [open_llm, *FALLBACK_LLMS]
    retries = [getattr(m, 'max_retries', 0) for m in chain]
    assert all(r <= RETRIES_IN_CHAIN for r in retries[:-1]) and retries[-1] == RETRIES_LAST
    assert RETRIES_IN_CHAIN < RETRIES_LAST


def test_is_permanent_error():
    resp = lambda code: httpx.Response(code, request=httpx.Request('POST', 'https://api.groq.com'))
    assert is_permanent_error(ModelNotFoundError('404 model not found'))
    assert is_permanent_error(groq.AuthenticationError('bad key', response=resp(401), body=None))
    assert not is_permanent_error(groq.InternalServerError('boom', response=resp(500), body=None))
    assert not is_permanent_error(ConnectionError('down'))


def test_openrouter_timeout_is_in_milliseconds():
    # ChatOpenRouter tính timeout bằng mili giây: đặt 60 (tưởng là giây) thì mọi request timeout sau 60ms,
    # SDK thử lại liên tục và agent treo vài phút
    assert open_llm.request_timeout >= 10_000


def test_other_llm_timeouts_are_in_seconds():
    assert 10 <= groq_llm.request_timeout <= 300
    assert 10 <= google_llm.timeout <= 300
    assert 10 <= google_llm_alt.timeout <= 300


def test_fallback_chain_has_distinct_models():
    # Model trùng tên với model chính hay với nhau thì dùng chung quota -> dự phòng vô ích
    names = [model_name(m) for m in [open_llm, *FALLBACK_LLMS]]
    assert len(names) == len(set(names)) == 5
    assert FALLBACK_LLMS[0] is nvidia_llm  # nhanh nhất trong các model dự phòng (đo 29/10/2026)


def test_nvidia_errors_are_read_from_message():
    # ChatNVIDIA ném Exception chung, mã HTTP nằm ở đầu thông báo
    assert nvidia_status(Exception('[429] Too Many Requests\nRequestID: x')) == 429
    assert nvidia_status(ValueError('[429] không phải lỗi của NVIDIA')) is None
    assert is_rate_limited(Exception('[429] Too Many Requests'))
    assert is_permanent_error(Exception('[401] Unauthorized\nPlease check or regenerate your API key.'))
    assert not is_permanent_error(Exception('[500] Internal Server Error'))
    assert not is_rate_limited(Exception('[503] Service Unavailable'))
