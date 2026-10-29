import httpx
import groq
from langchain_core.exceptions import ModelNotFoundError

from bank_faqs.llm import (FALLBACK_LLMS, RETRIES_IN_CHAIN, RETRIES_LAST, google_llm, google_llm_alt, groq_llm,
                           is_permanent_error, model_name, open_llm)


def test_only_last_model_in_chain_retries_more():
    # Model đứng trước chỉ thử lại 1 lần rồi nhường model sau; model cuối không còn ai đỡ nên thử lại nhiều hơn
    chain = [open_llm, *FALLBACK_LLMS]
    assert [m.max_retries for m in chain] == [RETRIES_IN_CHAIN] * (len(chain) - 1) + [RETRIES_LAST]
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
    assert len(names) == len(set(names)) == 4
