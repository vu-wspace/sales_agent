from bank_faqs.llm import FALLBACK_LLMS, google_llm, google_llm_alt, groq_llm, model_name, open_llm


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
