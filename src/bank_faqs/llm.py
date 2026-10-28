import groq
from langchain_core.exceptions import ModelRateLimitError
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openrouter import ChatOpenRouter
from openrouter import errors as openrouter_errors

groq_llm = ChatGroq(
    model='qwen/qwen3.8-27b',
    temperature=0,
    timeout=60,
    max_retries=3,
    reasoning_format='hidden',
    # Câu trả lời thực tế ~15-120 token. Không đặt thì Groq tính mỗi request 2048 token đầu ra,
    # vượt giới hạn gói free (1000 token đầu ra/phút) -> lỗi 429.
    max_tokens=512,
)

google_llm = ChatGoogleGenerativeAI(
    model='gemini-3.1-flash-lite',
    temperature=0,
    timeout=60,
    max_retries=3,
)

# Gemini gói free tính quota riêng cho từng model -> model thứ 2 là thêm 1 phần quota khi model trên hết
google_llm_alt = ChatGoogleGenerativeAI(
    model='gemini-3.5-flash-lite',
    temperature=0,
    timeout=60,
    max_retries=3,
)

open_llm = ChatOpenRouter(
    model='nvidia/nemotron-3-ultra-550b-a55b:free',
    temperature=0,
    max_tokens=1024,
    # ChatOpenRouter tính timeout bằng MILI GIÂY (khác ChatGroq/ChatGoogleGenerativeAI tính bằng giây).
    # Đặt 60 là 60ms -> mọi request đều timeout rồi SDK thử lại liên tục, agent treo vài phút.
    timeout=60_000,
    max_retries=3,
)

# Model chính hỏng/hết quota thì thử lần lượt các model này (xem fallback.py)
FALLBACK_LLMS = [google_llm, google_llm_alt, groq_llm]

def model_name(llm) -> str:
    return getattr(llm, 'model', None) or getattr(llm, 'model_name', None) or type(llm).__name__

def is_rate_limited(e: Exception) -> bool:
    # Gemini (langchain-google-genai) ném ModelRateLimitError chung của langchain_core.
    # Groq ném lỗi riêng: 429, hoặc 413 "Request too large" kèm code rate_limit_exceeded.
    # OpenRouter (SDK openrouter) cũng ném lỗi riêng, không kế thừa ModelError.
    if isinstance(e, (ModelRateLimitError, groq.RateLimitError, openrouter_errors.TooManyRequestsResponseError)):
        return True
    if not isinstance(e, groq.APIError):
        return False
    body = e.body if isinstance(e.body, dict) else {}
    err = body.get('error')
    return (err if isinstance(err, dict) else body).get('code') == 'rate_limit_exceeded'

def is_daily_limit(e: Exception) -> bool:
    # Hết quota NGÀY (OpenRouter: free-models-per-day, Gemini: ...PerDay...): chờ vài phút cũng vô ích
    text = str(e)
    return is_rate_limited(e) and ('per-day' in text or 'PerDay' in text or 'per day' in text.lower())
