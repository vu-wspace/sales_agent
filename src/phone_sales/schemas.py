"""Hợp đồng /chat dùng chung giữa service be (api.py) và service agent (agent_api.py)."""
from pydantic import BaseModel, Field, field_validator

from phone_sales import pii
from phone_sales.textnorm import normalize_user_text


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
