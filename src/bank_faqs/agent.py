import bank_faqs  # noqa: F401 - phải import đầu tiên khi chạy file trực tiếp (xem __init__.py)
from langchain.agents import create_agent
from langgraph.types import Checkpointer
from bank_faqs import llm
from bank_faqs.fallback import FallbackMiddleware
from bank_faqs.history import TrimHistoryMiddleware
from bank_faqs.prompts import get_system_prompt
from bank_faqs.tools import get_agent_tools

def build_agent(checkpointer: Checkpointer = None, fallback: bool = True):
    # checkpointer (vd: InMemorySaver) giúp agent nhớ hội thoại theo thread_id - api.py dùng cách này.
    # Không truyền khi chạy qua `langgraph dev` vì server tự quản lý bộ nhớ.
    middleware = [TrimHistoryMiddleware()]
    if fallback:
        # OpenRouter gói free hết quota ngày (hoặc lỗi) thì thử lần lượt Gemini 3.1 -> Gemini 3.5 -> Groq
        # thay vì báo lỗi cho khách. Eval tắt fallback để điểm số phản ánh đúng model chính.
        middleware.append(FallbackMiddleware(*llm.FALLBACK_LLMS))
    return create_agent(
        model=llm.open_llm,
        tools=get_agent_tools('agent_main'),
        system_prompt=get_system_prompt('agent_main'),
        middleware=middleware,
        checkpointer=checkpointer,
    )

agent = build_agent()

if __name__ == "__main__":
    tests = [
        'iphone 18 pro max 256gb màu đen giá bao nhiêu', # nên gọi get_product_details
        'tầm 10 triệu mua máy samsung nào',               # nên gọi search_products
        'mua máy rồi không ưng có trả lại được không',    # nên gọi search_policy
        'nên mua vàng hay cổ phiếu',                      # nên từ chối
    ]
    for q in tests:
        res = agent.invoke({'messages': [{'role': 'user', 'content': q}]})
        called = [tc['name'] for m in res['messages'] for tc in (getattr(m, 'tool_calls', None) or [])]
        print(f"\nKhách: {q}\nTool đã gọi: {called}\nTrả lời: {res['messages'][-1].text}")
