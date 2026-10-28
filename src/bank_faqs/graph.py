import bank_faqs  # noqa: F401 - phải import đầu tiên khi chạy file trực tiếp (xem __init__.py)
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Checkpointer
from bank_faqs import llm
from bank_faqs.conversation import (AgentState, append_unique, decide_stage, describe, needs_handoff,
                                    products_mentioned, track_tool_results, track_user_message)
from bank_faqs.fallback import FallbackMiddleware
from bank_faqs.history import trim_history
from bank_faqs.prompts import get_system_prompt
from bank_faqs.tools import get_agent_tools

# Agent tự dựng bằng StateGraph, có trạng thái hội thoại bán hàng (conversation.py) và giỏ hàng:
#   START -> track_user -> model -> (có tool_call) tools -> track_tools -> model ... -> track_reply -> END
TOOLS = get_agent_tools('agent_graph')
SYSTEM_PROMPT = get_system_prompt('agent_graph')

def _last_user_text(state: AgentState) -> str:
    human = [m for m in state.get('messages', []) if isinstance(m, HumanMessage)]
    return human[-1].text if human else ''

def track_user(state: AgentState) -> dict:
    # Đọc tin nhắn mới của khách: nhu cầu, lo ngại, cảm xúc, có đòi gặp nhân viên không
    info = track_user_message(_last_user_text(state))
    after = {**state, 'sentiment_history': [*state.get('sentiment_history', []), *info['sentiment_history']]}
    after['handoff'] = needs_handoff(after, info['wants_handoff'])
    return {
        'slots': info['slots'],
        'objections': info['objections'],
        'sentiment_history': info['sentiment_history'],
        'handoff': after['handoff'],
        'stage': decide_stage(after, info['wants_compare'], info['wants_close']),
    }

def _stage_now(state: AgentState) -> str:
    info = track_user_message(_last_user_text(state))
    return decide_stage(state, info['wants_compare'], info['wants_close'])

def track_tools(state: AgentState) -> dict:
    # Sau mỗi lần chạy tool: nhu cầu suy ra từ tham số tool; giỏ hàng do tool tự ghi -> tính lại bước (vd closing)
    return {**track_tool_results(state.get('messages', [])), 'stage': _stage_now(state)}

def track_reply(state: AgentState) -> dict:
    # Sau câu trả lời cuối của lượt: ghi các sản phẩm đã thật sự giới thiệu cho khách
    shown = products_mentioned(state.get('messages', []))
    after = {**state, 'shown_products': append_unique(state.get('shown_products'), shown)}
    return {'shown_products': shown, 'stage': _stage_now(after)}

def build_graph(fallback: bool = True, checkpointer: Checkpointer = None):
    # checkpointer (vd InMemorySaver) lưu cả trạng thái hội thoại theo thread_id - api.py dùng cách này.
    # Không truyền khi chạy qua `langgraph dev` vì server tự quản lý bộ nhớ.
    primary = llm.open_llm
    models = [primary, *llm.FALLBACK_LLMS] if fallback else [primary]
    # parallel_tool_calls chỉ truyền cho OpenRouter; model dự phòng dùng mặc định của hãng
    bound = {id(m): m.bind_tools(TOOLS, parallel_tool_calls=False) if m is primary else m.bind_tools(TOOLS)
             for m in models}
    chain = FallbackMiddleware(*models[1:]) if fallback else None

    def call_model(state: AgentState) -> dict:
        system = SystemMessage(content=f'{SYSTEM_PROMPT}\n\nTrạng thái hội thoại:\n{describe(state)}')
        messages = [system] + trim_history(state['messages'])
        if chain is None:
            return {'messages': [bound[id(primary)].invoke(messages)]}
        return {'messages': [chain.call(primary, lambda m: bound[id(m)].invoke(messages))]}

    builder = StateGraph(AgentState)
    builder.add_node('track_user', track_user)
    builder.add_node('model', call_model)
    builder.add_node('tools', ToolNode(TOOLS))
    builder.add_node('track_tools', track_tools)
    builder.add_node('track_reply', track_reply)
    builder.add_edge(START, 'track_user')
    builder.add_edge('track_user', 'model')
    # có tool_call -> 'tools', không thì câu trả lời cuối -> 'track_reply' -> END
    builder.add_conditional_edges('model', tools_condition, {'tools': 'tools', END: 'track_reply'})
    builder.add_edge('tools', 'track_tools')
    builder.add_edge('track_tools', 'model')
    builder.add_edge('track_reply', END)
    return builder.compile(checkpointer=checkpointer)

graph = build_graph()

if __name__ == "__main__":
    from langgraph.checkpoint.memory import InMemorySaver
    g = build_graph(checkpointer=InMemorySaver())
    config = {'configurable': {'thread_id': 'demo'}}
    for q in ['tầm 30 triệu muốn mua máy chụp ảnh đẹp', 'galaxy s26 bản 256gb màu đen giá sao', 'ok chốt máy đó cho em']:
        res = g.invoke({'messages': [{'role': 'user', 'content': q}]}, config)
        print(f"\nKhách: {q}\nTrả lời: {res['messages'][-1].text}")
        print({k: res.get(k) for k in ('stage', 'slots', 'shown_products', 'cart', 'objections', 'handoff')})
