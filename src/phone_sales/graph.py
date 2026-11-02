import phone_sales  # noqa: F401 - phải import đầu tiên khi chạy file trực tiếp (xem __init__.py)
import logging
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Checkpointer
from phone_sales import llm
from phone_sales.conversation import (AgentState, append_unique, cross_sell_offers, decide_stage, describe,
                                    needs_handoff, products_mentioned, track_tool_results, track_user_message)
from phone_sales.fallback import FallbackMiddleware
from phone_sales.grounding import correction, sources_of, ungrounded
from phone_sales.history import trim_history
from phone_sales.prompts import get_system_prompt
from phone_sales.textnorm import clean_answer, normalize_user_text
from phone_sales.tools import get_agent_tools

log = logging.getLogger(__name__)

# Agent tự dựng bằng StateGraph, có trạng thái hội thoại bán hàng (conversation.py) và giỏ hàng:
#   START -> track_user -> model -> (có tool_call) tools -> track_tools -> model ... -> track_reply -> END
TOOLS = get_agent_tools('agent_graph')
SYSTEM_PROMPT = get_system_prompt('agent_graph')

def _last_user_text(state: AgentState) -> str:
    human = [m for m in state.get('messages', []) if isinstance(m, HumanMessage)]
    # api.py đã chuẩn hóa; chuẩn hóa lại cho trường hợp gọi graph trực tiếp (langgraph dev, eval)
    return normalize_user_text(human[-1].text) if human else ''

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
        'declined': info['declined'],
        'order_form': False,  # form chỉ hiện ở lượt gọi show_order_form
        'stage': decide_stage(after, info['wants_compare'], info['wants_close']),
    }

def _stage_now(state: AgentState) -> str:
    info = track_user_message(_last_user_text(state))
    return decide_stage(state, info['wants_compare'], info['wants_close'])

def track_tools(state: AgentState) -> dict:
    # Sau mỗi lần chạy tool: nhu cầu suy ra từ tham số tool; giỏ hàng do tool tự ghi -> tính lại bước (vd closing)
    return {**track_tool_results(state.get('messages', [])), 'stage': _stage_now(state)}

def track_reply(state: AgentState) -> dict:
    # Sau câu trả lời cuối của lượt: ghi các sản phẩm đã thật sự giới thiệu cho khách và bước bán kèm đã mời
    messages = state.get('messages', [])
    shown = products_mentioned(messages)
    after = {**state, 'shown_products': append_unique(state.get('shown_products'), shown)}
    return {'shown_products': shown, 'cross_sell': cross_sell_offers(messages), 'stage': _stage_now(after)}

def build_graph(fallback: bool = True, checkpointer: Checkpointer = None):
    # checkpointer (vd InMemorySaver) lưu cả trạng thái hội thoại theo thread_id - api.py dùng cách này.
    # Không truyền khi chạy qua `langgraph dev` vì server tự quản lý bộ nhớ.
    primary = llm.open_llm
    models = [primary, *llm.FALLBACK_LLMS] if fallback else [primary]
    # parallel_tool_calls chỉ truyền cho OpenRouter; model dự phòng dùng mặc định của hãng
    bound = {id(m): m.bind_tools(TOOLS, parallel_tool_calls=False) if m is primary else m.bind_tools(TOOLS)
             for m in models}
    chain = FallbackMiddleware(*models[1:]) if fallback else None

    def invoke(messages: list) -> AIMessage:
        if chain is None:
            return bound[id(primary)].invoke(messages)
        return chain.call(primary, lambda m: bound[id(m)].invoke(messages))

    def call_model(state: AgentState) -> dict:
        status = describe(state)
        system = SystemMessage(content=f'{SYSTEM_PROMPT}\n\nTrạng thái hội thoại:\n{status}')
        messages = [system] + trim_history(state['messages'])
        reply = invoke(messages)
        if reply.tool_calls:
            return {'messages': [reply]}
        # Câu trả lời cuối: số liệu (giá, pin, bảo hành...) phải có trong kết quả tool, trạng thái hoặc tin của khách.
        # Có số không rõ nguồn thì cho model viết lại 1 lần; bản nháp sai không lưu vào lịch sử.
        sources = sources_of(state['messages'], status)
        bad = ungrounded(reply.text, sources)
        if bad:
            log.warning('Câu trả lời có số liệu không rõ nguồn %s, yêu cầu model viết lại', bad)
            reply = invoke([*messages, reply, HumanMessage(correction(bad))])
            if reply.tool_calls:
                return {'messages': [reply]}  # model đi tra cứu thêm: tool chạy rồi model trả lời lại từ đầu
            if still := ungrounded(reply.text, sources):
                log.warning('Viết lại vẫn còn số liệu không rõ nguồn %s', still)
        return {'messages': [reply.model_copy(update={'content': clean_answer(reply.text)})]}

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
