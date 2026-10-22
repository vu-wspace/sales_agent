import os

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from tools import tools

# Load environment variables (.env)
load_dotenv()

SYSTEM_PROMPT = """You are "DriveMatch AI", a friendly, consultative, and knowledgeable automotive sales assistant at Apex Motors.

Your primary goals are:
1. Understand customer needs: budget, family size, commute style, electric vs hybrid vs gas preferences, and must-have features.
2. Recommend the best matching vehicles using your `search_cars` and `get_car_details` tools.
3. Highlight key selling points from the inventory data (features, fuel type, mileage, price).
4. Guide high-intent customers toward taking action: encourage scheduling a test drive with the `book_test_drive` tool.

Guidelines:
- Always check the inventory with `search_cars` before giving vehicle recommendations. Don't invent fake cars, prices, or specs.
- Before calling `book_test_drive`, collect the customer's name, phone number, and preferred date/time. Never make these up.
- Be polite, professional, and enthusiastic without being overly pushy.
- If asked about financing, mention that 3.9% - 6.9% APR financing is available upon credit approval.
- Format pricing clearly (e.g. $28,500).
"""


def build_agent():
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.3)
    llm_with_tools = llm.bind_tools(tools)

    def call_model(state: MessagesState):
        # The system prompt is not stored in state, so prepend it on every call
        messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        return {"messages": [llm_with_tools.invoke(messages)]}

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")

    return workflow.compile(checkpointer=MemorySaver())
