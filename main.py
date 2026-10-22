import uuid

from langchain_core.messages import HumanMessage

from agent import build_agent


def main():
    print("=" * 65)
    print(" 🚗 Welcome to Apex Motors - Car Sales AI Assistant 🚗")
    print(" Powered by LangChain, LangGraph & LangSmith")
    print(" (Type 'quit' or 'exit' to end the conversation)")
    print("=" * 65 + "\n")

    agent = build_agent()
    # Unique thread_id persists conversation history in LangGraph's checkpointer
    thread_id = str(uuid.uuid4())
    config = {
        "configurable": {"thread_id": thread_id},
        "metadata": {"conversation_id": thread_id, "agent_type": "car_sales_agent"},
    }

    while True:
        try:
            user_input = input("Customer: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("\nAssistant: Thanks for stopping by Apex Motors! Have a great day!")
            break

        try:
            response = agent.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
            print(f"\nAssistant: {response['messages'][-1].content}\n")
        except KeyboardInterrupt:
            print("\nSession ended.")
            break
        except Exception as e:
            print(f"\n[Error]: {e}\n")


if __name__ == "__main__":
    main()
