# 🚗 Car Sales Agent (LangChain + LangGraph + LangSmith)

A simple, extensible automotive dealership sales agent built with **LangChain**, **LangGraph**, and **LangSmith**.

---

## 🌟 Key Features

1. **Consultative Car Sales System Prompt**: Acts as a friendly sales specialist at Apex Motors, qualifying buyer needs and suggesting inventory matching budget, vehicle type, and fuel preference.
2. **LangGraph State Machine**: Manages the multi-turn sales conversation state, function routing, and execution flow.
3. **Inventory & Scheduling Tools**:
   - `search_cars`: Query active car inventory by max price, body type (SUV, Sedan, Truck), make, or fuel type (Hybrid, Electric, Gasoline).
   - `get_car_details`: Fetch full features and specs by vehicle ID.
   - `book_test_drive`: Schedule test drives and dealership appointments.
4. **LangSmith Observability**: Automatic tracing for tool invocations, prompt iterations, token metrics, and multi-turn threads.

---

## 📁 Repository Structure

```text
sales_agent/
├── agent.py          # LangGraph graph definition (state, model call, tool node)
├── tools.py          # Inventory data and tool definitions
├── main.py           # Interactive CLI chat loop
├── requirements.txt  # Dependency list
└── .env.example      # Environment variable template
```

---

## 🚀 Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and set your API keys:
```bash
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o-mini   # optional
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=car-sales-agent
```

### 3. Run the Agent CLI
```bash
python main.py
```

---

## 💬 Example Interactions

**Customer:** *"I'm looking for a hybrid or electric car under $40k."*  
**Assistant:** Queries inventory using `search_cars`, displays options like the *2024 Toyota Camry Hybrid* or *2024 Tesla Model 3*, highlighting key specs and features.

**Customer:** *"Can I book a test drive for the Honda CR-V this Saturday?"*  
**Assistant:** Triggers `book_test_drive(car_id='CAR002', ...)` and confirms appointment details.

---

## 🛠️ LangSmith Integration

All runs automatically trace to [LangSmith](https://smith.langchain.com/) when `LANGSMITH_TRACING=true` is enabled in `.env`. You can monitor:
- Tool execution parameters and return values.
- Latency and token usage per conversation round.
- Turn-by-turn conversation graphs and state updates.
