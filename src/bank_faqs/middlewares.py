# Manage gpt-6-sol's context window, summarize with gpt-6-luna
SummarizationMiddleware(
    model="openai:gpt-6-sol",
    trigger=("fraction", 0.8),
    summarizer="openai:gpt-6-luna",
)

# Summarize with a non-LLM service
async def summarize(messages: Sequence[BaseMessage]) -> str:
    return await my_summarization_service(get_buffer_string(messages))

SummarizationMiddleware(
    model="openai:gpt-6-luna",
    trigger=("fraction", 0.8),
    summarizer=summarize,
)
