from langchain_openai import ChatOpenAI
from config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL_NAME, LLM_TEMPERATURE

def get_llm():
    """Creates and returns a chat model client for the configured provider."""
    return ChatOpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        model=LLM_MODEL_NAME,
        temperature=LLM_TEMPERATURE,
    )


def get_llm_response(prompt: str) -> str:
    """
    Sends a fully-built, grounded prompt to the LLM and returns its answer.
    """
    llm = get_llm()
    response = llm.invoke(prompt)
    return response.content