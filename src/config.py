import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def get_llm(temperature: float = 0) -> ChatOpenAI:
    """
    Return a ChatOpenAI instance.

    Behaviour is controlled entirely by env vars — no URL substring guessing:
      OPENAI_BASE_URL   → custom endpoint (leave blank for standard OpenAI)
      USE_RESPONSES_API → set to "true" to use the /responses endpoint
      AZURE_API_KEY     → set to "true" to send the api-key header for Azure
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY must be set in .env")

    base_url = os.getenv("OPENAI_BASE_URL")  # optional

    kwargs = dict(
        model=os.getenv("MODEL_NAME", "gpt-5-mini"),
        temperature=temperature,
        api_key=api_key,
    )

    if base_url:
        kwargs["base_url"] = base_url

    if os.getenv("AZURE_API_KEY", "").lower() == "true":
        kwargs["default_headers"] = {"api-key": api_key}

    if os.getenv("USE_RESPONSES_API", "").lower() == "true":
        kwargs["use_responses_api"] = True

    return ChatOpenAI(**kwargs)
