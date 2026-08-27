import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def get_llm(temperature: float = 0) -> ChatOpenAI:
    """
    Return a ChatOpenAI instance.
      * Sends 'api-key' header
      * Uses /responses endpoint 
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
        if "azure" in base_url:
            kwargs["default_headers"] = {"api-key": api_key}
            kwargs["use_responses_api"] = True

    return ChatOpenAI(**kwargs)
