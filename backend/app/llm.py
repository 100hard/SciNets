import os
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel

from app.telemetry.cost_tracker import CostCallbackHandler
from app.config import config

def get_llm(temperature: float = 0.0, model_name: str | None = None) -> BaseChatModel:
    """
    Get the primary reasoning LLM.
    Defaults to GPT-5 Mini.
    """
    model = model_name or getattr(config, "LLM_MODEL", "gpt-5-mini")
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        callbacks=[CostCallbackHandler()]
    )

def get_cheap_llm(temperature: float = 0.0) -> BaseChatModel:
    """
    Get the cost-effective LLM.
    Defaults to GPT-5 Mini.
    """
    model = getattr(config, "CHEAP_LLM_MODEL", "gpt-5-mini")
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        callbacks=[CostCallbackHandler()]
    )

