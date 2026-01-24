import os
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel

from app.telemetry.cost_tracker import CostCallbackHandler

def get_llm(temperature: float = 0.0, model_name: str = "gpt-5-mini") -> BaseChatModel:
    """
    Get the primary reasoning LLM.
    Defaults to GPT-5 Mini (released Aug 2025).
    """
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        callbacks=[CostCallbackHandler()]
    )

def get_cheap_llm(temperature: float = 0.0) -> BaseChatModel:
    """
    Get the cost-effective LLM.
    Defaults to GPT-5 Mini.
    """
    return ChatOpenAI(
        model="gpt-5-mini",
        temperature=temperature,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        callbacks=[CostCallbackHandler()]
    )
