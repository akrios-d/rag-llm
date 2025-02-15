import logging
import os

from typing import Any

# Import different LLM implementations
from langchain_ollama import ChatOllama
from langchain_community.chat_models import ChatOpenAI

from common.config import LLM_MODEL

logger = logging.getLogger(__name__)

def get_llm() -> Any:
    """
    Factory function to return an LLM instance based on configuration.

    Returns:
        An instance of an LLM configured based on the LLM_PROVIDER environment variable.

    Raises:
        ValueError: If an unsupported LLM provider is specified.
    """
    llm_provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    if llm_provider == "ollama":
        logger.info("Using ChatOllama as the LLM provider.")
        return ChatOllama(model=LLM_MODEL)
    elif llm_provider == "openai":
        logger.info("Using ChatOpenAI as the LLM provider.")
        # Optionally pass additional parameters like temperature, model name, etc.
        return ChatOpenAI(model=LLM_MODEL)
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}")