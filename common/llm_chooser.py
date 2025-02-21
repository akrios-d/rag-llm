import logging
import os

from typing import Union

# Import different LLM implementations
from langchain_ollama import ChatOllama
from langchain_community.chat_models import ChatOpenAI

from common.config import LLM_MODEL

logger = logging.getLogger(__name__)

def get_llm() -> Union[ChatOllama, ChatOpenAI]:
    """
    Factory function to return an LLM instance based on configuration.

    Returns:
        An instance of an LLM configured based on the LLM_PROVIDER environment variable.

    Raises:
        ValueError: If an unsupported LLM provider is specified.
    """
    llm_provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    logger.debug(f"Fetching LLM provider from environment: {llm_provider}")
    
    if not LLM_MODEL:
        raise ValueError("LLM_MODEL is not set. Please configure it in the environment or settings.")
                         
    if llm_provider == "ollama":
        logger.info(f"Using ChatOllama as the LLM provider with model: {LLM_MODEL}")
        return ChatOllama(model=LLM_MODEL)
    
    if llm_provider == "openai":
        logger.info(f"Using ChatOpenAI as the LLM provider with model: {LLM_MODEL}")
        # Optionally pass additional parameters like temperature, model name, etc.
        return ChatOpenAI(model=LLM_MODEL)
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}")