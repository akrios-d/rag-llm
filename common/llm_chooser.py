import logging

from typing import Union

# Import different LLM implementations
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace
from common.config import LLM_MODEL, LLM_PROVIDER

logger = logging.getLogger(__name__)

def get_llm() -> Union[ChatOllama, ChatOpenAI, ChatHuggingFace]:
    """
    Factory function to return an LLM instance based on configuration.

    Returns:
        An instance of an LLM configured based on the LLM_PROVIDER environment variable.

    Raises:
        ValueError: If an unsupported LLM provider is specified.
    """
    logger.debug(f"Fetching LLM provider from environment: {LLM_PROVIDER}")
    
    if not LLM_MODEL:
        raise ValueError("LLM_MODEL is not set. Please configure it in the environment or settings.")
                         
    if LLM_PROVIDER == "ollama":
        logger.info(f"Using ChatOllama as the LLM provider with model: {LLM_MODEL}")
        return ChatOllama(model=LLM_MODEL)
    
    if LLM_PROVIDER == "openai":
        logger.info(f"Using ChatOpenAI as the LLM provider with model: {LLM_MODEL}")
        # Optionally pass additional parameters like temperature, model name, etc.
        return ChatOpenAI(model=LLM_MODEL, temperature=0.7)
    
    if LLM_PROVIDER == "huggingface":
        logger.info(f"Using huggingface as the LLM provider with model: {LLM_MODEL}")
        # Optionally pass additional parameters like temperature, model name, etc.
        return ChatHuggingFace(model=LLM_MODEL, temperature=0.7)
    
    else:
        raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")