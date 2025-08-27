import logging
from typing import Union, Dict, Type, Any

# Import different LLM implementations
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace
from common.config import LLM_MODEL, LLM_PROVIDER

logger = logging.getLogger(__name__)

# Map providers to their classes and default params
LLM_PROVIDERS: Dict[str, Dict[str, Any]] = {
    "ollama": {
        "class": ChatOllama,
        "params": {"temperature": 0.3, "max_tokens": 512},
    },
    "openai": {
        "class": ChatOpenAI,
        "params": {"temperature": 0.7},
    },
    "huggingface": {
        "class": ChatHuggingFace,
        "params": {"temperature": 0.7},
    },
}

def get_llm() -> Union[ChatOllama, ChatOpenAI, ChatHuggingFace]:
    """
    Factory function to return an LLM instance based on configuration.

    Returns:
        An instance of a configured LLM based on the LLM_PROVIDER environment variable.

    Raises:
        ValueError: If LLM_MODEL or LLM_PROVIDER is not configured properly.
    """
    if not LLM_MODEL:
        raise ValueError("LLM_MODEL is not set. Please configure it in the environment or settings.")

    provider = LLM_PROVIDER.lower() if LLM_PROVIDER else None
    logger.debug(f"Fetching LLM provider from environment: {provider}")

    if provider not in LLM_PROVIDERS:
        raise ValueError(
            f"Unsupported LLM provider: {provider}. "
            f"Supported providers are: {', '.join(LLM_PROVIDERS.keys())}"
        )

    provider_config = LLM_PROVIDERS[provider]
    llm_class: Type = provider_config["class"]
    default_params: Dict[str, Any] = provider_config.get("params", {})

    logger.info(f"Using {llm_class.__name__} as the LLM provider with model: {LLM_MODEL}")
    return llm_class(model=LLM_MODEL, **default_params)
