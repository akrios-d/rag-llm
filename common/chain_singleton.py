from common.prompt import create_chain
import logging

logger = logging.getLogger(__name__)

class ChainSingleton:
    _instance = None
    _chain = None

    @classmethod
    def get_instance(cls):
        """Returns the Singleton instance of the ChainSingleton."""
        if ChainSingleton._instance is None:
            ChainSingleton._instance = ChainSingleton()
        return ChainSingleton._instance


    def initialize_chain(self, retriever, llm, memory):
        """Initializes the chain."""
        if ChainSingleton._chain is None:
            ChainSingleton._chain = create_chain(retriever, llm, memory)
        if ChainSingleton._chain is None:
            logger.error("Failed to create a valid chain.")
            return False
        logger.info("Processing chain created successfully.")

    def get_chain(self):
        """Returns the initialized chain."""
        return ChainSingleton._chain

    def clear_chain(self):
        """Clears the chain if you need to reinitialize it."""
        ChainSingleton._chain = None