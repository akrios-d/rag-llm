import logging
from common import chain_singleton
from initialize import initialize_resources, chat_manager

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_chat_history_with_error_handling() -> list:
    """Load chat history and handle any potential errors."""
    try:
        return chat_manager.load_chat_history()
    except Exception as e:
        logger.error(f"Error loading chat history: {e}")
        return []  # Return empty list if loading fails

def handle_user_input() -> str:
    """Prompt the user for input and handle exit conditions."""
    try:
        user_input = input("\nAsk a question (or type 'exit' to quit): ").strip()
        if user_input.lower() in {"exit", "quit"}:
            return "exit"  # Return a special exit signal
        return user_input
    except (KeyboardInterrupt, EOFError) as e:
        logger.info("Interactive session terminated by user.")
        return "exit"  # Gracefully handle exit on user interruption

def process_user_question(user_question: str, chat_history: list) -> None:
    """Process the user's question and log the response."""
    try:
        # Ensure the question isn't empty
        if not user_question:
            logger.info("Empty question provided. Please enter a valid question.")
            return

        # Invoke the chain with the user's question and get the response
        chain_instance = chain_singleton.ChainSingleton.get_instance()

        # Now, access the chain via the instance
        chain = chain_instance.get_chain()

        response =chain.invoke(input={"question": user_question})
        logger.info(f"Processing response for: {user_question}")
        answer = response  # Adjust if response needs processing
        print(f"\nAnswer: {answer}")

        # Append the question and answer to chat history
        chat_manager.append_to_history(user_question, answer)
    except Exception as e:
        logger.error(f"Error processing question: {e}")

def main() -> None:
    """
    Main function to run the interactive document query CLI application.
    """

    # Initialize resources and handle failure
    if not initialize_resources():
        logger.error("Failed to initialize resources. Exiting.")
        return
    
    # Load chat history
    chat_history = load_chat_history_with_error_handling()

    logger.info("Starting interactive session. Type 'exit' to quit.")
    
    # Main loop for interactive session
    while True:
        user_question = handle_user_input()
        if user_question == "exit":
            break
        
        process_user_question(user_question, chat_history)
    
    logger.info("Exiting. Goodbye!")

if __name__ == "__main__":
    main()