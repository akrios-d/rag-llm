from flask import Flask, request, jsonify
import logging
from common import chain_singleton
from initialize import initialize_resources, chat_manager

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize resources on startup
if not initialize_resources():
    logger.error("Resource initialization failed. The application may not work properly.")

@app.route("/ask", methods=["POST"])
def ask_question():
    """
    Endpoint to process a user's question.
    
    Expects JSON payload with key 'question'. Returns the answer in JSON format.
    """
    data = request.get_json(silent=True)  # Prevents throwing an exception on invalid JSON
    if not data or "question" not in data:
        logger.warning("Invalid or missing 'question' in request payload.")
        return jsonify({"error": "Invalid or missing 'question'"}), 400

    user_question = data["question"].strip()
    if not user_question:
        return jsonify({"error": "Question cannot be empty"}), 400

    # Load chat history safely
    chat_history = []
    try:
        chat_history = chat_manager.load_chat_history()
    except Exception as e:
        logger.error(f"Error loading chat history: {e}")

    # Process the question
    try:
        chain_instance = chain_singleton.ChainSingleton.get_instance()
        chain = chain_instance.get_chain()
        response = chain.invoke(input={"question": user_question})
        answer = response  
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        return jsonify({"error": "Failed to process question"}), 500

    # Update chat history safely
    try:
        chat_manager.append_to_history(chat_history, user_question, answer)
        chat_manager.save_chat_history(chat_history)
    except Exception as e:
        logger.error(f"Error saving chat history: {e}")

    return jsonify({"answer": answer})

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)  # Allows external access