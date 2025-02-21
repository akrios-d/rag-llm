from flask import Flask, request, jsonify
import logging

from initialize import initialize_resources, chain, chat_manager
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.before_first_request
def startup() -> None:
    """
    Startup function to initialize resources before handling any requests.
    """
    if not initialize_resources():
        logger.error("Resource initialization failed. The application may not work properly.")

@app.route("/ask", methods=["POST"])
def ask_question():
    """
    Endpoint to process a user's question.
    
    Expects JSON payload with key 'question'. Returns the answer in JSON format.
    """
    try:
        data = request.get_json(force=True)
    except Exception as e:
        logger.error(f"Invalid JSON input: {e}")
        return jsonify({"error": "Invalid JSON input"}), 400

    user_question = data.get("question", "").strip()
    if not user_question:
        return jsonify({"error": "Question is required"}), 400

    # Load chat history
    try:
        chat_history = chat_manager.load_chat_history()
    except Exception as e:
        logger.error(f"Error loading chat history: {e}")
        chat_history = []

    # Ensure the processing chain is initialized
    if chain is None:
        return jsonify({"error": "Processing chain not initialized"}), 500

    # Process the question
    try:
        response = chain.invoke(input={"question": user_question})
        # Adjust response extraction as needed.
        answer = response  
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        return jsonify({"error": "Failed to process question"}), 500

    # Update chat history
    try:
        chat_manager.append_to_history(chat_history, user_question, answer)
        chat_manager.save_chat_history(chat_history)
    except Exception as e:
        logger.error(f"Error saving chat history: {e}")

    return jsonify({"answer": answer})

@app.route("/health", methods=["GET"])
def health_check():
    """
    Health check endpoint.
    """
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    app.run(debug=True)