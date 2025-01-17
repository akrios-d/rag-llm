from flask import Flask, request, jsonify
import os
from common.chat_history import ChatHistoryManager
from common.document_loader import load_documents
from common.vectorstore import create_vectorstore
from common.retriever import create_retriever, create_chain
from common.config import LLM_MODEL
from langchain.memory import ConversationBufferMemory
from langchain_ollama import ChatOllama
from common.config import DATA_DIR, SESSION_FILE

app = Flask(__name__)

chat_manager = ChatHistoryManager(session_file=SESSION_FILE)

@app.route("/ask", methods=["POST"])
def ask_question():
    user_question = request.json.get("question", "")
    if not user_question:
        return jsonify({"error": "Question is required"}), 400

    chat_history = chat_manager.load_chat_history()
    documents = load_documents()

    llm = ChatOllama(model=LLM_MODEL)
    memory = ConversationBufferMemory(input_key="question", memory_key="history")

    if not documents:
        return jsonify({"error": "No documents available"}), 500

    vector_db = create_vectorstore(documents)
    retriever = create_retriever(vector_db, llm, True)

    # Create the chain
    chain = create_chain(retriever, llm, memory)

    #context = retriever.get_relevant_documents(user_question)
    response =  chain.invoke(input={"question": user_question})

    answer = response#response['choices'][0]['message']['content']
    
    chat_manager.append_to_history(chat_history, user_question, answer)
    chat_manager.save_chat_history(chat_history)

    return jsonify({"answer": answer})

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    app.run(debug=True)