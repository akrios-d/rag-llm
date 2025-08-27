from common.document_loader import load_documents
from common.llm_chooser import get_llm
from common.vectorstore import create_vectorstore

from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph

# -------------------------------
# Load and process your documents
# -------------------------------
documents = load_documents()
doc_vectorstore = create_vectorstore(documents)
doc_retriever = doc_vectorstore.as_retriever()

# -------------------------------
# Define your LLM
# -------------------------------
llm = get_llm()

# -------------------------------
# Define your prompt templates
# -------------------------------
MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an AI language model assistant. Your task is to generate five different versions "
               "of the given user question to retrieve relevant documents from a vector database. "
               "By generating multiple perspectives on the user question, your goal is to help the user "
               "overcome some of the limitations of the distance-based similarity search."),
    ("user", "Original question: {question}")
])

RAG_PROMPT_TEMPLATE = """You are an AI assistant tasked with answering questions strictly based on the provided documents.
Do not use external knowledge or provide answers unrelated to the content retrieved from the documents.

Documents retrieved:
{context}

Conversation so far:
{history}

User Question: {question}

Provide a detailed and accurate response based on the documents above."""

# -------------------------------
# Define a ChatState class to hold conversation memory and retrieval state.
# -------------------------------
class ChatState:
    def __init__(self, query, documents=None, retrieved_prompts=None, chat_history=None):
        self.query = query
        self.documents = documents if documents is not None else []
        self.retrieved_prompts = retrieved_prompts if retrieved_prompts is not None else []
        self.chat_history = chat_history if chat_history is not None else []


# -------------------------------
# Define the LangGraph workflow
# -------------------------------
graph = StateGraph(ChatState)

# Define and add nodes
def retrieve_documents(state):
    """
    Generate multiple paraphrased queries, deduplicate them, retrieve documents for each,
    and then rerank the retrieved documents by similarity to the original query.
    """
    prompt_text = MULTI_QUERY_PROMPT.format_messages(question=state.query)
    multi_query_response = llm(prompt_text)
    multi_query_versions = list(set(
        [line.strip() for line in multi_query_response.content.split("\n") if line.strip()]
    ))

    retrieved_docs = []
    for query_version in multi_query_versions:
        retrieved_docs.extend(doc_retriever.get_relevant_documents(query_version))
    
    # Return a ChatState object instead of a dict
    return ChatState(
        query=state.query,
        documents=retrieved_docs,
        chat_history=state.chat_history
    )


graph.add_node("retrieve_documents", retrieve_documents)

def generate_response(state):
    """
    Generate a final answer based on retrieved documents and conversation history.
    """
    chat_history = state.chat_history
    context = "\n".join([doc.page_content for doc in state.documents])
    input_prompt = RAG_PROMPT_TEMPLATE.format(
        context=context,
        history="\n".join(chat_history),
        question=state.query
    )
    response = llm(input_prompt)
    updated_memory = chat_history + [f"User: {state.query}", f"AI: {response}"]
    
    # Return a ChatState object instead of a dict
    return ChatState(
        query=state.query,
        documents=state.documents,
        chat_history=updated_memory
    )


graph.add_node("generate_response", generate_response)

# -------------------------------
# Connect the graph nodes.
# -------------------------------
graph.add_edge("retrieve_documents", "generate_response")
graph.set_entry_point("retrieve_documents")

# Modify the flow of the graph nodes manually
state1 = ChatState(query="What is quantum entanglement?")
# Step 1: Retrieve documents
state1 = retrieve_documents(state1)

# Step 2: Generate response based on documents
state1 = generate_response(state1)

# Print the final response
print("Response:", state1.chat_history[-1])  # The most recent AI response
