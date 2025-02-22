import logging
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

logger = logging.getLogger(__name__)

from common.config import USE_MULTIQUERY

# Multi-query retriever prompt
MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an AI language model assistant. Your task is to generate five different versions "
               "of the given user question to retrieve relevant documents from a vector database. "
               "By generating multiple perspectives on the user question, your goal is to help the user "
               "overcome some of the limitations of the distance-based similarity search."),
    ("user", "Original question: {question}")
])

# RAG chain prompt
RAG_PROMPT_TEMPLATE = """You are an AI assistant tasked with answering questions strictly based on the provided documents.
Do not use external knowledge or provide answers unrelated to the content retrieved from the documents.

Documents retrieved:
{context}

Conversation so far:
{history}

User Question: {question}

Provide a detailed and accurate response based on the documents above."""


def create_retriever(vector_db, llm):
    """Creates a retriever, optionally using MultiQueryRetriever."""

    if not vector_db:
        raise ValueError("Vector database cannot be None.")
    
    if USE_MULTIQUERY:
        logger.info("Using MultiQueryRetriever...")
        return MultiQueryRetriever.from_llm(vector_db.as_retriever(), llm, prompt=MULTI_QUERY_PROMPT)
    else:
        logger.info("Using basic retriever...")
        return vector_db.as_retriever()

def create_chain(retriever, llm, memory):
    """Creates the end-to-end RAG chain with memory."""

    if not retriever:
        raise ValueError("Retriever cannot be None.")
    if not llm:
        raise ValueError("LLM cannot be None.")
    if not memory:
        raise ValueError("Memory module cannot be None.")
    
    logger.info("Creating RAG chain...")

    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    return (
        {"context": retriever, "history": memory.load_memory_variables, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
