import logging
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

logger = logging.getLogger(__name__)

def create_retriever(vector_db, llm, use_multiquery):
    """Creates a retriever, optionally using MultiQueryRetriever."""
    if use_multiquery:
        logger.info("Using MultiQueryRetriever...")
        query_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI language model assistant. Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector database. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search."),
            ("user", "Original question: {question}")
        ])
        return MultiQueryRetriever.from_llm(vector_db.as_retriever(), llm, prompt=query_prompt)
    else:
        logger.info("Using basic retriever...")
        return vector_db.as_retriever()

def create_chain(retriever, llm, memory):
    """Creates the end-to-end RAG chain with memory."""
    template = """You are an AI assistant tasked with answering questions strictly based on the provided documents.
    Do not use external knowledge or provide answers unrelated to the content retrieved from the documents.
    
    Documents retrieved:
    {context}
    
    Conversation so far:
    {history}
    
    User Question: {question}
    
    Provide a detailed and accurate response based on the documents above."""
    prompt = ChatPromptTemplate.from_template(template)
    return (
        {"context": retriever, "history": memory.load_memory_variables, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
