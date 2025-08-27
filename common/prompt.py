import logging
from langchain.memory import ConversationBufferMemory
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

Provide a detailed and accurate response based on the documents above.""".strip()

# REQUIREMENTS_ANALYST
SYSTEM_REQUIREMENTS_ANALYST = (
"""
You are a senior Requirements Analyst. Your job is to iteratively ask concise, high-signal questions to capture everything needed to produce an excellent technical document for Confluence.


Guidelines:
- Ask one to three focused questions per turn.
- Use checklists and options when appropriate.
- Confirm constraints and acceptance criteria.
- Identify gaps, risks, dependencies.
- Stop asking about areas that are already sufficiently covered.
- If the user types 'generate', output nothing except a brief confirmation like: "Generating the document now." and end questioning.
"""
).strip()

# SYSTEM_DOC_WRITER
SYSTEM_DOC_WRITER = (
"""
You are a world-class technical writer creating a clear, skimmable, Confluence-ready document in Markdown.
Include only sections that are relevant and well-supported by the provided requirements. Avoid fluff.


Must-have structure (omit if truly N/A and explain why in Notes):
# <Title>
> Short abstract / executive summary


## Goals & Non-Goals
## Scope
## Stakeholders & Users
## Functional Requirements
## Non-Functional Requirements (Latency, Throughput, Privacy, Security, Compliance, Reliability, Observability)
## Architecture Overview
### Data Sources & Connectors
### Components & Sequence
## API / Interface Design (if applicable)
## Retrieval & Generation Workflow (for RAG systems)
## Evaluation & Metrics (e.g., accuracy, latency, cost)
## Deployment & Operations
## Security & Access Control
## Risks & Limitations
## Open Questions
## Project Plan & Milestones
## References


Write in crisp bullet points, tables where helpful, and short paragraphs. Make thoughtful assumptions to fill small gaps and label them as such.
"""
).strip()

def create_memory():
    return ConversationBufferMemory(input_key="question", memory_key="history")

def create_retriever(vector_db, llm):
    """Creates a MultiQueryRetriever."""

    if not vector_db:
        raise ValueError("Vector database cannot be None.")
    
    logger.info("Using MultiQueryRetriever...")
    return MultiQueryRetriever.from_llm(vector_db.as_retriever(), llm, prompt=MULTI_QUERY_PROMPT)

def simple_retriever(vector_db, llm):
    if not vector_db:
        raise ValueError("Vector database cannot be None.")
    logger.info("Using basic retriever...")
    return vector_db.as_retriever()

def create_chain(vector_db, llm):
    """Creates the end-to-end RAG chain with memory."""

    if not llm:
        raise ValueError("LLM cannot be None.")
    
    retriever = create_retriever(vector_db, llm)

    logger.info("Creating RAG chain...")

    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    
    return (
        {"context": retriever, "history": create_memory().load_memory_variables, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

def create_chain_analyst(vector_db, llm):
    """Creates the end-to-end Analyst chain with memory."""

    if not llm:
        raise ValueError("LLM cannot be None.")
    
    retriever = simple_retriever(vector_db, llm)

    logger.info("Creating Analyst chain...")

    prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_REQUIREMENTS_ANALYST),
            ("human", "Context so far (chronological transcript):\n{history}\n\nUser's latest reply:\n{input}\n\nGiven the context, ask the next best 1-3 questions to close gaps. Focus on missing details. If the user wrote 'generate', reply exactly 'Generating the document now.'"),
            ])
    
    return (
        {"context": retriever, "history": create_memory().load_memory_variables, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

def create_chain_writer(vector_db, llm):
    """Creates the end-to-end Writer chain with memory."""

    if not llm:
        raise ValueError("LLM cannot be None.")
    
    retriever = simple_retriever(vector_db, llm)

    logger.info("Creating Analyst chain...")

    prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_REQUIREMENTS_ANALYST),
            ("human", "Context so far (chronological transcript):\n{history}\n\nUser's latest reply:\n{input}\n\nGiven the context, ask the next best 1-3 questions to close gaps. Focus on missing details. If the user wrote 'generate', reply exactly 'Generating the document now.'"),
            ])
    
    return (
        {"context": retriever, "history": create_memory().load_memory_variables, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )