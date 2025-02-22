from langchain_community.vectorstores import PGVector, Chroma, ElasticVectorSearch
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from common.config import (
    DB_TYPE, CHROMA_COLLECTION_NAME, POSTGRES_CONNECTION_STRING, 
    EMBEDDING_MODEL, EMBEDDING_MODEL_NAME, ELASTICSEARCH_URL, ELASTICSEARCH_INDEX,
    ELASTICSEARCH_USERNAME, ELASTICSEARCH_PASSWORD
)
import logging

logger = logging.getLogger(__name__)

def chunk_documents(documents, chunk_size=512, chunk_overlap=50):
    """
    Splits documents into smaller chunks of text.

    Args:
        documents (list): List of documents to be chunked.
        chunk_size (int): The size of each chunk.
        chunk_overlap (int): The amount of overlap between chunks.

    Returns:
        List of chunked documents.
    """
    logger.info("Starting document chunking process...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunked_documents = text_splitter.split_documents(documents)
    logger.info("Document chunking completed. Total chunks created: %d", len(chunked_documents))
    return chunked_documents


def get_embedding_model():
    """
    Returns the appropriate embedding model based on the configuration.

    Returns:
        The embedding model (either HuggingFaceEmbeddings or OpenAIEmbeddings).
    
    Raises:
        ValueError: If an unsupported embedding model is specified.
    """
    logger.info("Selecting embedding model...")
    if EMBEDDING_MODEL == "huggingface":
        logger.info("Using Hugging Face embedding model: %s", EMBEDDING_MODEL_NAME)
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME, 
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )
    elif EMBEDDING_MODEL == "openai":
        logger.info("Using OpenAI embedding model: %s", EMBEDDING_MODEL_NAME)
        return OpenAIEmbeddings()
    else:
        logger.error("Unsupported embedding model: %s", EMBEDDING_MODEL)
        raise ValueError("Unsupported embedding model!")


def create_vectorstore(documents):
    """
    Creates a vector store (Chroma, PostgreSQL, or Elasticsearch) from the documents.

    Args:
        documents (list): The list of documents to be added to the vector store.

    Returns:
        The created vector store (Chroma, PGVector, or ElasticVectorSearch).

    Raises:
        ValueError: If an unsupported database type is specified.
    """
    logger.info("Creating vector store...")

    # Ensure documents are properly chunked
    logger.info("Chunking documents...")
    documents = chunk_documents(documents)

    # Get the embedding model
    embedding = get_embedding_model()

    # Create vector store based on DB_TYPE
    if DB_TYPE == "chroma":
        logger.info("Using Chroma as the vector store.")
        vectorstore = Chroma.from_documents(documents, embedding, collection_name=CHROMA_COLLECTION_NAME)

    elif DB_TYPE == "postgres":
        logger.info(f"Using PostgreSQL (PGVector) with connection: {POSTGRES_CONNECTION_STRING}")
        vectorstore = PGVector.from_documents(documents, embedding, connection_string=POSTGRES_CONNECTION_STRING)

    elif DB_TYPE == "elasticsearch":

        logger.info(f"Using Elasticsearch at {ELASTICSEARCH_URL}, index: {ELASTICSEARCH_INDEX}")
        vectorstore = ElasticVectorSearch.from_documents(
            documents,
            embedding,
            es_url=ELASTICSEARCH_URL,
            index_name=ELASTICSEARCH_INDEX,
            http_auth=(ELASTICSEARCH_USERNAME, ELASTICSEARCH_PASSWORD)  # Use auth from .env
        )

    else:
        logger.error("Unsupported database type: %s", DB_TYPE)
        raise ValueError("Unsupported database type!")

    logger.info("Vector store created successfully.")
    return vectorstore