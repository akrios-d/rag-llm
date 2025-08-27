import logging
import hashlib
from typing import List, Callable, Dict, Any

from elasticsearch import Elasticsearch
from sqlalchemy import create_engine

from langchain_community.vectorstores import Chroma, ElasticsearchStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector

from common.config import (
    DB_TYPE, CHROMA_COLLECTION_NAME, POSTGRES_CONNECTION_STRING, 
    EMBEDDING_MODEL, EMBEDDING_MODEL_NAME, ELASTICSEARCH_URL, ELASTICSEARCH_INDEX,
    ELASTICSEARCH_USERNAME, ELASTICSEARCH_PASSWORD, POSTGRES_HOST
)

logger = logging.getLogger(__name__)

# ---------- Utility Functions ---------- #

def generate_stable_id(doc) -> str:
    """Generate a stable SHA-256 ID for document consistency."""
    return hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()

def chunk_documents(documents: List, chunk_size: int = 512, chunk_overlap: int = 50) -> List:
    """Split documents into smaller chunks of text."""
    logger.info("Chunking %d documents (chunk_size=%d, overlap=%d)...", len(documents), chunk_size, chunk_overlap)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    logger.info("Created %d total chunks.", len(chunks))
    return chunks

def get_embedding_model():
    """Select embedding model based on config."""
    if EMBEDDING_MODEL == "huggingface":
        logger.info("Using HuggingFace embeddings: %s", EMBEDDING_MODEL_NAME)
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": False},
        )
    elif EMBEDDING_MODEL == "openai":
        logger.info("Using OpenAI embeddings: %s", EMBEDDING_MODEL_NAME)
        return OpenAIEmbeddings()
    else:
        msg = f"Unsupported embedding model: {EMBEDDING_MODEL}"
        logger.error(msg)
        raise ValueError(msg)

# ---------- Elasticsearch Helpers ---------- #

def reset_elasticsearch_index():
    """Reset (delete + recreate) the Elasticsearch index."""
    es_client = Elasticsearch(
        ELASTICSEARCH_URL,
        http_auth=(ELASTICSEARCH_USERNAME, ELASTICSEARCH_PASSWORD),
    )

    try:
        if es_client.indices.exists(index=ELASTICSEARCH_INDEX):
            logger.warning("Index '%s' exists. Dropping...", ELASTICSEARCH_INDEX)
            es_client.indices.delete(index=ELASTICSEARCH_INDEX)

        settings = {
            "settings": {"number_of_shards": 1, "number_of_replicas": 1},
            "mappings": {
                "properties": {
                    "page_content": {"type": "text"},
                    "metadata": {"type": "object"},
                }
            },
        }
        es_client.indices.create(index=ELASTICSEARCH_INDEX, body=settings)
        logger.info("Created fresh index: %s", ELASTICSEARCH_INDEX)
    except Exception as e:
        logger.exception("Failed resetting Elasticsearch index.")
        raise RuntimeError(f"Elasticsearch index reset failed: {e}") from e

# ---------- Vector Store Registry ---------- #

def _create_chroma(documents, embedding):
    return Chroma.from_documents(documents, embedding, collection_name=CHROMA_COLLECTION_NAME)

def _create_postgres(documents, embedding):
    engine = create_engine(POSTGRES_CONNECTION_STRING)
    vectorstore = PGVector(connection=engine, embeddings=embedding)

    ids = [generate_stable_id(doc) for doc in documents]
    unique_ids = list(dict.fromkeys(ids))  # preserve order + remove duplicates

    logger.info("Performing upsert for %d unique documents into PGVector.", len(unique_ids))
    vectorstore.delete(unique_ids)
    vectorstore.add_texts(
        texts=[doc.page_content for doc in documents],
        metadatas=[doc.metadata for doc in documents],
        ids=unique_ids,
        conflict_action="update",
    )
    return vectorstore

def _create_elasticsearch(documents, embedding):
    reset_elasticsearch_index()
    return ElasticsearchStore.from_documents(
        documents,
        embedding,
        es_url=ELASTICSEARCH_URL,
        es_user=ELASTICSEARCH_USERNAME,
        es_password=ELASTICSEARCH_PASSWORD,
        index_name=ELASTICSEARCH_INDEX,
    )

VECTORSTORE_REGISTRY: Dict[str, Callable[[List, Any], Any]] = {
    "chroma": _create_chroma,
    "postgres": _create_postgres,
    "elasticsearch": _create_elasticsearch,
    # future extension:
    # "pinecone": _create_pinecone,
    # "weaviate": _create_weaviate,
    # "milvus": _create_milvus,
}

# ---------- Factory Function ---------- #

def create_vectorstore(documents: List):
    """
    Create a vector store from documents using the configured DB_TYPE.
    """
    logger.info("Initializing vector store with DB_TYPE=%s", DB_TYPE)
    documents = chunk_documents(documents)
    embedding = get_embedding_model()

    if DB_TYPE not in VECTORSTORE_REGISTRY:
        supported = ", ".join(VECTORSTORE_REGISTRY.keys())
        msg = f"Unsupported DB_TYPE: {DB_TYPE}. Supported: [{supported}]"
        logger.error(msg)
        raise ValueError(msg)

    return VECTORSTORE_REGISTRY[DB_TYPE](documents, embedding)
