from langchain_community.vectorstores import PGVector, Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from common.config import DB_TYPE, CHROMA_COLLECTION_NAME, POSTGRES_CONNECTION_STRING, EMBEDDING_MODEL, EMBEDDING_MODEL_NAME

def create_vectorstore(documents):
    if EMBEDDING_MODEL == "huggingface":
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False} 
        embedding = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME, 
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
    elif EMBEDDING_MODEL == "openai":
        embedding = OpenAIEmbeddings()
    else:
        raise ValueError("Unsupported embedding model!")

    if DB_TYPE == "chroma":
        return Chroma.from_documents(documents, embedding, collection_name=CHROMA_COLLECTION_NAME)
    elif DB_TYPE == "postgres":
        return PGVector.from_documents(documents, embedding, connection_string=POSTGRES_CONNECTION_STRING)
    else:
        raise ValueError("Unsupported database type!")
