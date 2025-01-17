import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Configuration variables
DB_TYPE = os.getenv("DB_TYPE", "chroma").lower()
POSTGRES_CONNECTION_STRING = os.getenv("POSTGRES_CONNECTION_STRING", "postgresql://user:password@localhost:5432/mydatabase")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "my-collection")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "huggingface").lower()
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
DATA_DIR = os.getenv("DATA_DIR", "./data/")
SESSION_FILE = os.getenv("SESSION_FILE", os.path.join(DATA_DIR, "chat_history.json"))
CONFLUENCE_API_URL = os.getenv("CONFLUENCE_API_URL")
CONFLUENCE_API_KEY = os.getenv("CONFLUENCE_API_KEY")
CONFLUENCE_PAGE_IDS = os.getenv("CONFLUENCE_PAGE_IDS", "").split(",")
MANTIS_API_URL = os.getenv("MANTIS_API_URL")
MANTIS_API_KEY = os.getenv("MANTIS_API_KEY")