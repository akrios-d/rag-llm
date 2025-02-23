import os
import logging
from dotenv import load_dotenv
from pathlib import Path
from typing import List, Optional

# Load environment variables from .env file
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

# === Helper Functions for Environment Variables ===
def get_env_str(key: str, default: Optional[str] = None) -> Optional[str]:
    """Fetches an environment variable as a string, with an optional default."""
    value = os.getenv(key, default)
    if value is None:
        logger.warning(f"Missing environment variable: {key}")
    return value

def get_env_int(key: str, default: int) -> int:
    """Fetches an environment variable as an integer, with a fallback default."""
    try:
        return int(os.getenv(key, default))
    except ValueError:
        logger.warning(f"Invalid integer for {key}, using default: {default}")
        return default

def get_env_bool(key: str, default: bool = False) -> bool:
    """Fetches an environment variable as a boolean."""
    value = os.getenv(key, str(default)).strip().lower()
    return value in ("1", "true", "yes")

def get_env_list(key: str, separator: str = ",", default: Optional[List[str]] = None) -> List[str]:
    """Fetches an environment variable as a list, splitting by a separator."""
    value = os.getenv(key, "")
    return value.split(separator) if value else (default or [])

# === Database Configuration ===
DB_TYPE = get_env_str("DB_TYPE", "chroma").lower()

# PostgreSQL Variables
POSTGRES_HOST = get_env_str("POSTGRES_HOST", "localhost")
POSTGRES_PORT = get_env_int("POSTGRES_PORT", 5432)
POSTGRES_DB = get_env_str("POSTGRES_DB", "mydatabase")
POSTGRES_USER = get_env_str("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = get_env_str("POSTGRES_PASSWORD", "password")

# Dynamically build PostgreSQL connection string
POSTGRES_CONNECTION_STRING = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

# Chroma Variables
CHROMA_COLLECTION_NAME = get_env_str("CHROMA_COLLECTION_NAME", "my-collection")

# Elasticsearch Variables
ELASTICSEARCH_URL = get_env_str("ELASTICSEARCH_URL", "http://elasticsearch:9200")
ELASTICSEARCH_INDEX = get_env_str("ELASTICSEARCH_INDEX", "my_index")
ELASTICSEARCH_USERNAME = get_env_str("ELASTICSEARCH_USERNAME", "user")
ELASTICSEARCH_PASSWORD = get_env_str("ELASTICSEARCH_PASSWORD", "password")

# === Embedding Model Configuration ===
EMBEDDING_MODEL = get_env_str("EMBEDDING_MODEL", "huggingface").lower()
EMBEDDING_MODEL_NAME = get_env_str("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# === LLM Model Configuration ===
LLM_MODEL = get_env_str("LLM_MODEL", "llama3.2")
LLM_PROVIDER = get_env_str("LLM_PROVIDER", "ollama")

# === Data Storage ===
DATA_DIR = Path(get_env_str("DATA_DIR", "./data/"))
DATA_DIR.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
SESSION_FILE = get_env_str("SESSION_FILE", str(DATA_DIR / "chat_history.json"))

# === Confluence & Mantis Configuration ===
CONFLUENCE_API_URL = get_env_str("CONFLUENCE_API_URL")
CONFLUENCE_API_KEY = get_env_str("CONFLUENCE_API_KEY")
CONFLUENCE_API_USER = get_env_str("CONFLUENCE_API_USER")
CONFLUENCE_PAGE_IDS = get_env_list("CONFLUENCE_PAGE_IDS")

MANTIS_API_URL = get_env_str("MANTIS_API_URL")
MANTIS_API_KEY = get_env_str("MANTIS_API_KEY")

# DOCUMENTS
USE_HISTORY = get_env_bool("USE_HISTORY", False)
USE_MANTIS = get_env_bool("USE_MANTIS", False)
USE_CONFLUENCE = get_env_bool("USE_CONFLUENCE", False)
USE_MULTIQUERY = get_env_bool("USE_MULTIQUERY", True)

#OPEN AI
OPENAI_API_KEY = get_env_str("OPENAI_API_KEY")

# === Log Missing Critical Variables ===
critical_vars = {
    "POSTGRES_CONNECTION_STRING": POSTGRES_CONNECTION_STRING,
    "MANTIS_API_URL": MANTIS_API_URL,
    "MANTIS_API_KEY": MANTIS_API_KEY,
    "CONFLUENCE_API_URL": CONFLUENCE_API_URL,
    "CONFLUENCE_API_KEY": CONFLUENCE_API_KEY,
    "ELASTICSEARCH_URL": ELASTICSEARCH_URL,
}

for var_name, value in critical_vars.items():
    if not value:
        logger.warning(f"Critical environment variable {var_name} is missing or empty!")
