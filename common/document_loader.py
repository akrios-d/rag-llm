import os
import glob
import json
import logging
from langchain.schema import Document
from langchain_community.document_loaders import TextLoader, UnstructuredHTMLLoader, UnstructuredPDFLoader
import requests
from common.config import CONFLUENCE_API_URL, CONFLUENCE_API_KEY, CONFLUENCE_API_USER, MANTIS_API_URL, MANTIS_API_KEY, SESSION_FILE, DATA_DIR

logger = logging.getLogger(__name__)

def load_documents(from_confluence=False, from_mantis=False, use_history=False):
    documents = []

    documents.extend(load_local_files())

    if from_confluence:
        documents.extend(fetch_confluence_pages())
    if from_mantis:
        documents.extend(fetch_mantis_issues())
    if use_history:
        documents.extend(use_history())
    
    return documents

def fetch_confluence_pages():
    """Fetches the content of a Confluence page using the REST API."""
    documents = []
    CONFLUENCE_AUTH = (CONFLUENCE_API_USER, CONFLUENCE_API_KEY)  # Replace with actual Confluence username and API token

    for page_id in os.getenv("CONFLUENCE_PAGE_IDS", "").split(","):
        url = f"{CONFLUENCE_API_URL}/{page_id}?expand=body.storage"
        response = requests.get(url, auth=CONFLUENCE_AUTH )
        if response.status_code == 200:
            content = response.json()["body"]["storage"]["value"]
            documents.append(Document(page_content=content, metadata={"source": f"Confluence - {page_id}"}))
        else:
            logger.error(f"Error fetching Confluence page {page_id}. Status: {response.status_code}")
    return documents

def fetch_mantis_issues():
    documents = []
    headers = {"Authorization": f"Bearer {MANTIS_API_KEY}"}
    response = requests.get(f"{MANTIS_API_URL}/issues", headers=headers)
    if response.status_code == 200:
        issues = response.json()
        for issue in issues:
            content = f"{issue['summary']}\n{issue['description']}"
            documents.append(Document(page_content=content, metadata={"source": "Mantis"}))
    else:
        logger.error(f"Error fetching Mantis issues. Status: {response.status_code}")
    return documents

def load_chat_history():
    documents = []
     # Load chat history
    if os.path.exists(SESSION_FILE):
        with open(SESSION_FILE, 'r') as f:
            chat_history = json.load(f)
        for chat in chat_history:
            documents.append(Document(page_content=chat["response"], metadata={"source": "ChatHistory"}))
    return documents

def load_local_files():
    documents = []
    # Load local files
    for ext in ['*.pdf', '*.txt', '*.html']:
        for file_path in glob.glob(os.path.join(DATA_DIR, ext)):
            try:
                if file_path.endswith('.pdf'):
                    loader = UnstructuredPDFLoader(file_path)
                elif file_path.endswith('.txt'):
                    loader = TextLoader(file_path)
                elif file_path.endswith('.html'):
                    loader = UnstructuredHTMLLoader(file_path)
                else:
                    continue
                documents.extend(loader.load())
            except Exception as e:
                logger.error(f"Error loading file {file_path}: {e}")
    return documents