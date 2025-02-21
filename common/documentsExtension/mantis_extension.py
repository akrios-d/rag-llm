from typing import List
from langchain.schema import Document
import logging
import requests

logger = logging.getLogger(__name__)

from common.config import (
    MANTIS_API_URL, MANTIS_API_KEY
)


def fetch_mantis_issues() -> List[Document]:
    """Fetches issues from Mantis API."""
    documents = []
    headers = {"Authorization": f"Bearer {MANTIS_API_KEY}"}

    response = requests.get(f"{MANTIS_API_URL}/issues", headers=headers)
    if response.status_code == 200:
        issues = response.json()
        for issue in issues:
            content = f"{issue.get('summary', 'No summary')}\n{issue.get('description', 'No description')}"
            documents.append(Document(page_content=content, metadata={"source": "Mantis"}))
    else:
        logger.error(f"Error fetching Mantis issues. Status: {response.status_code}, Response: {response.text}")

    return documents