from typing import List
from langchain.schema import Document
import logging
import requests

logger = logging.getLogger(__name__)

from common.config import (
    CONFLUENCE_API_URL, CONFLUENCE_API_KEY, CONFLUENCE_API_USER, CONFLUENCE_PAGE_IDS
)

def get_confluence_auth():
    """Returns authentication tuple for Confluence API."""
    logger.debug("Retrieving Confluence authentication credentials.")
    return CONFLUENCE_API_USER, CONFLUENCE_API_KEY

def fetch_confluence_pages(paginate=False) -> List[Document]:
    """Fetches Confluence pages using the API."""
    documents = []
    auth = get_confluence_auth()

    logger.info(f"Starting to fetch Confluence pages with paginate={paginate}.")

    if paginate:
        url = f"{CONFLUENCE_API_URL}/rest/api/content"
        start = 0
        limit = 25  # Adjust as needed

        logger.debug("Pagination mode enabled. Starting with page 1.")

        while True:
            params = {"start": start, "limit": limit, "expand": "body.storage"}
            response = requests.get(url, auth=auth, params=params)

            if response.status_code == 200:
                data = response.json()
                logger.debug(f"Fetched page {start // limit + 1}. Processing pages...")

                for page in data.get("results", []):
                    content = page.get("body", {}).get("storage", {}).get("value", "")
                    page_id = page.get("id", "unknown")
                    documents.append(Document(page_content=content, metadata={"source": f"Confluence - {page_id}"}))
                    logger.info(f"Successfully fetched page {page_id}.")
                
                if "next" in data.get("_links", {}):
                    start += limit  # Move to the next page
                    logger.debug(f"Moving to next page: {start // limit + 1}")
                else:
                    logger.info("No more pages to fetch. Ending pagination.")
                    break  # No more pages to fetch
            else:
                logger.error(f"Failed to fetch Confluence pages. Status: {response.status_code}, Response: {response.text}")
                break
    else:
        # Non-paginated request for specific page IDs
        logger.debug("Fetching specific Confluence pages without pagination.")
        for page_id in CONFLUENCE_PAGE_IDS:
            url = f"{CONFLUENCE_API_URL}/{page_id}?expand=body.storage"
            response = requests.get(url, auth=auth)

            if response.status_code == 200:
                content = response.json().get("body", {}).get("storage", {}).get("value", "")
                documents.append(Document(page_content=content, metadata={"source": f"Confluence - {page_id}"}))
                logger.info(f"Successfully fetched page {page_id}.")
            else:
                logger.error(f"Error fetching Confluence page {page_id}. Status: {response.status_code}, Response: {response.text}")

    logger.info(f"Finished fetching Confluence pages. Total pages fetched: {len(documents)}.")
    return documents