# Intelligent Document Retrieval System

This repository contains a modularized document retrieval system with two interfaces:  
1. **Command-Line Interface (CLI)**  
2. **Backend API**  

Both systems support fetching documents from local files, Confluence pages, and MantisBT issues and use vector databases (Chroma or PostgreSQL) for efficient querying.  

---

## Features

- **Document Sources**:
  - Local files (`*.pdf`, `*.txt`, `*.html`)
  - Confluence pages
  - MantisBT issues

- **Vector Databases**:
  - [Chroma](https://www.trychroma.com/)
  - [PostgreSQL](https://www.postgresql.org/) with `pgvector`

- **Embeddings**:
  - Hugging Face
  - OpenAI

- **Query Options**:
  - Single-query retrieval
  - Multi-query retrieval

- **Session Management**:
  - Chat history stored as JSON files (for CLI)
  - Shared documents for multiple sessions

---

## Requirements

- Python 3.9+
- PostgreSQL (if using `pgvector`)
- Confluence API (optional)
- MantisBT API (optional)

---

## Configuration

### Environment variables:

| Variable                   | Description                                      | Default                                |
|----------------------------|--------------------------------------------------|----------------------------------------|
| `POSTGRES_CONNECTION_STRING` | PostgreSQL connection string                    | `postgresql://user:password@localhost:5432/mydatabase` |
| `CHROMA_COLLECTION_NAME`    | Collection name for Chroma                       | `my-collection`                        |
| `EMBEDDING_MODEL_NAME`      | Model name for HuggingFace embeddings             | `all-MiniLM-L6-v2`                     |
| `LLM_MODEL`             | LLM model name                               | `llama3.2`                             |
| `DATA_DIR`                  | Directory for storing documents and chat history | `./data/`                              |
| `SESSION_FILE`              | Path for saving chat history                     | `./data/chat_history.json`             |
| `CONFLUENCE_API_URL`        | Base URL for Confluence API                      | None                                   |
| `CONFLUENCE_API_KEY`        | API key for Confluence                           | None                                   |
| `CONFLUENCE_PAGE_IDS`       | Comma-separated list of Confluence page IDs      | None                                   |
| `MANTIS_API_URL`            | Base URL for MantisBT API                        | None                                   |
| `MANTIS_API_KEY`            | API key for MantisBT                             | None                                   |


   Running the CLI Version

    Start the CLI app:

python cli_app.py

Interactive Session:

    You'll be prompted to:
        Fetch documents from Confluence or MantisBT.
        Choose a vector database (Chroma/Postgres).
        Choose an embedding model (HuggingFace/OpenAI).
    Ask questions interactively, and the system will use the documents to provide answers.

Chat History:

    Chat history is saved to ./data/chat_history.json.
    This file will be included in subsequent sessions.
Running the Backend API Version

    Start the API server:

python api_app.py

API Endpoints:

    Health Check: GET /health

curl http://localhost:5000/health

Response:

{
  "status": "ok"
}

Ask a Question: POST /ask

curl -X POST -H "Content-Type: application/json" -d '{"question": "What is the project about?"}' http://localhost:5000/ask

Response:

{
  "question": "What is the project about?",
  "answer": "This project is about..."
}


Example Use Case

    Interactive CLI: Quickly load documents locally or via APIs and ask questions in a terminal.
    Backend API: Integrate document-based QA capabilities into a web or mobile app.

Contributing

    Fork the repository.
    Create a new branch.
    Submit a pull request.

Extending the System

    Add New Data Sources:
        Implement a new loader function in common/document_loader.py.
        Update the CLI and API to include the new source.

    Support Additional Embedding Models:
        Add the integration in common/vectorstore.py.

    Custom Retrieval Strategies:
        Modify or extend the logic in common/retriever.py.