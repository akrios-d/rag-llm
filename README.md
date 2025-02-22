# Intelligent Document Retrieval System

This repository contains a modularized document retrieval system with two interfaces:
1. **Command-Line Interface (CLI)**
2. **Backend API**

Both systems support fetching documents from local files, Confluence pages, and MantisBT issues, and use vector databases (Chroma, PostgreSQL, or Elasticsearch) for efficient querying.

## Features

- **Document Sources**:
  - Local files (`*.pdf`, `*.txt`, `*.html`)
  - Confluence pages
  - MantisBT issues
  - Chat history from previous sessions

- **Vector Databases**:
  - [Chroma](https://www.trychroma.com/)
  - [PostgreSQL](https://www.postgresql.org/) with `pgvector`
  - [Elasticsearch](https://www.elastic.co/elasticsearch)

- **Embeddings**:
  - Hugging Face
  - OpenAI
  - Ollama

- **Query Options**:
  - Single-query retrieval
  - Multi-query retrieval

- **Session Management**:
  - Chat history stored as JSON files (for CLI)
  - Shared documents for multiple sessions

---

## Requirements

- Python 3.12+
- PostgreSQL (if using `pgvector`)
- Confluence API (optional)
- MantisBT API (optional)
- Elasticsearch (optional)

---

## Configuration

The system reads configuration values from environment variables, typically stored in a `.env` file. Below are the configuration settings you can customize:

### Environment Variables:

| Variable                        | Description                                                | Default                                  |
|----------------------------------|------------------------------------------------------------|------------------------------------------|
| `DB_TYPE`                        | Type of the vector store (e.g., `chroma`, `postgresql`)     | `chroma`                                 |
| **PostgreSQL Configuration**     |                                                            |                                          |
| `POSTGRES_HOST`                  | Host for PostgreSQL                                         | `localhost`                              |
| `POSTGRES_PORT`                  | Port for PostgreSQL                                         | `5432`                                   |
| `POSTGRES_DB`                    | PostgreSQL database name                                    | `mydatabase`                             |
| `POSTGRES_USER`                  | PostgreSQL user name                                        | `postgres`                               |
| `POSTGRES_PASSWORD`              | PostgreSQL password                                         | `password`                               |
| `POSTGRES_CONNECTION_STRING`     | Dynamically generated connection string for PostgreSQL      | `postgresql://postgres:password@localhost:5432/mydatabase` |
| **Chroma Configuration**         |                                                            |                                          |
| `CHROMA_COLLECTION_NAME`         | Name for Chroma collection                                  | `my-collection`                          |
| **Elasticsearch Configuration**  |                                                            |                                          |
| `ELASTICSEARCH_URL`              | URL for Elasticsearch instance                              | `http://elasticsearch:9200`              |
| `ELASTICSEARCH_INDEX`            | Elasticsearch index name                                    | `my_index`                               |
| `ELASTICSEARCH_USERNAME`         | Username for Elasticsearch                                  | `user`                                   |
| `ELASTICSEARCH_PASSWORD`         | Password for Elasticsearch                                  | `password`                               |
| **Embedding Model Configuration**|                                                            |                                          |
| `EMBEDDING_MODEL`                | Embedding model to use (e.g., `huggingface`, `openai`)      | `huggingface`                            |
| `EMBEDDING_MODEL_NAME`           | Name of the embedding model (HuggingFace, OpenAI)           | `all-MiniLM-L6-v2`                       |
| **LLM Model Configuration**      |                                                            |                                          |
| `LLM_MODEL`                      | The LLM model name (e.g., `llama3.2`)                       | `llama3.2`                               |
| **Data Storage Configuration**   |                                                            |                                          |
| `DATA_DIR`                       | Directory for storing documents and chat history            | `./data/`                                |
| `SESSION_FILE`                   | Path for saving chat history (JSON format)                  | `./data/chat_history.json`               |
| **Confluence Configuration**     |                                                            |                                          |
| `CONFLUENCE_API_URL`             | Base URL for the Confluence API                             | None                                     |
| `CONFLUENCE_API_KEY`             | API key for Confluence                                      | None                                     |
| `CONFLUENCE_API_USER`            | API user for Confluence                                     | None                                     |
| `CONFLUENCE_PAGE_IDS`            | List of Confluence page IDs (comma-separated)               | None                                     |
| **MantisBT Configuration**       |                                                            |                                          |
| `MANTIS_API_URL`                 | Base URL for MantisBT API                                   | None                                     |
| `MANTIS_API_KEY`                 | API key for MantisBT                                        | None                                     |

---

### Critical Environment Variables

The following critical environment variables must be set for the system to function properly. Missing values will trigger warnings in the logs.

| Variable                        | Description                                                |
|----------------------------------|------------------------------------------------------------|
| `POSTGRES_CONNECTION_STRING`     | Connection string for PostgreSQL                           |
| `MANTIS_API_URL`                 | Base URL for MantisBT API                                  |
| `MANTIS_API_KEY`                 | API key for MantisBT                                       |
| `CONFLUENCE_API_URL`             | Base URL for Confluence API                                |
| `CONFLUENCE_API_KEY`             | API key for Confluence                                     |

---

### Configuration Handling

The system uses the `python-dotenv` package to load environment variables from a `.env` file. Additionally, the helper functions manage these environment variables with proper logging, validation, and fallback defaults.

- **Helper Functions**:
  - `get_env_str()`: Fetches a string environment variable with an optional default.
  - `get_env_int()`: Fetches an integer environment variable, with error handling for invalid values.
  - `get_env_bool()`: Fetches a boolean environment variable (supports `true`, `1`, `yes`).
  - `get_env_list()`: Fetches a list from a comma-separated string.

### Sample `.env` File:

Here is a sample `.env` file format for configuring the system:

```ini
DB_TYPE=postgresql
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=mydatabase
POSTGRES_USER=postgres
POSTGRES_PASSWORD=password
CHROMA_COLLECTION_NAME=my-collection
ELASTICSEARCH_URL=http://elasticsearch:9200
ELASTICSEARCH_INDEX=my_index
ELASTICSEARCH_USERNAME=user
ELASTICSEARCH_PASSWORD=password
EMBEDDING_MODEL=huggingface
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
LLM_MODEL=llama3.2
DATA_DIR=./data/
SESSION_FILE=./data/chat_history.json
CONFLUENCE_API_URL=
CONFLUENCE_API_KEY=
CONFLUENCE_API_USER=
CONFLUENCE_PAGE_IDS=
MANTIS_API_URL=
MANTIS_API_KEY=
```
This configuration setup ensures flexibility and ease of integration with various data sources, databases, and machine learning models.

---

## Running the CLI Version

1. **Start the CLI app:**

    To start the CLI app, run the following command:

    ```bash
    python cli_app.py
    ```

2. **Choosing Document Sources and Options:**

    In the `initialize.py` file, you can choose the sources and configurations that best suit your needs. 

    The `load_documents()` function is responsible for fetching documents, and it has the following signature:

    ```python
    def load_documents(from_confluence=False, from_mantis=False, use_history=False) -> List[Document]:
    ```

    On line 41 of `initialize.py`, you'll find the following:

    ```python
    documents = load_documents(use_history=True)
    ```

    - **`from_confluence`:** Set this to `True` if you want to fetch documents from Confluence.
    - **`from_mantis`:** Set this to `True` if you want to fetch documents from MantisBT.
    - **`use_history`:** Set this to `True` if you want to load chat history from previous sessions.

    For now, you can modify this file to activate the sources you wish to use. In the future, these options will be configurable through environment variables.

3. **Chat History:**

    - Chat history is saved to `./data/chat_history.json`.
    - This file will be included in subsequent sessions, allowing you to maintain continuity in conversations.

---

## Running the Backend API Version

1. **Start the API server:**

    ```bash
    python api_app.py
    ```

2. **API Endpoints:**

    - **Health Check:**  
      `GET /health`

      Example request:

      ```bash
      curl http://localhost:5000/health
      ```

      Example response:

      ```json
      {
        "status": "ok"
      }
      ```

    - **Ask a Question:**  
      `POST /ask`

      Example request:

      ```bash
      curl -X POST -H "Content-Type: application/json" -d '{"question": "What is the project about?"}' http://localhost:5000/ask
      ```

      Example response:

      ```json
      {
        "question": "What is the project about?",
        "answer": "This project is about..."
      }
      ```

---

## Example Use Case

- **Interactive CLI:** Quickly load documents locally or via APIs and ask questions in a terminal.
- **Backend API:** Integrate document-based QA capabilities into a web or mobile app.

---

## Contributing

1. Fork the repository.
2. Create a new branch.
3. Submit a pull request.

---

## Extending the System

- **Add New Data Sources:**
  - Implement a new loader function in `common/document_loader.py`.
  - Update the CLI and API to include the new source.

- **Support Additional Embedding Models:**
  - Add the integration in `common/vectorstore.py`.

- **Custom Retrieval Strategies:**
  - Modify or extend the logic in `common/prompt.py`

---

## Logging

The project uses Python’s logging module for better traceability and debugging. Ensure that logging is appropriately configured to capture warnings, errors, and important events. Logs are written to the console by default, but you can configure it to write to files as needed.

---

## Helper Functions

The following helper functions are available to manage various operations more efficiently:

- `get_env_str()`: Fetches environment variables as strings.
- `get_env_int()`: Fetches environment variables as integers with error handling.
- `get_env_bool()`: Fetches environment variables as booleans (from values like true, 1, yes).
- `get_env_list()`: Fetches environment variables as lists by splitting on commas.

These functions provide better validation and error handling, ensuring the application runs smoothly without missing or incorrect configuration.

---

## Docker Compose

The provided Docker Compose configuration includes services for **PostgreSQL**, **Elasticsearch**, and **Ollama**. However, you can choose which services to keep or modify according to your needs.

### Steps for Docker Compose:

1. Navigate to the `dockercompose` folder in the repository.
2. Run the following command to start the services defined in the Docker Compose configuration:

    ```bash
    docker-compose up -d
    ```

3. **Initialize the `ollama` container with a model**:

    Once you've started Docker Compose, you'll need to run the following command to initialize the `ollama` container with a model:

    ```bash
    docker exec -it ollama ollama run llama3.2
    ```

    This will ensure that the `llama3.2` model is available in your system.

After these steps, the system will be fully operational, and you can use the `ollama` model for document retrieval and querying as part of the Dockerized environment.