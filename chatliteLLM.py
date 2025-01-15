import requests
from langchain_community.document_loaders import UnstructuredPDFLoader, TextLoader, UnstructuredHTMLLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import Document
import logging
import os
from glob import glob

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurations
DATA_DIR = "./data/"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 300
COLLECTION_NAME = "rag-example"
USE_MULTIQUERY = True  # Set to True for MultiQueryRetriever

# Confluence API settings
CONFLUENCE_URL = "placeholder"
CONFLUENCE_PAGE_ID = ["placeholder","placeholder"]  # Replace with actual Confluence page ID
CONFLUENCE_AUTH = ("placeholder", "placeholder")  # Replace with actual Confluence username and API token
USE_CONFLUENCE = False

def fetch_confluence_page(page_id):
    """Fetches the content of a Confluence page using the REST API."""
    url = f"{CONFLUENCE_URL}/{page_id}?expand=body.storage"
    response = requests.get(url, auth=CONFLUENCE_AUTH)
    if response.status_code == 200:
        page_data = response.json()
        page_content = page_data['body']['storage']['value']    
        return page_content
    else:
        logger.error(f"Error fetching Confluence page: {response.status_code}")
        return None


def load_documents(directory, from_confluence=False):
    """Loads all documents from a directory or fetches from Confluence."""
    all_documents = []

    if from_confluence:
        logger.info("Fetching document from Confluence...")
        for confluenceDocument in CONFLUENCE_PAGE_ID:
            logger.info(confluenceDocument)
            confluence_content = fetch_confluence_page(confluenceDocument)
            if confluence_content:
                # Wrap the HTML content in a Document object
                all_documents.extend([Document(page_content=confluence_content, metadata={"source": "Confluence"})])
            else:
                logger.error("Failed to fetch Confluence content for pageId {confluenceDocument}")
                
        return all_documents

    # If not from Confluence, load from files in the directory
    supported_formats = ['*.pdf', '*.txt', '*.html']

    for format in supported_formats:
        files = glob(os.path.join(directory, format))
        for file_path in files:
            try:
                logger.info(f"Loading file: {file_path}")
                if file_path.endswith('.pdf'):
                    loader = UnstructuredPDFLoader(file_path)
                elif file_path.endswith('.txt'):
                    loader = TextLoader(file_path)
                elif file_path.endswith('.html'):
                    loader = UnstructuredHTMLLoader(file_path)
                else:
                    continue
                all_documents.extend(loader.load())
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
    return all_documents


def split_text(documents):
    """Splits text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return text_splitter.split_documents(documents)


def prepare_documents_for_embedding(chunks):
    """Prepares document chunks to ensure they are valid strings for embedding."""
    return [chunk.page_content for chunk in chunks if isinstance(chunk.page_content, str) and chunk.page_content.strip()]


def create_vector_db(chunks, embed_model, collection_name):
    """Creates a vector database using document chunks."""
    processed_chunks = prepare_documents_for_embedding(chunks)
    if not processed_chunks:
        logger.error("No valid chunks found for embedding. Exiting...")
        return None
    return Chroma.from_texts(
        texts=processed_chunks,
        embedding=OllamaEmbeddings(model=embed_model),
        collection_name=collection_name,
    )


def create_retriever(vector_db, llm, use_multiquery):
    """Creates a retriever, optionally using MultiQueryRetriever."""
    if use_multiquery:
        logger.info("Using MultiQueryRetriever...")
        query_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI language model assistant. Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector database. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search."),
            ("user", "Original question: {question}")
        ])
        return MultiQueryRetriever.from_llm(vector_db.as_retriever(), llm, prompt=query_prompt)
    else:
        logger.info("Using basic retriever...")
        return vector_db.as_retriever()

def create_chain(retriever, llm, memory):
    """Creates the end-to-end RAG chain with memory."""
    template = """You are an AI assistant tasked with answering questions strictly based on the provided documents.
    Do not use external knowledge or provide answers unrelated to the content retrieved from the documents.
    
    Documents retrieved:
    {context}
    
    Conversation so far:
    {history}
    
    User Question: {question}
    
    Provide a detailed and accurate response based on the documents above."""
    prompt = ChatPromptTemplate.from_template(template)
    return (
        {"context": retriever, "history": memory.load_memory_variables, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def main():
    # Load all documents from the data directory or Confluence
    logger.info("Loading documents...")
    documents = load_documents(DATA_DIR, from_confluence=USE_CONFLUENCE)  # Set to True to load from Confluence
    if not documents:
        logger.error("No documents loaded. Exiting...")
        return
    
    logger.info(f"Loaded {len(documents)} documents. Splitting into chunks...")
    chunks = split_text(documents)
    logger.info(f"Document split into {len(chunks)} chunks.")
    
    # Create vector database
    logger.info("Creating vector database...")
    vector_db = create_vector_db(chunks, EMBED_MODEL, COLLECTION_NAME)
    if vector_db is None:
        logger.error("Failed to create vector database. Exiting...")
        return
    
    # Initialize models and memory
    logger.info("Initializing models and memory...")
    llm = ChatOllama(model=CHAT_MODEL)
    memory = ConversationBufferMemory(input_key="question", memory_key="history")
    
    # Create retriever
    logger.info("Setting up retriever...")
    retriever = create_retriever(vector_db, llm, USE_MULTIQUERY)
    
    # Create the chain
    logger.info("Creating the processing chain...")
    chain = create_chain(retriever, llm, memory)
    
    # Interactive questioning
    logger.info("Ready for interactive questions. Type 'exit' to quit.")
    while True:
        user_question = input("\nAsk your question (or type 'exit' to quit): ").strip()
        if user_question.lower() == "exit":
            logger.info("Exiting...")
            break
        
        # Process the question
        logger.info(f"Processing question: {user_question}")
        response = chain.invoke(input={"question": user_question})
        
        # Print the response
        print(f"\Bot: {response}")

if __name__ == "__main__":
    main()
