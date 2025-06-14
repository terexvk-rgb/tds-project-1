# prepare_data_for_rag.py
import json
import os
import re
import time
from datetime import datetime
import chromadb
# Import the Google Generative AI client library
import google.generativeai as genai
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it in your .env file.")

# Configure the Google Generative AI client
genai.configure(api_key=GOOGLE_API_KEY)

# Define the embedding model for Gemini
# 'models/embedding-001' is a generally available embedding model.
# 'gemini-embedding-exp-03-07' is an experimental, higher-quality alternative if you want to try.
GEMINI_EMBEDDING_MODEL = "models/embedding-001" 

# Create the embedding function using GoogleGenerativeAiEmbeddingFunction from chromadb.utils.embedding_functions
gemini_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
    api_key=GOOGLE_API_KEY,
    model_name=GEMINI_EMBEDDING_MODEL
)

# ChromaDB persistence directory - this is where your vector database will be stored
CHROMA_DB_PATH = "chroma_db"
CHROMA_COLLECTION_NAME = "tds_course_data"

# --- Data Loading and Cleaning ---
def load_raw_data(file_path):
    """
    Loads raw data from a JSON file.
    Args:
        file_path (str): The path to the JSON file.
    Returns:
        list: A list of dictionaries, where each dictionary represents a raw document.
    """
    if not os.path.exists(file_path):
        print(f"Warning: Raw data file not found: {file_path}. Skipping.")
        return []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"Successfully loaded {len(data)} items from {file_path}")
            return data
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {file_path}: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while loading {file_path}: {e}")
        return []

def clean_text(text):
    """
    Performs basic text cleaning:
    - Removes excessive whitespace (multiple spaces, newlines).
    - Can be extended for more sophisticated cleaning (e.g., removing HTML tags if not done by BeautifulSoup,
      removing specific boilerplate text, handling special characters).
    Args:
        text (str): The input text.
    Returns:
        str: The cleaned text.
    """
    if not isinstance(text, str):
        return ""
    # Remove HTML tags (if any slipped through or if loading from 'cooked' without BeautifulSoup)
    cleaned_text = re.sub(r'<[^>]+>', '', text) 
    # Replace multiple newlines with single space, then multiple spaces with single space
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def chunk_text(data_items, chunk_size=1000, chunk_overlap=200):
    """
    Cleans text from data items and chunks them using RecursiveCharacterTextSplitter.
    Each chunk gets its own unique ID and retains original metadata.
    Args:
        data_items (list): List of dictionaries, each with 'text', 'source_url', 'type', 'title'.
        chunk_size (int): The maximum number of characters for each chunk.
        chunk_overlap (int): The number of characters to overlap between consecutive chunks.
    Returns:
        list: A list of dictionaries, where each dictionary represents a processed chunk.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        # Common separators to try to maintain semantic units
        separators=["\n\n", "\n", " ", ""], 
        add_start_index=True # Adds 'start_index' to metadata, useful for debugging
    )

    processed_chunks = []
    chunk_counter = 0

    for item_idx, item in enumerate(data_items):
        content_text = item.get('text', '')
        source_url = item.get('source_url', 'N/A')
        item_type = item.get('type', 'unknown')
        title = item.get('title', f"Document {item_idx}")

        # Perform cleaning on the raw text
        cleaned_text = clean_text(content_text)

        if not cleaned_text:
            # print(f"Skipping empty or invalid content for item {item_idx} from {source_url}")
            continue # Skip items with no valid text after cleaning

        # Split the cleaned text into chunks
        chunks = text_splitter.split_text(cleaned_text)

        for chunk_in_doc_idx, chunk_content in enumerate(chunks):
            # Ensure chunk_content is not empty after splitting, especially if original text was short
            if not chunk_content.strip():
                continue

            processed_chunks.append({
                'id': f"{item_type}_{item_idx}_{chunk_in_doc_idx}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{chunk_counter}", 
                # Unique ID for each chunk. Added datetime (with microseconds) and counter for extra uniqueness
                'content': chunk_content,
                'metadata': {
                    'source_url': source_url,
                    'type': item_type,
                    'title': title,
                    'chunk_index': chunk_in_doc_idx,
                    # You can add more metadata here, e.g., author for discourse posts
                    'original_item_id': item.get('id', f'item_{item_idx}') # Keep original item ID if available
                }
            })
            chunk_counter += 1 # Increment global chunk counter

    print(f"Original items processed: {len(data_items)}. Generated {len(processed_chunks)} usable chunks.")
    return processed_chunks

# --- Embedding and Storage in ChromaDB ---
def embed_and_store_data(chunks):
    """
    Generates embeddings for text chunks and stores them in a ChromaDB collection.
    Args:
        chunks (list): A list of processed chunk dictionaries.
    Returns:
        chromadb.api.models.Collection.Collection: The ChromaDB collection object.
    """
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # Get or create the collection
    try:
        # If collection exists, try to get it. This will use the existing embedding function.
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME, embedding_function=gemini_ef)
        print(f"Collection '{CHROMA_COLLECTION_NAME}' already exists. Attempting to add/upsert new data.")
        # If you want to force a fresh start every time, uncomment the lines below:
        # print(f"Deleting existing collection '{CHROMA_COLLECTION_NAME}' for a fresh start...")
        # client.delete_collection(name=CHROMA_COLLECTION_NAME)
        # collection = client.create_collection(name=CHROMA_COLLECTION_NAME, embedding_function=gemini_ef)
    except Exception as e: # Collection not found, create it
        print(f"Error getting collection (might not exist): {e}. Creating new collection '{CHROMA_COLLECTION_NAME}'...")
        collection = client.create_collection(name=CHROMA_COLLECTION_NAME, embedding_function=gemini_ef)

    documents_to_add = []
    metadatas_to_add = []
    ids_to_add = []

    for chunk in chunks:
        documents_to_add.append(chunk['content'])
        metadatas_to_add.append(chunk['metadata'])
        ids_to_add.append(chunk['id'])

    # Add in batches to avoid hitting API limits or memory issues with very large datasets
    # Google Generative AI embedding API also has rate limits, batching and delays help manage them.
    batch_size = 50 # Smaller batch size for potentially stricter free tier limits
    for i in range(0, len(documents_to_add), batch_size):
        batch_docs = documents_to_add[i:i+batch_size]
        batch_metadatas = metadatas_to_add[i:i+batch_size]
        batch_ids = ids_to_add[i:i+batch_size]

        print(f"Processing batch {int(i/batch_size) + 1} / {int(len(documents_to_add)/batch_size) + (1 if len(documents_to_add) % batch_size > 0 else 0)} ({len(batch_docs)} chunks)...")
        try:
            # upsert allows updating existing documents or adding new ones
            collection.upsert(
                documents=batch_docs,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            # Short delay to respect API rate limits. Adjust based on observed behavior.
            time.sleep(1) 
        except Exception as e:
            print(f"Error adding/upserting batch (IDs {batch_ids[0]} to {batch_ids[-1]}): {e}")
            # This is where a Gemini specific rate limit or quota error would show up.
            # You might need to wait and retry, or reduce total data if quota is strict.

    print(f"\nSuccessfully populated ChromaDB collection '{CHROMA_COLLECTION_NAME}' with {collection.count()} chunks.")
    return collection

# --- Main Execution Flow ---
if __name__ == "__main__":
    print("Starting data preparation for RAG pipeline using Gemini Embeddings...")

    # Define paths to your raw scraped JSON files
    course_content_file = os.path.join('data', 'course_content_raw.json')
    discourse_posts_file = os.path.join('data', 'discourse_posts_raw.json')

    # Step 1: Load raw scraped data
    course_content_raw = load_raw_data(course_content_file)
    discourse_posts_raw = load_raw_data(discourse_posts_file)

    # Combine all raw data
    all_raw_data = course_content_raw + discourse_posts_raw
    print(f"Total raw items loaded from all sources: {len(all_raw_data)}")

    # Step 2: Clean and chunk text
    processed_chunks = chunk_text(all_raw_data, chunk_size=1000, chunk_overlap=200)

    # Step 3: Embed and store in ChromaDB
    if processed_chunks:
        collection = embed_and_store_data(processed_chunks)
        print("\nData preparation complete. Vector database is ready!")
    else:
        print("No usable chunks processed. Data preparation skipped.")
    print(f"ChromaDB data is persisted at: {os.path.abspath(CHROMA_DB_PATH)}")

