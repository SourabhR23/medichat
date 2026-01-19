import chromadb
from huggingface_hub import InferenceClient
from typing import List
import os
import requests
from app.config import EMBEDDING_MODEL,HF_API_TOKEN   # e.g. "sentence-transformers/all-MiniLM-L6-v2"
from app.pdf_utils import clean_text
from dotenv import load_dotenv

load_dotenv()

# Create client ONCE (important)
hf_client = InferenceClient(
    model=EMBEDDING_MODEL,
    token=HF_API_TOKEN,
)

# ChromaDB Cloud Configuration
CHROMA_API_KEY = os.getenv('CHROMA_API_KEY', '')
CHROMA_TENANT = os.getenv('CHROMA_TENANT', '')
CHROMA_DATABASE = os.getenv('CHROMA_DATABASE', 'medibot')
CHROMA_COLLECTION = os.getenv('CHROMA_COLLECTION', 'medical_documents')

# Load the embedding model once at import time (efficient reuse)
# EMBEDDING_MODEL should be a HF model name, e.g.:
# "sentence-transformers/all-MiniLM-L6-v2" or "BAAI/bge-small-en-v1.5"
#embedding_model = SentenceTransformer(EMBEDDING_MODEL)

def get_chroma_client():
    """Get ChromaDB cloud client"""
    return chromadb.CloudClient(
        api_key=CHROMA_API_KEY,
        tenant=CHROMA_TENANT,
        database=CHROMA_DATABASE
    )

def _hf_embedding_request(texts: List[str]) -> List[List[float]]:
    if not HF_API_TOKEN:
        raise RuntimeError("HF_API_TOKEN is not set")

    # E5 models REQUIRE prefixes
    inputs = [f"passage: {t}" for t in texts]

    embeddings = hf_client.feature_extraction(inputs)

    return embeddings

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Get embeddings for texts using Hugging Face Inference API.

    Args:
        texts: List of input texts.

    Returns:
        List of embedding vectors (each a list[float]).
    """
    if not texts:
        return []

    cleaned_texts = [clean_text(t) for t in texts]

    try:
        return _hf_embedding_request(cleaned_texts)
    except Exception as e:
        print(f"Error generating embeddings from Hugging Face API: {e}")
        return []
    
def create_chroma_collection(texts: List[str], batch_size: int = 10):
    """Create ChromaDB collection and store documents in batches"""
    try:
        pass
        
        # Ensure collection exists and is not soft deleted
        collection = ensure_collection_exists()
        if not collection:
            return None
        
        # Process documents in batches
        total_stored = 0
        for i in range(0, len(texts), batch_size):
            batch_texts = [clean_text(text) for text in texts[i:i + batch_size]]
            batch_num = (i // batch_size) + 1
            total_batches = (len(texts) + batch_size - 1) // batch_size
            
            print(f"🧠 Processing batch {batch_num}/{total_batches} ({len(batch_texts)} documents)...")
            
            # Get embeddings for this batch
            embeddings = get_embeddings(batch_texts)
            
            if embeddings is None or len(embeddings) == 0:
                print(f"❌ Failed to get embeddings for batch {batch_num}, skipping...")
                continue

            if len(embeddings) != len(batch_texts):
                print(f"⚠️  Warning: Got {len(embeddings)} embeddings for {len(batch_texts)} texts in batch {batch_num}")
                # Truncate to match
                min_len = min(len(embeddings), len(batch_texts))
                embeddings = embeddings[:min_len]
                batch_texts = batch_texts[:min_len]
            
            # Prepare documents for storage
            batch_ids = [f"doc_{i+j}_{hash(text[:50])}" for j, text in enumerate(batch_texts)]
            
            # Add documents to collection
            print(f"💾 Storing batch {batch_num} ({len(batch_texts)} documents) in ChromaDB...")
            collection.add(
                documents=batch_texts,
                embeddings=embeddings,
                ids=batch_ids
            )
            
            total_stored += len(batch_texts)
            print(f"✅ Batch {batch_num} stored successfully. Total stored: {total_stored}")
        
        print(f"🎉 Successfully stored {total_stored}/{len(texts)} documents in ChromaDB collection '{CHROMA_COLLECTION}'")
        return collection
        
    except Exception as e:
        print(f"❌ Error creating ChromaDB collection: {e}")
        import traceback
        traceback.print_exc()
        return None

def clear_chroma_collection():
    """Clear all documents from ChromaDB collection"""
    try:
        client = get_chroma_client()
        try:
            # Try to get the collection first
            collection = client.get_collection(CHROMA_COLLECTION)
            # Delete the collection
            client.delete_collection(CHROMA_COLLECTION)
            print(f"🗑️  Cleared ChromaDB collection '{CHROMA_COLLECTION}'")
        except Exception as e:
            print(f"📂 Collection '{CHROMA_COLLECTION}' doesn't exist or is already empty: {e}")
        return True
    except Exception as e:
        print(f"❌ Error clearing ChromaDB collection: {e}")
        return False

def ensure_collection_exists():
    try:
        client = get_chroma_client()

        collection = client.get_or_create_collection(
            name=CHROMA_COLLECTION
        )

        print(f"📂 Using collection: {CHROMA_COLLECTION}")
        return collection

    except Exception as e:
        print(f"❌ Error ensuring collection exists: {e}")
        return None

def retrieve_relevant_docs(query: str, k: int = 10):
    """Retrieve relevant documents from ChromaDB"""
    try:
        # Ensure collection exists
        collection = ensure_collection_exists()
        if not collection:
            print("❌ Failed to ensure collection exists for retrieval")
            return []
        
        # Get query embedding
        query_embeddings = get_embeddings([query])
        
        if query_embeddings is None or len(query_embeddings) == 0:
            print("Failed to get query embedding")
            return []
        
        # Ensure we have a valid embedding
        if query_embeddings[0] is None or len(query_embeddings[0]) == 0:
            print("Empty query embedding")
            return []
        
        # Search for similar documents
        results = collection.query(
            query_embeddings=query_embeddings,
            n_results=k
        )
        
        # Format results for compatibility
        documents = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                # Create a document-like object for compatibility
                class Document:
                    def __init__(self, content):
                        self.page_content = content
                
                documents.append(Document(doc))
        
        return documents
        
    except Exception as e:
        print(f"Error retrieving documents from ChromaDB: {e}")
        return []

# Legacy function names for compatibility
def create_faiss_index(texts: List[str]):
    """Legacy function name - now uses ChromaDB"""
    return create_chroma_collection(texts)

def retrive_relevant_docs(vectorstore, query: str, k: int = 10):
    """Legacy function name - now uses ChromaDB"""
    return retrieve_relevant_docs(query, k)


