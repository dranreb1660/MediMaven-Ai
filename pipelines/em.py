"""
A safe implementation of the embedding generation process that focuses on preventing segmentation faults
by processing texts one by one, using CPU only, and implementing proper error handling.
"""

import os
import sys
import gc
import json
import time
import pickle
import logging
import numpy as np
import torch
import pymongo
from pymongo import MongoClient
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('embedding_safe.log')
    ]
)
logger = logging.getLogger(__name__)

# Constants
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
MAX_TOKEN_LENGTH = 256
MONGODB_URI = "mongodb://localhost:27017/"
DB_NAME = "medimaven"
COLLECTION_NAME = "context"
OUTPUT_DIR = "data/embeddings"

def connect_to_mongodb():
    """Connect to MongoDB and return the collection."""
    try:
        client = MongoClient(MONGODB_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        logger.info(f"Successfully connected to MongoDB collection: {COLLECTION_NAME}")
        return collection
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise

def get_contexts(collection):
    """Retrieve all contexts from MongoDB."""
    try:
        contexts = list(collection.find({}))
        logger.info(f"Retrieved {len(contexts)} contexts from MongoDB")
        return contexts
    except Exception as e:
        logger.error(f"Failed to retrieve contexts from MongoDB: {e}")
        raise

def chunk_text_by_tokens(text, tokenizer, max_tokens=MAX_TOKEN_LENGTH):
    """Split text into smaller chunks based on token count."""
    if not text:
        return []
    
    try:
        tokens = tokenizer.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk)
        
        return chunks
    except Exception as e:
        logger.error(f"Error during text chunking: {e}")
        return [text[:1000]]  # Return a truncated version as fallback

def chunk_documents(contexts, tokenizer):
    """Process all contexts and chunk them."""
    chunks = []
    chunk_to_doc_map = {}
    chunk_id = 0
    
    for context in tqdm(contexts, desc="Chunking documents"):
        try:
            doc_id = str(context.get('_id'))
            text = context.get('text', '')
            
            if not text:
                continue
                
            doc_chunks = chunk_text_by_tokens(text, tokenizer)
            
            for chunk in doc_chunks:
                chunks.append(chunk)
                chunk_to_doc_map[chunk_id] = doc_id
                chunk_id += 1
                
        except Exception as e:
            logger.error(f"Error processing document {context.get('_id')}: {e}")
            continue
    
    logger.info(f"Created {len(chunks)} chunks from {len(contexts)} documents")
    return chunks, chunk_to_doc_map

def generate_embeddings(chunks, model):
    """Generate embeddings one by one with error handling."""
    embeddings = []
    problem_indices = []
    
    for i, chunk in enumerate(tqdm(chunks, desc="Generating embeddings")):
        try:
            # Process one text at a time
            if not chunk or len(chunk.strip()) == 0:
                # Handle empty text
                embeddings.append(np.zeros(768))  # Use model dimension
                logger.warning(f"Empty text at index {i}")
                continue
                
            # Generate embedding
            embedding = model.encode(chunk, convert_to_tensor=False)
            embeddings.append(embedding)
            
            # Clean up memory
            if i % 100 == 0:
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
        except Exception as e:
            logger.error(f"Error generating embedding for chunk {i}: {e}")
            # Add placeholder embedding
            embeddings.append(np.zeros(768))
            problem_indices.append(i)
            
            # Wait a moment and clean up
            time.sleep(1)
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Log problem indices
    if problem_indices:
        logger.warning(f"Failed to generate embeddings for {len(problem_indices)} chunks")
        
    return np.array(embeddings)

def save_outputs(embeddings, chunks, chunk_to_doc_map):
    """Save the embeddings and mappings to files."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save embeddings
    embedding_path = os.path.join(OUTPUT_DIR, "embeddings_safe.npy")
    np.save(embedding_path, embeddings)
    
    # Save chunks
    chunks_path = os.path.join(OUTPUT_DIR, "chunks_safe.pkl")
    with open(chunks_path, 'wb') as f:
        pickle.dump(chunks, f)
    
    # Save mapping
    mapping_path = os.path.join(OUTPUT_DIR, "chunk_to_doc_map_safe.json")
    with open(mapping_path, 'w') as f:
        json.dump(chunk_to_doc_map, f)
    
    logger.info(f"Saved outputs to {OUTPUT_DIR}")
    return embedding_path, chunks_path, mapping_path

def main():
    """Main function to run the embedding pipeline safely."""
    start_time = time.time()
    logger.info("Starting safe embedding generation process")
    
    try:
        # Limit torch threads to avoid excessive parallelism
        torch.set_num_threads(1)
        logger.info(f"Torch threads limited to 1")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        logger.info("Tokenizer loaded successfully")
        
        # Load model with CPU only
        logger.info(f"Loading model: {MODEL_NAME} on CPU")
        model = SentenceTransformer(MODEL_NAME, device="cpu")
        logger.info("Model loaded successfully")
        
        # Connect to MongoDB and get contexts
        collection = connect_to_mongodb()
        contexts = get_contexts(collection)
        
        # Chunk documents
        logger.info("Chunking documents")
        chunks, chunk_to_doc_map = chunk_documents(contexts, tokenizer)
        
        # Generate embeddings
        logger.info("Generating embeddings")
        embeddings = generate_embeddings(chunks, model)
        
        # Save outputs
        logger.info("Saving outputs")
        embedding_path, chunks_path, mapping_path = save_outputs(
            embeddings, chunks, chunk_to_doc_map
        )
        
        duration = time.time() - start_time
        logger.info(f"Process completed in {duration:.2f} seconds")
        logger.info(f"Embedding shape: {embeddings.shape}")
        logger.info(f"Files saved at: \n- {embedding_path}\n- {chunks_path}\n- {mapping_path}")
        
    except Exception as e:
        logger.error(f"Critical error in embedding process: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()