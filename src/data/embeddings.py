"""
Embeddings Module
================

Extracted from my notebooks/v1.1/02_embed_chunks.ipynb
Contains embedding generation and vector storage logic.
"""

import os
import json
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
from tqdm import tqdm
import wandb
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models


class EmbeddingGenerator:
    """My main embedding generator using sentence transformers."""
    
    def __init__(
        self, 
        model_name: str = "pritamdeka/S-PubMedBert-MS-MARCO",
        vector_dim: int = 768,
        batch_size: int = 64
    ):
        """
        Initialize my embedding generator.
        
        Args:
            model_name: My chosen model (PubMedBert fine-tuned on MS-MARCO)
            vector_dim: Expected vector dimension
            batch_size: Batch size for encoding
        """
        self.model_name = model_name
        self.vector_dim = vector_dim
        self.batch_size = batch_size
        
        # Load my model
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Verify dimensions match
        actual_dim = self.model.get_sentence_embedding_dimension()
        if actual_dim != vector_dim:
            print(f"Warning: Expected dim {vector_dim}, got {actual_dim}")
            self.vector_dim = actual_dim
    
    def encode_texts(
        self, 
        texts: List[str], 
        normalize: bool = True,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of texts to encode
            normalize: Whether to normalize embeddings
            show_progress: Show progress bar
            
        Returns:
            Numpy array of embeddings
        """
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize
        )
    
    def encode_chunks_batched(
        self, 
        chunks_df: pd.DataFrame, 
        text_column: str = "text",
        batch_size: int = 512
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Encode chunks in batches - my optimized approach for large datasets.
        
        Args:
            chunks_df: DataFrame with text chunks
            text_column: Column containing text to embed
            batch_size: Processing batch size
            
        Returns:
            Tuple of (embeddings_array, chunk_ids)
        """
        vecs = []
        ids = []
        
        print(f"Encoding {len(chunks_df)} chunks in batches of {batch_size}")
        
        for i in tqdm(range(0, len(chunks_df), batch_size), desc="Embedding batches"):
            sub = chunks_df.iloc[i:i+batch_size]
            
            # Encode this batch
            emb = self.model.encode(
                sub[text_column].tolist(),
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            
            vecs.append(emb)
            ids.extend(sub['id'].tolist())
        
        # Stack all embeddings
        vectors = np.vstack(vecs)
        
        return vectors, ids
    
    def save_embeddings(
        self, 
        embeddings: np.ndarray, 
        output_path: Path,
        metadata: Optional[Dict] = None
    ) -> None:
        """Save my embeddings to numpy file with metadata."""
        np.save(output_path, embeddings)
        
        # Save metadata alongside
        if metadata:
            meta_path = output_path.with_suffix('.json')
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        print(f"✅ Saved embeddings: {embeddings.shape} → {output_path}")


class HybridEmbedder:
    """My hybrid embedding system - combines dense and sparse retrieval."""
    
    def __init__(
        self,
        dense_model: str = "pritamdeka/S-PubMedBert-MS-MARCO",
        use_bm25: bool = True
    ):
        """
        Initialize my hybrid embedder.
        
        Args:
            dense_model: Dense embedding model
            use_bm25: Whether to include BM25 sparse features
        """
        self.dense_embedder = EmbeddingGenerator(dense_model)
        self.use_bm25 = use_bm25
        
        if use_bm25:
            try:
                from rank_bm25 import BM25Okapi
                self.bm25_class = BM25Okapi
            except ImportError:
                print("Warning: rank_bm25 not installed, disabling BM25")
                self.use_bm25 = False
    
    def fit_bm25(self, texts: List[str]) -> None:
        """Fit BM25 on my corpus."""
        if not self.use_bm25:
            return
        
        # Tokenize texts for BM25
        tokenized_corpus = [text.lower().split() for text in texts]
        self.bm25 = self.bm25_class(tokenized_corpus)
        print(f"✅ Fitted BM25 on {len(texts)} documents")
    
    def get_bm25_scores(self, query: str, top_k: int = 100) -> np.ndarray:
        """Get BM25 scores for query."""
        if not self.use_bm25 or not hasattr(self, 'bm25'):
            return np.array([])
        
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        return scores
    
    def hybrid_search(
        self, 
        query: str, 
        dense_vectors: np.ndarray,
        texts: List[str],
        alpha: float = 0.7,
        top_k: int = 100
    ) -> List[int]:
        """
        My hybrid search combining dense and sparse signals.
        
        Args:
            query: Search query
            dense_vectors: Pre-computed dense embeddings
            texts: Original texts for BM25
            alpha: Weight for dense vs sparse (0.7 = 70% dense)
            top_k: Number of results
            
        Returns:
            Indices of top results
        """
        # Dense search
        query_vec = self.dense_embedder.encode_texts([query])[0]
        dense_scores = np.dot(dense_vectors, query_vec)
        
        # Normalize dense scores to [0, 1]
        dense_scores = (dense_scores + 1) / 2  # cosine to [0, 1]
        
        if self.use_bm25 and hasattr(self, 'bm25'):
            # Sparse search
            sparse_scores = self.get_bm25_scores(query)
            
            # Normalize sparse scores to [0, 1]
            if len(sparse_scores) > 0:
                sparse_scores = sparse_scores / (sparse_scores.max() + 1e-8)
            else:
                sparse_scores = np.zeros_like(dense_scores)
            
            # Combine scores
            hybrid_scores = alpha * dense_scores + (1 - alpha) * sparse_scores
        else:
            hybrid_scores = dense_scores
        
        # Get top-k indices
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        return top_indices.tolist()


class QdrantVectorStore:
    """My Qdrant vector store wrapper for production deployment."""
    
    def __init__(
        self,
        collection_name: str = "medimaven_chunks",
        client_config: Optional[Dict] = None,
        vector_dim: int = 768
    ):
        """
        Initialize my Qdrant client.
        
        Args:
            collection_name: My collection name
            client_config: Client configuration (defaults to in-memory)
            vector_dim: Vector dimensions
        """
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        
        # Setup client
        if client_config:
            self.client = QdrantClient(**client_config)
        else:
            # Default to in-memory for development
            self.client = QdrantClient(":memory:")
        
        self._setup_collection()
    
    def _setup_collection(self) -> None:
        """Setup my collection with proper configuration."""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            existing_names = [c.name for c in collections]
            
            if self.collection_name not in existing_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_dim, 
                        distance=models.Distance.COSINE
                    )
                )
                print(f"✅ Created collection: {self.collection_name}")
            else:
                print(f"✅ Using existing collection: {self.collection_name}")
                
        except Exception as e:
            print(f"Error setting up collection: {e}")
    
    def upsert_embeddings(
        self,
        chunks_df: pd.DataFrame,
        embeddings: np.ndarray,
        batch_size: int = 512,
        metadata_columns: List[str] = ["url", "section", "source"]
    ) -> None:
        """
        Upsert my embeddings to Qdrant in batches.
        
        Args:
            chunks_df: DataFrame with chunk metadata
            embeddings: Corresponding embeddings
            batch_size: Batch size for upserts
            metadata_columns: Columns to store as payload
        """
        print(f"Upserting {len(chunks_df)} vectors in batches of {batch_size}")
        
        for i in tqdm(range(0, len(chunks_df), batch_size), desc="Upserting to Qdrant"):
            sub = chunks_df.iloc[i:i+batch_size]
            sub_embeddings = embeddings[i:i+batch_size]
            
            # Prepare payload (metadata)
            available_cols = [col for col in metadata_columns if col in sub.columns]
            payloads = sub[available_cols].to_dict('records')
            
            # Upsert this batch
            self.client.upsert(
                collection_name=self.collection_name,
                points=models.Batch(
                    ids=sub['id'].tolist(),
                    vectors=sub_embeddings.tolist(),
                    payloads=payloads
                )
            )
        
        print(f"✅ Upserted {len(chunks_df)} vectors to Qdrant")
    
    def search(
        self, 
        query_vector: np.ndarray, 
        limit: int = 10,
        score_threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Search my vector store.
        
        Args:
            query_vector: Query embedding
            limit: Number of results
            score_threshold: Minimum score threshold
            
        Returns:
            List of search results with metadata
        """
        search_params = {
            "collection_name": self.collection_name,
            "query_vector": query_vector.tolist(),
            "limit": limit
        }
        
        if score_threshold:
            search_params["score_threshold"] = score_threshold
        
        results = self.client.search(**search_params)
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "id": result.id,
                "score": result.score,
                "payload": result.payload
            })
        
        return formatted_results
    
    def get_collection_info(self) -> Dict:
        """Get info about my collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": info.config.name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status
            }
        except Exception as e:
            return {"error": str(e)}


def log_embedding_run(
    model_name: str,
    chunks_count: int,
    embedding_shape: Tuple[int, int],
    processing_time: float,
    project_name: str = "medimaven-embeddings"
) -> None:
    """Log my embedding run to wandb for tracking."""
    
    run = wandb.init(
        project=project_name,
        job_type="embedding_generation",
        config={
            "model_name": model_name,
            "chunks_count": chunks_count,
            "embedding_dim": embedding_shape[1],
            "processing_time_minutes": processing_time / 60,
            "timestamp": datetime.datetime.now().isoformat()
        }
    )
    
    # Log metrics
    wandb.log({
        "chunks_processed": chunks_count,
        "embedding_dimension": embedding_shape[1],
        "processing_time_minutes": processing_time / 60,
        "throughput_chunks_per_minute": chunks_count / (processing_time / 60)
    })
    
    run.finish()
    print(f"✅ Logged embedding run to wandb: {project_name}")
