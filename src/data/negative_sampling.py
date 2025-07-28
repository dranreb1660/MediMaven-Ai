"""
Negative Sampling Module
========================

Extracted from my notebooks/v1.1/03_mine_hard_negs.ipynb
Contains logic for mining hard negatives using BM25 and vector similarities.
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from typing import List, Dict

class HardNegativeMiner:
    """Class to handle mining hard negatives for training."""
    
    def __init__(self, chunks_df: pd.DataFrame, text_column: str = "text"):
        """
        Initialize with chunks and prepare BM25 index by section.
        
        Args:
            chunks_df: DataFrame of text chunks
            text_column: Column name containing the text
        """
        self.chunks_df = chunks_df
        self.text_column = text_column
        self.bm25_index = {}
        
        # Prepare BM25 index for each section
        for sec, group in chunks_df.groupby('section'):
            tokens = [t.split() for t in group[text_column]]
            self.bm25_index[sec] = BM25Okapi(tokens)
            self.bm25_index[sec].docs_df = group  # Attach df for lookup
        print(f"BM25 index prepared for {len(self.bm25_index)} sections.")
    
    def mine_hard_negatives(self, query: str, pos_id: str, section: str, top_k: int = 5) -> List[Dict]:
        """
        Mine hard negatives for a given query and section.
        
        Args:
            query: The query text to search against
            pos_id: The positive example ID to exclude
            section: Section where to perform the search
            top_k: Number of top negatives to return
        
        Returns:
            List of hard negative examples
        """
        if section not in self.bm25_index:
            raise ValueError(f"Section {section} not indexed for BM25.")
        
        # Tokenize the query
        query_tokens = query.split()
        
        # Retrieve BM25 negatives
        bm25_negatives = self.bm25_index[section].get_top_n(query_tokens, n=top_k + 1)
        
        # Filter out the positive ID
        hard_negatives = [doc for doc in bm25_negatives if doc.name != pos_id][:top_k]
        
        return [
            {
                'id': neg.name,
                'text': neg[self.text_column],
                'score': self.bm25_index[section].calc_idf(query_tokens)  # Just a placeholder for score
            }
            for neg in hard_negatives
        ]


class CosineSimilaritySampler:
    """Handles sampling using cosine similarity between vectors."""
    
    def retrieve_similar(self, query_vector: np.ndarray, vectors: np.ndarray, top_k: int = 5) -> List[int]:
        """
        Retrieves indices of the top-k most similar vectors to the query.
        
        Args:
            query_vector: Embedding vector of the query
            vectors: Array of vectors to compare
            top_k: Number of top results to return
        
        Returns:
            List of indices for the most similar vectors
        """
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector.reshape(1, -1), vectors).flatten()
        
        # Get top-k indices (excluding self, assuming query is within `vectors`)
        top_indices = np.argsort(similarities)[::-1][:top_k+1]
        
        return top_indices
    
    def sample_hard_negatives(
        self,
        query_vector: np.ndarray,
        vectors: np.ndarray,
        positive_id: int,
        top_k: int = 5
    ) -> List[int]:
        """
        Sample hard negatives based on cosine similarity for training.
        
        Args:
            query_vector: Embedding of the positive query
            vectors: Complete set of vectors including query
            positive_id: Index of the positive example to exclude
            top_k: Number of negatives to sample
        
        Returns:
            List of negative sample indices
        """
        similar_indices = self.retrieve_similar(query_vector, vectors, top_k + 1)
        
        # Exclude the actual positive index
        negative_indices = [idx for idx in similar_indices if idx != positive_id][:top_k]
        
        return negative_indices


