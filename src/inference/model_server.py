"""
Model Server for MediMaven ML Models

This module provides production-ready model serving capabilities.
It handles model loading, caching, and inference for the LambdaMART ranker.

My implementation notes:
- Lazy loading to reduce memory usage
- Model caching for fast repeated inference
- Thread-safe operations for concurrent requests
- Integration with existing backend systems
"""

import pathlib
import logging
import threading
import time
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd

from ..models.ltr_models import LambdaMARTRanker
from ..data.embeddings import EmbeddingGenerator
from ..data.negative_sampling import HardNegativeMiner

logger = logging.getLogger(__name__)


class ModelServer:
    """
    Production model server for MediMaven ML models.
    
    My server architecture:
    - Singleton pattern to ensure single model instance
    - Thread-safe model loading and inference
    - Caching for model artifacts and embeddings
    - Health checks and monitoring hooks
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern for model server."""
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, model_dir: pathlib.Path, config: Dict[str, Any] = None):
        """Initialize model server."""
        # Prevent multiple initialization
        if hasattr(self, '_initialized'):
            return
        
        self.model_dir = pathlib.Path(model_dir)
        self.config = config or {}
        
        # Model components
        self.ranker: Optional[LambdaMARTRanker] = None
        self.embedding_generator: Optional[EmbeddingGenerator] = None
        self.bm25_miner: Optional[HardNegativeMiner] = None
        
        # Model metadata
        self.model_loaded = False
        self.load_time = None
        self.inference_count = 0
        
        # Thread safety
        self.inference_lock = threading.RLock()
        
        logger.info(f"Model server initialized with model_dir: {model_dir}")
        self._initialized = True
    
    def load_models(self, lazy: bool = True) -> None:
        """
        Load all required models.
        
        My loading strategy:
        - Lazy loading by default to reduce startup time
        - All models loaded together to ensure consistency
        - Error handling for missing model files
        
        Args:
            lazy: If True, models are loaded on first inference call
        """
        if self.model_loaded and not lazy:
            logger.info("Models already loaded")
            return
        
        if lazy:
            logger.info("Lazy loading enabled - models will load on first inference")
            return
        
        self._load_models_sync()
    
    def _load_models_sync(self) -> None:
        """Synchronously load all models."""
        with self.inference_lock:
            if self.model_loaded:
                return
            
            start_time = time.time()
            logger.info("Loading models...")
            
            try:
                # Load LambdaMART ranker
                ranker_path = self.model_dir / "ltr_lambdamart.txt"
                if ranker_path.exists():
                    self.ranker = LambdaMARTRanker()
                    self.ranker.load_model(ranker_path)
                    logger.info("LambdaMART ranker loaded")
                else:
                    logger.warning(f"LambdaMART model not found: {ranker_path}")
                
                # Load embedding generator
                embed_config = self.config.get('embedding', {})
                self.embedding_generator = EmbeddingGenerator(
                    model_name=embed_config.get('model_name', 'sentence-transformers/all-MiniLM-L6-v2'),
                    device=embed_config.get('device', 'cpu'),
                    batch_size=embed_config.get('batch_size', 32)
                )
                logger.info("Embedding generator loaded")
                
                # Load BM25 index
                bm25_path = self.model_dir / "bm25_index.pkl"
                if bm25_path.exists():
                    self.bm25_miner = HardNegativeMiner()
                    self.bm25_miner.load_bm25_index(bm25_path)
                    logger.info("BM25 index loaded")
                else:
                    logger.warning(f"BM25 index not found: {bm25_path}")
                
                self.model_loaded = True
                self.load_time = time.time() - start_time
                logger.info(f"All models loaded successfully in {self.load_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Failed to load models: {str(e)}")
                raise
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on loaded models.
        
        My health check implementation:
        - Verifies all models are loaded and functional
        - Returns detailed status and performance metrics
        - Can be used by monitoring systems
        """
        status = {
            'status': 'healthy',
            'model_loaded': self.model_loaded,
            'load_time': self.load_time,
            'inference_count': self.inference_count,
            'components': {}
        }
        
        try:
            # Check ranker
            if self.ranker and self.ranker.ranker:
                status['components']['ranker'] = 'loaded'
            else:
                status['components']['ranker'] = 'not_loaded'
                status['status'] = 'degraded'
            
            # Check embedding generator
            if self.embedding_generator:
                status['components']['embedding_generator'] = 'loaded'
            else:
                status['components']['embedding_generator'] = 'not_loaded'
                status['status'] = 'degraded'
            
            # Check BM25 index
            if self.bm25_miner and hasattr(self.bm25_miner, 'bm25'):
                status['components']['bm25_index'] = 'loaded'
            else:
                status['components']['bm25_index'] = 'not_loaded'
                status['status'] = 'degraded'
                
        except Exception as e:
            status['status'] = 'unhealthy'
            status['error'] = str(e)
        
        return status
    
    def compute_features(self, query: str, documents: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Compute ranking features for a query and candidate documents.
        
        My feature computation:
        - BM25 scores using loaded index
        - Dense cosine similarity with query embeddings
        - Document-level features like length
        
        Args:
            query: User query string
            documents: List of candidate documents with content
            
        Returns:
            DataFrame with computed features
        """
        if not self.model_loaded:
            self._load_models_sync()
        
        features = []
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embeddings([query])[0]
        
        for doc in documents:
            doc_content = doc.get('content', '')
            
            # Compute BM25 score
            if self.bm25_miner and hasattr(self.bm25_miner, 'bm25'):
                bm25_score = self.bm25_miner.get_bm25_scores([query], [doc_content])[0][0]
            else:
                bm25_score = 0.0
            
            # Compute dense cosine similarity
            doc_embedding = self.embedding_generator.generate_embeddings([doc_content])[0]
            cosine_sim = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            
            # Document features
            chunk_length = len(doc_content)
            
            features.append({
                'bm25_score': float(bm25_score),
                'dense_cosine': float(cosine_sim),
                'chunk_length': chunk_length
            })
        
        return pd.DataFrame(features)
    
    def rerank_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank search results using the trained LambdaMART model.
        
        My reranking pipeline:
        1. Ensure models are loaded
        2. Compute ranking features
        3. Apply LambdaMART model
        4. Sort and return reranked results
        
        Args:
            query: Original search query
            results: Initial search results to rerank
            
        Returns:
            Reranked results with scores
        """
        with self.inference_lock:
            if not self.model_loaded:
                self._load_models_sync()
            
            if not results:
                return []
            
            if not self.ranker:
                logger.warning("Ranker not available, returning original results")
                return results
            
            try:
                # Compute features
                features_df = self.compute_features(query, results)
                
                # Get ranking scores
                ranking_scores = self.ranker.predict(features_df)
                
                # Add scores to results and sort
                reranked_results = []
                for i, result in enumerate(results):
                    result_copy = result.copy()
                    result_copy['rank_score'] = float(ranking_scores[i])
                    reranked_results.append(result_copy)
                
                # Sort by ranking score (descending)
                reranked_results.sort(key=lambda x: x['rank_score'], reverse=True)
                
                self.inference_count += 1
                logger.debug(f"Reranked {len(results)} results for query")
                
                return reranked_results
                
            except Exception as e:
                logger.error(f"Reranking failed: {str(e)}")
                # Return original results on error
                return results


class InferenceEngine:
    """
    High-level inference engine that coordinates model server with business logic.
    
    My engine design:
    - Wraps ModelServer with additional business logic
    - Handles result filtering and post-processing
    - Integrates with existing backend APIs
    - Provides metrics and monitoring hooks
    """
    
    def __init__(self, model_server: ModelServer, config: Dict[str, Any] = None):
        """Initialize inference engine."""
        self.model_server = model_server
        self.config = config or {}
        
        # Processing parameters
        self.max_results = self.config.get('max_results', 20)
        self.min_score_threshold = self.config.get('min_score_threshold', 0.0)
        self.enable_caching = self.config.get('enable_caching', True)
        
        # Simple result cache
        self._result_cache = {} if self.enable_caching else None
        self._cache_ttl = self.config.get('cache_ttl', 300)  # 5 minutes
        
        logger.info("Inference engine initialized")
    
    def process_query(self, 
                     query: str, 
                     initial_results: List[Dict[str, Any]], 
                     metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a query with full inference pipeline.
        
        My processing flow:
        1. Check cache for recent results
        2. Apply ML reranking
        3. Filter and post-process results
        4. Cache results and return
        
        Args:
            query: User search query
            initial_results: Results from initial retrieval
            metadata: Additional query metadata
            
        Returns:
            Processed results with metadata
        """
        start_time = time.time()
        
        # Check cache
        cache_key = self._get_cache_key(query, len(initial_results))
        if self._result_cache and cache_key in self._result_cache:
            cached_result = self._result_cache[cache_key]
            if time.time() - cached_result['timestamp'] < self._cache_ttl:
                logger.debug("Returning cached results")
                return cached_result['results']
        
        # Apply ML reranking
        reranked_results = self.model_server.rerank_results(query, initial_results)
        
        # Filter results
        filtered_results = self._filter_results(reranked_results)
        
        # Prepare response
        response = {
            'query': query,
            'results': filtered_results[:self.max_results],
            'total_results': len(filtered_results),
            'processing_time': time.time() - start_time,
            'reranked': True,
            'metadata': metadata or {}
        }
        
        # Cache results
        if self._result_cache:
            self._result_cache[cache_key] = {
                'results': response,
                'timestamp': time.time()
            }
        
        logger.info(f"Processed query in {response['processing_time']:.3f}s")
        return response
    
    def _filter_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter results based on score threshold and other criteria."""
        filtered = []
        
        for result in results:
            # Score threshold filter
            if result.get('rank_score', 0) < self.min_score_threshold:
                continue
            
            # Add to filtered results
            filtered.append(result)
        
        return filtered
    
    def _get_cache_key(self, query: str, num_results: int) -> str:
        """Generate cache key for query and result count."""
        return f"{hash(query)}_{num_results}"
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get inference engine metrics."""
        return {
            'model_server_status': self.model_server.health_check(),
            'cache_size': len(self._result_cache) if self._result_cache else 0,
            'config': self.config
        }
