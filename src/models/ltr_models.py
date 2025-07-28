"""
LTR (Learning to Rank) Models for MediMaven

This module implements the LambdaMART ranking model for reranking search results.
Extracted from notebook 04a_ltr_train_lambdamart.ipynb

My notes:
- LambdaMART is a gradient boosting method specifically designed for ranking tasks
- Uses NDCG (Normalized Discounted Cumulative Gain) as the evaluation metric
- Features include BM25 scores, dense cosine similarity, and chunk length
- Need to handle query groups properly for ranking training
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import pathlib
from typing import Dict, List, Tuple, Optional, Any
import pickle
import logging

# Configure logging for my reference
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LambdaMARTRanker:
    """
    LambdaMART ranker for reranking medical document chunks.
    
    My implementation notes:
    - Uses LightGBM's LGBMRanker which implements the LambdaMART algorithm
    - Handles ranking task where we need to order chunks per query
    - Features are BM25 score, dense cosine similarity, and chunk length
    """
    
    def __init__(self, 
                 objective: str = "lambdarank",
                 metric: str = "ndcg",
                 num_leaves: int = 63,
                 learning_rate: float = 0.05,
                 n_estimators: int = 1000,
                 eval_at: List[int] = None,
                 random_state: int = 42):
        """
        Initialize LambdaMART ranker with hyperparameters.
        
        My configuration notes:
        - num_leaves=63: Controls tree complexity, prevents overfitting
        - learning_rate=0.05: Conservative learning rate for stable training
        - n_estimators=1000: Allow enough iterations with early stopping
        - eval_at=[10]: Evaluate NDCG@10 which matches our retrieval setup
        """
        self.objective = objective
        self.metric = metric
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.eval_at = eval_at or [10]
        self.random_state = random_state
        
        # Initialize the ranker - will be set during training
        self.ranker = None
        self.feature_columns = ['bm25_score', 'dense_cosine', 'chunk_length']
        
        logger.info(f"Initialized LambdaMART ranker with {self.n_estimators} estimators")
    
    def prepare_training_data(self, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[int], List[int]]:
        """
        Prepare training data for LambdaMART.
        
        My data prep notes:
        - Need to split by query groups to maintain ranking structure
        - Group sizes are essential for LightGBM ranking to work properly
        - Using simple train/validation split rather than time-based
        
        Args:
            features_df: DataFrame with columns [qid, chunk_id, label, bm25_score, dense_cosine, chunk_length]
            
        Returns:
            train_df, valid_df, train_groups, valid_groups
        """
        logger.info(f"Preparing training data from {len(features_df)} rows")
        
        # Get unique query IDs for splitting
        unique_qids = features_df['qid'].unique()
        np.random.seed(self.random_state)
        np.random.shuffle(unique_qids)
        
        # 80/20 split on queries (not rows) to maintain ranking structure
        split_idx = int(0.8 * len(unique_qids))
        train_qids = unique_qids[:split_idx]
        valid_qids = unique_qids[split_idx:]
        
        # Split data by query IDs
        train_df = features_df[features_df['qid'].isin(train_qids)].copy()
        valid_df = features_df[features_df['qid'].isin(valid_qids)].copy()
        
        # Sort by qid to ensure proper grouping
        train_df = train_df.sort_values('qid').reset_index(drop=True)
        valid_df = valid_df.sort_values('qid').reset_index(drop=True)
        
        # Calculate group sizes (number of documents per query)
        train_groups = train_df.groupby('qid').size().tolist()
        valid_groups = valid_df.groupby('qid').size().tolist()
        
        logger.info(f"Train: {len(train_df)} rows, {len(train_groups)} queries")
        logger.info(f"Valid: {len(valid_df)} rows, {len(valid_groups)} queries")
        
        return train_df, valid_df, train_groups, valid_groups
    
    def train(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the LambdaMART ranker.
        
        My training notes:
        - LightGBM requires group parameter for ranking tasks
        - eval_set and eval_group for validation during training
        - Early stopping based on validation NDCG@10
        
        Args:
            features_df: Training features DataFrame
            
        Returns:
            Dictionary with training metrics and model info
        """
        logger.info("Starting LambdaMART training")
        
        # Prepare training data
        train_df, valid_df, train_groups, valid_groups = self.prepare_training_data(features_df)
        
        # Initialize the ranker
        self.ranker = lgb.LGBMRanker(
            objective=self.objective,
            metric=self.metric,
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            random_state=self.random_state
        )
        
        # Train the model
        logger.info("Fitting LambdaMART model...")
        self.ranker.fit(
            train_df[self.feature_columns],
            train_df['label'],
            group=train_groups,
            eval_set=[(valid_df[self.feature_columns], valid_df['label'])],
            eval_group=[valid_groups],
            eval_at=self.eval_at,
            callbacks=[lgb.log_evaluation(period=100)]  # Log every 100 iterations
        )
        
        # Get training results
        best_score = self.ranker.best_score_['valid_0'][f'ndcg@{self.eval_at[0]}']
        best_iteration = self.ranker.best_iteration_
        
        training_info = {
            'best_score': best_score,
            'best_iteration': best_iteration,
            'num_train_queries': len(train_groups),
            'num_valid_queries': len(valid_groups),
            'feature_importance': dict(zip(
                self.feature_columns, 
                self.ranker.feature_importances_
            ))
        }
        
        logger.info(f"Training completed. Best NDCG@{self.eval_at[0]}: {best_score:.4f}")
        logger.info(f"Best iteration: {best_iteration}")
        
        return training_info
    
    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Predict ranking scores for given features.
        
        My prediction notes:
        - Returns ranking scores, not probabilities
        - Higher scores indicate better relevance
        - Can be used for reranking retrieved chunks
        
        Args:
            features_df: DataFrame with feature columns
            
        Returns:
            Array of ranking scores
        """
        if self.ranker is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        return self.ranker.predict(features_df[self.feature_columns])
    
    def rerank_results(self, query_results: List[Dict[str, Any]], query_text: str) -> List[Dict[str, Any]]:
        """
        Rerank search results using the trained model.
        
        My reranking implementation:
        - Takes initial retrieval results and reorders them
        - Assumes features are already computed and included in results
        - Returns reordered list with ranking scores added
        
        Args:
            query_results: List of result dictionaries with features
            query_text: Original query (for reference)
            
        Returns:
            Reranked results with 'rank_score' added
        """
        if self.ranker is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        if not query_results:
            return []
        
        # Create DataFrame from results
        features_data = []
        for result in query_results:
            features_data.append({
                'bm25_score': result.get('bm25_score', 0.0),
                'dense_cosine': result.get('dense_cosine', 0.0),
                'chunk_length': result.get('chunk_length', 0)
            })
        
        features_df = pd.DataFrame(features_data)
        
        # Get ranking scores
        rank_scores = self.predict(features_df)
        
        # Add scores to results and sort
        for i, result in enumerate(query_results):
            result['rank_score'] = float(rank_scores[i])
        
        # Sort by ranking score (descending)
        reranked_results = sorted(query_results, key=lambda x: x['rank_score'], reverse=True)
        
        logger.info(f"Reranked {len(reranked_results)} results for query")
        return reranked_results
    
    def save_model(self, model_path: pathlib.Path) -> None:
        """
        Save the trained model to disk.
        
        My saving notes:
        - Uses LightGBM's native save format (.txt)
        - More efficient than pickle for LightGBM models
        - Can be loaded directly by LightGBM
        """
        if self.ranker is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.ranker.booster_.save_model(str(model_path))
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: pathlib.Path) -> None:
        """
        Load a trained model from disk.
        
        My loading notes:
        - Creates new LGBMRanker and loads the booster
        - Need to reconstruct the wrapper properly
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Create ranker with same parameters
        self.ranker = lgb.LGBMRanker(
            objective=self.objective,
            metric=self.metric,
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            random_state=self.random_state
        )
        
        # Load the booster
        booster = lgb.Booster(model_file=str(model_path))
        self.ranker._Booster = booster
        
        logger.info(f"Model loaded from {model_path}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        My analysis notes:
        - Helps understand which features contribute most to ranking
        - Useful for feature engineering and model interpretation
        """
        if self.ranker is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        return dict(zip(self.feature_columns, self.ranker.feature_importances_))


class LTRTrainingPipeline:
    """
    Full training pipeline for LTR models.
    
    My pipeline notes:
    - Handles the complete flow from loading features to saving model
    - Includes validation and logging
    - Can integrate with experiment tracking (W&B)
    """
    
    def __init__(self, data_dir: pathlib.Path, config: Dict[str, Any] = None):
        """Initialize training pipeline."""
        self.data_dir = pathlib.Path(data_dir)
        self.config = config or {}
        
        # Default configuration
        self.default_config = {
            'model_params': {
                'num_leaves': 63,
                'learning_rate': 0.05,
                'n_estimators': 1000,
                'eval_at': [10]
            },
            'features_file': 'ltr_features.parquet',
            'model_file': 'ltr_lambdamart.txt'
        }
        
        # Merge configs
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
    
    def load_features(self) -> pd.DataFrame:
        """Load training features from parquet file."""
        features_path = self.data_dir / self.config['features_file']
        
        if not features_path.exists():
            raise FileNotFoundError(f"Features file not found: {features_path}")
        
        features_df = pd.read_parquet(features_path)
        logger.info(f"Loaded {len(features_df)} feature rows from {features_path}")
        
        return features_df
    
    def run_training(self, experiment_name: str = None) -> Dict[str, Any]:
        """
        Run the complete training pipeline.
        
        My pipeline flow:
        1. Load features
        2. Initialize and train model
        3. Save model
        4. Return training results
        
        Args:
            experiment_name: Optional experiment name for tracking
            
        Returns:
            Dictionary with training results and paths
        """
        logger.info(f"Starting LTR training pipeline")
        
        # Load features
        features_df = self.load_features()
        
        # Initialize ranker
        ranker = LambdaMARTRanker(**self.config['model_params'])
        
        # Train model
        training_info = ranker.train(features_df)
        
        # Save model
        model_path = self.data_dir / self.config['model_file']
        ranker.save_model(model_path)
        
        # Prepare results
        results = {
            'training_info': training_info,
            'model_path': str(model_path),
            'features_path': str(self.data_dir / self.config['features_file']),
            'config': self.config
        }
        
        logger.info("Training pipeline completed successfully")
        return results


# Utility functions for feature engineering
def compute_ranking_features(query_text: str, 
                           chunk_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Compute ranking features for a query and its candidate chunks.
    
    My feature computation notes:
    - This would integrate with the existing BM25 and dense retrieval
    - Features should match training data format
    - Called during inference to prepare data for reranking
    
    Args:
        query_text: The user query
        chunk_results: List of retrieved chunks with metadata
        
    Returns:
        Chunks with computed ranking features
    """
    # This is a placeholder - in practice this would integrate with
    # the BM25 scorer and dense embeddings from other modules
    
    for chunk in chunk_results:
        # Ensure required features exist
        if 'bm25_score' not in chunk:
            chunk['bm25_score'] = 0.0
        if 'dense_cosine' not in chunk:
            chunk['dense_cosine'] = 0.0
        if 'chunk_length' not in chunk:
            chunk['chunk_length'] = len(chunk.get('content', ''))
    
    return chunk_results


if __name__ == "__main__":
    # Example usage for testing
    import argparse
    
    parser = argparse.ArgumentParser(description="Train LambdaMART ranker")
    parser.add_argument("--data_dir", type=str, default="../../data/final",
                       help="Directory containing training features")
    parser.add_argument("--experiment", type=str, default="lambdamart_training",
                       help="Experiment name")
    
    args = parser.parse_args()
    
    # Run training pipeline
    pipeline = LTRTrainingPipeline(data_dir=args.data_dir)
    results = pipeline.run_training(experiment_name=args.experiment)
    
    print("Training completed!")
    print(f"Model saved to: {results['model_path']}")
    print(f"Best NDCG@10: {results['training_info']['best_score']:.4f}")
