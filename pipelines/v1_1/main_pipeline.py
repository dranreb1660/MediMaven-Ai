#!/usr/bin/env python3
"""
MediMaven ML Pipeline Orchestrator

This script coordinates the entire ML pipeline from data processing to model training.
It integrates all the modular components into a cohesive workflow.

My notes:
- Orchestrates data processing, embedding generation, negative sampling, and LTR training
- Uses configuration files to manage parameters
- Includes error handling and logging for production readiness
- Can be run end-to-end or in stages
"""

import argparse
import logging
import pathlib
import sys
import yaml
from typing import Dict, Any, Optional

# Add src to path so we can import our modules
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

from src.data.processors import DocumentProcessor, ChunkProcessor
from src.data.embeddings import EmbeddingGenerator, VectorStoreManager
from src.data.negative_sampling import HardNegativeMiner
from src.models.ltr_models import LTRTrainingPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MediMavenPipeline:
    """
    Main pipeline orchestrator for MediMaven ML system.
    
    My pipeline architecture:
    1. Data Processing: Clean and chunk documents
    2. Embedding Generation: Create dense embeddings and vector store
    3. Negative Sampling: Mine hard negatives for training data
    4. LTR Training: Train ranking model with generated features
    """
    
    def __init__(self, config_dir: pathlib.Path):
        """Initialize pipeline with configuration."""
        self.config_dir = pathlib.Path(config_dir)
        
        # Load configurations
        self.data_config = self._load_config('data_config.yaml')
        self.model_config = self._load_config('model_config.yaml')
        
        # Set up directories
        self.data_dir = pathlib.Path(self.data_config['data']['final_dir'])
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("MediMaven pipeline initialized")
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load YAML configuration file."""
        config_path = self.config_dir / config_file
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
    
    def run_data_processing(self) -> None:
        """
        Run data processing stage.
        
        My data processing flow:
        1. Initialize document processor with config
        2. Process raw documents into chunks
        3. Save processed chunks for embedding stage
        """
        logger.info("=== Starting Data Processing Stage ===")
        
        # Get processing config
        proc_config = self.data_config['processing']
        
        # Initialize processors
        doc_processor = DocumentProcessor()
        chunk_processor = ChunkProcessor(
            chunk_size=proc_config['chunk_size'],
            chunk_overlap=proc_config['chunk_overlap'],
            min_chunk_length=proc_config['min_chunk_length']
        )
        
        # Define paths
        raw_path = pathlib.Path(self.data_config['data']['raw_dir']) / self.data_config['data']['raw_documents']
        output_path = self.data_dir / self.data_config['data']['processed_chunks']
        
        # Process documents if raw file exists
        if raw_path.exists():
            # Load raw documents
            documents = doc_processor.load_documents(raw_path)
            logger.info(f"Loaded {len(documents)} documents")
            
            # Process and chunk documents
            all_chunks = []
            for doc in documents:
                # Clean document
                cleaned_doc = doc_processor.clean_text(doc['content'])
                
                # Create chunks
                chunks = chunk_processor.create_chunks(cleaned_doc, doc.get('doc_id', 'unknown'))
                all_chunks.extend(chunks)
            
            # Save processed chunks
            chunk_processor.save_chunks(all_chunks, output_path)
            logger.info(f"Saved {len(all_chunks)} chunks to {output_path}")
        else:
            logger.warning(f"Raw documents file not found: {raw_path}")
            logger.info("Skipping data processing stage")
    
    def run_embedding_generation(self) -> None:
        """
        Run embedding generation stage.
        
        My embedding flow:
        1. Load processed chunks
        2. Generate embeddings using sentence transformers
        3. Build vector store for similarity search
        4. Save embeddings and indexes
        """
        logger.info("=== Starting Embedding Generation Stage ===")
        
        # Get embedding config
        embed_config = self.data_config['embeddings']
        
        # Initialize embedding generator
        embedding_gen = EmbeddingGenerator(
            model_name=embed_config['model_name'],
            device=embed_config['device'],
            batch_size=embed_config['batch_size']
        )
        
        # Define paths
        chunks_path = self.data_dir / self.data_config['data']['processed_chunks']
        embeddings_path = self.data_dir / self.data_config['data']['embeddings']
        
        if chunks_path.exists():
            # Load chunks
            chunks_df = embedding_gen.load_chunks(chunks_path)
            logger.info(f"Loaded {len(chunks_df)} chunks for embedding")
            
            # Generate embeddings
            embeddings = embedding_gen.generate_embeddings(chunks_df['content'].tolist())
            
            # Save embeddings
            embedding_gen.save_embeddings(embeddings, chunks_df, embeddings_path)
            
            # Initialize vector store manager
            vector_config = embed_config['vector_store']
            vector_manager = VectorStoreManager(
                collection_name=vector_config['collection_name'],
                distance_metric=vector_config['distance_metric']
            )
            
            # Build vector store
            vector_manager.build_index(embeddings, chunks_df)
            logger.info("Vector store built successfully")
        else:
            logger.warning(f"Processed chunks file not found: {chunks_path}")
            logger.info("Skipping embedding generation stage")
    
    def run_negative_sampling(self) -> None:
        """
        Run negative sampling stage.
        
        My negative sampling flow:
        1. Load embeddings and build BM25 index
        2. Generate hard negative samples
        3. Create training features for LTR
        4. Save features for model training
        """
        logger.info("=== Starting Negative Sampling Stage ===")
        
        # Get sampling config
        neg_config = self.data_config['negative_sampling']
        bm25_config = self.data_config['bm25']
        
        # Initialize negative miner
        neg_miner = HardNegativeMiner(
            num_negatives=neg_config['num_negatives_per_query'],
            min_bm25_score=neg_config['min_bm25_score'],
            max_cosine_sim=neg_config['max_cosine_similarity'],
            random_seed=neg_config['random_seed']
        )
        
        # Define paths
        embeddings_path = self.data_dir / self.data_config['data']['embeddings']
        bm25_path = self.data_dir / self.data_config['data']['bm25_index']
        features_path = self.data_dir / self.data_config['data']['ltr_features']
        
        if embeddings_path.exists():
            # Load embeddings and chunks
            embeddings_data = neg_miner.load_embeddings(embeddings_path)
            chunks_df = embeddings_data['chunks_df']
            embeddings = embeddings_data['embeddings']
            
            # Build BM25 index
            neg_miner.build_bm25_index(
                chunks_df['content'].tolist(),
                k1=bm25_config['k1'],
                b=bm25_config['b']
            )
            
            # Save BM25 index
            neg_miner.save_bm25_index(bm25_path)
            
            # Generate sample queries (placeholder - in practice these would come from logs or annotations)
            sample_queries = [
                "What are the symptoms of diabetes?",
                "How to treat hypertension?",
                "Side effects of aspirin",
                "Diagnosis of heart disease",
                "Treatment for anxiety disorders"
            ]
            
            # Mine hard negatives and create features
            features_df = neg_miner.create_training_features(
                queries=sample_queries,
                chunks_df=chunks_df,
                embeddings=embeddings
            )
            
            # Save training features
            features_df.to_parquet(features_path, index=False)
            logger.info(f"Saved {len(features_df)} training features to {features_path}")
        else:
            logger.warning(f"Embeddings file not found: {embeddings_path}")
            logger.info("Skipping negative sampling stage")
    
    def run_ltr_training(self) -> None:
        """
        Run LTR model training stage.
        
        My LTR training flow:
        1. Load training features
        2. Initialize LambdaMART ranker
        3. Train model with validation
        4. Save trained model
        """
        logger.info("=== Starting LTR Training Stage ===")
        
        # Get model config
        ltr_config = self.model_config['lambdamart']
        
        # Initialize training pipeline
        pipeline_config = {
            'model_params': {
                'objective': ltr_config['objective'],
                'metric': ltr_config['metric'],
                'num_leaves': ltr_config['num_leaves'],
                'learning_rate': ltr_config['learning_rate'],
                'n_estimators': ltr_config['n_estimators'],
                'eval_at': ltr_config['eval_at'],
                'random_state': ltr_config['random_state']
            },
            'features_file': self.data_config['data']['ltr_features'],
            'model_file': 'ltr_lambdamart.txt'
        }
        
        ltr_pipeline = LTRTrainingPipeline(
            data_dir=self.data_dir,
            config=pipeline_config
        )
        
        # Run training
        results = ltr_pipeline.run_training(experiment_name='main_pipeline_training')
        
        logger.info("LTR training completed!")
        logger.info(f"Model saved to: {results['model_path']}")
        logger.info(f"Best NDCG@10: {results['training_info']['best_score']:.4f}")
        
        return results
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete end-to-end pipeline.
        
        My full pipeline orchestration:
        - Runs all stages in sequence
        - Handles errors gracefully
        - Returns summary of results
        """
        logger.info("=== Starting Full MediMaven ML Pipeline ===")
        
        results = {}
        
        try:
            # Run all stages
            self.run_data_processing()
            results['data_processing'] = 'completed'
            
            self.run_embedding_generation()
            results['embedding_generation'] = 'completed'
            
            self.run_negative_sampling()
            results['negative_sampling'] = 'completed'
            
            ltr_results = self.run_ltr_training()
            results['ltr_training'] = ltr_results
            
            logger.info("=== Full Pipeline Completed Successfully ===")
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {str(e)}")
            results['error'] = str(e)
            raise
        
        return results
    
    def run_stage(self, stage_name: str) -> None:
        """Run a specific pipeline stage."""
        stage_methods = {
            'data_processing': self.run_data_processing,
            'embedding_generation': self.run_embedding_generation,
            'negative_sampling': self.run_negative_sampling,
            'ltr_training': self.run_ltr_training
        }
        
        if stage_name not in stage_methods:
            raise ValueError(f"Unknown stage: {stage_name}. Available stages: {list(stage_methods.keys())}")
        
        logger.info(f"Running single stage: {stage_name}")
        stage_methods[stage_name]()


def main():
    """Main entry point for pipeline execution."""
    parser = argparse.ArgumentParser(description="MediMaven ML Pipeline")
    parser.add_argument(
        '--config_dir', 
        type=str, 
        default='config',
        help='Directory containing configuration files'
    )
    parser.add_argument(
        '--stage', 
        type=str, 
        choices=['data_processing', 'embedding_generation', 'negative_sampling', 'ltr_training', 'full'],
        default='full',
        help='Pipeline stage to run'
    )
    parser.add_argument(
        '--log_level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Initialize pipeline
    pipeline = MediMavenPipeline(config_dir=args.config_dir)
    
    # Run requested stage
    if args.stage == 'full':
        results = pipeline.run_full_pipeline()
        print("\n=== Pipeline Results ===")
        for stage, result in results.items():
            print(f"{stage}: {result}")
    else:
        pipeline.run_stage(args.stage)


if __name__ == "__main__":
    main()
