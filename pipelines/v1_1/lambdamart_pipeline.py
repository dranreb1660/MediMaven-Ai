#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import pathlib
from src.models.ltr_models import LTRTrainingPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
    Main pipeline for training the LambdaMART model.
    """
    logger.info("Starting LambdaMART model training pipeline")
    
    # Directory paths
    data_dir = pathlib.Path('../../data/final')
    
    # Initialize training pipeline
    pipeline = LTRTrainingPipeline(data_dir=data_dir)
    
    # Run training
    results = pipeline.run_training(experiment_name='lambdamart_training_v1')
    
    logger.info("Training completed successfully!")
    logger.info(f"Model saved to: {results['model_path']}")
    logger.info(f"Best NDCG@10: {results['training_info']['best_score']:.4f}")


if __name__ == "__main__":
    main()
