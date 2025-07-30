"""
Inference pipeline script for MediMaven.

This script handles the complete inference workflow from query to answer.
"""

import logging
import json
from typing import Dict, Any, List
from pathlib import Path

# Import from existing modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.inference.rag_engine import RAGPipeline
from src.inference.model_server import ModelServer

logger = logging.getLogger(__name__)


class MediMavenInferencePipeline:
    """
    Complete inference pipeline for MediMaven system.
    
    My inference pipeline design:
    - Query preprocessing and validation
    - Retrieval and ranking
    - Response generation and post-processing
    - Logging and monitoring integration
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize inference pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or "config/"
        self.config = self._load_config()
        
        # Initialize RAG pipeline
        self.rag_pipeline = RAGPipeline(config_path=self.config_path)
        
        logger.info("MediMaven inference pipeline initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load inference configuration."""
        # Placeholder configuration
        return {
            'max_query_length': 512,
            'min_confidence_threshold': 0.1,
            'response_format': 'structured',
            'enable_logging': True
        }
    
    def preprocess_query(self, query: str) -> str:
        """
        Preprocess user query before inference.
        
        My preprocessing approach:
        - Input validation and sanitization
        - Query length limits
        - Medical terminology normalization
        - Safety filtering
        
        Args:
            query: Raw user query
            
        Returns:
            Preprocessed query
        """
        # Basic preprocessing
        cleaned_query = query.strip()
        
        # Length validation
        max_length = self.config.get('max_query_length', 512)
        if len(cleaned_query) > max_length:
            cleaned_query = cleaned_query[:max_length]
            logger.warning(f"Query truncated to {max_length} characters")
        
        # Basic validation
        if not cleaned_query:
            raise ValueError("Empty query provided")
        
        logger.debug(f"Preprocessed query: {cleaned_query[:100]}...")
        return cleaned_query
    
    def postprocess_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post-process RAG response for final output.
        
        My post-processing approach:
        - Response formatting and structuring
        - Confidence score validation
        - Source attribution formatting
        - Safety checks and disclaimers
        
        Args:
            response: Raw RAG response
            
        Returns:
            Formatted response
        """
        processed_response = {
            'answer': response.get('answer', ''),
            'confidence': response.get('confidence', 0.0),
            'sources': response.get('sources', []),
            'metadata': response.get('metadata', {}),
            'timestamp': self._get_timestamp(),
            'disclaimer': self._get_medical_disclaimer()
        }
        
        # Confidence threshold check
        min_confidence = self.config.get('min_confidence_threshold', 0.1)
        if processed_response['confidence'] < min_confidence:
            processed_response['answer'] = (
                "I don't have enough confidence in this answer based on the available sources. "
                "Please consult with a healthcare professional for medical advice."
            )
            processed_response['low_confidence_warning'] = True
        
        logger.debug(f"Post-processed response with confidence {processed_response['confidence']:.2f}")
        return processed_response
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for response."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _get_medical_disclaimer(self) -> str:
        """Get medical disclaimer for responses."""
        return (
            "This information is for educational purposes only and should not replace "
            "professional medical advice. Please consult with a healthcare provider "
            "for personalized medical guidance."
        )
    
    def run_inference(self, query: str) -> Dict[str, Any]:
        """
        Run complete inference pipeline for a single query.
        
        Args:
            query: User query
            
        Returns:
            Complete inference result
        """
        logger.info(f"Running inference for query: {query[:50]}...")
        
        try:
            # Preprocess query
            processed_query = self.preprocess_query(query)
            
            # Run RAG inference
            rag_response = self.rag_pipeline.run_inference(processed_query)
            
            # Post-process response
            final_response = self.postprocess_response(rag_response)
            
            logger.info(f"Inference complete with confidence {final_response['confidence']:.2f}")
            return final_response
            
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            
            # Return error response
            return {
                'answer': "I apologize, but I encountered an error processing your query. Please try again.",
                'confidence': 0.0,
                'sources': [],
                'metadata': {'error': str(e)},
                'timestamp': self._get_timestamp(),
                'disclaimer': self._get_medical_disclaimer(),
                'error': True
            }
    
    def batch_inference(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Run inference for multiple queries.
        
        Args:
            queries: List of user queries
            
        Returns:
            List of inference results
        """
        logger.info(f"Running batch inference for {len(queries)} queries")
        
        results = []
        for i, query in enumerate(queries):
            logger.debug(f"Processing query {i+1}/{len(queries)}")
            result = self.run_inference(query)
            results.append(result)
        
        logger.info(f"Batch inference complete for {len(queries)} queries")
        return results


def run_single_inference(query: str, config_path: str = None) -> Dict[str, Any]:
    """
    Run inference for a single query.
    
    Args:
        query: User query
        config_path: Optional configuration path
        
    Returns:
        Inference result
    """
    pipeline = MediMavenInferencePipeline(config_path)
    return pipeline.run_inference(query)


def run_batch_inference(queries: List[str], config_path: str = None) -> List[Dict[str, Any]]:
    """
    Run inference for multiple queries.
    
    Args:
        queries: List of user queries
        config_path: Optional configuration path
        
    Returns:
        List of inference results
    """
    pipeline = MediMavenInferencePipeline(config_path)
    return pipeline.batch_inference(queries)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MediMaven Inference Pipeline")
    parser.add_argument("--query", type=str,
                       help="Single query to process")
    parser.add_argument("--queries-file", type=str,
                       help="File containing queries (one per line)")
    parser.add_argument("--output", type=str,
                       help="Output file for results")
    parser.add_argument("--config", type=str, default="config/",
                       help="Configuration directory")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Determine processing mode
    if args.query:
        # Single query mode
        result = run_single_inference(args.query, args.config)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Result saved to {args.output}")
        else:
            print("=" * 60)
            print(f"Query: {args.query}")
            print(f"Answer: {result['answer']}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Sources: {len(result['sources'])}")
            print("=" * 60)
    
    elif args.queries_file:
        # Batch processing mode
        with open(args.queries_file, 'r') as f:
            queries = [line.strip() for line in f if line.strip()]
        
        results = run_batch_inference(queries, args.config)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output}")
        else:
            for i, result in enumerate(results):
                print(f"\n--- Query {i+1} ---")
                print(f"Query: {queries[i]}")
                print(f"Answer: {result['answer'][:200]}...")
                print(f"Confidence: {result['confidence']:.2f}")
    
    else:
        print("Please provide either --query or --queries-file")
        parser.print_help()
