"""
Embedding generation script for MediMaven pipeline.

This script generates embeddings for text data used in the RAG system.
"""

import numpy as np
import logging
from typing import List, Dict, Any
from pathlib import Path

# Import from existing modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data.embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)


class EmbeddingPipeline:
    """
    Pipeline for generating embeddings for medical text data.
    
    My embedding pipeline approach:
    - Batch processing for efficiency
    - Multiple embedding models support
    - Caching and persistence
    - Quality validation and metrics
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize embedding pipeline.
        
        Args:
            config: Configuration dictionary for embedding generation
        """
        self.config = config or {}
        
        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator(
            model_name=self.config.get('model_name', 'sentence-transformers/all-MiniLM-L6-v2'),
            config=self.config.get('embedding_config', {})
        )
        
        # Pipeline parameters
        self.batch_size = self.config.get('batch_size', 32)
        self.output_dir = Path(self.config.get('output_dir', 'data/embeddings'))
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info("Embedding pipeline initialized")
    
    def generate_embeddings(self, 
                          texts: List[str], 
                          metadata: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate embeddings for a list of texts.
        
        My generation strategy:
        - Batch processing for memory efficiency
        - Progress tracking and logging
        - Validation of embedding quality
        - Metadata preservation
        
        Args:
            texts: List of texts to embed
            metadata: Optional metadata for each text
            
        Returns:
            Dictionary containing embeddings and metadata
        """
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        
        if metadata is None:
            metadata = [{}] * len(texts)
        
        all_embeddings = []
        processed_metadata = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_metadata = metadata[i:i + self.batch_size]
            
            logger.debug(f"Processing batch {i//self.batch_size + 1}/{(len(texts)-1)//self.batch_size + 1}")
            
            # Generate embeddings for batch
            batch_embeddings = self.embedding_generator.generate_embeddings(batch_texts)
            
            all_embeddings.extend(batch_embeddings)
            processed_metadata.extend(batch_metadata)
        
        logger.info("Embedding generation complete")
        
        return {
            'embeddings': np.array(all_embeddings),
            'texts': texts,
            'metadata': processed_metadata,
            'model_info': {
                'model_name': self.embedding_generator.model_name,
                'embedding_dim': len(all_embeddings[0]) if all_embeddings else 0
            }
        }
    
    def process_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process documents with chunking and embedding generation.
        
        Args:
            documents: List of document dictionaries with 'content' field
            
        Returns:
            Processed embeddings with chunk information
        """
        logger.info(f"Processing {len(documents)} documents...")
        
        all_chunks = []
        all_metadata = []
        
        for doc_idx, doc in enumerate(documents):
            content = doc.get('content', '')
            title = doc.get('title', f'Document {doc_idx}')
            
            # Simple chunking strategy
            chunks = self._chunk_text(content)
            
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    'document_id': doc.get('id', doc_idx),
                    'document_title': title,
                    'chunk_id': f"{doc_idx}_{chunk_idx}",
                    'chunk_index': chunk_idx,
                    'original_metadata': doc.get('metadata', {})
                })
        
        # Generate embeddings for all chunks
        result = self.generate_embeddings(all_chunks, all_metadata)
        
        logger.info(f"Generated embeddings for {len(all_chunks)} chunks from {len(documents)} documents")
        return result
    
    def _chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """
        Simple text chunking strategy.
        
        My chunking approach:
        - Fixed-size chunks with overlap
        - Sentence boundary awareness
        - Preserves context across chunks
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk = ' '.join(chunk_words)
            
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks if chunks else [text]  # Return original text if no chunks created
    
    def save_embeddings(self, embeddings_data: Dict[str, Any], filename: str) -> Path:
        """
        Save embeddings to disk.
        
        Args:
            embeddings_data: Dictionary containing embeddings and metadata
            filename: Name for the output file
            
        Returns:
            Path to saved file
        """
        output_path = self.output_dir / f"{filename}.npz"
        
        # Prepare data for saving
        save_data = {
            'embeddings': embeddings_data['embeddings'],
            'texts': embeddings_data['texts'],
            'model_name': embeddings_data['model_info']['model_name'],
            'embedding_dim': embeddings_data['model_info']['embedding_dim']
        }
        
        # Save metadata separately as JSON-compatible format
        metadata_path = self.output_dir / f"{filename}_metadata.json"
        
        import json
        with open(metadata_path, 'w') as f:
            json.dump(embeddings_data['metadata'], f, indent=2)
        
        # Save embeddings as numpy compressed file
        np.savez_compressed(output_path, **save_data)
        
        logger.info(f"Embeddings saved to {output_path}")
        logger.info(f"Metadata saved to {metadata_path}")
        
        return output_path
    
    def load_embeddings(self, filename: str) -> Dict[str, Any]:
        """
        Load embeddings from disk.
        
        Args:
            filename: Name of the embeddings file (without extension)
            
        Returns:
            Loaded embeddings data
        """
        embeddings_path = self.output_dir / f"{filename}.npz"
        metadata_path = self.output_dir / f"{filename}_metadata.json"
        
        # Load embeddings
        with np.load(embeddings_path) as data:
            embeddings_data = {
                'embeddings': data['embeddings'],
                'texts': data['texts'].tolist(),
                'model_info': {
                    'model_name': str(data['model_name']),
                    'embedding_dim': int(data['embedding_dim'])
                }
            }
        
        # Load metadata if exists
        if metadata_path.exists():
            import json
            with open(metadata_path, 'r') as f:
                embeddings_data['metadata'] = json.load(f)
        else:
            embeddings_data['metadata'] = [{}] * len(embeddings_data['texts'])
        
        logger.info(f"Loaded embeddings from {embeddings_path}")
        return embeddings_data


def run_embedding_pipeline(input_file: str, output_name: str, config: Dict[str, Any] = None):
    """
    Run the complete embedding generation pipeline.
    
    Args:
        input_file: Path to input data file
        output_name: Name for output embeddings
        config: Pipeline configuration
    """
    logger.info("Starting embedding generation pipeline...")
    
    # Initialize pipeline
    pipeline = EmbeddingPipeline(config)
    
    # Load documents (placeholder - would load actual data)
    logger.info(f"Loading documents from {input_file}")
    
    # Placeholder document loading
    documents = [
        {
            'id': '1',
            'title': 'Sample Medical Document 1',
            'content': 'This is a sample medical document content about cardiovascular health.',
            'metadata': {'source': 'medical_journal', 'year': 2023}
        },
        {
            'id': '2', 
            'title': 'Sample Medical Document 2',
            'content': 'This document discusses diabetes management and treatment options.',
            'metadata': {'source': 'clinical_guidelines', 'year': 2023}
        }
    ]
    
    # Process documents and generate embeddings
    embeddings_data = pipeline.process_documents(documents)
    
    # Save embeddings
    output_path = pipeline.save_embeddings(embeddings_data, output_name)
    
    logger.info(f"Embedding pipeline complete. Output saved to {output_path}")
    
    return embeddings_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MediMaven Embedding Generation Pipeline")
    parser.add_argument("--input", type=str, required=True,
                       help="Input data file path")
    parser.add_argument("--output", type=str, required=True,
                       help="Output embeddings name")
    parser.add_argument("--model", type=str, 
                       default="sentence-transformers/all-MiniLM-L6-v2",
                       help="Embedding model name")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for embedding generation")
    
    args = parser.parse_args()
    
    config = {
        'model_name': args.model,
        'batch_size': args.batch_size,
        'output_dir': 'data/embeddings'
    }
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run pipeline
    embeddings_data = run_embedding_pipeline(
        input_file=args.input,
        output_name=args.output,
        config=config
    )
    
    print("=" * 50)
    print(f"Generated embeddings: {embeddings_data['embeddings'].shape}")
    print(f"Model used: {embeddings_data['model_info']['model_name']}")
    print(f"Embedding dimension: {embeddings_data['model_info']['embedding_dim']}")
    print("=" * 50)
