import os
import sys
import gc
import torch
import argparse
from typing import Optional
from pipelines.rag_inference_pipeline import MedicalRAGSystem

def load_rag_system() -> MedicalRAGSystem:
    """Load the RAG system (ZenML will cache this)."""
    # Initialize system with memory optimization flags
    rag = MedicalRAGSystem()
    return rag

def run_inference(rag: MedicalRAGSystem, query: str) -> str:
    """Run a query through the RAG pipeline."""
    response =  rag.run(query)
    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a question.")
    parser.add_argument('-q', '--question', type=str, required=True, help='Question to ask the model')
    args = parser.parse_args()
    
    rag = load_rag_system()
    # Run with all configurations
    response = run_inference(rag, args.question)
    
    print(response)
