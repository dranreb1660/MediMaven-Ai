from pipelines.rag_inference_pipeline import MedicalRAGSystem
from src.backend.app.config import config

class RAGService:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = MedicalRAGSystem(
                llm_path=config.MODEL_PATH,
                faiss_index_path=config.FAISS_INDEX_PATH
            )
            # Warm-up
            cls._instance.embed_query("warmup")
        return cls._instance