from pydantic_settings import BaseSettings
from pathlib import Path
import os
# from dotenv import load_dotenv
# load_dotenv()
# -------------- CONFIGURATION -------------- #
# This file contains the configuration for the Medical RAG system.
# It uses Pydantic for settings management and environment variable loading.
#

class AppConfig(BaseSettings):
    # Raw environment values (relative paths from project root)
    MODEL_PATH: str
    FAISS_INDEX_PATH: str
    METADATA_PATH: str
    LTR_MODEL_PATH: str

    EMBED_MODEL_NAME: str
    HF_TOKEN: str
    WANDB_API_KEY: str
    NVIDIA_VISIBLE_DEVICES: str = "all"
    CUDA_VISIBLE_DEVICES: str = "0"
    TGI_API_URL: str = os.getenv("TGI_API_URL", "http://localhost:8080")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @property
    def BASE_DIR(self) -> Path:
        return Path(__file__).resolve().parent.parent.parent.parent

    @property
    def model_path(self) -> str:
        return str(self.BASE_DIR / self.MODEL_PATH)

    @property
    def faiss_index_path(self) -> str:
        return str(self.BASE_DIR / self.FAISS_INDEX_PATH)

    @property
    def metadata_path(self) -> str:
        return str(self.BASE_DIR / self.METADATA_PATH)

    @property
    def ltr_model_path(self) -> str:
        return str(self.BASE_DIR / self.LTR_MODEL_PATH)

config = AppConfig()
