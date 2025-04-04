import os
from pathlib import Path

class AppConfig:
    BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
    MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "models/llama_finetuned/gptqmodel_4bit"))
    FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", str(BASE_DIR / "fmodels/faiss_index.bin"))

config = AppConfig()

print(config.BASE_DIR)
print(config.MODEL_PATH)
print(config.FAISS_INDEX_PATH)