from pydantic import BaseModel

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str
    latency_ms: float
    model_version: str