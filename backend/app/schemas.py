# src/backend/app/schemas.py
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class Citation(BaseModel):
    id: str
    source: str
    url: Optional[str] = None
    rank: int

class ConversationMessage(BaseModel):
    user: str
    assistant: str

class ChatRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None
    history: Optional[List[ConversationMessage]] = None  # ADD THIS LINE

class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]
    latency: float
    conversation_id: Optional[str] = None
    messages: List[ConversationMessage]

# Rebuild all models in dependency order
Citation.model_rebuild()
ConversationMessage.model_rebuild()
ChatRequest.model_rebuild()
ChatResponse.model_rebuild()