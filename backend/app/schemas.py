# src/backend/app/schemas.py
from pydantic import BaseModel
from typing import List, Optional, Dict, Any  # Add explicit imports

class Citation(BaseModel):
    id: str
    source: str
    url: Optional[str] = None
    rank: int

# Explicitly define message type for messages field
class ConversationMessage(BaseModel):
    user: str
    assistant: str

class ChatRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None
    history: Optional[List[ConversationMessage]] = None

class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]  # Uses defined Citation model
    latency: float
    conversation_id: Optional[str] = None
    messages: List[ConversationMessage]  # Use explicit model instead of dict

# Rebuild all models in dependency order
Citation.model_rebuild()
ConversationMessage.model_rebuild()
ChatRequest.model_rebuild()
ChatResponse.model_rebuild()