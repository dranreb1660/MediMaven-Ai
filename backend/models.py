# src/backend/models.py
from sqlalchemy import Column, String, DateTime, JSON, func
from backend.services.db import Base

class Conversation(Base):
    __tablename__ = "conversations"

    cid        = Column(String, primary_key=True)
    user_id    = Column(String, index=True)
    preview    = Column(String, nullable=False)
    messages   = Column(JSON, nullable=False)          # last ≤ 50 turns
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(),
                         onupdate=func.now())
