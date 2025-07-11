from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field, validator
import re

class ContentSection(BaseModel):
    heading: str
    body: str
    type: str = "medical_content"

    @validator('body')
    def validate_body_length(cls, v):
        if len(v.strip()) < 50:
            raise ValueError('Content body is too short')
        return v.strip()

class MedicalArticle(BaseModel):
    url: str
    domain: str
    title: str
    content_sections: List[ContentSection]
    medical_entities: List[str] = Field(min_items=2)
    last_updated: Optional[datetime] = None
    
    @validator('url')
    def validate_url(cls, v):
        if not re.match(r'https?://', v):
            raise ValueError('Invalid URL format')
        return v
    
    @validator('title')
    def validate_title(cls, v):
        if not v or len(v.strip()) < 3:
            raise ValueError('Title is too short or empty')
        return v.strip()
    
    @validator('content_sections')
    def validate_sections(cls, v):
        if not v:
            raise ValueError('Article must have at least one content section')
        return v
    
    @validator('medical_entities')
    def validate_entities(cls, v):
        if not v or len(v) < 2:
            raise ValueError('Article must have at least two medical entities')
        return list(set(v))  # Deduplicate entities

