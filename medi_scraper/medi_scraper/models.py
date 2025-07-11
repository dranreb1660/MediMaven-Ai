from pydantic import BaseModel, HttpUrl, Field
from typing import List, Optional


class ContentSection(BaseModel):
    heading: str = Field(min_length=1)
    body: str    = Field(min_length=1)
    type: str    = "medical_content"


class MedicalItem(BaseModel):
    url:             HttpUrl
    domain:          str
    title:           str
    last_updated:    Optional[str] = None
    medical_entities: List[str] = []
    content_sections: List[ContentSection]