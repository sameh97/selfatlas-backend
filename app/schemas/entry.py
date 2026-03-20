from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class EntryCreate(BaseModel):
    raw_text: str
    source: str = "text"
    energy_level: Optional[int] = None   # 1-5
    context_tags: list[str] = []
    captured_at: datetime


class EntryOut(BaseModel):
    id: str
    raw_text: str
    source: str
    energy_level: Optional[int]
    context_tags: list[str]
    emotion_primary: Optional[str]
    valence: Optional[float]
    arousal: Optional[float]
    distress_flag: bool
    captured_at: datetime
    created_at: datetime


class EntryListOut(BaseModel):
    entries: list[EntryOut]
    total: int
    page: int
    per_page: int