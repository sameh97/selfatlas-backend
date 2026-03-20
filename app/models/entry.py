from datetime import datetime, timezone
from typing import Optional
from beanie import Document
from pydantic import Field
import uuid


class Entry(Document):
    user_id: str
    raw_text: str
    source: str = "text"                # text | voice
    energy_level: Optional[int] = None  # 1-5
    context_tags: list[str] = []

    # Filled after Gemini processes the entry
    emotion_primary: Optional[str] = None
    valence: Optional[float] = None     # -1.0 to 1.0
    arousal: Optional[float] = None     # 0.0 to 1.0
    distress_flag: bool = False

    # Pinecone vector ID — so we can link back
    vector_id: Optional[str] = None

    captured_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    class Settings:
        name = "entries"              # MongoDB collection name
        indexes = [
            "user_id",                # fast lookup by user
            "captured_at",            # fast sort by time
        ]