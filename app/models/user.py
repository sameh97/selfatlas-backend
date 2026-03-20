from datetime import datetime, timezone
from typing import Optional
from beanie import Document
from pydantic import Field, EmailStr


class User(Document):
    email: EmailStr
    password_hash: str
    display_name: Optional[str] = None
    timezone: str = "UTC"
    tier: str = "free"                # free | pro
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    class Settings:
        name = "users"
        indexes = [
            [("email", 1)],           # unique index on email
        ]