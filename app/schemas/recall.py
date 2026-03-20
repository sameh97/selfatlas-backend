from pydantic import BaseModel


class RecallRequest(BaseModel):
    query: str
    k: int = 5            


class MemoryChunk(BaseModel):
    entry_id: str
    raw_text: str
    captured_at: str
    emotion_primary: str | None
    valence: float | None


class RecallOut(BaseModel):
    query: str
    memories: list[MemoryChunk]
    synthesis: str              
