from pinecone import Pinecone
from app.core.config import settings


_client: Pinecone | None = None


def get_pinecone() -> Pinecone:
    """Lazy singleton — initializes once, reused everywhere."""
    global _client
    if _client is None:
        _client = Pinecone(api_key=settings.pinecone_api_key)
    return _client


def get_index():
    """Returns the Pinecone index object ready to upsert/query."""
    pc = get_pinecone()
    return pc.Index(settings.pinecone_index)