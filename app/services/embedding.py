from app.core.config import settings
from app.db.pinecone import get_index


def _embed(text: str) -> list[float]:
    from google import genai
    client = genai.Client(api_key=settings.google_api_key)
    result = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=text,
    )
    return result.embeddings[0].values


async def _aembed(text: str) -> list[float]:
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _embed, text)


async def embed_and_upsert(
    vector_id: str,
    text: str,
    user_id: str,
    entry_id: str,
    emotion_primary: str | None,
    valence: float | None,
    captured_at: str,
) -> None:
    vector = await _aembed(text)

    index = get_index()
    index.upsert(
        vectors=[
            {
                "id": vector_id,
                "values": vector,
                "metadata": {
                    "user_id": user_id,
                    "entry_id": entry_id,
                    "emotion_primary": emotion_primary or "unknown",
                    "valence": valence or 0.0,
                    "captured_at": captured_at,
                    "text_preview": text[:500],
                },
            }
        ],
        namespace=user_id,
    )


async def delete_vector(vector_id: str, user_id: str) -> None:
    index = get_index()
    index.delete(ids=[vector_id], namespace=user_id)


async def query_similar(query_text: str, user_id: str, k: int = 5):
    query_vector = await _aembed(query_text)
    index = get_index()
    
    print(f"DEBUG: Querying Pinecone namespace: '{user_id}' with k={k}")
    
    results = index.query(
        vector=query_vector,
        top_k=k,
        namespace=user_id,
        include_metadata=True,
    )
    
    print(f"DEBUG: Pinecone returned {len(results.matches)} matches.")
    return results.matches