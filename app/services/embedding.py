from app.core.llm import get_embeddings
from app.db.pinecone import get_index


async def embed_and_upsert(
    vector_id: str,
    text: str,
    user_id: str,
    entry_id: str,
    emotion_primary: str | None,
    valence: float | None,
    captured_at: str,
) -> None:
    """
    Embed text and upsert into Pinecone.
    vector_id = the MongoDB entry ID — links the two systems together.
    """
    embeddings_model = get_embeddings()

    # Get the 768-dim vector from Google
    vector = await embeddings_model.aembed_query(text)

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
                    # Store first 500 chars for recall context
                    # Pinecone metadata has a 40KB limit per vector
                    "text_preview": text[:500],
                },
            }
        ],
        namespace=user_id,   # isolate each user's vectors
    )


async def delete_vector(vector_id: str, user_id: str) -> None:
    """Remove a vector when user deletes an entry."""
    index = get_index()
    index.delete(ids=[vector_id], namespace=user_id)


async def query_similar(
    query_text: str,
    user_id: str,
    k: int = 5,
) -> list[dict]:
    """
    Embed a query and find the k most similar entries.
    Returns list of Pinecone matches with metadata.
    """
    embeddings_model = get_embeddings()
    query_vector = await embeddings_model.aembed_query(query_text)

    index = get_index()

    results = index.query(
        vector=query_vector,
        top_k=k,
        namespace=user_id,       # only search THIS user's vectors
        include_metadata=True,
    )

    return results.matches