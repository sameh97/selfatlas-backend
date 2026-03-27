from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.core.llm import get_flash
from app.models.entry import Entry
from app.services.embedding import query_similar


WITNESS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are Self Atlas — a personal emotional memory companion.
You have retrieved this person's actual past journal entries, shown below.
These are their real words, written in real moments.

Your role is to WITNESS — not advise, not fix, not diagnose.

Rules:
- Reference specific past moments by date naturally
- Name emotions they may not have named themselves — precisely, not clinically  
- Never say "you should" or give advice
- End with ONE open question rooted in what they shared
- Be warm and human — not like a product or a bot
- If no memories were found, say so simply and honestly

Their retrieved memories:
{context}""",
    ),
    ("human", "{query}"),
])

_chain = WITNESS_PROMPT | get_flash() | StrOutputParser()

from bson import ObjectId

async def recall(
    query: str, 
    user_id: str, 
    k: int = 5, 
    threshold: float = 0.60
) -> tuple[list[dict], str]:
    """
    RAG Pipeline:
    1. Search Pinecone for similar vectors.
    2. Filter by similarity score to remove 'less related' noise.
    3. Bridge String IDs to MongoDB ObjectIds.
    4. Fetch full entries and map them to the Pydantic schema.
    5. Synthesize a witnessing response using Gemini Flash.
    """

    # 1. Semantic search in Pinecone
    all_matches = await query_similar(query, user_id, k=k)

    # 2. Filter by similarity score (The "Noise" Filter)
    # Adjust this 0.75 based on your specific index metrics
    matches = [m for m in all_matches if m.score >= threshold]

    if not matches:
        return [], "I looked through your past entries, but I couldn't find any specific memories that match that right now."

    # 3. Bridge the Gap: Convert string IDs to BSON ObjectIds
    entry_ids = [
        ObjectId(m.metadata.get("entry_id")) 
        for m in matches 
        if m.metadata and m.metadata.get("entry_id")
    ]

    # 4. Fetch documents from MongoDB
    entries = await Entry.find(
        {"_id": {"$in": entry_ids}, "user_id": user_id}
    ).to_list()

    # 5. Map entries to preserve relevance order and match MemoryChunk schema
    entry_map = {str(e.id): e for e in entries}
    
    memory_chunks = []
    context_parts = []

    for m in matches:
        eid_str = m.metadata.get("entry_id")
        entry = entry_map.get(eid_str)
        
        if not entry:
            continue
            
        # Matches your MemoryChunk Pydantic model exactly
        memory_chunks.append({
            "entry_id": eid_str,
            "raw_text": entry.raw_text,
            "captured_at": entry.captured_at.isoformat(), # ISO string for Pydantic/Frontend
            "emotion_primary": entry.emotion_primary or "unknown",
            "valence": entry.valence or 0.0,
            "score": m.score
        })
        
        # Format context for the LLM
        date_str = entry.captured_at.strftime("%B %d, %Y")
        context_parts.append(f"[{date_str}]: {entry.raw_text}")

    # 6. Generate Synthesis
    context_text = "\n\n".join(context_parts)
    chain = WITNESS_PROMPT | get_flash() | StrOutputParser()

    synthesis = await chain.ainvoke({
        "query": query,
        "context": context_text
    })

    return memory_chunks, synthesis