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


async def recall(
    query: str,
    user_id: str,
    k: int = 5,
) -> tuple[list[dict], str]:
    """
    Full recall pipeline:
    1. Embed query → search Pinecone
    2. Fetch full entries from MongoDB
    3. Gemini synthesizes a witnessing response

    Returns (memory_chunks, synthesis)
    """

    # Step 1 — semantic search in Pinecone
    matches = await query_similar(query, user_id, k=k)

    if not matches:
        synthesis = await _chain.ainvoke({
            "context": "No relevant memories found.",
            "query": query,
        })
        return [], synthesis

    # Step 2 — fetch full entries from MongoDB by entry_id
    entry_ids = [m.metadata.get("entry_id") for m in matches if m.metadata]
    entries = await Entry.find(
        {"_id": {"$in": entry_ids}, "user_id": user_id}
    ).to_list()

    # Map id → entry for ordering
    entry_map = {str(e.id): e for e in entries}

    # Step 3 — build context block from retrieved entries
    memory_chunks = []
    context_parts = []

    for match in matches:
        entry_id = match.metadata.get("entry_id")
        entry = entry_map.get(entry_id)
        if not entry:
            continue

        date_str = entry.captured_at.strftime("%B %d, %Y at %I:%M %p")
        emotion = entry.emotion_primary or "unknown"
        valence = f"{entry.valence:+.1f}" if entry.valence is not None else "?"

        context_parts.append(
            f"--- {date_str} [{emotion}, valence: {valence}] ---\n"
            f"{entry.raw_text}"
        )

        memory_chunks.append({
            "entry_id": entry_id,
            "raw_text": entry.raw_text,
            "captured_at": entry.captured_at.isoformat(),
            "emotion_primary": entry.emotion_primary,
            "valence": entry.valence,
        })

    context = "\n\n".join(context_parts)

    # Step 4 — Gemini witnesses
    synthesis = await _chain.ainvoke({
        "context": context,
        "query": query,
    })

    return memory_chunks, synthesis
