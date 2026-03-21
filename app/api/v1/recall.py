from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from app.schemas.recall import RecallRequest, RecallOut, MemoryChunk
from app.services.recall import recall
from app.models.user import User
from app.core.security import get_current_user

router = APIRouter(prefix="/recall", tags=["recall"])


@router.post("", response_model=RecallOut)
async def recall_query(
    body: RecallRequest,
    current_user: User = Depends(get_current_user),
):
    memory_chunks, synthesis = await recall(
        query=body.query,
        user_id=str(current_user.id),
        k=body.k,
    )

    return RecallOut(
        query=body.query,
        memories=[MemoryChunk(**m) for m in memory_chunks],
        synthesis=synthesis,
    )


@router.post("/stream")
async def recall_stream(
    body: RecallRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Streams the Gemini response token by token.
    Frontend reads this as Server-Sent Events.
    """
    from app.services.embedding import query_similar
    from app.models.entry import Entry
    from app.core.llm import get_flash
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from app.services.recall import WITNESS_PROMPT

    # Get memories first
    matches = await query_similar(body.query, str(current_user.id), k=body.k)

    if not matches:
        context = "No relevant memories found."
    else:
        entry_ids = [m.metadata.get("entry_id") for m in matches if m.metadata]
        entries = await Entry.find(
            {"_id": {"$in": entry_ids}, "user_id": str(current_user.id)}
        ).to_list()
        entry_map = {str(e.id): e for e in entries}

        parts = []
        for match in matches:
            entry = entry_map.get(match.metadata.get("entry_id"))
            if not entry:
                continue
            date_str = entry.captured_at.strftime("%B %d, %Y at %I:%M %p")
            emotion = entry.emotion_primary or "unknown"
            parts.append(
                f"--- {date_str} [{emotion}] ---\n{entry.raw_text}"
            )
        context = "\n\n".join(parts)

    # Stream the response
    chain = WITNESS_PROMPT | get_flash() | StrOutputParser()

    async def generate():
        async for chunk in chain.astream({
            "context": context,
            "query": body.query,
        }):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )