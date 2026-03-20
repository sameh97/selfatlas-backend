from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Query
from app.models.entry import Entry
from app.models.user import User
from app.schemas.entry import EntryCreate, EntryOut, EntryListOut
from app.services.emotion import extract_emotion
from app.services.embedding import embed_and_upsert
from app.core.security import get_current_user

router = APIRouter(prefix="/entries", tags=["entries"])


@router.post("", response_model=EntryOut)
async def create_entry(
    body: EntryCreate,
    current_user: User = Depends(get_current_user),
):
    # 1. Save raw entry to MongoDB immediately
    entry = Entry(
        user_id=str(current_user.id),
        raw_text=body.raw_text,
        source=body.source,
        energy_level=body.energy_level,
        context_tags=body.context_tags,
        captured_at=body.captured_at,
    )
    await entry.insert()

    # 2. Extract emotion with Gemini
    try:
        emotion = await extract_emotion(body.raw_text)
        entry.emotion_primary = emotion.primary_emotion
        entry.valence = emotion.valence
        entry.arousal = emotion.arousal
        entry.distress_flag = emotion.distress_flag
    except Exception as e:
        # Don't fail the whole request if AI extraction fails
        print(f"⚠️ Emotion extraction failed: {e}")

    # 3. Embed and upsert to Pinecone
    try:
        vector_id = str(entry.id)
        await embed_and_upsert(
            vector_id=vector_id,
            text=body.raw_text,
            user_id=str(current_user.id),
            entry_id=str(entry.id),
            emotion_primary=entry.emotion_primary,
            valence=entry.valence,
            captured_at=entry.captured_at.isoformat(),
        )
        entry.vector_id = vector_id
    except Exception as e:
        print(f"⚠️ Embedding failed: {e}")

    # 4. Save updated entry with emotion + vector_id
    await entry.save()

    return _to_out(entry)


@router.get("", response_model=EntryListOut)
async def list_entries(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
):
    user_id = str(current_user.id)
    skip = (page - 1) * per_page

    total = await Entry.find(Entry.user_id == user_id).count()

    entries = await (
        Entry.find(Entry.user_id == user_id)
        .sort(-Entry.captured_at)
        .skip(skip)
        .limit(per_page)
        .to_list()
    )

    return EntryListOut(
        entries=[_to_out(e) for e in entries],
        total=total,
        page=page,
        per_page=per_page,
    )


@router.get("/{entry_id}", response_model=EntryOut)
async def get_entry(
    entry_id: str,
    current_user: User = Depends(get_current_user),
):
    entry = await Entry.get(entry_id)

    if not entry or entry.user_id != str(current_user.id):
        raise HTTPException(404, "Entry not found")

    return _to_out(entry)


@router.delete("/{entry_id}", status_code=204)
async def delete_entry(
    entry_id: str,
    current_user: User = Depends(get_current_user),
):
    from app.services.embedding import delete_vector

    entry = await Entry.get(entry_id)

    if not entry or entry.user_id != str(current_user.id):
        raise HTTPException(404, "Entry not found")

    # Remove from Pinecone first
    if entry.vector_id:
        await delete_vector(entry.vector_id, str(current_user.id))

    await entry.delete()


# ── Helper ────────────────────────────────────────────────────────────────────

def _to_out(entry: Entry) -> EntryOut:
    return EntryOut(
        id=str(entry.id),
        raw_text=entry.raw_text,
        source=entry.source,
        energy_level=entry.energy_level,
        context_tags=entry.context_tags,
        emotion_primary=entry.emotion_primary,
        valence=entry.valence,
        arousal=entry.arousal,
        distress_flag=entry.distress_flag,
        captured_at=entry.captured_at,
        created_at=entry.created_at,
    )