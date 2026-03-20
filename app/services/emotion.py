from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from app.core.llm import get_flash


class EmotionResult(BaseModel):
    primary_emotion: str = Field(
        description=(
            "The dominant emotion. One of: joy, sadness, anger, fear, "
            "anxiety, loneliness, excitement, calm, frustration, "
            "grief, shame, guilt, pride, overwhelm, or mixed."
        )
    )
    valence: float = Field(
        description="Emotional tone from -1.0 (very negative) to 1.0 (very positive).",
        ge=-1.0,
        le=1.0,
    )
    arousal: float = Field(
        description="Energy level from 0.0 (very calm) to 1.0 (very activated).",
        ge=0.0,
        le=1.0,
    )
    intensity: int = Field(
        description="How intense the emotion feels, from 1 (mild) to 5 (overwhelming).",
        ge=1,
        le=5,
    )
    distress_flag: bool = Field(
        description=(
            "True if the entry contains signs of crisis, hopelessness, "
            "self-harm ideation, or significant distress."
        )
    )


_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You analyze emotional content in personal journal entries. "
        "Be precise and compassionate. "
        "Only identify emotions clearly present in the text — never project.",
    ),
    ("human", "Analyze this journal entry:\n\n{text}"),
])

# Build once at module level — reused across all requests
_chain = _prompt | get_flash().with_structured_output(EmotionResult)


async def extract_emotion(text: str) -> EmotionResult:
    return await _chain.ainvoke({"text": text})