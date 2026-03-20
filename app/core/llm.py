import os
from langchain.chat_models import init_chat_model
from app.core.config import settings

os.environ["GOOGLE_API_KEY"] = settings.google_api_key


def get_flash():
    """Fast + cheap — emotion extraction, recall synthesis."""
    return init_chat_model("google_genai:gemini-2.5-flash", temperature=0.3)


def get_embeddings():
    """Google text-embedding-004 — 768 dims, matches Pinecone index."""
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=settings.google_api_key,
    )