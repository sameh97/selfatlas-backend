from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
from app.core.config import settings


client: AsyncIOMotorClient | None = None


async def connect_mongo():
    """Call this on FastAPI startup."""
    global client

    # Import all Beanie document models here
    from app.models.entry import Entry
    from app.models.user import User

    client = AsyncIOMotorClient(settings.mongo_url)

    await init_beanie(
        database=client[settings.mongo_db],
        document_models=[Entry, User],
    )
    print(f"✅ MongoDB connected → {settings.mongo_db}")


async def disconnect_mongo():
    """Call this on FastAPI shutdown."""
    global client
    if client:
        client.close()
        print("🔌 MongoDB disconnected")