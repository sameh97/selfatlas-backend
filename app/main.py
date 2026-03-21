from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.db.mongo import connect_mongo, disconnect_mongo
from app.api.v1.router import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await connect_mongo()
    yield
    # Shutdown
    await disconnect_mongo()


app = FastAPI(
    title="Self Atlas API",
    description="Your personal emotional memory system",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.is_dev else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
async def health():
    return {"status": "ok", "app": "self-atlas"}