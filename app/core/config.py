from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    
    access_token_expire_minutes: int = 10080  # 7 days
    
    # App
    app_env: str = "development"
    secret_key: str

    # Gemini
    google_api_key: str

    # MongoDB
    mongo_url: str = "mongodb://mongo:27017"
    mongo_db: str = "self_atlas"

    # Pinecone
    pinecone_api_key: str
    pinecone_index: str = "self-atlas"

    @property
    def is_dev(self) -> bool:
        return self.app_env == "development"


settings = Settings()