import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import AnyUrl, Field
from typing import Optional, List

env = os.getenv("APP_ENV", "development")
env_file = f".env.{env}"

class Settings(BaseSettings):
    database_url: AnyUrl
    # CORS origins can be overridden via CORS_ORIGINS env var (JSON list or comma-separated)
    cors_origins: List[str] = Field(default=["http://localhost:3000"])
    # JWT configuration
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    refresh_token_expire_days: int = 7

    # API keys
    google_api_key: str = Field(..., env="GOOGLE_API_KEY")
    tavily_api_key: Optional[str] = Field(None, env="TAVILY_API_KEY")
    cohere_api_key: Optional[str] = Field(None, env="COHERE_API_KEY")

    model_config = SettingsConfigDict(
        env_file=env_file,
        env_file_encoding="utf-8",
        extra="ignore",
    )

settings = Settings()
