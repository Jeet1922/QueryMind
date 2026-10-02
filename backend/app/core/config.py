from functools import lru_cache
from typing import Final

from pydantic import BaseModel, Field


class Settings(BaseModel):
    DATABASE_URL: str = Field(
        default="postgresql+psycopg://insightmesh:insightmesh@localhost:5432/insightmesh",
        description="PostgreSQL connection string used by the application.",
    )
    APP_ENV: str = Field(default="development")
    LOG_LEVEL: str = Field(default="INFO")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    import os

    return Settings(
        DATABASE_URL=os.getenv(
            "DATABASE_URL",
            "postgresql+psycopg://insightmesh:insightmesh@localhost:5432/insightmesh",
        ),
        APP_ENV=os.getenv("APP_ENV", "development"),
        LOG_LEVEL=os.getenv("LOG_LEVEL", "INFO"),
    )


settings: Final[Settings] = get_settings()
