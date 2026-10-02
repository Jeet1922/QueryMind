from functools import lru_cache
from pathlib import Path
from typing import Final

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).resolve().parents[3] / ".env")


class Settings(BaseModel):
    DATABASE_URL: str = Field(
        default="postgresql+psycopg://querymind:querymind@localhost:5432/querymind",
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
            "postgresql+psycopg://querymind:querymind@localhost:5432/querymind",
        ),
        APP_ENV=os.getenv("APP_ENV", "development"),
        LOG_LEVEL=os.getenv("LOG_LEVEL", "INFO"),
    )


settings: Final[Settings] = get_settings()
