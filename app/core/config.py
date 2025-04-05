# app/core/config.py
from pydantic_settings import BaseSettings
from pydantic import PostgresDsn, field_validator
from typing import List
from functools import lru_cache


class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Resume Jobs API"

    # Auth Settings
    SECRET_KEY: str = (
        "bcd2ea681c63e1e6a4362f267c18ceffc318f95111c52d420c11516bfa8dfa6a2e51c863d6521787f336b799e209f8eb648a66a3f1d4abee2e97235b32592d6b46b312bb046ab8a5a8d540dfd0f6b2a8936c64bb911e91bf54073e54abfceafe55b19fcf62c789f1704734253367e24074f7e8cb2182b228998951536a972b70bd7d30032600941206b1d21dd3e44c289a015037c5a480cf7b49390bab7cbe3e38d211aeaf9444ef4ac80b0df7f69dbc5d3e3ec1da30bdc5e54ae1585f3b683c58417b5463c3ab9d41882ca7737b58fae0c199c70a4eecfe485a0ad9f33cf798b56a1733c6c25a447541af5edda5dc6a713f7196813c992191837ced8da9476c"
    )
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Database Settings
    POSTGRES_SERVER: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_PORT: int = 5432
    DATABASE_URL: str | None = None

    # Application Settings
    DEBUG: bool = False
    TIMEZONE: str = "UTC"

    # Services
    JOBS_SERVICE_URL: str = "http://localhost:8000"
    AUTH_SERVICE_URL: str = "http://localhost:8000"

    # CORS
    CORS_ORIGINS: List[str] = ["*"]

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def assemble_db_connection(cls, v: str | None, info) -> str:
        if isinstance(v, str):
            return v
        return PostgresDsn.build(
            scheme="postgresql",
            username=info.data.get("POSTGRES_USER"),
            password=info.data.get("POSTGRES_PASSWORD"),
            host=info.data.get("POSTGRES_SERVER"),
            port=info.data.get("POSTGRES_PORT"),
            path=f"/{info.data.get('POSTGRES_DB') or ''}",
        )

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

if settings.DEBUG:
    print(settings.model_dump())  # Print settings only in debug mode
