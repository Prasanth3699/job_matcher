# from pydantic_settings import BaseSettings
# from pydantic import PostgresDsn, field_validator
# import os


# class Settings(BaseSettings):
#     POSTGRES_SERVER: str
#     POSTGRES_USER: str
#     POSTGRES_PASSWORD: str
#     POSTGRES_DB: str
#     POSTGRES_PORT: int = 5432

#     DATABASE_URL: str | None = None

#     DEBUG: bool = False
#     TIMEZONE: str = "UTC"
#     SECRET_KEY: str

#     @field_validator("DATABASE_URL", mode="before")
#     @classmethod
#     def assemble_db_connection(cls, v: str | None, info) -> str:
#         if isinstance(v, str):
#             return v
#         return PostgresDsn.build(
#             scheme="postgresql",
#             username=info.data.get("POSTGRES_USER"),
#             password=info.data.get("POSTGRES_PASSWORD"),
#             host=info.data.get("POSTGRES_SERVER"),
#             port=info.data.get("POSTGRES_PORT"),
#             path=f"/{info.data.get('POSTGRES_DB') or ''}",
#         )

#     class Config:
#         env_file = ".env"
#         case_sensitive = True


# settings = Settings()
# print(settings.model_dump())  # To check the loaded settings
