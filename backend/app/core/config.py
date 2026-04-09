from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    GROQ_API_KEY: str
    COHERE_API_KEY: str
    DATABASE_URL: str


settings = Settings()
