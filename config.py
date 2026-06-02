from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = ""
    database_url: str = "postgresql://agent:agent@localhost:5432/memory_agent"

    # Single-user mode for this build.
    user_id: str = ""
    user_name: str = "Uddeshya"

    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    memory_md_path: str = "./memory.md"

    working_memory_window: int = 10
    semantic_top_k: int = 5
    episodic_top_k: int = 3
    summary_refresh_every_k: int = 10


settings = Settings()
