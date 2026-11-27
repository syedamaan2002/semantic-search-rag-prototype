from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = ""
    chunk_size: int = 900
    chunk_overlap: int = 150
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

settings = Settings()
