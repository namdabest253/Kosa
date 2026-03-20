"""Application settings loaded from environment variables or .env file."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # OpenAI
    openai_api_key: str = ""
    extraction_model: str = "gpt-4o-mini"
    hypothesis_model: str = "gpt-4o"

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""

    # Ingestion
    batch_api: bool = True  # Use OpenAI Batch API for 50% cost savings on extraction

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = ["http://localhost:5173"]  # Vite dev server


settings = Settings()
