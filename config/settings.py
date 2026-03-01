from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # Neo4j Graph Settings
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password123"

    # OpenSearch Settings
    opensearch_host: str = "localhost"
    opensearch_port: int = 9200
    opensearch_user: str = "admin"
    opensearch_password: str = "Gr@phR@g!2026_OpenSearch!"
    opensearch_index_name: str = "graphrag-documents"

    # Ollama Settings
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "ibm/granite4:latest"
    ollama_embedding_model: str = "granite-embedding:278m"

    # Multi-LLM Routing Settings
    llm_provider: str = "ollama" # Options: ollama, openai, anthropic, gemini
    llm_temperature: float = 0.1
    llm_max_tokens: int = 4096
    llm_top_p: float = 1.0

    # API Keys for Cloud Providers
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""

    # Per-provider model overrides (optional)
    openai_model: str = "gpt-4o"
    anthropic_model: str = "claude-sonnet-4-20250514"
    gemini_model: str = "gemini-2.0-flash"
    openai_embedding_model: str = "text-embedding-3-small"
    gemini_embedding_model: str = "models/text-embedding-004"

    # Configuration for files
    input_dir: str = "input"
    output_dir: str = "output"
    
    # NLP parameters
    chunk_size: int = 500
    chunk_overlap: int = 50

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
