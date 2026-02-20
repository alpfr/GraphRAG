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
