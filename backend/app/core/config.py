from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    app_name: str = "Scinets API"
    environment: str = Field(default="development")
    api_prefix: str = "/api"

    # Postgres
    postgres_host: str = Field(default="postgres")
    postgres_port: int = Field(default=5432)
    postgres_db: str = Field(default="scinets")
    postgres_user: str = Field(default="scinets")
    postgres_password: str = Field(default="scinets")

    # MinIO
    minio_endpoint: str = Field(default="minio:9000")
    minio_access_key: str = Field(default="minioadmin")
    minio_secret_key: str = Field(default="minioadmin")
    minio_secure: bool = Field(default=False)
    minio_bucket_papers: str = Field(default="papers")
    minio_connect_max_attempts: int = Field(default=20)
    minio_connect_initial_delay_seconds: float = Field(default=1.0)
    minio_connect_max_delay_seconds: float = Field(default=5.0)

    # Qdrant
    qdrant_url: str = Field(default="http://qdrant:6333")
    qdrant_api_key: str | None = None

    # Embeddings
    embedding_model_name: str = Field(default="all-MiniLM-L6-v2")
    embedding_dimension: int = Field(default=384)
    embedding_batch_size: int = Field(default=32)
    qdrant_collection_name: str = Field(default="paper_sections")
    qdrant_upsert_batch_size: int = Field(default=256)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

