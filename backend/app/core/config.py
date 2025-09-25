from __future__ import annotations
from pydantic import Field
from pydantic_settings import BaseSettings


from typing import Optional
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
    qdrant_api_key: Optional[str] = None

    # Embeddings
    embedding_model_name: str = Field(default="all-MiniLM-L6-v2")
    embedding_dimension: int = Field(default=384)
    embedding_batch_size: int = Field(default=32)
    qdrant_collection_name: str = Field(default="paper_sections")
    qdrant_upsert_batch_size: int = Field(default=256)

    # NLP Pipelines
    nlp_pipeline_spec: str = Field(default="spacy:en_core_web_trf,scispacy:en_core_sci_lg")
    nlp_cache_dir: str = Field(default="/tmp/scinets/nlp_cache")
    nlp_max_workers: int = Field(default=0)  # 0 auto-detect
    nlp_min_span_score: float = Field(default=0.25)
    nlp_min_span_char_length: int = Field(default=3)
    nlp_overlap_score_margin: float = Field(default=0.05)

    # Tier-2 LLM
    openai_api_key: Optional[str] = Field(default=None)
    openai_organization: Optional[str] = Field(default=None)
    tier2_llm_model: Optional[str] = Field(default=None)
    tier2_llm_base_url: Optional[str] = Field(default=None)
    tier2_llm_completion_path: Optional[str] = Field(default=None)
    tier2_llm_temperature: float = Field(default=0.1)
    tier2_llm_top_p: float = Field(default=1.0)
    tier2_llm_timeout_seconds: float = Field(default=120.0)
    tier2_llm_max_sections: int = Field(default=24)
    tier2_llm_max_section_chars: int = Field(default=3500)
    tier2_llm_force_json: bool = Field(default=True)
    tier2_llm_max_output_tokens: int = Field(default=8129)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()