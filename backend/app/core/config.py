from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from typing import Dict, List, Optional

DEFAULT_STOPWORDS = [
    "a",
    "an",
    "the",
    "and",
    "or",
    "if",
    "but",
    "on",
    "in",
    "into",
    "by",
    "for",
    "of",
    "with",
    "we",
    "our",
    "is",
    "are",
    "was",
    "were",
    "this",
    "that",
    "these",
    "those",
    "using",
    "used",
    "use",
    "can",
    "may",
    "might",
    "should",
    "could",
    "to",
    "from",
    "as",
    "at",
    "be",
    "been",
    "it",
    "its",
    "their",
    "they",
    "them",
    "than",
    "then",
    "also",
    "such",
    "however",
    "between",
    "within",
    "through",
    "across",
    "each",
    "both",
    "either",
    "neither",
    "because",
    "due",
    "after",
    "before",
    "over",
    "under",
    "more",
    "most",
    "less",
    "least",
    "many",
    "much",
    "several",
    "various",
    "against",
    "include",
    "includes",
    "including",
    "based",
    "extend",
    "extends",
    "extending",
    "extended",
    "leveraging",
    "leverage",
    "leverages",
    "utilizing",
    "utilize",
    "utilizes",
    "via",
    "around",
    "among",
    "amongst",
    "towards",
    "toward",
    "accompanying",
    "accompanied",
    "accompanies",
    "compared",
    "comparing",
    "compare",
    "compares",
]

DEFAULT_FILLER_PREFIXES = [
    "baseline",
    "baselines",
    "compare",
    "compares",
    "compared",
    "comparing",
    "proposed",
    "propose",
    "proposes",
    "introduce",
    "introduces",
    "introducing",
    "novel",
    "new",
    "simple",
    "improved",
    "fast",
    "robust",
    "efficient",
    "effective",
    "powerful",
    "general",
]

DEFAULT_FILLER_SUFFIXES = [
    "approach",
    "approaches",
    "method",
    "methods",
    "technique",
    "techniques",
    "architecture",
    "architectures",
    "pipeline",
    "pipelines",
    "framework",
    "frameworks",
    "model",
    "models",
    "system",
    "systems",
    "strategy",
    "strategies",
    "procedure",
    "procedures",
    "scheme",
    "schemes",
]

DEFAULT_SCISPACY_MODELS = [
    "en_core_sci_sm",
    "en_core_sci_md",
    "en_core_web_sm",
]


class ConceptExtractionDomainOverride(BaseModel):
    provider_priority: Optional[List[str]] = None
    max_tokens: Optional[int] = None
    stopwords: Optional[List[str]] = None
    filler_prefixes: Optional[List[str]] = None
    filler_suffixes: Optional[List[str]] = None
    ner_model: Optional[str] = None
    llm_prompt: Optional[str] = None
    entity_hints: Dict[str, List[str]] = Field(default_factory=dict)


class ConceptExtractionTuning(BaseModel):
    max_tokens: int = 6
    stopwords: List[str] = Field(default_factory=lambda: list(DEFAULT_STOPWORDS))
    filler_prefixes: List[str] = Field(
        default_factory=lambda: list(DEFAULT_FILLER_PREFIXES)
    )
    filler_suffixes: List[str] = Field(
        default_factory=lambda: list(DEFAULT_FILLER_SUFFIXES)
    )


def _default_domain_overrides() -> Dict[str, ConceptExtractionDomainOverride]:
    return {
        "biology": ConceptExtractionDomainOverride(
            provider_priority=["scispacy", "domain_ner", "llm"],
            max_tokens=8,
            entity_hints={
                "organism": ["bacter", "coli", "saccharomyces", "arabidopsis"],
                "chemical": ["enzyme", "protein", "rna", "dna"],
            },
        ),
        "materials": ConceptExtractionDomainOverride(
            provider_priority=["domain_ner", "scispacy", "llm"],
            max_tokens=8,
            entity_hints={
                "material": [
                    "perovskite",
                    "graphene",
                    "nanotube",
                    "alloy",
                    "oxide",
                    "ceramic",
                ],
                "chemical": ["sulfide", "carbonate", "chloride", "lithium"],
            },
        ),
    }


class ConceptExtractionSettings(BaseModel):
    max_concepts: int = 50
    scispacy_models: List[str] = Field(
        default_factory=lambda: list(DEFAULT_SCISPACY_MODELS)
    )
    providers: List[str] = Field(default_factory=lambda: ["scispacy", "domain_ner", "llm"])
    llm_prompt: Optional[str] = None
    tuning: ConceptExtractionTuning = Field(default_factory=ConceptExtractionTuning)
    domain_overrides: Dict[str, ConceptExtractionDomainOverride] = Field(
        default_factory=_default_domain_overrides
    )


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

    # Canonicalization
    canonicalization_mapping_version: int = Field(default=1)

    # Graph
    graph_metadata_path: Optional[str] = Field(default=None)

    # Tier-2 LLM
    openai_api_key: Optional[str] = Field(default=None)
    openai_organization: Optional[str] = Field(default=None)
    tier2_llm_model: Optional[str] = Field(default=None)
    tier2_llm_base_url: Optional[str] = Field(default=None)
    tier2_llm_completion_path: Optional[str] = Field(default=None)
    tier2_llm_temperature: float = Field(default=0.1)
    tier2_llm_top_p: float = Field(default=1.0)
    tier2_llm_timeout_seconds: float = Field(default=120.0)
    tier2_llm_retry_attempts: int = Field(default=3)
    tier2_llm_max_sections: int = Field(default=24)
    tier2_llm_max_triples: int = Field(default=30)
    # Maximum characters per Tier-2 section chunk after formatting "[idx] sentence" lines.
    tier2_llm_section_chunk_chars: int = Field(default=3500)
    # Number of sentences to overlap between successive section chunks.
    tier2_llm_section_chunk_overlap_sentences: int = Field(default=1)
    # Hard limit on how many chunks can be produced from a single section.
    tier2_llm_max_chunks_per_section: int = Field(default=3)
    tier2_llm_max_section_chars: int = Field(default=3500)
    tier2_llm_force_json: bool = Field(default=True)
    tier2_llm_max_output_tokens: int = Field(default=8129)
    tier2_llm_system_prompt: str = Field(
        default=(
            "You extract scientific facts as (subject, relation, object) with evidence. "
            "Use precise noun phrases and exact character spans. Prioritize capturing every "
            "statement the passage explicitly supports with evidence; only skip when support is "
            "missing or spans cannot be resolved."
        )
    )
    # Tier-3 LLM
    tier3_llm_model: Optional[str] = Field(default=None)
    tier3_llm_base_url: Optional[str] = Field(default=None)
    tier3_llm_completion_path: Optional[str] = Field(default=None)
    tier3_llm_temperature: float = Field(default=0.1)
    tier3_llm_top_p: float = Field(default=1.0)
    tier3_llm_timeout_seconds: float = Field(default=120.0)
    tier3_llm_retry_attempts: int = Field(default=2)
    tier3_llm_force_json: bool = Field(default=True)
    tier3_llm_max_output_tokens: int = Field(default=4096)
    tier3_llm_max_triples: int = Field(default=20)
    tier3_llm_max_sentences: int = Field(default=40)
    tier3_llm_min_rule_hits: int = Field(default=1)
    tier3_llm_prompt: str = Field(
        default=(
            "You review scientific paper sentences and recover missing relations between "
            "methods, datasets, tasks, and reported metrics. Only emit relations that are "
            "explicitly supported by the supplied evidence sentences."
        )
    )
    concept_extraction: ConceptExtractionSettings = Field(
        default_factory=ConceptExtractionSettings
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
