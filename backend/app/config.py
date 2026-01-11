"""Centralized configuration for SciNets agents and tools."""

from pydantic_settings import BaseSettings


class AgentConfig(BaseSettings):
    """Configuration settings for SciNets agents.
    
    All settings can be overridden via environment variables with SCINETS_ prefix.
    """
    
    # Timeouts
    EXPLORER_TIMEOUT_SECONDS: int = 60
    EXPERIMENT_TIMEOUT_SECONDS: int = 60
    
    # Limits
    EXPERIMENT_MAX_TURNS: int = 5
    LITERATURE_BATCH_SIZE: int = 3
    MAX_PAPERS_DEFAULT: int = 5
    
    # Evidence
    HYPOTHESES_TO_EVIDENCE: int = 3
    EVIDENCE_PAPERS_LIMIT: int = 6
    
    # Memory thresholds
    MEMORY_SIMILARITY_THRESHOLD: float = 0.7
    
    # LLM settings
    DEFAULT_TEMPERATURE: float = 0.0
    HIGH_CREATIVITY_TEMPERATURE: float = 0.7
    LOW_CREATIVITY_TEMPERATURE: float = 0.2
    
    class Config:
        env_prefix = "SCINETS_"


# Global config instance
config = AgentConfig()
