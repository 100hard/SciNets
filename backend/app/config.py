"""Centralized configuration for SciNets agents and tools."""

import os
from pydantic_settings import BaseSettings


class AgentConfig(BaseSettings):
    """Configuration settings for SciNets agents.
    
    All settings can be overridden via environment variables with SCINETS_ prefix.
    """
    
    # =============================================================================
    # Auth & Security
    # =============================================================================
    SECRET_KEY: str = "dev-secret-key-change-in-prod"
    SESSION_EXPIRE_DAYS: int = 7
    MAGIC_LINK_EXPIRE_MINUTES: int = 15

    # =============================================================================
    # Feature Flags
    # =============================================================================
    ENABLE_EXPERIMENT_EXECUTION: bool = False
    ENABLE_DISCOVERY: bool = True

    # =============================================================================
    # Rate Limiting
    # =============================================================================
    MAX_DISCOVERIES_PER_WINDOW: int = 2
    WINDOW_LENGTH_DAYS: int = 7
    ADMIN_EMAILS: list[str] = [] # Set via env vars as json list or use default

    # =============================================================================
    # Email (SMTP)
    # =============================================================================
    SMTP_SERVER: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_EMAIL: str = "" # Set via SCINETS_SMTP_EMAIL
    SMTP_PASSWORD: str = "" # Set via SCINETS_SMTP_PASSWORD

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
