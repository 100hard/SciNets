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
    # Auth & Security
    # =============================================================================
    SECRET_KEY: str = "dev-secret-key-change-in-prod"
    SESSION_EXPIRE_DAYS: int = 7
    MAGIC_LINK_EXPIRE_MINUTES: int = 15
    
    # Deployment
    DATABASE_URL: str = "sqlite:///./scinets.db"
    FRONTEND_URL: str = "http://localhost:8080"

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
    
    # =============================================================================
    # Hardening & Limits
    # =============================================================================
    # Default to True for Public Demo safety
    DEMO_MODE: bool = True
    
    # Run Limits (Public Demo V1)
    MAX_AGENT_STEPS: int = 10      # Was 50 - Reduced to prevent runaway chains
    MAX_TOOL_CALLS: int = 5        # Was 20 - interactions are expensive
    MAX_TOTAL_TOKENS: int = 10000  # Was 100k - Strict budget per run
    MAX_RUN_TIME_SECONDS: int = 120 # Was 300 - 2 mins max per run

    # Requests
    MAX_INPUT_CHARS: int = 2000
    
    # =============================================================================
    # Security & Quotas
    # =============================================================================
    JWT_ISSUER: str = "scinets-auth"
    JWT_AUDIENCE: str = "scinets-frontend"
    
    SCINETS_READONLY_MODE: bool = False
    
    # Weekly Quota (Public Demo)
    MAX_RUNS_PER_USER_PER_WEEK: int = 2
    MAX_TOKENS_PER_USER_PER_WEEK: int = 200000 
    MAX_OPENALEX_CALLS_PER_USER_PER_DAY: int = 30
    
    # LLM settings
    DEFAULT_TEMPERATURE: float = 0.0
    HIGH_CREATIVITY_TEMPERATURE: float = 0.7
    LOW_CREATIVITY_TEMPERATURE: float = 0.2
    
    class Config:
        env_prefix = "SCINETS_"


# Global config instance
config = AgentConfig()
