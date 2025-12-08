"""
Structured logging configuration for SciNets V2.

This module sets up structlog for beautiful, searchable logs with:
- Colored console output for development
- JSON output for production
- Automatic request ID tracking
- Stack trace context for errors
- Timestamp and log level formatting

Usage:
    from app.logging_config import get_logger
    
    log = get_logger(__name__)
    log.info("hypothesis_generated", hypothesis_id=h.id, novelty_score=h.novelty_score)
"""

import logging
import sys
import os
from typing import Any

import structlog
from structlog.types import EventDict, Processor


def add_app_context(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Add application context to log entries."""
    # You can add global context here, like version, environment, etc.
    event_dict["app"] = "scinets-v2"
    return event_dict


def setup_logging(
    log_level: str = "INFO",
    json_logs: bool = False,
    include_timestamp: bool = True,
    log_file: str | None = None
) -> None:
    """
    Configure structlog for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_logs: If True, output JSON logs (for production). If False, use colored console output.
        include_timestamp: Include timestamp in logs
        log_file: Path to log file. Defaults to LOG_FILE env var if set.
    """
    if log_file is None:
        log_file = os.environ.get("LOG_FILE")

    # Configure timestamper
    timestamper = structlog.processors.TimeStamper(fmt="iso")
    
    # Shared processors for both structlog and stdlib logging
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        add_app_context,
    ]
    
    if include_timestamp:
        shared_processors.append(timestamper)
    
    # Add exception formatting
    shared_processors.append(structlog.processors.StackInfoRenderer())
    shared_processors.append(structlog.processors.format_exc_info)
    
    if json_logs:
        # Production: JSON output
        shared_processors.append(structlog.processors.dict_tracebacks)
        renderer = structlog.processors.JSONRenderer()
    else:
        # Development: Colored console output
        renderer = structlog.dev.ConsoleRenderer(
            colors=True,
            exception_formatter=structlog.dev.RichTracebackFormatter(
                show_locals=True,
                max_frames=10,
            )
        )
    
    # Configure structlog
    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure stdlib logging to use structlog
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )
    
    root_logger = logging.getLogger()
    # Clear existing handlers to prevent duplicates
    if root_logger.handlers:
        root_logger.handlers.clear()
        
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File Handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    root_logger.setLevel(log_level.upper())
    
    # Silence noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured structlog logger
        
    Example:
        >>> log = get_logger(__name__)
        >>> log.info("user_action", user_id=123, action="login")
        >>> log.error("processing_failed", error=str(e), item_id=456)
    """
    return structlog.get_logger(name)
