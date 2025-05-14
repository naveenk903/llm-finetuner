"""Logging configuration for domain-llm-ft."""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

def get_logger(
    name: str,
    level: Optional[str] = None,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """Get a logger with consistent formatting.
    
    Args:
        name: Logger name (usually __name__)
        level: Log level (defaults to env var LOG_LEVEL or INFO)
        log_file: Optional file path to write logs to
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:  # Only add handlers if none exist
        level = level or os.getenv("LOG_LEVEL", "INFO")
        logger.setLevel(getattr(logging, level.upper()))
        
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
    return logger 