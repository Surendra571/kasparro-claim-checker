"""
Logging configuration for the service.
"""

import logging
import sys

from app.core.config import settings


def setup_logging() -> None:
    """Configures structured stream logging."""
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d - %(message)s"
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
