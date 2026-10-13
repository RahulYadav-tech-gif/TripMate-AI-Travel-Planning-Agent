"""Logging setup for TripMate.

Sets up console logging and writes logs to tripmate.log.
"""

import logging
from pathlib import Path
from typing import Optional

from src.config import Settings, get_settings


def setup_logging(settings: Optional[Settings] = None, console_level: Optional[int] = None) -> logging.Logger:
    """Sets up root logging handlers for console and tripmate.log."""
    settings = settings or get_settings()
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers if already set up
    if root_logger.handlers:
        root_logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    log_file_path = Path(settings.log_file)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(str(log_file_path), encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    c_level = console_level if console_level is not None else log_level
    console_handler.setLevel(c_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    logger = logging.getLogger("tripmate")
    logger.info("Logging initialized. Output file: %s", settings.log_file)
    return logger
