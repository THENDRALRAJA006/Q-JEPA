"""Structured logging setup for Surakshit Bharat Q-JEPA."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int = logging.INFO,
               log_file: Optional[Path] = None) -> logging.Logger:
    """Return a configured logger with console (and optional file) handler."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    logger.addHandler(ch)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        logger.addHandler(fh)

    logger.propagate = False
    return logger
