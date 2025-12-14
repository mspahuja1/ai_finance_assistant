"""Centralized Logging Configuration"""
import logging
import os
from logging.handlers import RotatingFileHandler

LOG_DIR = "/Users/mandeep/myprojects/ai_finance_assistant/logs"
os.makedirs(LOG_DIR, exist_ok=True)

def create_logger(name: str, filename: str):
    """Create a logger with rotating file handler"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    handler = RotatingFileHandler(
        os.path.join(LOG_DIR, filename),
        maxBytes=1_000_000,  # 1 MB
        backupCount=5
    )
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )
    handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(handler)

    return logger