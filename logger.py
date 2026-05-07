"""
Structured logging configuration for Finora backend.
Outputs JSON in production, human-readable in development.
"""
import logging
import sys
import json
from datetime import datetime, timezone
from config import LOG_LEVEL, ENVIRONMENT


class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter for production / Cloud Run."""
    def format(self, record):
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "request_id"):
            log_entry["request_id"] = record.request_id
        if hasattr(record, "user_id"):
            log_entry["user_id"] = record.user_id
        return json.dumps(log_entry)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        if ENVIRONMENT == "production":
            handler.setFormatter(JSONFormatter())
        else:
            handler.setFormatter(logging.Formatter(
                "%(asctime)s │ %(levelname)-7s │ %(name)-20s │ %(message)s",
                datefmt="%H:%M:%S"
            ))
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    return logger
