import logging
import os
import sys
from logging.handlers import RotatingFileHandler

from src.config import DEFAULT_LOG_FILE, DEFAULT_LOG_LEVEL, LOG_DIR

_CONFIGURED = False

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: str | None = None,
    log_file: str | os.PathLike | None = None,
    enable_file: bool = True,
) -> None:
    """Configure root logging for console and optional rotating file output."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    resolved_level = getattr(logging, (level or os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL)).upper(), logging.INFO)
    resolved_file = log_file or os.getenv("LOG_FILE", str(DEFAULT_LOG_FILE))

    root = logging.getLogger()
    root.setLevel(resolved_level)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    if enable_file and os.getenv("LOG_TO_FILE", "true").lower() in ("1", "true", "yes"):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            resolved_file,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Keep uvicorn access logs visible at INFO without duplicating app logs.
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)

    _CONFIGURED = True
    logging.getLogger(__name__).info(
        "Logging configured | level=%s | file=%s",
        logging.getLevelName(resolved_level),
        resolved_file if enable_file else "disabled",
    )
