import logging
import os
from collections import deque
from logging.handlers import RotatingFileHandler
from typing import List, Optional

# Directory for persistent logs
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "pipeline.log")
MAX_LOG_BYTES = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3

# In-memory circular buffer for Streamlit UI or live display
_LOG_BUFFER: deque = deque(maxlen=1500)


class MemoryLogHandler(logging.Handler):
    """Logging handler that retains recent log entries in an in-memory deque."""

    def __init__(self, buffer: deque):
        super().__init__()
        self.buffer = buffer

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.buffer.append(msg)
        except Exception:
            self.handleError(record)


# Create log directory
os.makedirs(LOG_DIR, exist_ok=True)

# Shared formatter
_FORMATTER = logging.Formatter(
    fmt="[%(asctime)s] [%(levelname)-7s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Root pipeline logger setup
_is_configured = False


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    max_bytes: int = MAX_LOG_BYTES,
    backup_count: int = BACKUP_COUNT,
    enable_console: bool = True,
) -> None:
    """Initialize handlers for console, rotating file, and in-memory log buffer."""
    global _is_configured
    if _is_configured:
        return

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root_logger = logging.getLogger("rag_pipeline")
    root_logger.setLevel(numeric_level)
    root_logger.propagate = False  # Avoid duplicate messages in root logger

    # 1. Console Handler
    if enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(_FORMATTER)
        root_logger.addHandler(console_handler)

    # 2. File Handler (UTF-8 safe for Windows)
    target_file = log_file or LOG_FILE
    target_dir = os.path.dirname(target_file)
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)

    file_handler = RotatingFileHandler(
        filename=target_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(_FORMATTER)
    root_logger.addHandler(file_handler)

    # 3. In-Memory Handler for UI display
    memory_handler = MemoryLogHandler(_LOG_BUFFER)
    memory_handler.setLevel(numeric_level)
    memory_handler.setFormatter(_FORMATTER)
    root_logger.addHandler(memory_handler)

    _is_configured = True


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the 'rag_pipeline' namespace."""
    if not _is_configured:
        setup_logging()
    return logging.getLogger(f"rag_pipeline.{name}")


def get_recent_logs(limit: int = 200) -> List[str]:
    """Retrieve the most recent log lines from the in-memory buffer."""
    items = list(_LOG_BUFFER)
    return items[-limit:] if limit else items


def clear_log_buffer() -> None:
    """Clear the in-memory log buffer."""
    _LOG_BUFFER.clear()
