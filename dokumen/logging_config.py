"""Centralized logging configuration for Dokumen CLI.

Provides:
- Configurable log levels via CLI flags or environment variables
- Console and file output options
- JSON format support for --output json mode
- Colored console output for human readability
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

# Type for decorated functions
F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class LogConfig:
    """Configuration for logging setup."""

    level: str = "INFO"
    log_file: Optional[Path] = None
    json_format: bool = False
    # Extra context to include in all log messages
    context: dict[str, Any] = field(default_factory=dict)


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging output."""

    def __init__(self, context: Optional[dict[str, Any]] = None) -> None:
        super().__init__()
        self.context = context or {}

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add global context
        log_entry.update(self.context)

        # Add extra fields from record
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for human readability."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, context: Optional[dict[str, Any]] = None) -> None:
        super().__init__()
        self.context = context or {}

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        # Color for level
        color = self.COLORS.get(record.levelname, "")

        # Build extra fields string
        extra_parts = []
        if hasattr(record, "extra_fields") and record.extra_fields:
            for k, v in record.extra_fields.items():
                extra_parts.append(f"{k}={v}")

        extra_str = " " + " ".join(extra_parts) if extra_parts else ""

        # Format message
        message = (
            f"{timestamp} {color}[{record.levelname:7}]{self.RESET} "
            f"{record.getMessage()}{extra_str}"
        )

        # Add exception if present
        if record.exc_info:
            message += "\n" + self.formatException(record.exc_info)

        return message


class ContextLogger(logging.LoggerAdapter):
    """Logger adapter that supports extra fields in log calls."""

    # Reserved kwargs that should not be extracted as extra fields
    RESERVED_KWARGS = frozenset(
        {
            "exc_info",
            "stack_info",
            "stacklevel",
            "extra",
            # Prevent conflicts with LoggerAdapter internals
            "level",
            "msg",
            "args",
        }
    )

    def process(self, msg: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Process log message to include extra fields."""
        extra = kwargs.get("extra", {})

        # Extract extra_fields from kwargs
        extra_fields = {}
        for key in list(kwargs.keys()):
            if key not in self.RESERVED_KWARGS:
                extra_fields[key] = kwargs.pop(key)

        if extra_fields:
            extra["extra_fields"] = extra_fields
            kwargs["extra"] = extra

        return msg, kwargs


# Global config reference
_config: Optional[LogConfig] = None


def setup_logging(config: LogConfig) -> None:
    """Initialize logging with the given configuration.

    Args:
        config: LogConfig instance with logging settings
    """
    global _config
    _config = config

    # Convert string level to logging constant
    numeric_level = getattr(logging, config.level.upper(), logging.INFO)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(numeric_level)

    # Choose formatter based on config
    if config.json_format:
        formatter = JsonFormatter(config.context)
    else:
        formatter = ColoredFormatter(config.context)

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Add file handler if specified
    if config.log_file:
        file_handler = logging.FileHandler(config.log_file)
        file_handler.setLevel(numeric_level)
        # Always use JSON format for file output
        file_handler.setFormatter(JsonFormatter(config.context))
        root_logger.addHandler(file_handler)

    # Set levels for noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)

    # Log initialization
    logger = get_logger(__name__)
    logger.debug(
        "logging.init",
        log_level=config.level,
        log_json_format=config.json_format,
        log_file=str(config.log_file) if config.log_file else None,
    )


def setup_logging_from_env() -> None:
    """Initialize logging from environment variables.

    Environment variables:
        DOKUMEN_LOG_LEVEL: Log level (DEBUG, INFO, WARNING, ERROR)
        DOKUMEN_LOG_FILE: Path to log file
        DOKUMEN_LOG_JSON: Set to "true" for JSON format
    """
    config = LogConfig(
        level=os.environ.get("DOKUMEN_LOG_LEVEL", "INFO"),
        log_file=(
            Path(os.environ["DOKUMEN_LOG_FILE"]) if os.environ.get("DOKUMEN_LOG_FILE") else None
        ),
        json_format=os.environ.get("DOKUMEN_LOG_JSON", "").lower() == "true",
    )
    setup_logging(config)


def get_logger(name: str) -> ContextLogger:
    """Get a logger with context support.

    Args:
        name: Logger name (typically __name__)

    Returns:
        ContextLogger with extra field support
    """
    return ContextLogger(logging.getLogger(name), {})


def log_timing(logger: ContextLogger, event: str) -> Callable[[F], F]:
    """Decorator to log function execution time.

    Args:
        logger: Logger to use for timing logs
        event: Event name prefix for the log message

    Example:
        @log_timing(logger, "test.execute")
        def run_test():
            ...
    """
    import time

    def decorator(func: F) -> F:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start) * 1000
                logger.debug(f"{event}.complete", duration_ms=round(duration_ms, 2))
                return result
            except Exception as e:
                duration_ms = (time.perf_counter() - start) * 1000
                logger.error(f"{event}.error", duration_ms=round(duration_ms, 2), error=str(e))
                raise

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start) * 1000
                logger.debug(f"{event}.complete", duration_ms=round(duration_ms, 2))
                return result
            except Exception as e:
                duration_ms = (time.perf_counter() - start) * 1000
                logger.error(f"{event}.error", duration_ms=round(duration_ms, 2), error=str(e))
                raise

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator
