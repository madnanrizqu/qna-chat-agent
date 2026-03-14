"""Centralized logging configuration for the QnA Chat Agent."""

import contextvars
import logging
import sys
import uuid
from datetime import datetime, timezone

from config import settings


class AppLogger:
    """Application logger with environment-aware configuration."""

    def __init__(self, name: str = "app", log_filter: logging.Filter | None = None):
        """Initialize the logger.

        Args:
            name: Logger name (defaults to 'app').
            log_filter: Optional logging filter to inject (defaults to RequestIDFilter).
        """
        self._logger = logging.getLogger(name)
        self._filter = log_filter if log_filter is not None else RequestIDFilter()
        self._setup()

    def _setup(self) -> None:
        """Configure the logger with handlers and formatters."""
        # Prevent duplicate handlers if already configured
        if self._logger.handlers:
            return

        # Set log level based on environment
        level = (
            logging.DEBUG if settings.environment == "development" else logging.WARNING
        )
        self._logger.setLevel(level)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        handler.addFilter(self._filter)

        formatter = ColoredFormatter(
            "%(asctime)s | [%(request_id)s] | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)

        self._logger.addHandler(handler)

        self._logger.propagate = False

    def debug(self, msg: str, *args, **kwargs) -> None:
        """Log a debug message."""
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        """Log an info message."""
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        """Log a warning message."""
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        """Log an error message."""
        self._logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs) -> None:
        """Log a critical message."""
        self._logger.critical(msg, *args, **kwargs)


class RequestIDContext:
    """Manages request ID in async-safe context."""

    def __init__(self):
        self._context_var: contextvars.ContextVar[str] = contextvars.ContextVar(
            "request_id", default=""
        )

    def get(self) -> str:
        """Retrieve the current request ID from context."""
        return self._context_var.get()

    def set(self, request_id: str) -> None:
        """Store the request ID in context."""
        self._context_var.set(request_id)

    def generate(self) -> str:
        """Generate a new short request ID."""
        return str(uuid.uuid4())[:8]


request_id_context = RequestIDContext()


class RequestIDFilter(logging.Filter):
    """Injects request_id attribute into every log record."""

    def filter(self, record):
        record.request_id = request_id_context.get() or "-"
        return True


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors matching uvicorn's style."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\x1b[36m",  # Cyan
        "INFO": "\x1b[32m",  # Green
        "WARNING": "\x1b[33m",  # Yellow
        "ERROR": "\x1b[31m",  # Red
        "CRITICAL": "\x1b[31;1m",  # Bold Red
    }
    RESET = "\x1b[0m"

    def formatTime(self, record, datefmt=None):
        """Override to use UTC timestamps."""
        ct = datetime.fromtimestamp(record.created, tz=timezone.utc)
        if datefmt:
            s = ct.strftime(datefmt)
        else:
            s = ct.isoformat(timespec="seconds")
        return s

    def format(self, record):
        """Format with colors matching uvicorn."""
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname:<7}{self.RESET}"

        return super().format(record)


class RequestIDMiddleware:
    """Middleware to assign and propagate request IDs."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        """ASGI middleware implementation."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Extract X-Request-ID header if present
        headers = dict(scope.get("headers", []))
        request_id = headers.get(b"x-request-id")

        if request_id:
            request_id = request_id.decode("utf-8")
        else:
            request_id = request_id_context.generate()

        # Store in context
        request_id_context.set(request_id)

        # Inject request ID into response headers
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = message.setdefault("headers", [])
                headers.append((b"x-request-id", request_id.encode("utf-8")))
            await send(message)

        await self.app(scope, receive, send_wrapper)


logger = AppLogger()
