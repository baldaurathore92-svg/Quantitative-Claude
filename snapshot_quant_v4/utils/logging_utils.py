"""Non-blocking logging with credential redaction.

Two production concerns are addressed here.

**Latency.** A ``logging.FileHandler`` performs a synchronous write. Calling it
from the engine thread couples compute latency to disk latency and makes the
per-snapshot budget unpredictable. All handlers are therefore driven by a
:class:`logging.handlers.QueueListener` running on its own thread; the engine
only ever appends to an in-memory queue.

**Secrets.** The configuration carries an API key, a client PIN and a TOTP
secret. A stack trace from the SmartAPI library can easily contain them.
:class:`RedactingFilter` replaces every registered secret with a fixed marker
before the record reaches any handler.
"""

from __future__ import annotations

import logging
import logging.handlers
import queue
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Final

_REDACTION: Final[str] = "***REDACTED***"
_MIN_SECRET_LENGTH: Final[int] = 4
_LOG_FORMAT: Final[str] = (
    "%(asctime)s.%(msecs)03d %(levelname)-8s %(name)-38s %(message)s"
)
_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"


class RedactingFilter(logging.Filter):
    """Remove registered secret values from log records.

    The filter rewrites ``record.msg`` and stringifies ``record.args`` when a
    secret is present. Rewriting is only performed when a secret actually
    occurs in the formatted text, so the common path costs one substring scan
    per secret and no allocation.
    """

    def __init__(self, secrets: Iterable[str] = ()) -> None:
        super().__init__()
        self._secrets: list[str] = []
        for secret in secrets:
            self.add_secret(secret)

    def add_secret(self, secret: str | None) -> None:
        """Register a value that must never appear in the logs."""
        if not secret or len(secret) < _MIN_SECRET_LENGTH:
            return
        if secret not in self._secrets:
            self._secrets.append(secret)

    def filter(self, record: logging.LogRecord) -> bool:
        if not self._secrets:
            return True
        try:
            message = record.getMessage()
        except (TypeError, ValueError):
            # A malformed format string must not prevent logging entirely.
            record.args = None
            return True
        if not any(secret in message for secret in self._secrets):
            return True
        for secret in self._secrets:
            message = message.replace(secret, _REDACTION)
        record.msg = message
        record.args = None
        return True


@dataclass(slots=True)
class LoggingHandle:
    """Owns the logging queue listener so it can be shut down cleanly."""

    listener: logging.handlers.QueueListener
    redactor: RedactingFilter
    log_file: Path | None

    def add_secret(self, secret: str | None) -> None:
        """Register an additional secret after logging has been configured."""
        self.redactor.add_secret(secret)

    def close(self) -> None:
        """Flush and stop the listener thread. Safe to call more than once."""
        self.listener.stop()
        for handler in self.listener.handlers:
            handler.flush()
            if isinstance(handler, logging.FileHandler):
                handler.close()

    def __enter__(self) -> LoggingHandle:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def configure_logging(
    *,
    level: str = "INFO",
    log_file: str | None = None,
    console: bool = False,
    secrets: Iterable[str] = (),
    queue_size: int = 8192,
) -> LoggingHandle:
    """Install a non-blocking, redacting logging pipeline on the root logger.

    Parameters
    ----------
    level:
        Root log level name.
    log_file:
        Optional path for a file handler. Parent directories are created.
    console:
        Whether to also emit to ``stderr``. Defaults to ``False`` because the
        live console renderer owns ``stdout``/the terminal, and interleaved log
        lines would corrupt the display.
    secrets:
        Values to redact.
    queue_size:
        Bound on the in-memory queue. A bounded queue means a stalled disk can
        never grow memory without limit; overflow drops the record and is
        reported once per occurrence by the queue handler.

    Returns
    -------
    LoggingHandle
        Handle whose :meth:`LoggingHandle.close` must be called on shutdown.
    """
    numeric_level = logging.getLevelNamesMapping().get(level.upper())
    if numeric_level is None:
        raise ValueError(f"unknown log level: {level!r}")

    redactor = RedactingFilter(secrets)
    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    handlers: list[logging.Handler] = []
    resolved_path: Path | None = None
    if log_file:
        resolved_path = Path(log_file).expanduser()
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(resolved_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.addFilter(redactor)
        handlers.append(file_handler)
    if console or not handlers:
        stream_handler = logging.StreamHandler(stream=sys.stderr)
        stream_handler.setFormatter(formatter)
        stream_handler.addFilter(redactor)
        handlers.append(stream_handler)

    record_queue: queue.Queue[logging.LogRecord] = queue.Queue(maxsize=queue_size)
    queue_handler = logging.handlers.QueueHandler(record_queue)
    queue_handler.addFilter(redactor)

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(queue_handler)
    root.setLevel(numeric_level)

    listener = logging.handlers.QueueListener(
        record_queue, *handlers, respect_handler_level=True
    )
    listener.start()
    return LoggingHandle(listener=listener, redactor=redactor, log_file=resolved_path)


def get_logger(name: str) -> logging.Logger:
    """Return a module logger.

    A thin wrapper so that call sites never construct logger names ad hoc and
    the hierarchy stays consistent with the package layout.
    """
    return logging.getLogger(name)


__all__ = ["LoggingHandle", "RedactingFilter", "configure_logging", "get_logger"]
