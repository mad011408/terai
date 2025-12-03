"""
Logging utilities for AI Terminal Agent.
"""

from typing import Any, Dict, Optional, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import sys
import json
import os
import traceback
from pathlib import Path


class LogLevel(Enum):
    """Log levels."""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


@dataclass
class LogRecord:
    """A log record."""
    timestamp: datetime
    level: LogLevel
    message: str
    logger_name: str
    module: Optional[str] = None
    function: Optional[str] = None
    line: Optional[int] = None
    extra: Optional[Dict[str, Any]] = None
    exception: Optional[str] = None


class LogFormatter:
    """
    Formats log records.
    """

    def __init__(self, format_string: Optional[str] = None,
                 use_colors: bool = True,
                 json_format: bool = False):
        self.format_string = format_string or "[{timestamp}] {level} [{name}] {message}"
        self.use_colors = use_colors
        self.json_format = json_format

        self.colors = {
            LogLevel.DEBUG: "\033[36m",    # Cyan
            LogLevel.INFO: "\033[32m",     # Green
            LogLevel.WARNING: "\033[33m",  # Yellow
            LogLevel.ERROR: "\033[31m",    # Red
            LogLevel.CRITICAL: "\033[35m", # Magenta
        }
        self.reset = "\033[0m"

    def format(self, record: LogRecord) -> str:
        """Format a log record."""
        if self.json_format:
            return self._format_json(record)
        return self._format_text(record)

    def _format_text(self, record: LogRecord) -> str:
        """Format as text."""
        timestamp = record.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        level = record.level.name.ljust(8)

        if self.use_colors:
            color = self.colors.get(record.level, "")
            level = f"{color}{level}{self.reset}"

        message = self.format_string.format(
            timestamp=timestamp,
            level=level,
            name=record.logger_name,
            message=record.message,
            module=record.module or "",
            function=record.function or "",
            line=record.line or ""
        )

        if record.extra:
            extra_str = " ".join(f"{k}={v}" for k, v in record.extra.items())
            message += f" [{extra_str}]"

        if record.exception:
            message += f"\n{record.exception}"

        return message

    def _format_json(self, record: LogRecord) -> str:
        """Format as JSON."""
        data = {
            "timestamp": record.timestamp.isoformat(),
            "level": record.level.name,
            "logger": record.logger_name,
            "message": record.message,
        }

        if record.module:
            data["module"] = record.module
        if record.function:
            data["function"] = record.function
        if record.line:
            data["line"] = record.line
        if record.extra:
            data["extra"] = record.extra
        if record.exception:
            data["exception"] = record.exception

        return json.dumps(data)


class LogHandler:
    """
    Base log handler.
    """

    def __init__(self, level: LogLevel = LogLevel.DEBUG,
                 formatter: Optional[LogFormatter] = None):
        self.level = level
        self.formatter = formatter or LogFormatter()

    def should_handle(self, record: LogRecord) -> bool:
        """Check if handler should process record."""
        return record.level.value >= self.level.value

    def handle(self, record: LogRecord) -> None:
        """Handle a log record."""
        if self.should_handle(record):
            self.emit(record)

    def emit(self, record: LogRecord) -> None:
        """Emit a log record. Override in subclasses."""
        raise NotImplementedError


class ConsoleHandler(LogHandler):
    """
    Handler that outputs to console.
    """

    def __init__(self, stream=None, **kwargs):
        super().__init__(**kwargs)
        self.stream = stream or sys.stderr

    def emit(self, record: LogRecord) -> None:
        """Emit to console."""
        try:
            message = self.formatter.format(record)
            self.stream.write(message + "\n")
            self.stream.flush()
        except Exception:
            pass


class FileHandler(LogHandler):
    """
    Handler that outputs to file.
    """

    def __init__(self, filename: str, mode: str = "a",
                 max_bytes: int = 10 * 1024 * 1024,
                 backup_count: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.filename = Path(filename)
        self.mode = mode
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self._file = None

        # Ensure directory exists
        self.filename.parent.mkdir(parents=True, exist_ok=True)

    def _open(self):
        """Open the file."""
        if self._file is None or self._file.closed:
            self._file = open(self.filename, self.mode, encoding='utf-8')
        return self._file

    def _close(self):
        """Close the file."""
        if self._file and not self._file.closed:
            self._file.close()

    def _should_rotate(self) -> bool:
        """Check if rotation is needed."""
        if not self.filename.exists():
            return False
        return self.filename.stat().st_size >= self.max_bytes

    def _rotate(self) -> None:
        """Rotate log files."""
        self._close()

        # Rotate existing backups
        for i in range(self.backup_count - 1, 0, -1):
            src = Path(f"{self.filename}.{i}")
            dst = Path(f"{self.filename}.{i + 1}")
            if src.exists():
                src.rename(dst)

        # Rename current file
        if self.filename.exists():
            self.filename.rename(Path(f"{self.filename}.1"))

    def emit(self, record: LogRecord) -> None:
        """Emit to file."""
        try:
            if self._should_rotate():
                self._rotate()

            f = self._open()
            message = self.formatter.format(record)
            f.write(message + "\n")
            f.flush()
        except Exception:
            pass


class Logger:
    """
    Main logger class.
    """

    _loggers: Dict[str, "Logger"] = {}

    def __init__(self, name: str, level: LogLevel = LogLevel.DEBUG):
        self.name = name
        self.level = level
        self.handlers: list[LogHandler] = []
        self.parent: Optional[Logger] = None
        self.propagate = True

    @classmethod
    def get_logger(cls, name: str) -> "Logger":
        """Get or create a logger."""
        if name not in cls._loggers:
            cls._loggers[name] = Logger(name)
        return cls._loggers[name]

    def add_handler(self, handler: LogHandler) -> None:
        """Add a handler."""
        if handler not in self.handlers:
            self.handlers.append(handler)

    def remove_handler(self, handler: LogHandler) -> None:
        """Remove a handler."""
        if handler in self.handlers:
            self.handlers.remove(handler)

    def set_level(self, level: LogLevel) -> None:
        """Set logging level."""
        self.level = level

    def _log(self, level: LogLevel, message: str,
             extra: Optional[Dict[str, Any]] = None,
             exc_info: bool = False) -> None:
        """Internal log method."""
        if level.value < self.level.value:
            return

        # Get caller info
        frame = sys._getframe(2)
        module = frame.f_globals.get("__name__")
        function = frame.f_code.co_name
        line = frame.f_lineno

        exception = None
        if exc_info:
            exception = traceback.format_exc()

        record = LogRecord(
            timestamp=datetime.now(),
            level=level,
            message=message,
            logger_name=self.name,
            module=module,
            function=function,
            line=line,
            extra=extra,
            exception=exception
        )

        # Handle with all handlers
        for handler in self.handlers:
            handler.handle(record)

        # Propagate to parent
        if self.propagate and self.parent:
            self.parent._handle(record)

    def _handle(self, record: LogRecord) -> None:
        """Handle a record from child logger."""
        for handler in self.handlers:
            handler.handle(record)

        if self.propagate and self.parent:
            self.parent._handle(record)

    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self._log(LogLevel.DEBUG, message, kwargs.get("extra"), kwargs.get("exc_info", False))

    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self._log(LogLevel.INFO, message, kwargs.get("extra"), kwargs.get("exc_info", False))

    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self._log(LogLevel.WARNING, message, kwargs.get("extra"), kwargs.get("exc_info", False))

    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self._log(LogLevel.ERROR, message, kwargs.get("extra"), kwargs.get("exc_info", False))

    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        self._log(LogLevel.CRITICAL, message, kwargs.get("extra"), kwargs.get("exc_info", False))

    def exception(self, message: str, **kwargs) -> None:
        """Log exception with traceback."""
        self._log(LogLevel.ERROR, message, kwargs.get("extra"), exc_info=True)


class ContextLogger:
    """
    Logger with context information.
    """

    def __init__(self, logger: Logger, context: Dict[str, Any]):
        self.logger = logger
        self.context = context

    def _merge_extra(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Merge context with extra."""
        merged = self.context.copy()
        if extra:
            merged.update(extra)
        return merged

    def debug(self, message: str, **kwargs) -> None:
        kwargs["extra"] = self._merge_extra(kwargs.get("extra"))
        self.logger.debug(message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        kwargs["extra"] = self._merge_extra(kwargs.get("extra"))
        self.logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        kwargs["extra"] = self._merge_extra(kwargs.get("extra"))
        self.logger.warning(message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        kwargs["extra"] = self._merge_extra(kwargs.get("extra"))
        self.logger.error(message, **kwargs)

    def critical(self, message: str, **kwargs) -> None:
        kwargs["extra"] = self._merge_extra(kwargs.get("extra"))
        self.logger.critical(message, **kwargs)


# Global logger instance
_root_logger: Optional[Logger] = None


def get_logger(name: str = "ai_terminal_agent") -> Logger:
    """Get a logger instance."""
    return Logger.get_logger(name)


def setup_logging(
    level: Union[str, LogLevel] = LogLevel.INFO,
    log_file: Optional[str] = None,
    json_format: bool = False,
    use_colors: bool = True
) -> Logger:
    """
    Setup logging configuration.

    Args:
        level: Log level (string or LogLevel)
        log_file: Optional file to log to
        json_format: Use JSON format
        use_colors: Use colors in console output

    Returns:
        Root logger
    """
    global _root_logger

    if isinstance(level, str):
        level = LogLevel[level.upper()]

    _root_logger = Logger.get_logger("ai_terminal_agent")
    _root_logger.set_level(level)

    # Console handler
    console_formatter = LogFormatter(use_colors=use_colors, json_format=json_format)
    console_handler = ConsoleHandler(formatter=console_formatter, level=level)
    _root_logger.add_handler(console_handler)

    # File handler if specified
    if log_file:
        file_formatter = LogFormatter(use_colors=False, json_format=json_format)
        file_handler = FileHandler(log_file, formatter=file_formatter, level=level)
        _root_logger.add_handler(file_handler)

    return _root_logger


class LoggerAdapter:
    """
    Adapter for using with async contexts.
    """

    def __init__(self, logger: Logger):
        self.logger = logger
        self._context: Dict[str, Any] = {}

    def bind(self, **kwargs) -> "LoggerAdapter":
        """Bind context to logger."""
        new_adapter = LoggerAdapter(self.logger)
        new_adapter._context = {**self._context, **kwargs}
        return new_adapter

    def unbind(self, *keys) -> "LoggerAdapter":
        """Unbind context keys."""
        new_adapter = LoggerAdapter(self.logger)
        new_adapter._context = {k: v for k, v in self._context.items() if k not in keys}
        return new_adapter

    def _log(self, method: str, message: str, **kwargs) -> None:
        extra = kwargs.pop("extra", {})
        extra.update(self._context)
        getattr(self.logger, method)(message, extra=extra, **kwargs)

    def debug(self, message: str, **kwargs) -> None:
        self._log("debug", message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        self._log("info", message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        self._log("warning", message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        self._log("error", message, **kwargs)

    def critical(self, message: str, **kwargs) -> None:
        self._log("critical", message, **kwargs)

