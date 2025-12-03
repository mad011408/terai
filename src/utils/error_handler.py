"""
Error handling utilities for AI Terminal Agent.
"""

from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import traceback
import functools
import asyncio

from .logger import get_logger, Logger


T = TypeVar('T')


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories."""
    VALIDATION = "validation"
    CONFIGURATION = "configuration"
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    TOOL_EXECUTION = "tool_execution"
    AGENT_EXECUTION = "agent_execution"
    MODEL_ERROR = "model_error"
    INTERNAL = "internal"
    USER_INPUT = "user_input"


class AgentError(Exception):
    """Base exception for agent errors."""

    def __init__(self, message: str, code: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 category: ErrorCategory = ErrorCategory.INTERNAL,
                 recoverable: bool = True):
        super().__init__(message)
        self.message = message
        self.code = code or "AGENT_ERROR"
        self.details = details or {}
        self.severity = severity
        self.category = category
        self.recoverable = recoverable
        self.timestamp = datetime.now()
        self.traceback = traceback.format_exc()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "error": self.code,
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "recoverable": self.recoverable,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }


class ValidationError(AgentError):
    """Validation error."""

    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        kwargs.setdefault("code", "VALIDATION_ERROR")
        kwargs.setdefault("category", ErrorCategory.VALIDATION)
        kwargs.setdefault("severity", ErrorSeverity.LOW)
        super().__init__(message, **kwargs)
        self.field = field
        self.details["field"] = field


class ConfigurationError(AgentError):
    """Configuration error."""

    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        kwargs.setdefault("code", "CONFIG_ERROR")
        kwargs.setdefault("category", ErrorCategory.CONFIGURATION)
        kwargs.setdefault("severity", ErrorSeverity.HIGH)
        kwargs.setdefault("recoverable", False)
        super().__init__(message, **kwargs)
        self.config_key = config_key
        self.details["config_key"] = config_key


class NetworkError(AgentError):
    """Network error."""

    def __init__(self, message: str, status_code: Optional[int] = None,
                 url: Optional[str] = None, **kwargs):
        kwargs.setdefault("code", "NETWORK_ERROR")
        kwargs.setdefault("category", ErrorCategory.NETWORK)
        super().__init__(message, **kwargs)
        self.status_code = status_code
        self.url = url
        self.details["status_code"] = status_code
        self.details["url"] = url


class RateLimitError(AgentError):
    """Rate limit error."""

    def __init__(self, message: str, retry_after: Optional[int] = None, **kwargs):
        kwargs.setdefault("code", "RATE_LIMIT_ERROR")
        kwargs.setdefault("category", ErrorCategory.RATE_LIMIT)
        kwargs.setdefault("severity", ErrorSeverity.MEDIUM)
        super().__init__(message, **kwargs)
        self.retry_after = retry_after
        self.details["retry_after"] = retry_after


class TimeoutError(AgentError):
    """Timeout error."""

    def __init__(self, message: str, timeout: Optional[float] = None, **kwargs):
        kwargs.setdefault("code", "TIMEOUT_ERROR")
        kwargs.setdefault("category", ErrorCategory.TIMEOUT)
        super().__init__(message, **kwargs)
        self.timeout = timeout
        self.details["timeout"] = timeout


class ToolError(AgentError):
    """Tool execution error."""

    def __init__(self, message: str, tool_name: Optional[str] = None, **kwargs):
        kwargs.setdefault("code", "TOOL_ERROR")
        kwargs.setdefault("category", ErrorCategory.TOOL_EXECUTION)
        super().__init__(message, **kwargs)
        self.tool_name = tool_name
        self.details["tool_name"] = tool_name


class ModelError(AgentError):
    """Model error."""

    def __init__(self, message: str, model_name: Optional[str] = None, **kwargs):
        kwargs.setdefault("code", "MODEL_ERROR")
        kwargs.setdefault("category", ErrorCategory.MODEL_ERROR)
        super().__init__(message, **kwargs)
        self.model_name = model_name
        self.details["model_name"] = model_name


@dataclass
class ErrorRecord:
    """Record of an error occurrence."""
    error: AgentError
    context: Dict[str, Any] = field(default_factory=dict)
    handled: bool = False
    recovery_attempted: bool = False
    recovery_successful: bool = False


class ErrorHandler:
    """
    Centralized error handling.
    """

    def __init__(self, logger: Optional[Logger] = None):
        self.logger = logger or get_logger("error_handler")
        self.error_history: List[ErrorRecord] = []
        self.handlers: Dict[Type[Exception], Callable] = {}
        self.recovery_strategies: Dict[ErrorCategory, Callable] = {}
        self.max_history = 1000

    def register_handler(self, error_type: Type[Exception],
                        handler: Callable[[Exception], Any]) -> None:
        """Register a custom error handler."""
        self.handlers[error_type] = handler

    def register_recovery(self, category: ErrorCategory,
                         strategy: Callable[[AgentError], bool]) -> None:
        """Register a recovery strategy for error category."""
        self.recovery_strategies[category] = strategy

    def handle(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> ErrorRecord:
        """
        Handle an error.

        Args:
            error: The error to handle
            context: Additional context

        Returns:
            Error record
        """
        context = context or {}

        # Convert to AgentError if needed
        if not isinstance(error, AgentError):
            agent_error = AgentError(
                message=str(error),
                code="UNHANDLED_ERROR",
                details={"original_type": type(error).__name__}
            )
        else:
            agent_error = error

        # Create record
        record = ErrorRecord(error=agent_error, context=context)

        # Log the error
        self._log_error(agent_error, context)

        # Check for custom handler
        for error_type, handler in self.handlers.items():
            if isinstance(error, error_type):
                try:
                    handler(error)
                    record.handled = True
                except Exception as e:
                    self.logger.error(f"Error handler failed: {e}")

        # Attempt recovery
        if agent_error.recoverable:
            record.recovery_attempted = True
            record.recovery_successful = self._attempt_recovery(agent_error)

        # Store in history
        self._store_error(record)

        return record

    def _log_error(self, error: AgentError, context: Dict[str, Any]) -> None:
        """Log an error."""
        log_method = self.logger.error
        if error.severity == ErrorSeverity.CRITICAL:
            log_method = self.logger.critical
        elif error.severity == ErrorSeverity.HIGH:
            log_method = self.logger.error
        elif error.severity == ErrorSeverity.MEDIUM:
            log_method = self.logger.warning
        else:
            log_method = self.logger.info

        log_method(
            f"[{error.code}] {error.message}",
            extra={
                "category": error.category.value,
                "severity": error.severity.value,
                "recoverable": error.recoverable,
                "context": context
            }
        )

    def _attempt_recovery(self, error: AgentError) -> bool:
        """Attempt to recover from error."""
        strategy = self.recovery_strategies.get(error.category)
        if strategy:
            try:
                return strategy(error)
            except Exception as e:
                self.logger.error(f"Recovery strategy failed: {e}")
        return False

    def _store_error(self, record: ErrorRecord) -> None:
        """Store error in history."""
        self.error_history.append(record)
        if len(self.error_history) > self.max_history:
            self.error_history = self.error_history[-self.max_history:]

    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        total = len(self.error_history)
        by_category = {}
        by_severity = {}
        recovery_rate = 0

        for record in self.error_history:
            cat = record.error.category.value
            sev = record.error.severity.value

            by_category[cat] = by_category.get(cat, 0) + 1
            by_severity[sev] = by_severity.get(sev, 0) + 1

        recovery_attempted = sum(1 for r in self.error_history if r.recovery_attempted)
        recovery_successful = sum(1 for r in self.error_history if r.recovery_successful)
        if recovery_attempted > 0:
            recovery_rate = recovery_successful / recovery_attempted

        return {
            "total_errors": total,
            "by_category": by_category,
            "by_severity": by_severity,
            "recovery_attempted": recovery_attempted,
            "recovery_successful": recovery_successful,
            "recovery_rate": recovery_rate
        }

    def clear_history(self) -> None:
        """Clear error history."""
        self.error_history.clear()


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0,
                 max_delay: float = 60.0, exponential_base: float = 2.0,
                 retry_on: Optional[List[Type[Exception]]] = None):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.retry_on = retry_on or [NetworkError, RateLimitError, TimeoutError]

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for attempt."""
        delay = self.base_delay * (self.exponential_base ** attempt)
        return min(delay, self.max_delay)

    def should_retry(self, error: Exception) -> bool:
        """Check if should retry on error."""
        return any(isinstance(error, t) for t in self.retry_on)


def handle_errors(
    error_handler: Optional[ErrorHandler] = None,
    reraise: bool = True,
    default: Any = None
):
    """
    Decorator for error handling.

    Args:
        error_handler: Error handler to use
        reraise: Whether to reraise the error
        default: Default value to return on error
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            handler = error_handler or ErrorHandler()
            try:
                return func(*args, **kwargs)
            except Exception as e:
                handler.handle(e)
                if reraise:
                    raise
                return default

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            handler = error_handler or ErrorHandler()
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                handler.handle(e)
                if reraise:
                    raise
                return default

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator


def retry(config: Optional[RetryConfig] = None):
    """
    Decorator for retrying failed operations.

    Args:
        config: Retry configuration
    """
    config = config or RetryConfig()
    logger = get_logger("retry")

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_error = None
            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if not config.should_retry(e) or attempt == config.max_retries:
                        raise

                    delay = config.get_delay(attempt)
                    logger.warning(
                        f"Retry {attempt + 1}/{config.max_retries} after {delay}s: {e}"
                    )
                    import time
                    time.sleep(delay)

            raise last_error

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            last_error = None
            for attempt in range(config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if not config.should_retry(e) or attempt == config.max_retries:
                        raise

                    delay = config.get_delay(attempt)
                    logger.warning(
                        f"Retry {attempt + 1}/{config.max_retries} after {delay}s: {e}"
                    )
                    await asyncio.sleep(delay)

            raise last_error

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator


class ErrorBoundary:
    """
    Context manager for error boundaries.
    """

    def __init__(self, handler: Optional[ErrorHandler] = None,
                 fallback: Optional[Callable[[], Any]] = None,
                 reraise: bool = False):
        self.handler = handler or ErrorHandler()
        self.fallback = fallback
        self.reraise = reraise
        self.error: Optional[Exception] = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val:
            self.error = exc_val
            self.handler.handle(exc_val)
            if self.fallback:
                self.fallback()
            if self.reraise:
                return False
            return True
        return False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_val:
            self.error = exc_val
            self.handler.handle(exc_val)
            if self.fallback:
                if asyncio.iscoroutinefunction(self.fallback):
                    await self.fallback()
                else:
                    self.fallback()
            if self.reraise:
                return False
            return True
        return False

