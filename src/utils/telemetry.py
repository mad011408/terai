"""
Telemetry and metrics collection for AI Terminal Agent.
"""

from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from contextlib import contextmanager
import asyncio
import time
import uuid
import json
import functools


class EventType(Enum):
    """Types of telemetry events."""
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    AGENT_ERROR = "agent_error"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    MODEL_REQUEST = "model_request"
    MODEL_RESPONSE = "model_response"
    USER_INPUT = "user_input"
    SYSTEM_OUTPUT = "system_output"
    GUARDRAIL_CHECK = "guardrail_check"
    MEMORY_ACCESS = "memory_access"
    CUSTOM = "custom"


@dataclass
class TelemetryEvent:
    """A telemetry event."""
    event_id: str
    event_type: EventType
    timestamp: datetime
    name: str
    data: Dict[str, Any] = field(default_factory=dict)
    duration_ms: Optional[float] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "name": self.name,
            "data": self.data,
            "duration_ms": self.duration_ms,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "tags": self.tags
        }


@dataclass
class Metric:
    """A metric measurement."""
    name: str
    value: float
    timestamp: datetime
    unit: str = ""
    tags: Dict[str, str] = field(default_factory=dict)


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Span:
    """A trace span."""
    span_id: str
    name: str
    trace_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    parent_span_id: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[TelemetryEvent] = field(default_factory=list)
    status: str = "ok"
    error: Optional[str] = None

    @property
    def duration_ms(self) -> Optional[float]:
        """Get span duration in milliseconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return None

    def add_event(self, name: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Add event to span."""
        event = TelemetryEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.CUSTOM,
            timestamp=datetime.now(),
            name=name,
            data=data or {},
            trace_id=self.trace_id,
            span_id=self.span_id
        )
        self.events.append(event)

    def set_attribute(self, key: str, value: Any) -> None:
        """Set span attribute."""
        self.attributes[key] = value

    def set_error(self, error: Exception) -> None:
        """Set span error."""
        self.status = "error"
        self.error = str(error)

    def end(self) -> None:
        """End the span."""
        self.end_time = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "span_id": self.span_id,
            "name": self.name,
            "trace_id": self.trace_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "parent_span_id": self.parent_span_id,
            "attributes": self.attributes,
            "events": [e.to_dict() for e in self.events],
            "status": self.status,
            "error": self.error
        }


class Tracer:
    """
    Distributed tracing.
    """

    def __init__(self, service_name: str = "ai-terminal-agent"):
        self.service_name = service_name
        self.active_traces: Dict[str, List[Span]] = {}
        self._current_span: Optional[Span] = None

    def start_trace(self) -> str:
        """Start a new trace."""
        trace_id = str(uuid.uuid4())
        self.active_traces[trace_id] = []
        return trace_id

    def start_span(self, name: str, trace_id: Optional[str] = None,
                   parent_span_id: Optional[str] = None) -> Span:
        """Start a new span."""
        if trace_id is None:
            trace_id = self.start_trace()

        span = Span(
            span_id=str(uuid.uuid4()),
            name=name,
            trace_id=trace_id,
            start_time=datetime.now(),
            parent_span_id=parent_span_id or (self._current_span.span_id if self._current_span else None)
        )

        if trace_id in self.active_traces:
            self.active_traces[trace_id].append(span)

        self._current_span = span
        return span

    def end_span(self, span: Span) -> None:
        """End a span."""
        span.end()
        if span == self._current_span:
            # Find parent span
            if span.parent_span_id and span.trace_id in self.active_traces:
                for s in self.active_traces[span.trace_id]:
                    if s.span_id == span.parent_span_id:
                        self._current_span = s
                        return
            self._current_span = None

    def get_current_span(self) -> Optional[Span]:
        """Get current active span."""
        return self._current_span

    def get_trace(self, trace_id: str) -> List[Span]:
        """Get all spans for a trace."""
        return self.active_traces.get(trace_id, [])

    def end_trace(self, trace_id: str) -> List[Span]:
        """End a trace and return all spans."""
        spans = self.active_traces.pop(trace_id, [])
        for span in spans:
            if not span.end_time:
                span.end()
        return spans

    @contextmanager
    def span(self, name: str, trace_id: Optional[str] = None):
        """Context manager for spans."""
        span = self.start_span(name, trace_id)
        try:
            yield span
        except Exception as e:
            span.set_error(e)
            raise
        finally:
            self.end_span(span)


class MetricsCollector:
    """
    Collects and aggregates metrics.
    """

    def __init__(self):
        self.counters: Dict[str, float] = {}
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = {}
        self.timers: Dict[str, List[float]] = {}
        self._history: List[Metric] = []
        self._max_history = 10000

    def increment(self, name: str, value: float = 1.0,
                 tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter."""
        key = self._make_key(name, tags)
        self.counters[key] = self.counters.get(key, 0) + value
        self._record(name, self.counters[key], "count", tags)

    def decrement(self, name: str, value: float = 1.0,
                 tags: Optional[Dict[str, str]] = None) -> None:
        """Decrement a counter."""
        self.increment(name, -value, tags)

    def gauge(self, name: str, value: float,
             tags: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge value."""
        key = self._make_key(name, tags)
        self.gauges[key] = value
        self._record(name, value, "", tags)

    def histogram(self, name: str, value: float,
                 tags: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram value."""
        key = self._make_key(name, tags)
        if key not in self.histograms:
            self.histograms[key] = []
        self.histograms[key].append(value)
        self._record(name, value, "", tags)

    def timer(self, name: str, duration_ms: float,
             tags: Optional[Dict[str, str]] = None) -> None:
        """Record a timer value."""
        key = self._make_key(name, tags)
        if key not in self.timers:
            self.timers[key] = []
        self.timers[key].append(duration_ms)
        self._record(name, duration_ms, "ms", tags)

    @contextmanager
    def time(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for timing operations."""
        start = time.time()
        try:
            yield
        finally:
            duration_ms = (time.time() - start) * 1000
            self.timer(name, duration_ms, tags)

    def _make_key(self, name: str, tags: Optional[Dict[str, str]]) -> str:
        """Make unique key from name and tags."""
        if not tags:
            return name
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}:{tag_str}"

    def _record(self, name: str, value: float, unit: str,
               tags: Optional[Dict[str, str]]) -> None:
        """Record metric to history."""
        metric = Metric(
            name=name,
            value=value,
            timestamp=datetime.now(),
            unit=unit,
            tags=tags or {}
        )
        self._history.append(metric)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    def get_counter(self, name: str, tags: Optional[Dict[str, str]] = None) -> float:
        """Get counter value."""
        key = self._make_key(name, tags)
        return self.counters.get(key, 0)

    def get_gauge(self, name: str, tags: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Get gauge value."""
        key = self._make_key(name, tags)
        return self.gauges.get(key)

    def get_histogram_stats(self, name: str,
                           tags: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """Get histogram statistics."""
        key = self._make_key(name, tags)
        values = self.histograms.get(key, [])
        if not values:
            return {}

        sorted_values = sorted(values)
        n = len(sorted_values)

        return {
            "count": n,
            "min": sorted_values[0],
            "max": sorted_values[-1],
            "mean": sum(values) / n,
            "p50": sorted_values[n // 2],
            "p90": sorted_values[int(n * 0.9)],
            "p99": sorted_values[int(n * 0.99)] if n > 100 else sorted_values[-1]
        }

    def get_timer_stats(self, name: str,
                       tags: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """Get timer statistics."""
        key = self._make_key(name, tags)
        values = self.timers.get(key, [])
        if not values:
            return {}

        return self.get_histogram_stats(name, tags)

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        return {
            "counters": self.counters.copy(),
            "gauges": self.gauges.copy(),
            "histograms": {k: self.get_histogram_stats(k) for k in self.histograms},
            "timers": {k: self.get_timer_stats(k) for k in self.timers}
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self.counters.clear()
        self.gauges.clear()
        self.histograms.clear()
        self.timers.clear()


class Telemetry:
    """
    Main telemetry interface.
    """

    def __init__(self, service_name: str = "ai-terminal-agent",
                 enabled: bool = True,
                 sample_rate: float = 1.0):
        self.service_name = service_name
        self.enabled = enabled
        self.sample_rate = sample_rate
        self.tracer = Tracer(service_name)
        self.metrics = MetricsCollector()
        self.events: List[TelemetryEvent] = []
        self._exporters: List[Callable[[TelemetryEvent], None]] = []
        self._max_events = 10000

    def add_exporter(self, exporter: Callable[[TelemetryEvent], None]) -> None:
        """Add event exporter."""
        self._exporters.append(exporter)

    def record_event(self, event_type: EventType, name: str,
                    data: Optional[Dict[str, Any]] = None,
                    tags: Optional[Dict[str, str]] = None) -> Optional[TelemetryEvent]:
        """Record a telemetry event."""
        if not self.enabled:
            return None

        # Apply sampling
        import random
        if random.random() > self.sample_rate:
            return None

        current_span = self.tracer.get_current_span()

        event = TelemetryEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.now(),
            name=name,
            data=data or {},
            trace_id=current_span.trace_id if current_span else None,
            span_id=current_span.span_id if current_span else None,
            tags=tags or {}
        )

        self.events.append(event)
        if len(self.events) > self._max_events:
            self.events = self.events[-self._max_events:]

        # Export
        for exporter in self._exporters:
            try:
                exporter(event)
            except Exception:
                pass

        return event

    def record_agent_start(self, agent_name: str, task: str) -> Optional[TelemetryEvent]:
        """Record agent start."""
        return self.record_event(
            EventType.AGENT_START,
            f"agent.{agent_name}.start",
            {"task": task}
        )

    def record_agent_end(self, agent_name: str, success: bool,
                        duration_ms: float) -> Optional[TelemetryEvent]:
        """Record agent end."""
        self.metrics.timer(f"agent.{agent_name}.duration", duration_ms)
        self.metrics.increment(f"agent.{agent_name}.total")
        if success:
            self.metrics.increment(f"agent.{agent_name}.success")
        else:
            self.metrics.increment(f"agent.{agent_name}.failure")

        return self.record_event(
            EventType.AGENT_END,
            f"agent.{agent_name}.end",
            {"success": success, "duration_ms": duration_ms}
        )

    def record_tool_call(self, tool_name: str, params: Dict[str, Any]) -> Optional[TelemetryEvent]:
        """Record tool call."""
        self.metrics.increment(f"tool.{tool_name}.calls")
        return self.record_event(
            EventType.TOOL_CALL,
            f"tool.{tool_name}.call",
            {"params": params}
        )

    def record_tool_result(self, tool_name: str, success: bool,
                          duration_ms: float) -> Optional[TelemetryEvent]:
        """Record tool result."""
        self.metrics.timer(f"tool.{tool_name}.duration", duration_ms)
        if not success:
            self.metrics.increment(f"tool.{tool_name}.errors")

        return self.record_event(
            EventType.TOOL_RESULT,
            f"tool.{tool_name}.result",
            {"success": success, "duration_ms": duration_ms}
        )

    def record_model_request(self, model: str, tokens: int) -> Optional[TelemetryEvent]:
        """Record model request."""
        self.metrics.increment(f"model.{model}.requests")
        self.metrics.increment(f"model.{model}.input_tokens", tokens)

        return self.record_event(
            EventType.MODEL_REQUEST,
            f"model.{model}.request",
            {"input_tokens": tokens}
        )

    def record_model_response(self, model: str, tokens: int,
                             duration_ms: float) -> Optional[TelemetryEvent]:
        """Record model response."""
        self.metrics.increment(f"model.{model}.output_tokens", tokens)
        self.metrics.timer(f"model.{model}.latency", duration_ms)

        return self.record_event(
            EventType.MODEL_RESPONSE,
            f"model.{model}.response",
            {"output_tokens": tokens, "duration_ms": duration_ms}
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get telemetry statistics."""
        return {
            "service": self.service_name,
            "enabled": self.enabled,
            "total_events": len(self.events),
            "metrics": self.metrics.get_all_metrics(),
            "active_traces": len(self.tracer.active_traces)
        }


def traced(name: Optional[str] = None):
    """
    Decorator for tracing functions.
    """
    def decorator(func: Callable) -> Callable:
        span_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            telemetry = get_telemetry()
            with telemetry.tracer.span(span_name) as span:
                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("success", True)
                    return result
                except Exception as e:
                    span.set_error(e)
                    raise

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            telemetry = get_telemetry()
            with telemetry.tracer.span(span_name) as span:
                try:
                    result = await func(*args, **kwargs)
                    span.set_attribute("success", True)
                    return result
                except Exception as e:
                    span.set_error(e)
                    raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator


def timed(name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
    """
    Decorator for timing functions.
    """
    def decorator(func: Callable) -> Callable:
        metric_name = name or f"function.{func.__name__}.duration"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            telemetry = get_telemetry()
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                duration_ms = (time.time() - start) * 1000
                telemetry.metrics.timer(metric_name, duration_ms, tags)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            telemetry = get_telemetry()
            start = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                duration_ms = (time.time() - start) * 1000
                telemetry.metrics.timer(metric_name, duration_ms, tags)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator


# Global telemetry instance
_telemetry: Optional[Telemetry] = None


def get_telemetry() -> Telemetry:
    """Get global telemetry instance."""
    global _telemetry
    if _telemetry is None:
        _telemetry = Telemetry()
    return _telemetry


def setup_telemetry(service_name: str = "ai-terminal-agent",
                   enabled: bool = True,
                   sample_rate: float = 1.0) -> Telemetry:
    """Setup global telemetry."""
    global _telemetry
    _telemetry = Telemetry(service_name, enabled, sample_rate)
    return _telemetry

