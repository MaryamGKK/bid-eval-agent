# services/observer.py
"""Observability service using OpenTelemetry with comprehensive tracing and error analysis."""

import time
import traceback
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from contextlib import contextmanager
from enum import Enum

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import Status, StatusCode, SpanKind

from models import RAGTrace
from config import RAG_CONFIG

# Initialize OpenTelemetry
resource = Resource.create({
    "service.name": "bid-eval-agent",
    "service.version": "1.0.0",
    "service.namespace": "bid-evaluation",
    "deployment.environment": "development"
})
provider = TracerProvider(resource=resource)

# Use console exporter for POC (can switch to OTLP for production)
processor = BatchSpanProcessor(ConsoleSpanExporter())
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

# Get tracer
tracer = trace.get_tracer("bid-eval-agent", "1.0.0")


class EventStatus(Enum):
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    PENDING = "pending"


@dataclass
class ErrorInfo:
    """Detailed error information."""
    error_type: str
    message: str
    stack_trace: str
    timestamp: str
    span_id: Optional[str] = None
    trace_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpanInfo:
    """Comprehensive span information."""
    span_id: str
    trace_id: str
    parent_span_id: Optional[str]
    name: str
    kind: str
    status: str
    status_code: int
    start_time: str
    end_time: Optional[str]
    duration_ms: Optional[float]
    attributes: Dict[str, Any]
    events: List[Dict[str, Any]]
    links: List[str]
    resource: Dict[str, Any]


@dataclass
class Event:
    """Single observable event with full trace context."""
    timestamp: str
    event_type: str
    node: str
    status: str
    data: Dict[str, Any]
    duration_ms: Optional[float] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    span_kind: Optional[str] = None
    error: Optional[ErrorInfo] = None
    attributes: Dict[str, Any] = field(default_factory=dict)


class Observer:
    """Track and log evaluation events with comprehensive OpenTelemetry tracing."""
    
    def __init__(self):
        self.events: List[Event] = []
        self.errors: List[ErrorInfo] = []
        self.spans: List[SpanInfo] = []
        self._timers: Dict[str, float] = {}
        self._active_spans: Dict[str, Any] = {}
        self.tracer = tracer
        self._root_span = None
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        
        # Metrics
        self._metrics = {
            "total_events": 0,
            "successful_events": 0,
            "failed_events": 0,
            "warnings": 0,
            "total_latency_ms": 0,
            "latencies": [],
            "errors_by_type": {},
            "events_by_type": {}
        }
    
    @contextmanager
    def span(self, name: str, attributes: Dict[str, Any] = None, kind: SpanKind = SpanKind.INTERNAL):
        """Create a span context manager with comprehensive tracing."""
        self._start_time = self._start_time or time.perf_counter()
        
        with self.tracer.start_as_current_span(
            name, 
            kind=kind,
            attributes=self._sanitize_attributes(attributes or {})
        ) as span:
            span_context = span.get_span_context()
            trace_id = format(span_context.trace_id, '032x')
            span_id = format(span_context.span_id, '016x')
            
            # Get parent span ID
            parent_span_id = None
            parent_context = trace.get_current_span().get_span_context()
            if parent_context and parent_context.span_id != span_context.span_id:
                parent_span_id = format(parent_context.span_id, '016x')
            
            start_time = time.perf_counter()
            
            # Record span start
            span_info = SpanInfo(
                span_id=span_id,
                trace_id=trace_id,
                parent_span_id=parent_span_id,
                name=name,
                kind=str(kind),
                status="UNSET",
                status_code=0,
                start_time=datetime.now().isoformat(),
                end_time=None,
                duration_ms=None,
                attributes=attributes or {},
                events=[],
                links=[],
                resource=dict(resource.attributes)
            )
            
            try:
                yield span
                span.set_status(Status(StatusCode.OK))
                span_info.status = "OK"
                span_info.status_code = StatusCode.OK.value
                
            except Exception as e:
                # Record error
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                
                error_info = ErrorInfo(
                    error_type=type(e).__name__,
                    message=str(e),
                    stack_trace=traceback.format_exc(),
                    timestamp=datetime.now().isoformat(),
                    span_id=span_id,
                    trace_id=trace_id,
                    context={"span_name": name, "attributes": attributes}
                )
                self.errors.append(error_info)
                self._metrics["failed_events"] += 1
                self._metrics["errors_by_type"][type(e).__name__] = \
                    self._metrics["errors_by_type"].get(type(e).__name__, 0) + 1
                
                span_info.status = "ERROR"
                span_info.status_code = StatusCode.ERROR.value
                span_info.events.append({
                    "name": "exception",
                    "timestamp": datetime.now().isoformat(),
                    "attributes": {
                        "exception.type": type(e).__name__,
                        "exception.message": str(e),
                        "exception.stacktrace": traceback.format_exc()
                    }
                })
                
                raise
            finally:
                end_time = time.perf_counter()
                duration_ms = (end_time - start_time) * 1000
                
                span_info.end_time = datetime.now().isoformat()
                span_info.duration_ms = round(duration_ms, 2)
                self.spans.append(span_info)
                
                self._metrics["latencies"].append(duration_ms)
                self._metrics["total_latency_ms"] += duration_ms
    
    def _sanitize_attributes(self, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize attributes for OpenTelemetry (only primitive types allowed)."""
        sanitized = {}
        for key, value in attributes.items():
            if isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, (list, tuple)):
                # Convert list to string representation
                sanitized[key] = str(value)
            else:
                sanitized[key] = str(value)
        return sanitized
    
    def log(self, event_type: str, node: str, data: Dict[str, Any] = None, 
            status: str = "success", error: Exception = None):
        """Log an event with full trace context and optional error."""
        current_span = trace.get_current_span()
        trace_id = None
        span_id = None
        parent_span_id = None
        span_kind = None
        
        if current_span and current_span.is_recording():
            ctx = current_span.get_span_context()
            trace_id = format(ctx.trace_id, '032x')
            span_id = format(ctx.span_id, '016x')
            span_kind = str(current_span.kind) if hasattr(current_span, 'kind') else "INTERNAL"
            
            # Add as span event
            event_attrs = {"node": node, "status": status}
            if data:
                for k, v in data.items():
                    if isinstance(v, (str, int, float, bool)):
                        event_attrs[k] = v
                    else:
                        event_attrs[k] = str(v)
            current_span.add_event(event_type, attributes=event_attrs)
        
        # Create error info if error provided
        error_info = None
        if error:
            error_info = ErrorInfo(
                error_type=type(error).__name__,
                message=str(error),
                stack_trace=traceback.format_exc(),
                timestamp=datetime.now().isoformat(),
                span_id=span_id,
                trace_id=trace_id,
                context=data or {}
            )
            self.errors.append(error_info)
            status = "error"
        
        event = Event(
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            node=node,
            status=status,
            data=data or {},
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            span_kind=span_kind,
            error=error_info,
            attributes=self._sanitize_attributes(data or {})
        )
        self.events.append(event)
        
        # Update metrics
        self._metrics["total_events"] += 1
        self._metrics["events_by_type"][event_type] = \
            self._metrics["events_by_type"].get(event_type, 0) + 1
        
        if status == "success":
            self._metrics["successful_events"] += 1
        elif status == "error":
            self._metrics["failed_events"] += 1
        elif status == "warning":
            self._metrics["warnings"] += 1
    
    def log_error(self, event_type: str, node: str, error: Exception, data: Dict[str, Any] = None):
        """Log an error event with full stack trace."""
        self.log(event_type, node, data, status="error", error=error)
    
    def start_timer(self, key: str):
        """Start a timer for duration tracking."""
        self._timers[key] = time.perf_counter()
    
    def stop_timer(self, key: str, event_type: str, node: str, data: Dict[str, Any] = None,
                   status: str = "success"):
        """Stop timer and log event with duration."""
        start = self._timers.pop(key, None)
        duration_ms = None
        if start:
            duration_ms = (time.perf_counter() - start) * 1000
        
        current_span = trace.get_current_span()
        trace_id = None
        span_id = None
        span_kind = None
        
        if current_span and current_span.is_recording():
            ctx = current_span.get_span_context()
            trace_id = format(ctx.trace_id, '032x')
            span_id = format(ctx.span_id, '016x')
            span_kind = str(current_span.kind) if hasattr(current_span, 'kind') else "INTERNAL"
            
            if duration_ms:
                current_span.set_attribute(f"{event_type}.duration_ms", duration_ms)
        
        event = Event(
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            node=node,
            status=status,
            data=data or {},
            duration_ms=round(duration_ms, 2) if duration_ms else None,
            trace_id=trace_id,
            span_id=span_id,
            span_kind=span_kind
        )
        self.events.append(event)
        
        # Update metrics
        self._metrics["total_events"] += 1
        if duration_ms:
            self._metrics["latencies"].append(duration_ms)
            self._metrics["total_latency_ms"] += duration_ms
        if status == "success":
            self._metrics["successful_events"] += 1
    
    def get_events(self, event_type: str = None) -> List[Dict]:
        """Get all events with full trace context."""
        events = self.events
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        return [
            {
                "timestamp": e.timestamp,
                "type": e.event_type,
                "node": e.node,
                "status": e.status,
                "data": e.data,
                **({"duration_ms": e.duration_ms} if e.duration_ms else {}),
                **({"trace_id": e.trace_id} if e.trace_id else {}),
                **({"span_id": e.span_id} if e.span_id else {}),
                **({"parent_span_id": e.parent_span_id} if e.parent_span_id else {}),
                **({"span_kind": e.span_kind} if e.span_kind else {}),
                **({"error": {
                    "type": e.error.error_type,
                    "message": e.error.message,
                    "stack_trace": e.error.stack_trace
                }} if e.error else {}),
                "attributes": e.attributes
            }
            for e in events
        ]
    
    def get_errors(self) -> List[Dict]:
        """Get all recorded errors."""
        return [
            {
                "error_type": e.error_type,
                "message": e.message,
                "stack_trace": e.stack_trace,
                "timestamp": e.timestamp,
                "trace_id": e.trace_id,
                "span_id": e.span_id,
                "context": e.context
            }
            for e in self.errors
        ]
    
    def get_spans(self) -> List[Dict]:
        """Get all recorded spans with full details."""
        return [
            {
                "span_id": s.span_id,
                "trace_id": s.trace_id,
                "parent_span_id": s.parent_span_id,
                "name": s.name,
                "kind": s.kind,
                "status": s.status,
                "status_code": s.status_code,
                "start_time": s.start_time,
                "end_time": s.end_time,
                "duration_ms": s.duration_ms,
                "attributes": s.attributes,
                "events": s.events,
                "resource": s.resource
            }
            for s in self.spans
        ]
    
    def get_rag_trace(self) -> RAGTrace:
        """Get RAG trace configuration."""
        return RAGTrace(
            vector_store=RAG_CONFIG["vector_store"],
            embedding_model=RAG_CONFIG["embedding_model"],
            documents_retrieved_per_company=RAG_CONFIG["documents_retrieved_per_company"],
            retrieval_confidence_threshold=RAG_CONFIG["retrieval_confidence_threshold"]
        )
    
    def get_error_analysis(self) -> Dict[str, Any]:
        """Get comprehensive error analysis."""
        if not self.errors:
            return {
                "total_errors": 0,
                "error_rate": 0,
                "errors_by_type": {},
                "errors_by_span": {},
                "recent_errors": [],
                "root_causes": []
            }
        
        # Group errors by type
        errors_by_type = {}
        errors_by_span = {}
        for error in self.errors:
            errors_by_type[error.error_type] = errors_by_type.get(error.error_type, 0) + 1
            span_key = error.context.get("span_name", "unknown")
            errors_by_span[span_key] = errors_by_span.get(span_key, 0) + 1
        
        # Calculate error rate
        total_events = self._metrics["total_events"]
        error_rate = (len(self.errors) / total_events * 100) if total_events > 0 else 0
        
        # Identify potential root causes
        root_causes = []
        for error in self.errors:
            if "API" in error.error_type or "Connection" in error.error_type:
                root_causes.append(f"External service failure: {error.error_type}")
            elif "Timeout" in error.error_type:
                root_causes.append(f"Operation timeout in {error.context.get('span_name', 'unknown')}")
            elif "Validation" in error.error_type or "Value" in error.error_type:
                root_causes.append(f"Data validation issue: {error.message[:100]}")
        
        return {
            "total_errors": len(self.errors),
            "error_rate": round(error_rate, 2),
            "errors_by_type": errors_by_type,
            "errors_by_span": errors_by_span,
            "recent_errors": [
                {
                    "type": e.error_type,
                    "message": e.message[:200],
                    "timestamp": e.timestamp,
                    "span": e.context.get("span_name", "unknown")
                }
                for e in self.errors[-5:]  # Last 5 errors
            ],
            "root_causes": list(set(root_causes))
        }
    
    def get_latency_analysis(self) -> Dict[str, Any]:
        """Get latency analysis with percentiles."""
        latencies = self._metrics["latencies"]
        
        if not latencies:
            return {
                "count": 0,
                "total_ms": 0,
                "avg_ms": 0,
                "min_ms": 0,
                "max_ms": 0,
                "p50_ms": 0,
                "p90_ms": 0,
                "p99_ms": 0
            }
        
        sorted_latencies = sorted(latencies)
        count = len(sorted_latencies)
        
        def percentile(data, p):
            idx = int(len(data) * p / 100)
            return data[min(idx, len(data) - 1)]
        
        return {
            "count": count,
            "total_ms": round(sum(latencies), 2),
            "avg_ms": round(sum(latencies) / count, 2),
            "min_ms": round(min(latencies), 2),
            "max_ms": round(max(latencies), 2),
            "p50_ms": round(percentile(sorted_latencies, 50), 2),
            "p90_ms": round(percentile(sorted_latencies, 90), 2),
            "p99_ms": round(percentile(sorted_latencies, 99), 2)
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive observability summary."""
        search_events = [e for e in self.events if e.event_type == "search"]
        score_events = [e for e in self.events if e.event_type == "score"]
        llm_events = [e for e in self.events if e.event_type == "llm"]
        
        # Get unique trace IDs
        trace_ids = list(set(e.trace_id for e in self.events if e.trace_id))
        
        return {
            "total_events": self._metrics["total_events"],
            "successful_events": self._metrics["successful_events"],
            "failed_events": self._metrics["failed_events"],
            "warnings": self._metrics["warnings"],
            "search_count": len(search_events),
            "score_count": len(score_events),
            "llm_calls": len(llm_events),
            "total_duration_ms": round(self._metrics["total_latency_ms"], 2),
            "trace_ids": trace_ids,
            "spans_count": len(self.spans),
            "errors_count": len(self.errors),
            "events_by_type": self._metrics["events_by_type"],
            "latency_analysis": self.get_latency_analysis(),
            "error_analysis": self.get_error_analysis(),
            "rag_trace": self.get_rag_trace().to_dict()
        }
    
    def clear(self):
        """Clear all events and reset metrics."""
        self.events = []
        self.errors = []
        self.spans = []
        self._timers = {}
        self._active_spans = {}
        self._start_time = None
        self._end_time = None
        self._metrics = {
            "total_events": 0,
            "successful_events": 0,
            "failed_events": 0,
            "warnings": 0,
            "total_latency_ms": 0,
            "latencies": [],
            "errors_by_type": {},
            "events_by_type": {}
        }
