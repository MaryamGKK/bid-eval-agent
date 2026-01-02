# services/observer.py
"""LangSmith-style observability with detailed tracing for every operation."""

import time
import traceback
import hashlib
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from contextlib import contextmanager
from enum import Enum

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
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
# Traces stored in-memory only (no console output)
# For production: add OTLP exporter to send to observability backend
# processor = BatchSpanProcessor(OTLPSpanExporter())
# provider.add_span_processor(processor)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("bid-eval-agent", "1.0.0")


class RunType(Enum):
    """LangSmith-style run types."""
    CHAIN = "chain"
    LLM = "llm"
    TOOL = "tool"
    RETRIEVER = "retriever"
    EMBEDDING = "embedding"
    PARSER = "parser"
    PROMPT = "prompt"


class RunStatus(Enum):
    """Run status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"


@dataclass
class TokenUsage:
    """Token usage for LLM calls."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens
        }


@dataclass
class Run:
    """LangSmith-style Run object with full trace details."""
    id: str
    name: str
    run_type: str
    start_time: str
    end_time: Optional[str] = None
    status: str = "pending"
    
    # Trace context
    trace_id: str = ""
    parent_run_id: Optional[str] = None
    child_run_ids: List[str] = field(default_factory=list)
    
    # Input/Output
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Optional[Dict[str, Any]] = None
    
    # Execution details
    error: Optional[str] = None
    error_type: Optional[str] = None
    stack_trace: Optional[str] = None
    
    # Timing
    latency_ms: Optional[float] = None
    first_token_ms: Optional[float] = None
    
    # LLM specific
    model: Optional[str] = None
    model_parameters: Dict[str, Any] = field(default_factory=dict)
    token_usage: Optional[TokenUsage] = None
    prompts: List[str] = field(default_factory=list)
    
    # Tool specific
    tool_name: Optional[str] = None
    tool_input: Optional[Dict] = None
    tool_output: Optional[Any] = None
    
    # Retriever specific
    query: Optional[str] = None
    documents: List[Dict] = field(default_factory=list)
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    feedback: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "run_type": self.run_type,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "latency_ms": self.latency_ms,
            "trace_id": self.trace_id,
            "parent_run_id": self.parent_run_id,
            "child_run_ids": self.child_run_ids,
            "inputs": self._truncate_dict(self.inputs),
            "outputs": self._truncate_dict(self.outputs) if self.outputs else None,
            "error": self.error,
            "error_type": self.error_type,
            "model": self.model,
            "model_parameters": self.model_parameters,
            "token_usage": self.token_usage.to_dict() if self.token_usage else None,
            "prompts": self.prompts[:2] if self.prompts else [],  # First 2 prompts
            "tool_name": self.tool_name,
            "query": self.query,
            "documents_count": len(self.documents),
            "tags": self.tags,
            "metadata": self.metadata,
            "feedback": self.feedback
        }
    
    def _truncate_dict(self, d: Dict, max_len: int = 1000) -> Dict:
        """Truncate long values in dict for display."""
        if not d:
            return d
        result = {}
        for k, v in d.items():
            if isinstance(v, str) and len(v) > max_len:
                result[k] = v[:max_len] + "..."
            elif isinstance(v, dict):
                result[k] = self._truncate_dict(v, max_len)
            elif isinstance(v, list) and len(v) > 10:
                result[k] = v[:10] + ["..."]
            else:
                result[k] = v
        return result


class Observer:
    """LangSmith-style observability with detailed run tracking."""
    
    def __init__(self):
        self.runs: Dict[str, Run] = {}
        self.run_tree: List[str] = []  # Root run IDs
        self._timers: Dict[str, float] = {}
        self._current_trace_id: Optional[str] = None
        self._run_stack: List[str] = []  # Stack of active run IDs
        self.tracer = tracer
        
        # Legacy event compatibility
        self.events: List[Dict] = []
        self.errors: List[Dict] = []
        self.spans: List[Dict] = []
        
        # Metrics
        self._metrics = {
            "total_runs": 0,
            "successful_runs": 0,
            "failed_runs": 0,
            "total_latency_ms": 0,
            "total_tokens": 0,
            "runs_by_type": {},
            "latencies": []
        }
    
    def _generate_id(self) -> str:
        """Generate unique run ID."""
        return hashlib.sha256(
            f"{datetime.now().isoformat()}-{time.perf_counter()}".encode()
        ).hexdigest()[:16]
    
    def _generate_trace_id(self) -> str:
        """Generate unique trace ID."""
        return hashlib.sha256(
            f"trace-{datetime.now().isoformat()}-{time.perf_counter()}".encode()
        ).hexdigest()[:32]
    
    def start_trace(self, name: str = "evaluation") -> str:
        """Start a new trace (root run)."""
        self._current_trace_id = self._generate_trace_id()
        return self._current_trace_id
    
    @contextmanager
    def trace_run(
        self,
        name: str,
        run_type: RunType = RunType.CHAIN,
        inputs: Dict[str, Any] = None,
        tags: List[str] = None,
        metadata: Dict[str, Any] = None
    ):
        """Context manager for tracing a run (LangSmith style)."""
        run_id = self._generate_id()
        
        # Set trace ID if not set
        if not self._current_trace_id:
            self._current_trace_id = self._generate_trace_id()
        
        # Get parent run ID
        parent_run_id = self._run_stack[-1] if self._run_stack else None
        
        # Create run
        run = Run(
            id=run_id,
            name=name,
            run_type=run_type.value,
            start_time=datetime.now().isoformat(),
            status="running",
            trace_id=self._current_trace_id,
            parent_run_id=parent_run_id,
            inputs=inputs or {},
            tags=tags or [],
            metadata=metadata or {}
        )
        
        # Add to parent's children
        if parent_run_id and parent_run_id in self.runs:
            self.runs[parent_run_id].child_run_ids.append(run_id)
        else:
            self.run_tree.append(run_id)
        
        self.runs[run_id] = run
        self._run_stack.append(run_id)
        
        start_time = time.perf_counter()
        
        # OpenTelemetry span
        with self.tracer.start_as_current_span(
            name,
            kind=SpanKind.INTERNAL,
            attributes={"run_id": run_id, "run_type": run_type.value}
        ) as span:
            try:
                yield run
                
                # Success
                run.status = "success"
                run.end_time = datetime.now().isoformat()
                run.latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                
                span.set_status(Status(StatusCode.OK))
                self._metrics["successful_runs"] += 1
                
            except Exception as e:
                # Error
                run.status = "error"
                run.error = str(e)
                run.error_type = type(e).__name__
                run.stack_trace = traceback.format_exc()
                run.end_time = datetime.now().isoformat()
                run.latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
                
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                self._metrics["failed_runs"] += 1
                
                self.errors.append({
                    "run_id": run_id,
                    "error_type": run.error_type,
                    "message": run.error,
                    "stack_trace": run.stack_trace,
                    "timestamp": run.end_time
                })
                
                raise
            finally:
                self._run_stack.pop()
                self._metrics["total_runs"] += 1
                self._metrics["runs_by_type"][run_type.value] = \
                    self._metrics["runs_by_type"].get(run_type.value, 0) + 1
                if run.latency_ms:
                    self._metrics["latencies"].append(run.latency_ms)
                    self._metrics["total_latency_ms"] += run.latency_ms
    
    def log_llm_call(
        self,
        name: str,
        model: str,
        prompts: List[Dict[str, str]],
        response: str,
        token_usage: Dict[str, int] = None,
        model_parameters: Dict[str, Any] = None,
        latency_ms: float = None
    ) -> str:
        """Log an LLM call with full details."""
        run_id = self._generate_id()
        parent_run_id = self._run_stack[-1] if self._run_stack else None
        
        usage = TokenUsage(
            prompt_tokens=token_usage.get("prompt_tokens", 0) if token_usage else 0,
            completion_tokens=token_usage.get("completion_tokens", 0) if token_usage else 0,
            total_tokens=token_usage.get("total_tokens", 0) if token_usage else 0
        )
        
        run = Run(
            id=run_id,
            name=name,
            run_type=RunType.LLM.value,
            start_time=datetime.now().isoformat(),
            end_time=datetime.now().isoformat(),
            status="success",
            trace_id=self._current_trace_id or self._generate_trace_id(),
            parent_run_id=parent_run_id,
            inputs={"messages": prompts},
            outputs={"response": response[:2000] if len(response) > 2000 else response},
            model=model,
            model_parameters=model_parameters or {},
            token_usage=usage,
            prompts=[p.get("content", "")[:500] for p in prompts],
            latency_ms=latency_ms,
            tags=["llm", model.split("/")[-1] if "/" in model else model]
        )
        
        if parent_run_id and parent_run_id in self.runs:
            self.runs[parent_run_id].child_run_ids.append(run_id)
        
        self.runs[run_id] = run
        self._metrics["total_tokens"] += usage.total_tokens
        
        # Legacy event
        self.events.append({
            "timestamp": run.start_time,
            "type": "llm",
            "node": "complete",
            "status": "success",
            "data": {
                "model": model,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens
            },
            "duration_ms": latency_ms,
            "trace_id": run.trace_id,
            "span_id": run_id
        })
        
        return run_id
    
    def log_tool_call(
        self,
        name: str,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_output: Any,
        latency_ms: float = None,
        error: Exception = None
    ) -> str:
        """Log a tool call."""
        run_id = self._generate_id()
        parent_run_id = self._run_stack[-1] if self._run_stack else None
        
        run = Run(
            id=run_id,
            name=name,
            run_type=RunType.TOOL.value,
            start_time=datetime.now().isoformat(),
            end_time=datetime.now().isoformat(),
            status="error" if error else "success",
            trace_id=self._current_trace_id or self._generate_trace_id(),
            parent_run_id=parent_run_id,
            inputs=tool_input,
            outputs={"result": tool_output} if not error else None,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output if not error else None,
            latency_ms=latency_ms,
            error=str(error) if error else None,
            error_type=type(error).__name__ if error else None,
            tags=["tool", tool_name]
        )
        
        if parent_run_id and parent_run_id in self.runs:
            self.runs[parent_run_id].child_run_ids.append(run_id)
        
        self.runs[run_id] = run
        
        # Legacy event
        self.events.append({
            "timestamp": run.start_time,
            "type": "tool",
            "node": tool_name,
            "status": "error" if error else "success",
            "data": {"input": tool_input, "output_preview": str(tool_output)[:200]},
            "duration_ms": latency_ms,
            "trace_id": run.trace_id,
            "span_id": run_id
        })
        
        return run_id
    
    def log_retriever_call(
        self,
        name: str,
        query: str,
        documents: List[Dict[str, Any]],
        latency_ms: float = None
    ) -> str:
        """Log a retriever/search call."""
        run_id = self._generate_id()
        parent_run_id = self._run_stack[-1] if self._run_stack else None
        
        run = Run(
            id=run_id,
            name=name,
            run_type=RunType.RETRIEVER.value,
            start_time=datetime.now().isoformat(),
            end_time=datetime.now().isoformat(),
            status="success",
            trace_id=self._current_trace_id or self._generate_trace_id(),
            parent_run_id=parent_run_id,
            inputs={"query": query},
            outputs={"documents_count": len(documents)},
            query=query,
            documents=documents[:5],  # Store first 5 docs
            latency_ms=latency_ms,
            tags=["retriever", "search"]
        )
        
        if parent_run_id and parent_run_id in self.runs:
            self.runs[parent_run_id].child_run_ids.append(run_id)
        
        self.runs[run_id] = run
        
        # Legacy event
        self.events.append({
            "timestamp": run.start_time,
            "type": "search",
            "node": "retriever",
            "status": "success",
            "data": {"query": query[:100], "documents_found": len(documents)},
            "duration_ms": latency_ms,
            "trace_id": run.trace_id,
            "span_id": run_id
        })
        
        return run_id
    
    def update_run_output(self, run_id: str, outputs: Dict[str, Any]):
        """Update a run's outputs."""
        if run_id in self.runs:
            self.runs[run_id].outputs = outputs
    
    def add_run_feedback(self, run_id: str, score: float, comment: str = None):
        """Add feedback to a run."""
        if run_id in self.runs:
            self.runs[run_id].feedback.append({
                "score": score,
                "comment": comment,
                "timestamp": datetime.now().isoformat()
            })
    
    # Legacy compatibility methods
    @contextmanager
    def span(self, name: str, attributes: Dict[str, Any] = None, kind: SpanKind = SpanKind.INTERNAL):
        """Legacy span context manager - now wraps trace_run."""
        with self.trace_run(name, RunType.CHAIN, inputs=attributes) as run:
            # Create a mock span object for compatibility
            class MockSpan:
                def set_attribute(self, key, value):
                    run.metadata[key] = value
            yield MockSpan()
    
    def log(self, event_type: str, node: str, data: Dict[str, Any] = None, 
            status: str = "success", error: Exception = None):
        """Legacy log method."""
        run_id = self._generate_id()
        
        self.events.append({
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "node": node,
            "status": status,
            "data": data or {},
            "trace_id": self._current_trace_id,
            "span_id": run_id
        })
        
        if error:
            self.errors.append({
                "run_id": run_id,
                "error_type": type(error).__name__,
                "message": str(error),
                "timestamp": datetime.now().isoformat()
            })
    
    def start_timer(self, key: str):
        """Start a timer."""
        self._timers[key] = time.perf_counter()
    
    def stop_timer(self, key: str, event_type: str, node: str, data: Dict[str, Any] = None,
                   status: str = "success"):
        """Stop timer and log event with duration."""
        start = self._timers.pop(key, None)
        duration_ms = round((time.perf_counter() - start) * 1000, 2) if start else None
        
        self.events.append({
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "node": node,
            "status": status,
            "data": data or {},
            "duration_ms": duration_ms,
            "trace_id": self._current_trace_id
        })
        
        if duration_ms:
            self._metrics["latencies"].append(duration_ms)
    
    def get_events(self, event_type: str = None) -> List[Dict]:
        """Get all events."""
        events = self.events
        if event_type:
            events = [e for e in events if e.get("type") == event_type]
        return events
    
    def get_runs(self) -> List[Dict]:
        """Get all runs as dictionaries."""
        return [run.to_dict() for run in self.runs.values()]
    
    def get_run_tree(self) -> List[Dict]:
        """Get run tree structure for visualization."""
        def build_tree(run_id: str, depth: int = 0) -> Dict:
            run = self.runs.get(run_id)
            if not run:
                return None
            
            return {
                **run.to_dict(),
                "depth": depth,
                "children": [
                    build_tree(child_id, depth + 1) 
                    for child_id in run.child_run_ids
                    if child_id in self.runs
                ]
            }
        
        return [build_tree(root_id) for root_id in self.run_tree if root_id in self.runs]
    
    def get_trace(self, trace_id: str = None) -> Dict:
        """Get full trace details."""
        tid = trace_id or self._current_trace_id
        trace_runs = [r for r in self.runs.values() if r.trace_id == tid]
        
        if not trace_runs:
            return {}
        
        # Calculate totals
        total_latency = sum(r.latency_ms or 0 for r in trace_runs)
        total_tokens = sum(
            r.token_usage.total_tokens if r.token_usage else 0 
            for r in trace_runs
        )
        
        return {
            "trace_id": tid,
            "start_time": min(r.start_time for r in trace_runs),
            "end_time": max(r.end_time or r.start_time for r in trace_runs),
            "total_latency_ms": round(total_latency, 2),
            "total_tokens": total_tokens,
            "total_runs": len(trace_runs),
            "status": "error" if any(r.status == "error" for r in trace_runs) else "success",
            "runs_by_type": {
                rt: len([r for r in trace_runs if r.run_type == rt])
                for rt in set(r.run_type for r in trace_runs)
            },
            "runs": self.get_run_tree()
        }
    
    def get_rag_trace(self) -> RAGTrace:
        """Get RAG trace configuration."""
        return RAGTrace(
            vector_store=RAG_CONFIG["vector_store"],
            embedding_model=RAG_CONFIG["embedding_model"],
            documents_retrieved_per_company=RAG_CONFIG["documents_retrieved_per_company"],
            retrieval_confidence_threshold=RAG_CONFIG["retrieval_confidence_threshold"]
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive observability summary."""
        search_events = [e for e in self.events if e.get("type") == "search"]
        score_events = [e for e in self.events if e.get("type") == "score"]
        llm_events = [e for e in self.events if e.get("type") == "llm"]
        
        latencies = self._metrics["latencies"]
        
        return {
            "total_events": len(self.events),
            "total_runs": self._metrics["total_runs"],
            "successful_runs": self._metrics["successful_runs"],
            "failed_runs": self._metrics["failed_runs"],
            "search_count": len(search_events),
            "score_count": len(score_events),
            "llm_calls": len(llm_events),
            "total_tokens": self._metrics["total_tokens"],
            "total_duration_ms": round(self._metrics["total_latency_ms"], 2),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
            "p50_latency_ms": round(sorted(latencies)[len(latencies)//2], 2) if latencies else 0,
            "p99_latency_ms": round(sorted(latencies)[int(len(latencies)*0.99)] if latencies else 0, 2),
            "trace_ids": [self._current_trace_id] if self._current_trace_id else [],
            "runs_by_type": self._metrics["runs_by_type"],
            "errors_count": len(self.errors),
            "rag_trace": self.get_rag_trace().to_dict() if hasattr(self, 'get_rag_trace') else {}
        }
    
    def clear(self):
        """Clear all data."""
        self.runs = {}
        self.run_tree = []
        self._timers = {}
        self._current_trace_id = None
        self._run_stack = []
        self.events = []
        self.errors = []
        self.spans = []
        self._metrics = {
            "total_runs": 0,
            "successful_runs": 0,
            "failed_runs": 0,
            "total_latency_ms": 0,
            "total_tokens": 0,
            "runs_by_type": {},
            "latencies": []
        }
