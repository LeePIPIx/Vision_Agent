"""Request-scoped ordered call tracing for LangGraph execution."""

import json
import logging
import threading
from contextvars import ContextVar
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any
from uuid import uuid4

from utils.logger import get_logger


logger = get_logger("medical_agent.call_trace")

_SENSITIVE_PARTS = ("api_key", "apikey", "authorization", "password", "secret", "token")
_MAX_STRING_LENGTH = 1200
_current_trace: ContextVar["CallTrace | None"] = ContextVar("call_trace", default=None)


def _truncate(value: str, limit: int = _MAX_STRING_LENGTH) -> str:
    normalized = value.replace("\x00", "").strip()
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit]}... <truncated {len(normalized) - limit} chars>"


def _safe_value(value: Any, depth: int = 0) -> Any:
    if depth >= 4:
        return _truncate(str(value), 240)
    if isinstance(value, dict):
        output = {}
        for key, item in list(value.items())[:30]:
            key_text = str(key)
            if any(part in key_text.lower() for part in _SENSITIVE_PARTS):
                output[key_text] = "***"
            else:
                output[key_text] = _safe_value(item, depth + 1)
        if len(value) > 30:
            output["_truncated_fields"] = len(value) - 30
        return output
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        output = [_safe_value(item, depth + 1) for item in items[:20]]
        if len(items) > 20:
            output.append(f"<truncated {len(items) - 20} items>")
        return output
    if isinstance(value, str):
        return _truncate(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _truncate(str(value))


@dataclass
class CallTrace:
    session_id: str
    graph: str
    request_id: str = field(default_factory=lambda: f"req-{uuid4().hex[:10]}")
    started_at: float = field(default_factory=perf_counter)
    events: list[dict] = field(default_factory=list)
    _sequence: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(
        self,
        component: str,
        action: str,
        details: Any = None,
        level: int = logging.INFO,
    ) -> dict:
        from medical_app.context import checkpoint
        checkpoint()
        with self._lock:
            self._sequence += 1
            elapsed_ms = round((perf_counter() - self.started_at) * 1000, 1)
            event = {
                "sequence": self._sequence,
                "elapsed_ms": elapsed_ms,
                "request_id": self.request_id,
                "session_id": self.session_id,
                "graph": self.graph,
                "component": component,
                "action": action,
                "details": _safe_value(details) if details is not None else {},
            }
            detail_text = json.dumps(event["details"], ensure_ascii=False, separators=(",", ":"))
            event["line"] = (
                f"[CALL_TRACE] request={self.request_id} session={self.session_id} "
                f"seq={self._sequence:02d} +{elapsed_ms:.1f}ms "
                f"component={component} action={action} details={detail_text}"
            )
            self.events.append(event)

        from medical_app.context import current_task
        ctx = current_task.get()
        if ctx:
            stages = {
                ("supervisor", "model_request"): ("routing", "识别请求意图"),
                ("model", "request"): ("generating", "模型正在处理请求"),
                ("tool", "execute_start"): ("tool_running", "正在执行工具"),
                ("tool", "execute_success"): ("tool_completed", "工具执行完成"),
            }
            if (component, action) in stages:
                ctx.event(*stages[(component, action)])
        logger.log(level, "request=%s component=%s action=%s elapsed_ms=%s", self.request_id, component, action, event["elapsed_ms"])
        return event


def start_call_trace(session_id: str, graph: str, details: Any = None):
    trace = CallTrace(session_id=session_id, graph=graph)
    token = _current_trace.set(trace)
    trace.add("request", "start", details)
    return trace, token


def finish_call_trace(token) -> None:
    _current_trace.reset(token)


def get_current_trace() -> CallTrace | None:
    return _current_trace.get()


def trace_event(
    component: str,
    action: str,
    details: Any = None,
    level: int = logging.INFO,
) -> dict | None:
    trace = get_current_trace()
    if trace is None:
        return None
    return trace.add(component, action, details, level)
