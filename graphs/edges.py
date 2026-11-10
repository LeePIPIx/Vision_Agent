import logging
from graphs.state import AgentState
from typing import Literal
from utils.call_trace import trace_event

# 连续工具失败上限（可在 state 中覆盖）
DEFAULT_MAX_CONSECUTIVE_ERRORS = 3


def _too_many_errors(state: AgentState) -> bool:
    """检查是否超过连续错误阈值（熔断）"""
    max_err = state.get("max_consecutive_errors", DEFAULT_MAX_CONSECUTIVE_ERRORS)
    consecutive = state.get("consecutive_errors", 0)
    return consecutive >= max_err


def should_continue(state: AgentState) -> Literal["tool_execute", "__end__"]:
    """
    ReAct 模式路由：
    - 连续错误过多 → 强制终止（熔断保护）
    - 有 tool_calls → 执行工具
    - 无 tool_calls → 正常结束
    """
    last_msg = state["messages"][-1]
    if last_msg.get("role") == "assistant" and last_msg.get("tool_calls"):
        if _too_many_errors(state):
            trace_event(
                "router",
                "error_limit_reached",
                {
                    "decision": "end",
                    "consecutive_errors": state.get("consecutive_errors", 0),
                },
                level=logging.WARNING,
            )
            return "__end__"
        trace_event(
            "router",
            "decision",
            {"next": "tool_execute", "tool_call_count": len(last_msg.get("tool_calls", []))},
        )
        return "tool_execute"
    trace_event("router", "decision", {"next": "end", "reason": "final_response"})
    return "__end__"


def should_loop_or_summarize(state: AgentState) -> Literal["tool_execute", "summarize"]:
    """
    Plan-Execute 模式路由：
    - 连续错误过多 → 强制进入总结（熔断保护）
    - 有 tool_calls → 执行工具
    - 无 tool_calls → 进入总结节点
    """
    last_msg = state["messages"][-1]
    if last_msg.get("role") == "assistant" and last_msg.get("tool_calls"):
        if _too_many_errors(state):
            trace_event(
                "router",
                "error_limit_reached",
                {
                    "decision": "summarize",
                    "consecutive_errors": state.get("consecutive_errors", 0),
                },
                level=logging.WARNING,
            )
            return "summarize"
        trace_event(
            "router",
            "decision",
            {"next": "tool_execute", "tool_call_count": len(last_msg.get("tool_calls", []))},
        )
        return "tool_execute"
    trace_event("router", "decision", {"next": "summarize", "reason": "agent_finished"})
    return "summarize"
