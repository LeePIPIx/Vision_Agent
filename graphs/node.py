import json
import logging
import os
import re
from functools import lru_cache
from html import unescape
from time import perf_counter
from uuid import uuid4

from openai import OpenAI

from graphs.state import AgentState
from tools.ToolRegistry import ToolRegistry
from utils.call_trace import trace_event
from tools import (
    GetCurrentTimeTool,
    LesionAnalysisTool,
    LoadNiigzTool,
    SegmentationTool,
    WebSearchTool,
)

# 全局单例，所有node共享
_register = ToolRegistry()
_register.register(GetCurrentTimeTool, LesionAnalysisTool, LoadNiigzTool, SegmentationTool, WebSearchTool)
@lru_cache(maxsize=1)
def _get_client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url="https://api.deepseek.com", timeout=60, max_retries=1)


_DSML_INVOKE_RE = re.compile(
    r'<\s*\|+\s*DSML\s*\|+\s*invoke\b(?P<attrs>[^>]*)>'
    r'(?P<body>.*?)'
    r'</\s*\|+\s*DSML\s*\|+\s*invoke\s*>',
    re.IGNORECASE | re.DOTALL,
)
_DSML_PARAMETER_RE = re.compile(
    r'<\s*\|+\s*DSML\s*\|+\s*parameter\b(?P<attrs>[^>]*)>'
    r'(?P<value>.*?)'
    r'</\s*\|+\s*DSML\s*\|+\s*parameter\s*>',
    re.IGNORECASE | re.DOTALL,
)
_ATTRIBUTE_RE = re.compile(r'([\w-]+)\s*=\s*(["\'])(.*?)\2', re.DOTALL)


def _parse_attributes(source: str) -> dict[str, str]:
    return {key.lower(): unescape(value) for key, _, value in _ATTRIBUTE_RE.findall(source)}


def _parse_dsml_tool_calls(content: str) -> list[dict]:
    """Convert DeepSeek's textual DSML tool protocol into OpenAI tool calls."""
    if "DSML" not in content or "tool_calls" not in content:
        return []

    normalized = content.replace("｜", "|")
    calls = []
    for invoke in _DSML_INVOKE_RE.finditer(normalized):
        invoke_attrs = _parse_attributes(invoke.group("attrs"))
        tool_name = invoke_attrs.get("name", "").strip()
        if not tool_name:
            continue

        arguments = {}
        for parameter in _DSML_PARAMETER_RE.finditer(invoke.group("body")):
            parameter_attrs = _parse_attributes(parameter.group("attrs"))
            parameter_name = parameter_attrs.get("name", "").strip()
            if not parameter_name:
                continue

            value = unescape(parameter.group("value").strip())
            if parameter_attrs.get("string", "").lower() != "true":
                try:
                    value = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    pass
            arguments[parameter_name] = value

        calls.append({
            "id": f"call_dsml_{uuid4().hex[:16]}",
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": json.dumps(arguments, ensure_ascii=False),
            },
        })

    return calls


def agent_think(
    state: AgentState,
    system_prompt: str | None = None,
    with_tools: bool = True,
    phase: str = "agent",
) -> dict:
    """
    通用的 agent 节点函数。

    参数:
        state:         LangGraph 注入的 AgentState
        system_prompt: 如果提供，会在本次调用时前置注入到消息列表，
                       用于"切换模式"（如从规划模式切换到总结模式）。
                       为 None 时沿用 messages 中已有的 system prompt。
        with_tools:    True  → 可调用工具（agent ⇄ tool 循环节点）
                       False → 纯文本生成（最终总结节点）
        phase:         写入调用日志的模型阶段名称。
    """
    messages = state["messages"]
    if system_prompt is not None:
        if state.get("case_context"):
            system_prompt += "\n当前病例已验证的结构化结果：\n" + json.dumps(state["case_context"], ensure_ascii=False)
        system_prompt += "\n医学数值只能引用工具或病例结果；没有测量就明确未完成。通用 YOLO 输出不代表医学病灶。输出报告必须标明待人工复核。"
        history_without_system = [
            message
            for message in messages
            if not isinstance(message, dict) or message.get("role") != "system"
        ]
        messages = [{"role": "system", "content": system_prompt}] + history_without_system

    kwargs: dict = {
        "model": "deepseek-v4-flash",
        "messages": messages,
        "extra_body": {"thinking": {"type": "disabled"}},
    }
    if with_tools:
        kwargs["tools"] = _register.openai_schema
        kwargs["tool_choice"] = "auto"

    trace_event(
        "model",
        "request",
        {
            "phase": phase,
            "model": kwargs["model"],
            "message_count": len(messages),
            "tools_enabled": with_tools,
        },
    )
    started_at = perf_counter()
    try:
        response = _get_client().chat.completions.create(**kwargs)
    except Exception as exc:
        trace_event(
            "model",
            "error",
            {
                "phase": phase,
                "duration_ms": round((perf_counter() - started_at) * 1000, 1),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
            level=logging.ERROR,
        )
        raise
    msg = response.choices[0].message

    assistant_message = {
        "role": "assistant",
        "content": msg.content or "",
    }
    if msg.tool_calls:
        assistant_message["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]
    else:
        dsml_tool_calls = _parse_dsml_tool_calls(assistant_message["content"])
        if dsml_tool_calls:
            assistant_message["content"] = ""
            assistant_message["tool_calls"] = dsml_tool_calls
            trace_event(
                "model",
                "dsml_converted",
                {"phase": phase, "tool_call_count": len(dsml_tool_calls)},
            )

    tool_call_details = []
    for call in assistant_message.get("tool_calls", []):
        function = call.get("function", {})
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                pass
        tool_call_details.append({
            "call_id": call.get("id", ""),
            "name": function.get("name", ""),
            "arguments": arguments,
        })

    trace_event(
        "model",
        "response",
        {
            "phase": phase,
            "duration_ms": round((perf_counter() - started_at) * 1000, 1),
            "finish_reason": response.choices[0].finish_reason,
            "tool_calls": tool_call_details,
            "content": assistant_message["content"],
        },
    )
    return {"messages": [assistant_message]}

def plan_generate(state: AgentState, system_prompt: str) -> dict:
    """
    规划节点：调用 LLM 生成执行计划（禁用工具，纯文本输出）。

    与 agent_think 的区别：
    - agent_think：带工具，在 plan_and_execute_graph 中身兼"规划+执行"两职
    - plan_generate：不带工具，只负责生成计划，计划存入 state["plan"]
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": state["user_query"]},
    ]

    trace_event(
        "model",
        "request",
        {
            "phase": "planner",
            "model": "deepseek-v4-flash",
            "message_count": len(messages),
            "tools_enabled": False,
        },
    )
    started_at = perf_counter()
    try:
        response = _get_client().chat.completions.create(
            model="deepseek-v4-flash",
            messages=messages,
            extra_body={"thinking": {"type": "disabled"}},
            tool_choice="none",
        )
    except Exception as exc:
        trace_event(
            "model",
            "error",
            {
                "phase": "planner",
                "duration_ms": round((perf_counter() - started_at) * 1000, 1),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
            level=logging.ERROR,
        )
        raise
    msg = response.choices[0].message
    content = msg.content or ""
    trace_event(
        "model",
        "response",
        {
            "phase": "planner",
            "duration_ms": round((perf_counter() - started_at) * 1000, 1),
            "finish_reason": response.choices[0].finish_reason,
            "content": content,
        },
    )

    return {
        "messages": [{
            "role": "assistant",
            "content": content,
        }],
        "plan": content,
    }


LAYA_MODEL = "convaiinnovations/laya-multilingual"
LAYA_QUESTIONS = {
    "workflow": {
        "type": "choice",
        "instructions": "判断用户当前业务意图。参考 recent_requests 和 has_measurements。缺少被引用的结果、影像或具体目标时选择 clarification。",
        "criteria": {
            "knowledge": "问候、时间查询、一般知识问答",
            "image_analysis": "请求对提供的图像进行检测或分割",
            "result_explanation": "解释或追问当前病例已有的分割和测量结果",
            "report": "根据已有病例结果生成或总结报告",
            "clarification": "缺少必要信息，无法明确分析对象或执行目标",
        },
    }
}


@lru_cache(maxsize=1)
def _laya_router():
    import laya

    return laya.load(os.getenv("LAYA_MODEL_PATH", LAYA_MODEL))


def supervisor_decide(state: AgentState) -> dict:
    """
    Supervisor 路由节点：LLM 分析 query → 输出路由目标。

    不操作 messages，只设置 next_agent 用于条件边路由。
    """
    query = state["user_query"]
    trace_event(
        "supervisor",
        "model_request",
        {"model": LAYA_MODEL, "query": query},
    )
    started_at = perf_counter()
    try:
        recent = [m.get("content", "")[:300] for m in state.get("messages", [])[-4:] if m.get("role") == "user"]
        result = _laya_router().predict({"query": query, "recent_requests": recent, "has_measurements": bool(state.get("case_context"))}, LAYA_QUESTIONS)
    except Exception as exc:
        trace_event(
            "supervisor",
            "model_error",
            {
                "duration_ms": round((perf_counter() - started_at) * 1000, 1),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
            level=logging.ERROR,
        )
        raise
    answer = result["answers"]["workflow"]
    raw = answer["choice"]
    intent = raw.strip().lower()
    # 中文短句上的零样本分类会高置信度误判，先执行明确的业务约束。
    normalized = query.strip().lower()
    has_results = bool(state.get("case_context"))
    if re.search(r"分割|检测|识别|标注", normalized) and re.search(r"图片|图像|影像|ct|mri|脾脏|\.nii|\.jpe?g|\.png", normalized):
        intent = "image_analysis"
    elif has_results and re.search(r"报告|总结", normalized) and re.search(r"生成|整理|撰写|输出", normalized):
        intent = "report"
    elif has_results and re.search(r"刚才|已有|之前|这个|这些|结果|测量|体积|连通区域", normalized):
        intent = "result_explanation"
    elif re.fullmatch(r"(帮我处理一下|它怎么样|帮我看看|这个怎么样)[？?。!！\s]*", normalized):
        intent = "clarification"
    mapping = {"knowledge": "react", "image_analysis": "plan_execute", "result_explanation": "react", "report": "plan_llm", "clarification": "clarify"}
    route = mapping.get(intent, "clarify")
    if intent in {"result_explanation", "report"} and not state.get("case_context"):
        route = "clarify"

    trace_event(
        "supervisor",
        "route_selected",
        {
            "route": route,
            "raw_route": raw,
            "intent": intent,
            "confidence": answer.get("confidence"),
            "duration_ms": round((perf_counter() - started_at) * 1000, 1),
        },
    )
    return {"next_agent": route}


# ---------------------------------------------------------------------------
# 错误恢复辅助函数
# ---------------------------------------------------------------------------

def _is_error_result(result: str) -> bool:
    """判断工具返回是否为错误（以已知错误前缀开头）"""
    return result.startswith(("[错误]", "[工具执行异常]", "[参数验证失败]"))


def _format_tool_error(tool_id: str, tool_name: str, error_type: str, error_detail: str) -> dict:
    """
    构建增强的错误消息，包含参数 schema 以帮助 LLM 自我纠正。

    LLM 看到这种格式的错误后，可以：
    1. 对比自己传的参数和 schema 的差异
    2. 修正参数名拼写或类型后重试
    3. 如果是未知工具，改用 schema 中存在的工具名
    """
    parts = [
        "⚠️ 工具调用失败 —— 请分析原因并修正后重试",
        f"失败工具: {tool_name}",
        f"错误类型: {error_type}",
        f"错误详情: {error_detail}",
    ]

    tool_cls = _register.get(tool_name)
    if tool_cls:
        try:
            schema = tool_cls.Parameters.model_json_schema()
            parts.append(f"正确参数格式 (JSON Schema): {json.dumps(schema, ensure_ascii=False)}")
        except Exception:
            pass
    else:
        parts.append(f"可用工具列表: {', '.join(_register.tool_names)}")

    parts.append("请修正参数后重新调用该工具，或改用其他工具完成任务。")
    return {"role": "tool", "tool_call_id": tool_id, "content": "\n\n".join(parts)}


def tool_execute(state: AgentState) -> dict:
    """
    执行工具调用，并返回增强的错误消息（支持 LLM 自我纠正）。

    容错策略：
    - 单次失败返回详细的错误上下文 + 参数 schema → LLM 可自我纠正
    - 连续失败计数记录到 state.consecutive_errors → edges 层做熔断
    - 成功时重置计数器
    """
    last_msg = state["messages"][-1]
    if "tool_calls" not in last_msg:
        trace_event("tool", "batch_skipped", {"reason": "no_tool_calls"})
        return {"messages": []}

    tool_calls = last_msg.get("tool_calls", [])
    tool_messages = []
    error_count = 0

    for tool in tool_calls:
        function = tool.get("function", {})
        name = function.get("name", "")
        tool_id = tool.get("id", "")

        # --- 获取参数 ---
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as e:
                trace_event(
                    "tool",
                    "arguments_error",
                    {
                        "call_id": tool_id,
                        "name": name,
                        "arguments": arguments,
                        "error": str(e),
                    },
                    level=logging.ERROR,
                )
                tool_messages.append(
                    _format_tool_error(tool_id, name, "JSON解析失败", str(e))
                )
                error_count += 1
                continue

        # --- 工具不存在 ---
        if name not in _register.tool_names:
            trace_event(
                "tool",
                "not_registered",
                {"call_id": tool_id, "name": name, "arguments": arguments},
                level=logging.ERROR,
            )
            tool_messages.append(
                _format_tool_error(tool_id, name, "工具未注册", f"'{name}' 不在工具列表中")
            )
            error_count += 1
            continue

        # --- 执行工具 ---
        trace_event(
            "tool",
            "execute_start",
            {"call_id": tool_id, "name": name, "arguments": arguments},
        )
        started_at = perf_counter()
        try:
            tool_result = _register.execute(name, **arguments)
        except Exception as e:
            trace_event(
                "tool",
                "execute_error",
                {
                    "call_id": tool_id,
                    "name": name,
                    "duration_ms": round((perf_counter() - started_at) * 1000, 1),
                    "error_type": type(e).__name__,
                    "error": str(e),
                },
                level=logging.ERROR,
            )
            tool_messages.append(
                _format_tool_error(tool_id, name, type(e).__name__, str(e))
            )
            error_count += 1
            continue

        # --- 检查返回值是否为错误 ---
        if isinstance(tool_result, str) and _is_error_result(tool_result):
            error_count += 1
            trace_event(
                "tool",
                "execute_failed",
                {
                    "call_id": tool_id,
                    "name": name,
                    "duration_ms": round((perf_counter() - started_at) * 1000, 1),
                    "result": tool_result,
                },
                level=logging.WARNING,
            )
            # 提取错误类型以便 LLM 理解
            tool_messages.append(
                _format_tool_error(tool_id, name, "参数或执行错误", tool_result)
            )
        else:
            trace_event(
                "tool",
                "execute_success",
                {
                    "call_id": tool_id,
                    "name": name,
                    "duration_ms": round((perf_counter() - started_at) * 1000, 1),
                    "result": tool_result,
                },
            )
            tool_messages.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "content": json.dumps(tool_result, ensure_ascii=False),
            })

    # --- 更新连续错误计数 ---
    prev_errors = state.get("consecutive_errors", 0)
    if error_count > 0:
        consecutive_errors = prev_errors + error_count
    else:
        consecutive_errors = 0  # 全部成功 → 归零

    trace_event(
        "tool",
        "batch_complete",
        {
            "tool_call_count": len(tool_calls),
            "error_count": error_count,
            "consecutive_errors": consecutive_errors,
        },
        level=logging.WARNING if error_count else logging.INFO,
    )

    return {
        "messages": tool_messages,
        "consecutive_errors": consecutive_errors,
    }

   
    
    
