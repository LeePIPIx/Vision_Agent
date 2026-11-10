from functools import partial

from langgraph.graph import StateGraph, START, END
from graphs.state import AgentState
from graphs.node import agent_think, tool_execute, plan_generate, supervisor_decide
from graphs.edges import should_continue, should_loop_or_summarize
from agent.react_prompts import MEDICAL_AGENT_SYSTEM_PROMPT
from agent.structured_plan_prompts import STRUCTURED_SUMMARY_PROMPT
from agent.plan_llm_tool_prompts import PLANNING_SYSTEM_PROMPT, EXECUTION_SYSTEM_PROMPT


PLAN_EXECUTE_AGENT_PROMPT = MEDICAL_AGENT_SYSTEM_PROMPT + """

## Plan-Execute 执行约束
- 先在内部判断完成任务需要哪些步骤，然后直接使用原生 tool call 执行所需工具。
- 不要只输出 JSON、计划、伪代码或“准备调用工具”等文字来代替实际工具调用。
- 用户提供图片路径并要求检测、分割、姿态、旋转框或分类时，必须调用 segmentation_tool。
- 在收到对应的 role=tool 结果之前，禁止声称工具已执行、文件已生成或给出检测数量。
- 工具执行完成后，基于真实工具结果回答；不得编造工具未返回的类别、数量、置信度或文件路径。
"""


def _make_init_state():
    """Append only the current user turn and reset turn-scoped state."""
    def init_state(state: AgentState) -> dict:
        return {
            "messages": [{"role": "user", "content": state["user_query"]}],
            "plan": "",
            "consecutive_errors": 0,
        }
    return init_state


def react_graph(system_prompt: str = MEDICAL_AGENT_SYSTEM_PROMPT, checkpointer=None):
    """
    ReAct 模式：边想边做，适合简单、单步工具调用。

    流程: init → agent ⇄ tool → END

    参数:
        system_prompt: 系统提示词
        checkpointer:  LangGraph checkpointer（如 MemorySaver()），传入即启用会话记忆
    """
    g = StateGraph(AgentState)

    g.add_node("init", _make_init_state())
    g.add_node(
        "agent",
        partial(agent_think, system_prompt=system_prompt, phase="react_agent"),
    )
    g.add_node("tool", tool_execute)

    g.add_edge(START, "init")
    g.add_edge("init", "agent")
    g.add_conditional_edges(
        "agent",
        should_continue,
        {"tool_execute": "tool", "__end__": END},
    )
    g.add_edge("tool", "agent")

    return g.compile(checkpointer=checkpointer)


def plan_and_execute_graph(system_prompt: str = PLAN_EXECUTE_AGENT_PROMPT, checkpointer=None):
    """
    Plan-Execute 模式：先规划再执行，最后用专用总结 prompt 生成报告。

    流程: init → agent ⇄ tool → summarize → END

    参数:
        system_prompt: 系统提示词
        checkpointer:  LangGraph checkpointer，传入即启用会话记忆
    """
    g = StateGraph(AgentState)

    # agent 节点 —— 每次调用都注入本工作流的 prompt，带工具
    plan_agent = partial(
        agent_think,
        system_prompt=system_prompt,
        phase="plan_execute_agent",
    )
    # summarize 节点 —— 注入总结 prompt 切换模式，不带工具
    summarize = partial(
        agent_think,
        system_prompt=STRUCTURED_SUMMARY_PROMPT,
        with_tools=False,
        phase="summarizer",
    )

    g.add_node("init", _make_init_state())
    g.add_node("agent", plan_agent)
    g.add_node("tool", tool_execute)
    g.add_node("summarize", summarize)

    g.add_edge(START, "init")
    g.add_edge("init", "agent")
    g.add_conditional_edges(
        "agent",
        should_loop_or_summarize,
        {"tool_execute": "tool", "summarize": "summarize"},
    )
    g.add_edge("tool", "agent")
    g.add_edge("summarize", END)

    return g.compile(checkpointer=checkpointer)


def plan_llm_tool_graph(checkpointer=None):
    """
    Plan-Execute 模式：显式规划 → 逐步执行 → 总结。

    与 plan_and_execute_graph 的关键区别：
    - plan_and_execute_graph: init 注入规划 prompt，agent 节点同时做规划+执行（隐式）
    - plan_llm_tool_graph:   planner 节点独立生成计划，executor 只负责执行（显式）

    流程: init → planner → executor ⇄ tool → summarize → END

    参数:
        checkpointer: LangGraph checkpointer，传入即启用会话记忆
    """
    g = StateGraph(AgentState)

    # ---- 节点定义 ----
    # planner:  禁用工具，只生成计划
    planner = partial(plan_generate, system_prompt=PLANNING_SYSTEM_PROMPT)
    # executor: 启用工具，参照计划逐步执行
    executor = partial(
        agent_think,
        system_prompt=EXECUTION_SYSTEM_PROMPT,
        phase="executor",
    )
    # summarize: 禁用工具，整合所有结果生成最终报告
    summarize = partial(
        agent_think,
        system_prompt=STRUCTURED_SUMMARY_PROMPT,
        with_tools=False,
        phase="summarizer",
    )

    g.add_node("init", _make_init_state())       # plan_llm 不在 init 中注入 system prompt
    g.add_node("planner", planner)
    g.add_node("executor", executor)
    g.add_node("tool", tool_execute)
    g.add_node("summarize", summarize)

    # ---- 连线 ----
    g.add_edge(START, "init")
    g.add_edge("init", "planner")          # 先规划
    g.add_edge("planner", "executor")       # 规划完 → 执行
    g.add_conditional_edges(
        "executor",
        should_loop_or_summarize,           # 有 tool_call → 继续执行，否则 → 总结
        {"tool_execute": "tool", "summarize": "summarize"},
    )
    g.add_edge("tool", "executor")          # 工具结果 → 回到执行器
    g.add_edge("summarize", END)            # 总结完 → 结束

    return g.compile(checkpointer=checkpointer)

def build_graph(checkpointer=None):
    """
    Supervisor 路由图：LLM 分析 query → 路由到最合适的 Agent 流程。

    流程: START → supervisor → react / plan_execute / plan_llm → END

    三条流程直接展开在同一张状态图中。这样每个节点只返回本轮新增消息，
    不会将 compiled 子图的完整历史再次通过 messages reducer 追加到父图。

    参数:
        checkpointer: LangGraph checkpointer，传入即启用会话记忆。
    """
    g = StateGraph(AgentState)

    g.add_node("supervisor", supervisor_decide)
    g.add_node("clarify", lambda state: {"messages": [
        {"role": "user", "content": state["user_query"]},
        {"role": "assistant", "content": "请补充要分析的影像或具体目标。如果你想解读之前的结果，请在对应病例会话中提问；CT 脾脏分割也可直接选择页面中的处理方式。"},
    ]})
    g.add_edge("clarify", END)

    # ReAct 分支
    g.add_node("react_init", _make_init_state())
    g.add_node(
        "react_agent",
        partial(
            agent_think,
            system_prompt=MEDICAL_AGENT_SYSTEM_PROMPT,
            phase="react_agent",
        ),
    )
    g.add_node("react_tool", tool_execute)

    # Plan-Execute 分支
    g.add_node("plan_execute_init", _make_init_state())
    g.add_node(
        "plan_execute_agent",
        partial(
            agent_think,
            system_prompt=PLAN_EXECUTE_AGENT_PROMPT,
            phase="plan_execute_agent",
        ),
    )
    g.add_node("plan_execute_tool", tool_execute)
    g.add_node(
        "plan_execute_summarize",
        partial(
            agent_think,
            system_prompt=STRUCTURED_SUMMARY_PROMPT,
            with_tools=False,
            phase="summarizer",
        ),
    )

    # Plan-LLM 分支
    g.add_node("plan_llm_init", _make_init_state())
    g.add_node(
        "plan_llm_planner",
        partial(plan_generate, system_prompt=PLANNING_SYSTEM_PROMPT),
    )
    g.add_node(
        "plan_llm_executor",
        partial(
            agent_think,
            system_prompt=EXECUTION_SYSTEM_PROMPT,
            phase="executor",
        ),
    )
    g.add_node("plan_llm_tool", tool_execute)
    g.add_node(
        "plan_llm_summarize",
        partial(
            agent_think,
            system_prompt=STRUCTURED_SUMMARY_PROMPT,
            with_tools=False,
            phase="summarizer",
        ),
    )

    g.add_edge(START, "supervisor")

    def route_supervisor(state: AgentState):
        """根据 supervisor 的决策路由到对应流程。"""
        return state["next_agent"]

    g.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "clarify": "clarify",
            "react": "react_init",
            "plan_execute": "plan_execute_init",
            "plan_llm": "plan_llm_init",
        },
    )

    g.add_edge("react_init", "react_agent")
    g.add_conditional_edges(
        "react_agent",
        should_continue,
        {"tool_execute": "react_tool", "__end__": END},
    )
    g.add_edge("react_tool", "react_agent")

    g.add_edge("plan_execute_init", "plan_execute_agent")
    g.add_conditional_edges(
        "plan_execute_agent",
        should_loop_or_summarize,
        {
            "tool_execute": "plan_execute_tool",
            "summarize": "plan_execute_summarize",
        },
    )
    g.add_edge("plan_execute_tool", "plan_execute_agent")
    g.add_edge("plan_execute_summarize", END)

    g.add_edge("plan_llm_init", "plan_llm_planner")
    g.add_edge("plan_llm_planner", "plan_llm_executor")
    g.add_conditional_edges(
        "plan_llm_executor",
        should_loop_or_summarize,
        {
            "tool_execute": "plan_llm_tool",
            "summarize": "plan_llm_summarize",
        },
    )
    g.add_edge("plan_llm_tool", "plan_llm_executor")
    g.add_edge("plan_llm_summarize", END)

    return g.compile(checkpointer=checkpointer)
