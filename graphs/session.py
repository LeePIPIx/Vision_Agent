"""
会话管理器：封装 LangGraph 图 + MemorySaver，提供多轮对话能力。

用法:
    from graphs.session import SessionManager

    sess = SessionManager()                    # 默认使用 supervisor 路由图
    sess = SessionManager(graph="react")       # 指定使用 react 图
    sess = SessionManager(graph="plan_llm")    # 指定使用 plan_llm 图

    answer1 = sess.chat("现在几点了？")
    answer2 = sess.chat("那明天是周几？")      # 自动携带上文

    sess.new_session()                         # 开启新会话，清空历史
    sess.info                                  # 查看会话元信息
"""

import logging
from uuid import uuid4

from langgraph.checkpoint.memory import MemorySaver
from utils.call_trace import finish_call_trace, start_call_trace, trace_event


class SessionManager:
    """
    管理单个会话的多轮对话。

    核心职责:
    - 持有 MemorySaver 作为图的状态持久化后端
    - 管理 thread_id，LangGraph 按 thread_id 隔离不同会话
    - 提供 chat() 方法，自动携带历史上下文

    注意:
    - MemorySaver 是内存存储，进程重启后会话丢失
    - 需要持久化时，可替换为 SqliteSaver（pip install langgraph-checkpoint-sqlite）
    """

    def __init__(self, graph: str = "supervisor", checkpointer=None):
        """
        参数:
            graph: 图类型 — "supervisor" | "react" | "plan_execute" | "plan_llm"
        """
        self._memory = checkpointer if checkpointer is not None else MemorySaver()
        # 定义对话图类型，决定使用哪种图来处理多轮对话
        self._graph_type = graph
        # 根据图类型构建对应的 LangGraph 图实例
        self.graph = self._build_graph(graph, self._memory)
        self.thread_id = f"sess-{uuid4().hex[:8]}"

        # 保存最后一次执行结果，方便调试
        self.last_result: dict = {}
        self.last_route: str = ""
        self.last_trace: list[dict] = []
        self.last_request_id: str = ""

    @staticmethod
    def _build_graph(graph_type: str, checkpointer: MemorySaver):
        """根据类型名构建对应的编译图"""
        from graphs.build_graph import (
            build_graph,
            react_graph,
            plan_and_execute_graph,
            plan_llm_tool_graph,
        )

        builders = {
            "supervisor": build_graph,
            "react": react_graph,
            "plan_execute": plan_and_execute_graph,
            "plan_llm": plan_llm_tool_graph,
        }

        if graph_type not in builders:
            raise ValueError(f"未知图类型: {graph_type}，可选: {list(builders.keys())}")

        return builders[graph_type](checkpointer=checkpointer)

    # ============================
    # 核心接口
    # ============================

    def chat(self, user_query: str, max_iterations: int = 12, case_context=None) -> str:
        """
        发送消息并获取回复。自动携带当前会话的完整历史。

        参数:
            user_query: 用户输入的自然语言问题

        返回:
            Agent 的最终回答文本
        """
        config = {"configurable": {"thread_id": self.thread_id}, "recursion_limit": max_iterations * 3 + 8}
        previous_message_count = len(self.last_result.get("messages", []))
        trace, trace_token = start_call_trace(
            session_id=self.thread_id,
            graph=self._graph_type,
            details={"query": user_query},
        )
        self.last_request_id = trace.request_id

        try:
            trace_event(
                "langgraph",
                "invoke_start",
                {
                    "thread_id": self.thread_id,
                    "previous_message_count": previous_message_count,
                },
            )
            result = self.graph.invoke(
                {"user_query": user_query, "case_context": case_context or {}},
                config,
            )

            self.last_result = result
            self.last_route = result.get("next_agent", "")

            messages = result.get("messages", [])
            answer = ""
            if messages:
                last_msg = messages[-1]
                answer = last_msg.get("content", "") if isinstance(last_msg, dict) else str(last_msg)

            trace_event(
                "request",
                "complete",
                {
                    "route": self.last_route,
                    "message_count": len(messages),
                    "current_turn_message_count": len(messages) - previous_message_count,
                    "answer": answer,
                },
            )
            return answer
        except Exception as exc:
            trace_event(
                "request",
                "error",
                {"error_type": type(exc).__name__, "error": str(exc)},
                level=logging.ERROR,
            )
            raise
        finally:
            self.last_trace = [dict(event) for event in trace.events]
            finish_call_trace(trace_token)

    def new_session(self) -> str:
        """
        开启全新会话（清空历史上下文）。

        返回:
            新会话的 thread_id
        """
        self.thread_id = f"sess-{uuid4().hex[:8]}"
        self.last_result = {}
        self.last_route = ""
        self.last_trace = []
        self.last_request_id = ""
        return self.thread_id

    # ============================
    # 辅助接口
    # ============================

    @property
    def info(self) -> dict:
        """返回当前会话的元信息"""
        return {
            "thread_id": self.thread_id,
            "graph_type": self._graph_type,
            "last_route": self.last_route,
            "last_request_id": self.last_request_id,
            "message_count": len(self.last_result.get("messages", [])),
        }

    def print_history(self):
        """打印当前会话的完整对话历史（调试用）"""
        messages = self.last_result.get("messages", [])
        if not messages:
            print("(无历史消息)")
            return

        for i, msg in enumerate(messages):
            role = msg.get("role", "?") if isinstance(msg, dict) else "?"
            content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
            display = content[:200] + "..." if len(content) > 200 else content
            print(f"[{i}] {role}: {display}")
