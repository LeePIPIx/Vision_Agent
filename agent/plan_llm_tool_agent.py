import json
import os
import sys

# 将项目根目录加入 sys.path，支持直接运行此脚本
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from agent.plan_llm_tool_prompts import PLANNING_SYSTEM_PROMPT, EXECUTION_SYSTEM_PROMPT
from tools import SegmentationTool
from tools import LesionAnalysisTool
from tools import LoadNiigzTool
from tools import WebSearchTool
from tools import GetCurrentTimeTool
from openai.types.chat import ChatCompletionMessage
from utils.logger import get_logger
from tools.ToolRegistry import ToolRegistry

logger = get_logger(__name__)


class PlanMedicalAgent:
    """
    基于 Plan-Execute 模式的医疗 Agent：
    1. 规划阶段：禁用工具，LLM 先输出执行计划
    2. 执行阶段：启用工具，LLM 按计划逐步执行并追踪进度
    """

    def __init__(self, tools: list, model: str = "deepseek-v4-flash"):
        self.agent_tools = ToolRegistry()
        self.agent_tools.register(*tools)
        self.model = model
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://api.deepseek.com",
        )
        logger.info(f"[PlanAgent] 初始化完成，model={model}，tools={[t.name for t in tools]}")

    # ========== LLM 调用底层 ==========

    def __build_assistant_message(self, assistant_message: ChatCompletionMessage) -> dict:
        """将 LLM 响应构建为标准 message dict"""
        return {
            "role": "assistant",
            "content": assistant_message.content or "",
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                for tool_call in (assistant_message.tool_calls or [])
            ],
        }

    def __call_llm(self, messages: list, with_tools: bool = True) -> ChatCompletionMessage:
        """
        调用 LLM。
        - with_tools=True: 正常调用，可选择调用工具
        - with_tools=False: 禁用工具，只输出文本（规划阶段使用）
        """
        import time

        start = time.time()
        kwargs = {
            "model": self.model,
            "messages": messages,
            "extra_body": {"thinking": {"type": "disabled"}},
        }
        if with_tools:
            kwargs["tools"] = self.agent_tools.openai_schema
            kwargs["tool_choice"] = "auto"
        else:
            kwargs["tool_choice"] = "none"

        response = self.client.chat.completions.create(**kwargs)
        elapsed = time.time() - start
        msg = response.choices[0].message

        usage = getattr(response, "usage", None)
        if usage:
            logger.debug(
                f"[LLM] 耗时 {elapsed:.2f}s，"
                f"input={usage.prompt_tokens}，output={usage.completion_tokens}"
            )
        else:
            logger.debug(f"[LLM] 耗时 {elapsed:.2f}s")
        return msg

    def __execute_tool(self, tool_call) -> str:
        """执行单个工具调用并返回结果"""
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        logger.info(f"[PlanAgent] 调用工具: {function_name}，参数={arguments}")

        if function_name not in self.agent_tools.tool_names:
            logger.error(f"[PlanAgent] 未找到工具: {function_name}")
            return f"[错误] 未找到工具: {function_name}"

        result = self.agent_tools.execute(function_name, **arguments)
        logger.debug(f"[PlanAgent] 工具返回: {json.dumps(result, ensure_ascii=False)[:200]}")
        return result

    # ========== Phase 1: 规划 ==========

    def _create_plan(self, query: str) -> str:
        """
        规划阶段：调用 LLM 生成执行计划（禁用工具）。
        返回计划文本。
        """
        logger.info("[PlanAgent] === 规划阶段：生成执行计划 ===")

        plan_messages = [
            {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        response_msg = self.__call_llm(plan_messages, with_tools=False)
        plan_text = response_msg.content or ""

        logger.info(f"[PlanAgent] 计划生成完成，长度={len(plan_text)}")
        logger.debug(f"[PlanAgent] 计划内容:\n{plan_text[:500]}...")
        return plan_text

    # ========== Phase 2: 执行 ==========

    def _execute_plan(self, query: str, plan_text: str, max_iterations: int = 15) -> str:
        """
        执行阶段：将计划注入上下文，启用工具，按计划逐步执行。
        返回最终回答。
        """
        logger.info(f"[PlanAgent] === 执行阶段：按计划逐步执行（max_iterations={max_iterations}）===")

        messages = [
            {"role": "system", "content": EXECUTION_SYSTEM_PROMPT},
            {"role": "user", "content": query},
            {"role": "assistant", "content": plan_text},
            {
                "role": "user",
                "content": "好的，请严格按照你制定的计划逐步执行。从步骤 1 开始，每完成一步汇报进度。",
            },
        ]

        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"[PlanAgent] === 第 {iteration}/{max_iterations} 轮迭代 ===")

            assistant_message = self.__call_llm(messages, with_tools=True)
            messages.append(self.__build_assistant_message(assistant_message))

            if assistant_message.tool_calls:
                # 有工具调用 → 执行并反馈结果
                for tool_call in assistant_message.tool_calls:
                    tool_result = self.__execute_tool(tool_call)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    })
            else:
                # 无工具调用 → 最终回答
                logger.info(f"[PlanAgent] 执行完成，共 {iteration} 轮")
                return assistant_message.content or ""

        logger.warning(f"[PlanAgent] 达到最大迭代次数 {max_iterations}")
        return "Error: 达到最大迭代次数，未能完成所有计划步骤。"

    # ========== 主入口 ==========

    def call(self, query: str, max_iterations: int = 15) -> str:
        """
        完整的 Plan-Execute 流程：
        1. 生成执行计划
        2. 按计划逐步执行
        """
        logger.info(f"[PlanAgent] 收到查询，length={len(query)}")

        # Phase 1: 规划
        plan_text = self._create_plan(query)

        # Phase 2: 执行
        final_answer = self._execute_plan(query, plan_text, max_iterations)

        return final_answer


# ============================================================
# 测试入口
# ============================================================
if __name__ == "__main__":
    tools = [LoadNiigzTool, SegmentationTool, LesionAnalysisTool, WebSearchTool, GetCurrentTimeTool]
    agent = PlanMedicalAgent(tools=tools)

    # 测试 1：简单查询
    print("=" * 60)
    print("测试 1：简单时间查询")
    print("=" * 60)
    result = agent.call("现在几点了")
    print("[模型输出]:", result)
    print()

    # 测试 2：知识查询
    print("=" * 60)
    print("测试 2：医学知识查询")
    print("=" * 60)
    result = agent.call("肺结节的常见病因是什么？")
    print("[模型输出]:", result)
