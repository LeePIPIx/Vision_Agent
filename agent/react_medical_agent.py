import json
import os
import sys

# 将项目根目录加入 sys.path，支持直接运行此脚本
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from agent.react_prompts import MEDICAL_AGENT_SYSTEM_PROMPT
from tools import SegmentationTool
from tools import LesionAnalysisTool
from tools import LoadNiigzTool
from tools import WebSearchTool
from tools import GetCurrentTimeTool
from openai.types.chat import ChatCompletionMessage
from utils.logger import get_logger
from tools.ToolRegistry import ToolRegistry

logger = get_logger(__name__)


class MedicalAgent:
    def __init__(self, tools: list, model: str = "deepseek-v4-flash"):
        self.agent_tools = ToolRegistry()
        self.agent_tools.register(*tools)
        self.model = model
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://api.deepseek.com",
        )
        self.system_prompt = MEDICAL_AGENT_SYSTEM_PROMPT
        logger.info(f"[Agent] 初始化完成，model={model}，tools={[t.name for t in tools]}")

    def __build_messages(self, assistant_message: ChatCompletionMessage) -> dict:
        return {
            "role": "assistant",
            "content": assistant_message.content or "",
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                }
                for tool_call in (assistant_message.tool_calls or [])
            ]
        }
        
    def __build_LLm(self, messages: list) -> str:
        import time
        start = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            extra_body={"thinking": {"type": "disabled"}},
            tools=self.agent_tools.openai_schema,
            tool_choice="auto"
        )
        elapsed = time.time() - start
        msg = response.choices[0].message
        # 记录 token 用量（如果响应中包含）
        usage = getattr(response, "usage", None)
        if usage:
            logger.debug(
                f"[LLM] 耗时 {elapsed:.2f}s，"
                f"input-tokens={usage.prompt_tokens}，output-tokens={usage.completion_tokens}"
            )
        else:
            logger.debug(f"[LLM] 耗时 {elapsed:.2f}s")
        return msg
    
    def call(self, query: str, max_iterations=10) -> str:
        logger.info(f"[Agent] 收到查询，length={len(query)}，max_iterations={max_iterations}")

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query}
        ]
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"[Agent] === 第 {iteration}/{max_iterations} 轮迭代 ===")

            assistant_message = self.__build_LLm(messages)
            # 添加模型的初始回复（可能包含工具调用）到历史消息中
            messages.append(self.__build_messages(assistant_message))

            # 模型决定调用工具
            if assistant_message.tool_calls:
                tool_call = assistant_message.tool_calls[0]
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                logger.info(f"[Agent] LLM 决定调用工具: {function_name}，参数={arguments}")

                tool_result = None
                if function_name in self.agent_tools.tool_names:
                    tool_result = self.agent_tools.execute(function_name, **arguments)

                if tool_result is None:
                    logger.error(f"[Agent] 未找到工具: {function_name}")
                    return f"[错误] 未找到工具: {function_name}"

                logger.debug(f"[Agent] 工具返回: {json.dumps(tool_result, ensure_ascii=False)[:200]}")
                # 工具执行结果
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_result, ensure_ascii=False)
                })
            else:
                logger.info(f"[Agent] LLM 返回最终回答，结束（共 {iteration} 轮）")
                return assistant_message.content

        logger.warning(f"[Agent] 达到最大迭代次数 {max_iterations}，返回兜底消息")
        return "Error: 达到最大迭代次数,未能得到最终回答。"


if __name__ == "__main__":
    tools = [LoadNiigzTool, SegmentationTool, LesionAnalysisTool, WebSearchTool, GetCurrentTimeTool]
    agent = MedicalAgent(tools=tools)
    query = "现在几点了"
    result = agent.call(query)
    print("[模型输出]:", result)