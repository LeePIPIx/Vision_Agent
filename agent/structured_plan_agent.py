import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from agent.structured_plan_prompts import STRUCTURED_PLAN_PROMPT, STRUCTURED_SUMMARY_PROMPT
from utils.logger import get_logger
from tools.ToolRegistry import ToolRegistry

logger = get_logger(__name__)


class StructuredPlanAgent:
    """
    基于「LLM 输出 JSON 计划 → Python 直接执行 → LLM 最终总结」的 Agent。

    流程：
    1. LLM 接收用户 query，输出 JSON 格式的执行计划
    2. Python 解析 JSON，按顺序调用工具，处理步骤间依赖
    3. 将所有工具结果喂给 LLM，生成最终的综合回答

    与 PlanMedicalAgent 的核心区别：
    - PlanMedicalAgent：计划以文本形式注入 LLM 上下文，LLM 在每一步决定调用哪个工具
    - StructuredPlanAgent：计划是结构化 JSON，Python 直接按计划调用工具，LLM 不参与执行
    """

    def __init__(self, tools: list, model: str = "deepseek-v4-flash"):
        self.agent_tools = ToolRegistry()
        self.agent_tools.register(*tools)
        self.model = model
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://api.deepseek.com",
        )
        self._plan_max_retries = 2
        logger.info(f"[StructuredPlanAgent] 初始化完成，model={model}，tools={[t.name for t in tools]}")

    # ============================
    # LLM 底层调用
    # ============================

    def _call_llm(
        self,
        messages: list,
        json_mode: bool = False,
        with_tools: bool = False,
    ) -> str:
        """
        调用 LLM 并返回消息文本。
        - json_mode: 启用 JSON 模式（response_format）
        - with_tools: 是否携带工具定义
        """
        import time

        start = time.time()
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "extra_body": {"thinking": {"type": "disabled"}},
        }

        if with_tools:
            kwargs["tools"] = self.agent_tools.openai_schema
            kwargs["tool_choice"] = "auto"
        else:
            # JSON 模式和 tool_choice="none" 一起使用
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            kwargs["tool_choice"] = "none"

        response = self.client.chat.completions.create(**kwargs)
        elapsed = time.time() - start
        msg = response.choices[0].message
        content = msg.content or ""

        usage = getattr(response, "usage", None)
        if usage:
            logger.debug(
                f"[LLM] 耗时 {elapsed:.2f}s，"
                f"input={usage.prompt_tokens}，output={usage.completion_tokens}"
            )
        else:
            logger.debug(f"[LLM] 耗时 {elapsed:.2f}s")
        return content

    # ============================
    # Phase 1: JSON 计划生成
    # ============================

    def _generate_plan(self, query: str) -> dict:
        """
        调用 LLM 生成 JSON 执行计划。
        返回解析后的 dict，结构为 {"task_analysis": "...", "steps": [...]}
        """
        logger.info("[StructuredPlanAgent] === Phase 1: 生成 JSON 计划 ===")

        messages = [
            {"role": "system", "content": STRUCTURED_PLAN_PROMPT},
            {"role": "user", "content": query},
        ]

        for attempt in range(1, self._plan_max_retries + 1):
            raw = self._call_llm(messages, json_mode=True, with_tools=False)
            logger.debug(f"[StructuredPlanAgent] LLM 原始输出 (尝试 {attempt}):\n{raw[:500]}")

            plan = self._parse_json(raw)
            if plan and self._validate_plan(plan):
                logger.info(
                    f"[StructuredPlanAgent] 计划解析成功，共 {len(plan['steps'])} 个步骤"
                )
                return plan

            logger.warning(f"[StructuredPlanAgent] JSON 解析/校验失败 (尝试 {attempt})，准备重试...")
            # 重试时在消息后面追加纠错提示
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": "你输出的不是合法的 JSON，请严格按照要求的 JSON 格式重新输出，不要用 Markdown 代码块包裹。",
            })

        # 兜底：返回一个极简计划
        logger.error("[StructuredPlanAgent] 多次重试后仍无法解析 JSON，使用兜底计划")
        return {
            "task_analysis": "规划失败，使用默认策略",
            "steps": [
                {
                    "step_id": 1,
                    "description": "搜索相关知识",
                    "tool": "web_search_tool",
                    "parameters": {"query": query},
                    "depends_on": None,
                }
            ],
        }

    def _parse_json(self, raw: str) -> dict | None:
        """尝试从 LLM 输出中提取 JSON 对象"""
        # 尝试 1：直接解析
        try:
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            pass

        # 尝试 2：提取 ```json ... ``` 代码块中的内容
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # 尝试 3：查找第一个 { 和最后一个 } 之间的内容
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                pass

        return None

    def _validate_plan(self, plan: dict) -> bool:
        """校验计划结构是否合法"""
        if not isinstance(plan, dict):
            return False
        if "steps" not in plan or not isinstance(plan["steps"], list):
            return False
        if len(plan["steps"]) == 0:
            return False

        known_tools = set(self.agent_tools.tool_names)
        for step in plan["steps"]:
            if not isinstance(step, dict):
                return False
            if "tool" not in step or step["tool"] not in known_tools:
                logger.warning(f"[StructuredPlanAgent] 步骤 {step.get('step_id', '?')} 的工具 '{step.get('tool', '?')}' 不在已知工具列表中")
                return False
            if "step_id" not in step:
                return False
        return True

    # ============================
    # Phase 2: Python 按计划执行工具
    # ============================

    def _execute_plan(self, plan: dict) -> list[dict]:
        """
        按 JSON 计划逐步调用工具。
        处理步骤间依赖（depends_on + 占位符替换）。
        返回每一步的执行结果列表。
        """
        logger.info(f"[StructuredPlanAgent] === Phase 2: Python 执行计划（共 {len(plan['steps'])} 步）===")

        step_results: dict[int, str] = {}  # step_id → 工具返回的原始字符串
        step_records: list[dict] = []       # 详细的步骤执行记录

        for step in plan["steps"]:
            step_id = step["step_id"]
            tool_name = step["tool"]
            description = step.get("description", "")
            depends_on = step.get("depends_on")
            parameters: dict = step.get("parameters", {})

            logger.info(f"[StructuredPlanAgent] 执行步骤 {step_id}: {description} (工具: {tool_name})")

            # --- 处理依赖 ---
            if depends_on is not None:
                dep_id = depends_on
                if dep_id not in step_results:
                    err = f"依赖的步骤 {dep_id} 未找到或未执行成功，跳过本步骤"
                    logger.warning(f"[StructuredPlanAgent] {err}")
                    step_records.append({
                        "step_id": step_id,
                        "description": description,
                        "tool": tool_name,
                        "success": False,
                        "result": err,
                    })
                    continue

                # 替换参数中的占位符
                dep_result = step_results[dep_id]
                for key, value in parameters.items():
                    if isinstance(value, str) and re.search(r"\{\{output_of_step_\d+\}\}", value):
                        parameters[key] = re.sub(
                            r"\{\{output_of_step_\d+\}\}",
                            dep_result,
                            value,
                        )

            # --- 执行工具 ---
            try:
                result = self.agent_tools.execute(tool_name, **parameters)
                step_results[step_id] = result
                step_records.append({
                    "step_id": step_id,
                    "description": description,
                    "tool": tool_name,
                    "parameters": parameters,
                    "success": True,
                    "result": result,
                })
                logger.info(f"[StructuredPlanAgent] 步骤 {step_id} 执行成功")
                result_preview = json.dumps(result, ensure_ascii=False) if isinstance(result, dict) else str(result)
                logger.debug(f"[StructuredPlanAgent] 结果摘要: {result_preview[:200]}")

            except Exception as e:
                err_msg = f"工具执行异常: {type(e).__name__}: {e}"
                logger.error(f"[StructuredPlanAgent] 步骤 {step_id} 失败: {err_msg}")
                step_results[step_id] = err_msg
                step_records.append({
                    "step_id": step_id,
                    "description": description,
                    "tool": tool_name,
                    "parameters": parameters,
                    "success": False,
                    "result": err_msg,
                })

        succeeded = sum(1 for r in step_records if r["success"])
        logger.info(f"[StructuredPlanAgent] 执行完毕: {succeeded}/{len(step_records)} 步骤成功")
        return step_records

    # ============================
    # Phase 3: LLM 最终总结
    # ============================

    def _summarize(self, query: str, plan: dict, step_records: list[dict]) -> str:
        """
        将所有工具结果喂给 LLM，生成最终的综合回答。
        """
        logger.info("[StructuredPlanAgent] === Phase 3: LLM 生成最终总结 ===")

        # 构建结构化的执行摘要
        steps_text_parts = []
        for r in step_records:
            status = "✅ 成功" if r["success"] else "❌ 失败"
            steps_text_parts.append(
                f"步骤 {r['step_id']}: {r['description']} [{status}]\n"
                f"工具: {r['tool']}\n"
                f"结果: {r['result']}"
            )
        steps_text = "\n\n".join(steps_text_parts)

        user_content = f"""## 用户问题
{query}

## 执行计划
任务分析: {plan.get('task_analysis', 'N/A')}
共 {len(step_records)} 个步骤，已全部执行完毕。

## 各步骤执行结果
{steps_text}

请根据以上信息，为用户生成一份综合性的最终回答。"""

        messages = [
            {"role": "system", "content": STRUCTURED_SUMMARY_PROMPT},
            {"role": "user", "content": user_content},
        ]

        answer = self._call_llm(messages, json_mode=False, with_tools=False)
        logger.info(f"[StructuredPlanAgent] 最终回答生成完成，长度={len(answer)}")
        return answer

    # ============================
    # 主入口
    # ============================

    def call(self, query: str) -> str:
        """
        完整的 Plan → Execute → Summarize 流程：
        1. LLM 生成 JSON 计划
        2. Python 按计划执行工具
        3. LLM 根据结果生成最终回答
        """
        logger.info(f"[StructuredPlanAgent] 收到查询，length={len(query)}")

        # Phase 1: LLM 生成 JSON 计划
        plan = self._generate_plan(query)

        # Phase 2: Python 解析并执行
        step_records = self._execute_plan(plan)

        # Phase 3: LLM 总结
        final_answer = self._summarize(query, plan, step_records)

        return final_answer


# ============================================================
# 测试入口
# ============================================================
if __name__ == "__main__":
    from tools import (
        SegmentationTool,
        LoadNiigzTool,
        LesionAnalysisTool,
        WebSearchTool,
        GetCurrentTimeTool,
    )

    tools = [LoadNiigzTool, SegmentationTool, LesionAnalysisTool, WebSearchTool, GetCurrentTimeTool]
    agent = StructuredPlanAgent(tools=tools)

    print("=" * 60)
    print("测试 1：简单时间查询")
    print("=" * 60)
    result = agent.call("现在几点了")
    print("[模型输出]:", result)
    print()

    print("=" * 60)
    print("测试 2：医学知识查询")
    print("=" * 60)
    result = agent.call("肺结节的常见病因是什么？")
    print("[模型输出]:", result)
