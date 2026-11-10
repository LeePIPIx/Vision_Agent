from typing import TypedDict, Annotated, Sequence
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    # ========== 输入层 ==========
    case_context: dict
    user_query: str           # 用户的自然语言问题
    image_path: str           # 影像/图片路径（.nii.gz / .jpg 等）
    iteration_count: int      # agent最大迭代轮数

    # ========== 路由层 ==========
    task_type: str            # 任务类型：image_info / segmentation / analysis / report / full_analysis
    next_agent: str           # supervisor 路由决策：react / plan_execute / plan_llm

    # ========== 规划层 ==========
    plan: str                 # Plan-Execute 模式下 planner 节点生成的执行计划

    # ========== 工具结果层 ==========
    image_info: dict | None          # LoadNiigzTool 的结果 {shape, affine, dtype, ...}
    segmentation_result: dict | None  # SegmentationTool 的结果 {task, objects, save_path, ...}
    lesion_stats: dict | None         # LesionAnalysisTool 的结果
    rag_context: str | None           # WebSearchTool / RAG 检索到的医学知识

    # ========== 报告层 ==========
    report: str | None        # 最终生成的结构化报告

    # ========== 消息层（LangGraph 特有，自动追加） ==========
    messages: Annotated[list, operator.add]  # 对话历史，LangGraph 自动管理

    # ========== 容错层 ==========
    consecutive_errors: int   # 连续工具调用失败次数（成功时归零，超阈值强制终止）
    max_consecutive_errors: int  # 连续错误上限（默认 3）

    # ========== 输出层 ==========
    errors: list[str]         # 执行过程中的错误信息
    final_answer: str         # 最终返回给用户的回答
