"""Small, explicit smoke set; not a clinical or production accuracy benchmark."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graphs.node import LAYA_QUESTIONS, _laya_router, supervisor_decide

CASES = [
    ("你好", False, "knowledge"), ("现在几点了", False, "knowledge"),
    ("你能做什么", False, "knowledge"), ("CT 是什么", False, "knowledge"),
    ("请检测这张图片中的物体", False, "image_analysis"),
    ("分割上传图片中的目标", False, "image_analysis"),
    ("分析上传的腹部 CT 并分割脾脏", False, "image_analysis"),
    ("这些分割测量结果是什么意思", True, "result_explanation"),
    ("报告里的连通区域数代表病灶数量吗", True, "result_explanation"),
    ("刚才计算的体积是多少", True, "result_explanation"),
    ("根据刚才的分割结果生成报告", True, "report"),
    ("请把这个病例的定量结果整理成报告", True, "report"),
    ("帮我处理一下", False, "clarification"),
    ("它怎么样", False, "clarification"),
]


def main():
    agent = _laya_router()
    results = []
    for query, context, expected in CASES:
        answer = agent.predict({"query": query, "recent_requests": [], "has_measurements": context}, LAYA_QUESTIONS)["answers"]["workflow"]
        from unittest.mock import patch
        with patch("graphs.node._laya_router", return_value=agent):
            decision = supervisor_decide({"user_query":query,"case_context":{"measurements":True} if context else {},"messages":[]})
        expected_route = {"knowledge":"react","image_analysis":"plan_execute","result_explanation":"react","report":"plan_llm","clarification":"clarify"}[expected]
        results.append({"query":query,"has_measurements":context,"expected":expected,"expected_route":expected_route,"actual_route":decision["next_agent"],**answer})
    result = {"scope":"14 hand-written smoke examples, not an independent validation set",
              "raw_correct":sum(r["expected"] == r["choice"] for r in results), "effective_correct":sum(r["expected_route"] == r["actual_route"] for r in results), "total":len(results), "results":results}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
