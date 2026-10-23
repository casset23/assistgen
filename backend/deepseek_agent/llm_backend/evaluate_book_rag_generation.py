from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any

from app.services.book_rag_chat_service import BookRAGChatService

DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "data" / 'book_rag_generation_eval.json'
DEFAULT_QUERIES_PATH = Path(__file__).resolve().parent / "data" / 'book_rag_retrieval_queries.json'
DEFAULT_QUERIES = [
    {
        "query": "什么是导航？",
        "expected_keywords": ["导航", "定义"],
        "required_keywords": ["导航"],
        "intent_keywords": ["定义", "是", "概念"],
    },
    {
        "query": "惯性导航系统有什么特点？",
        "expected_keywords": ["惯性导航", "特点"],
        "required_keywords": ["惯性导航"],
        "intent_keywords": ["特点", "优点", "缺点", "概念"],
    },
    {
        "query": "导航和制导有什么区别？",
        "expected_keywords": ["导航", "制导", "区别"],
        "required_keywords": ["导航", "制导"],
        "intent_keywords": ["区别", "不同", "联系"],
    },
    {
        "query": "什么是地理坐标系？",
        "expected_keywords": ["地理坐标系"],
        "required_keywords": ["地理坐标系"],
        "intent_keywords": ["定义", "是", "概念"],
    },
    {
        "query": "墨卡托海图有什么特点？",
        "expected_keywords": ["墨卡托", "海图"],
        "required_keywords": ["墨卡托", "海图"],
        "intent_keywords": ["特点", "性质", "定义"],
    },
]

DEFINITION_CUES = ("定义", "是什么", "什么是", "概念", "是指", "称为")
COMPARISON_CUES = ("区别", "不同", "联系", "比较")
FEATURE_CUES = ("特点", "特征", "优点", "缺点", "性质")
LOW_CONFIDENCE_PATTERNS = (
    "暂无充分依据",
    "信息不足",
    "无法确定",
    "未检索到",
)
WHITESPACE_PATTERN = re.compile(r"\s+")


class GenerationQualityEvaluator:
    def __init__(
        self,
        index_id: str,
        top_k: int = 3,
        document_type: str = "textbook",
        noise_level: str = "medium",
        answer_style: str = "balanced",
        retrieval_mode: str = "balanced",
    ):
        self.index_id = index_id
        self.top_k = top_k
        self.chat_service = BookRAGChatService(
            document_type=document_type,
            noise_level=noise_level,
            answer_style=answer_style,
            retrieval_mode=retrieval_mode,
        )

    async def evaluate_queries(self, queries: list[dict[str, Any]]) -> dict[str, Any]:
        results = []
        for item in queries:
            results.append(await self._evaluate_single_query(item))

        overview = self._build_overview(results)
        suggestions = self._build_suggestions(overview, results)
        return {
            "index_id": self.index_id,
            "top_k": self.top_k,
            "query_count": len(results),
            "overview": overview,
            "results": results,
            "suggestions": suggestions,
        }

    async def _evaluate_single_query(self, query_item: dict[str, Any]) -> dict[str, Any]:
        query = str(query_item.get("query") or "").strip()
        expected_keywords = [str(x) for x in query_item.get("expected_keywords") or [] if str(x).strip()]
        required_keywords = [str(x) for x in query_item.get("required_keywords") or [] if str(x).strip()]
        intent_keywords = [str(x) for x in query_item.get("intent_keywords") or [] if str(x).strip()]

        generation_result = await self.chat_service.generate(
            query=query,
            index_id=self.index_id,
            top_k=self.top_k,
        )
        answer = self._normalize_text(str(generation_result.get("answer") or ""))
        references = generation_result.get("references") or []
        profile = generation_result.get("profile") or {}

        query_type = self._detect_query_type(query, intent_keywords)
        expected_keyword_hits = [keyword for keyword in expected_keywords if keyword and keyword in answer]
        required_keyword_hits = [keyword for keyword in required_keywords if keyword and keyword in answer]
        intent_hits = [keyword for keyword in intent_keywords if keyword and keyword in answer]
        low_confidence = any(pattern in answer for pattern in LOW_CONFIDENCE_PATTERNS)
        answer_nonempty = bool(answer)
        required_covered = len(required_keyword_hits) == len(required_keywords) if required_keywords else answer_nonempty
        intent_supported = self._intent_supported(query_type, intent_hits, answer)
        answer_supported = required_covered and intent_supported and not low_confidence
        reference_count = len(references)
        top_reference = references[0] if references else {}

        return {
            "query": query,
            "query_type": query_type,
            "answer": answer,
            "profile": profile,
            "reference_count": reference_count,
            "top_reference": {
                "section_title": top_reference.get("section_title"),
                "start_page": top_reference.get("start_page"),
                "end_page": top_reference.get("end_page"),
                "snippet": top_reference.get("snippet"),
            },
            "expected_keywords": expected_keywords,
            "required_keywords": required_keywords,
            "intent_keywords": intent_keywords,
            "expected_keyword_hits": expected_keyword_hits,
            "required_keyword_hits": required_keyword_hits,
            "intent_hits": intent_hits,
            "low_confidence": low_confidence,
            "answer_nonempty": answer_nonempty,
            "required_covered": required_covered,
            "intent_supported": intent_supported,
            "answer_supported": answer_supported,
        }

    def _build_overview(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        if not results:
            return {
                "answer_supported_rate": 0.0,
                "required_coverage_rate": 0.0,
                "intent_support_rate": 0.0,
                "low_confidence_rate": 0.0,
                "avg_reference_count": 0.0,
            }

        total = len(results)
        return {
            "answer_supported_rate": round(sum(1 for item in results if item["answer_supported"]) / total, 3),
            "required_coverage_rate": round(sum(1 for item in results if item["required_covered"]) / total, 3),
            "intent_support_rate": round(sum(1 for item in results if item["intent_supported"]) / total, 3),
            "low_confidence_rate": round(sum(1 for item in results if item["low_confidence"]) / total, 3),
            "avg_reference_count": round(mean(item["reference_count"] for item in results), 3),
        }

    def _build_suggestions(self, overview: dict[str, Any], results: list[dict[str, Any]]) -> list[str]:
        suggestions: list[str] = []
        if overview["answer_supported_rate"] < 0.7:
            suggestions.append("生成回答整体支撑度偏低，建议优先检查 prompt 是否过于保守，或回看检索片段是否被截断过短。")
        if overview["required_coverage_rate"] < 0.8:
            suggestions.append("回答中对核心术语覆盖不足，建议加强“必须先回答主概念”的提示约束。")
        if overview["intent_support_rate"] < 0.8:
            suggestions.append("回答结构与问题意图不够匹配，建议继续区分定义题、特点题、比较题的作答模板。")
        if overview["low_confidence_rate"] > 0.2:
            suggestions.append("保守拒答比例偏高，建议检查 top_k 是否过小，或检索上下文是否缺少直接证据句。")
        if not suggestions:
            suggestions.append("当前生成层已具备联调基础，可以进入接口联调和前端接入阶段，并继续观察具体题型表现。")
        return suggestions

    def _detect_query_type(self, query: str, intent_keywords: list[str]) -> str:
        combined = " ".join([query, *intent_keywords])
        if any(word in combined for word in COMPARISON_CUES):
            return "comparison"
        if any(word in combined for word in FEATURE_CUES):
            return "feature"
        if any(word in combined for word in DEFINITION_CUES):
            return "definition"
        return "general"

    def _intent_supported(self, query_type: str, intent_hits: list[str], answer: str) -> bool:
        if query_type == "comparison":
            return bool(intent_hits) or ("不同" in answer and "联系" in answer)
        if query_type == "feature":
            return bool(intent_hits) or ("特点" in answer or "具有" in answer)
        if query_type == "definition":
            return bool(intent_hits) or ("是" in answer or "是指" in answer)
        return bool(answer)

    def _normalize_text(self, text: str) -> str:
        return WHITESPACE_PATTERN.sub(" ", text).strip()


def load_queries(queries_path: Path | None) -> list[dict[str, Any]]:
    if queries_path and queries_path.exists():
        return json.loads(queries_path.read_text(encoding="utf-8"))
    return DEFAULT_QUERIES


async def main() -> None:
    parser = argparse.ArgumentParser(description="评估教材 RAG 生成质量")
    parser.add_argument("--index-id", type=str, required=True, help="已有索引 ID")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="评测结果输出路径")
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES_PATH, help="查询配置 JSON 路径")
    parser.add_argument("--top-k", type=int, default=3, help="生成前检索的 top-k")
    parser.add_argument("--document-type", type=str, default="textbook", choices=["general", "textbook", "paper", "manual", "faq"], help="简单模式：资料类型")
    parser.add_argument("--noise-level", type=str, default="medium", choices=["low", "medium", "high"], help="简单模式：资料噪声级别")
    parser.add_argument("--answer-style", type=str, default="balanced", choices=["faithful", "balanced", "summary"], help="简单模式：回答风格")
    parser.add_argument("--retrieval-mode", type=str, default="balanced", choices=["stable", "balanced", "aggressive"], help="简单模式：检索风格")
    args = parser.parse_args()

    queries = load_queries(args.queries)
    evaluator = GenerationQualityEvaluator(
        index_id=args.index_id.strip(),
        top_k=args.top_k,
        document_type=args.document_type,
        noise_level=args.noise_level,
        answer_style=args.answer_style,
        retrieval_mode=args.retrieval_mode,
    )
    result = await evaluator.evaluate_queries(queries)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    overview = result["overview"]
    print(f"生成评估完成，index_id={result['index_id']}")
    print(f"[Generate] answer_supported_rate: {overview['answer_supported_rate']}")
    print(f"[Generate] required_coverage_rate: {overview['required_coverage_rate']}")
    print(f"[Generate] intent_support_rate: {overview['intent_support_rate']}")
    print(f"[Generate] low_confidence_rate: {overview['low_confidence_rate']}")
    print(f"结果已写入: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())

