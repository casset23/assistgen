from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from evaluate_book_rag_generation import (
    DEFAULT_OUTPUT_PATH as DEFAULT_GENERATION_OUTPUT_PATH,
    GenerationQualityEvaluator,
    load_queries as load_generation_queries,
)
from evaluate_book_rag_retrieval import (
    DEFAULT_CHUNKS_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_RETRIEVAL_OUTPUT_PATH,
    RetrievalQualityEvaluator,
)

DEFAULT_QUERIES_PATH = Path(__file__).resolve().parent / "data" / 'book_rag_retrieval_queries.json'
DEFAULT_COMBINED_OUTPUT_PATH = Path(__file__).resolve().parent / "data" / 'book_rag_full_eval.json'


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 3)


def _summarize_retrieval_by_query_type(retrieval_result: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in retrieval_result.get("results", {}).get("hybrid_rrf", []):
        query_type = _detect_query_type_from_query_item(item)
        grouped.setdefault(query_type, []).append(item)

    summary: dict[str, Any] = {}
    for query_type, items in grouped.items():
        total = len(items)
        summary[query_type] = {
            "query_count": total,
            "top1_strict_hit_rate": _safe_rate(sum(1 for item in items if item.get("top1_strict_relevant")), total),
            "top3_strict_hit_rate": _safe_rate(sum(1 for item in items if item.get("top3_strict_relevant")), total),
            "top3_usable_rate": _safe_rate(sum(1 for item in items if item.get("top3_usable")), total),
            "avg_strict_relevant_per_query": round(
                sum(int(item.get("strict_relevant_count", 0)) for item in items) / total,
                3,
            ) if total else 0.0,
            "avg_partial_relevant_per_query": round(
                sum(int(item.get("partial_relevant_count", 0)) for item in items) / total,
                3,
            ) if total else 0.0,
        }
    return summary


def _summarize_generation_by_query_type(generation_result: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in generation_result.get("results", []):
        query_type = str(item.get("query_type") or _detect_query_type_from_query_item(item))
        grouped.setdefault(query_type, []).append(item)

    summary: dict[str, Any] = {}
    for query_type, items in grouped.items():
        total = len(items)
        summary[query_type] = {
            "query_count": total,
            "answer_supported_rate": _safe_rate(sum(1 for item in items if item.get("answer_supported")), total),
            "required_coverage_rate": _safe_rate(sum(1 for item in items if item.get("required_covered")), total),
            "intent_support_rate": _safe_rate(sum(1 for item in items if item.get("intent_supported")), total),
            "low_confidence_rate": _safe_rate(sum(1 for item in items if item.get("low_confidence")), total),
            "avg_reference_count": round(
                sum(int(item.get("reference_count", 0)) for item in items) / total,
                3,
            ) if total else 0.0,
        }
    return summary


def _detect_query_type_from_query_item(item: dict[str, Any]) -> str:
    query = str(item.get("query") or "")
    intent_keywords = [str(x) for x in item.get("intent_keywords") or []]
    combined = " ".join([query, *intent_keywords])

    if any(word in combined for word in ("区别", "不同", "联系", "比较", "相同点", "不同点")):
        return "comparison"
    if any(word in combined for word in ("特点", "特征", "优点", "缺点", "性质", "基本要求", "要求")):
        return "feature"
    if any(word in combined for word in ("组成", "包括", "部分", "分类", "分为", "内容")):
        return "structure"
    if any(word in combined for word in ("作用", "用途", "意义", "为什么", "原因")):
        return "purpose"
    if any(word in combined for word in ("方法", "过程", "如何", "条件")):
        return "method"
    if any(word in combined for word in ("定义", "是什么", "什么是", "概念", "是指", "称为")):
        return "definition"
    return "general"


async def main() -> None:
    parser = argparse.ArgumentParser(description="一键运行教材 RAG 检索与生成评测")
    parser.add_argument("--index-id", type=str, required=True, help="已有索引 ID")
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES_PATH, help="查询配置 JSON 路径")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PATH, help="book_rag_chunks.json 路径")
    parser.add_argument("--retrieval-output", type=Path, default=DEFAULT_RETRIEVAL_OUTPUT_PATH, help="检索评测结果输出路径")
    parser.add_argument("--generation-output", type=Path, default=DEFAULT_GENERATION_OUTPUT_PATH, help="生成评测结果输出路径")
    parser.add_argument("--combined-output", type=Path, default=DEFAULT_COMBINED_OUTPUT_PATH, help="汇总结果输出路径")
    parser.add_argument("--retrieval-top-k", type=int, default=5, help="检索评测 top-k")
    parser.add_argument("--generation-top-k", type=int, default=3, help="生成评测 top-k")
    parser.add_argument("--document-type", type=str, default="textbook", choices=["general", "textbook", "paper", "manual", "faq"], help="简单模式：资料类型")
    parser.add_argument("--noise-level", type=str, default="medium", choices=["low", "medium", "high"], help="简单模式：资料噪声级别")
    parser.add_argument("--answer-style", type=str, default="balanced", choices=["faithful", "balanced", "summary"], help="简单模式：回答风格")
    parser.add_argument("--retrieval-mode", type=str, default="balanced", choices=["stable", "balanced", "aggressive"], help="简单模式：检索风格")
    args = parser.parse_args()

    queries = load_generation_queries(args.queries)

    retrieval_evaluator = RetrievalQualityEvaluator(
        index_id=args.index_id.strip(),
        top_k=args.retrieval_top_k,
        chunks_path=args.chunks,
    )
    generation_evaluator = GenerationQualityEvaluator(
        index_id=args.index_id.strip(),
        top_k=args.generation_top_k,
        document_type=args.document_type,
        noise_level=args.noise_level,
        answer_style=args.answer_style,
        retrieval_mode=args.retrieval_mode,
    )

    retrieval_result = await retrieval_evaluator.evaluate_queries(queries)
    generation_result = await generation_evaluator.evaluate_queries(queries)
    retrieval_by_type = _summarize_retrieval_by_query_type(retrieval_result)
    generation_by_type = _summarize_generation_by_query_type(generation_result)

    args.retrieval_output.parent.mkdir(parents=True, exist_ok=True)
    args.retrieval_output.write_text(json.dumps(retrieval_result, ensure_ascii=False, indent=2), encoding="utf-8")
    args.generation_output.parent.mkdir(parents=True, exist_ok=True)
    args.generation_output.write_text(json.dumps(generation_result, ensure_ascii=False, indent=2), encoding="utf-8")

    combined_result = {
        "index_id": args.index_id.strip(),
        "query_count": len(queries),
        "retrieval_top_k": args.retrieval_top_k,
        "generation_top_k": args.generation_top_k,
        "profile": {
            "document_type": args.document_type,
            "noise_level": args.noise_level,
            "answer_style": args.answer_style,
            "retrieval_mode": args.retrieval_mode,
        },
        "overview": {
            "retrieval": retrieval_result.get("overview", {}),
            "generation": generation_result.get("overview", {}),
        },
        "by_query_type": {
            "retrieval": retrieval_by_type,
            "generation": generation_by_type,
        },
        "outputs": {
            "retrieval_output": str(args.retrieval_output),
            "generation_output": str(args.generation_output),
        },
    }

    args.combined_output.parent.mkdir(parents=True, exist_ok=True)
    args.combined_output.write_text(json.dumps(combined_result, ensure_ascii=False, indent=2), encoding="utf-8")

    retrieval_overview = retrieval_result["overview"]["hybrid_rrf"]
    generation_overview = generation_result["overview"]

    print(f"完整评测完成，index_id={args.index_id.strip()}")
    print(
        f"[Retrieval] top1_strict_hit_rate: {retrieval_overview['top1_strict_hit_rate']} | "
        f"top3_strict_hit_rate: {retrieval_overview['top3_strict_hit_rate']} | "
        f"top3_usable_rate: {retrieval_overview['top3_usable_rate']}"
    )
    print(
        f"[Generate] answer_supported_rate: {generation_overview['answer_supported_rate']} | "
        f"required_coverage_rate: {generation_overview['required_coverage_rate']} | "
        f"intent_support_rate: {generation_overview['intent_support_rate']} | "
        f"low_confidence_rate: {generation_overview['low_confidence_rate']}"
    )
    if generation_by_type:
        print("[ByType][Generate]")
        for query_type, metrics in generation_by_type.items():
            print(
                f"  - {query_type}: count={metrics['query_count']}, "
                f"answer_supported_rate={metrics['answer_supported_rate']}, "
                f"low_confidence_rate={metrics['low_confidence_rate']}"
            )
    if retrieval_by_type:
        print("[ByType][Retrieval]")
        for query_type, metrics in retrieval_by_type.items():
            print(
                f"  - {query_type}: count={metrics['query_count']}, "
                f"top1_strict_hit_rate={metrics['top1_strict_hit_rate']}, "
                f"top3_usable_rate={metrics['top3_usable_rate']}"
            )
    print(f"检索结果已写入: {args.retrieval_output}")
    print(f"生成结果已写入: {args.generation_output}")
    print(f"汇总结果已写入: {args.combined_output}")


if __name__ == "__main__":
    asyncio.run(main())

