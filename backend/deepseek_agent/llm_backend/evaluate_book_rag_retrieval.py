from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from app.services.embedding_service import EmbeddingService
from app.services.profile_resolver_service import ProfileResolver, SimpleProfileConfig

DEFAULT_CHUNKS_PATH = Path(__file__).resolve().parent / "data" / 'book_rag_chunks.json'
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "data" / 'book_rag_retrieval_eval.json'
DEFAULT_QUERIES_PATH = Path(__file__).resolve().parent / "data" / 'book_rag_retrieval_queries.json'
DEFAULT_QUERIES = [
    {
        "query": "什么是导航？",
        "expected_keywords": ["导航", "定义"],
        "expected_sections": ["绪论"],
        "required_keywords": ["导航"],
        "intent_keywords": ["定义", "是", "概念"],
    },
    {
        "query": "惯性导航系统有什么特点？",
        "expected_keywords": ["惯性导航", "特点"],
        "expected_sections": ["惯性导航"],
        "required_keywords": ["惯性导航"],
        "intent_keywords": ["特点", "优点", "缺点", "概念"],
    },
    {
        "query": "导航和制导有什么区别？",
        "expected_keywords": ["导航", "制导", "区别"],
        "expected_sections": ["绪论"],
        "required_keywords": ["导航", "制导"],
        "intent_keywords": ["区别", "不同", "联系"],
    },
    {
        "query": "什么是地理坐标系？",
        "expected_keywords": ["地理坐标系"],
        "expected_sections": ["坐标系"],
        "required_keywords": ["地理坐标系"],
        "intent_keywords": ["定义", "是", "概念"],
    },
    {
        "query": "墨卡托海图有什么特点？",
        "expected_keywords": ["墨卡托", "海图"],
        "expected_sections": ["海图"],
        "required_keywords": ["墨卡托", "海图"],
        "intent_keywords": ["特点", "性质", "定义"],
    },
]

DEFINITION_CUES = ("定义", "是", "指", "概念", "称为", "叫作")
COMPARISON_CUES = ("区别", "不同", "联系", "比较")
FEATURE_CUES = ("特点", "特征", "优点", "缺点", "性质")
EXPLANATION_CUES = ("包括", "分为", "组成", "作用", "过程", "条件", "方法", "用途", "分类")
LOW_VALUE_SECTION_KEYWORDS = ("参考文献", "价值导向", "哲学", "科学思想")
LOW_VALUE_SNIPPET_PATTERNS = (
    r"\[[0-9]+\]\s*",
    r"出版社",
    r"译\.\s*北京",
    r"参考文献",
)
RRF_K = 60
SPARSE_CANDIDATE_MULTIPLIER = 4
TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_.-]+")


class RetrievalQualityEvaluator:
    def __init__(
        self,
        index_id: str,
        top_k: int = 5,
        chunks_path: Path | None = None,
        retrieval_profile_path: str | None = None,
        chunking_config_path: str | None = None,
    ):
        self.index_id = index_id
        self.top_k = top_k
        self.embedding_service = EmbeddingService(
            chunking_config_path=chunking_config_path,
            retrieval_profile_path=retrieval_profile_path,
        )
        self.chunks_path = chunks_path
        self._sparse_documents: list[dict[str, Any]] = []
        self._idf_map: dict[str, float] = {}
        self._sparse_ready = False

    async def evaluate_queries(self, queries: list[dict[str, Any]]) -> dict[str, Any]:
        if self.chunks_path is not None:
            self._prepare_sparse_index(self.chunks_path)

        dense_results: list[dict[str, Any]] = []
        hybrid_results: list[dict[str, Any]] = []
        for item in queries:
            dense_result = await self._evaluate_single_query(item, mode="dense")
            hybrid_result = await self._evaluate_single_query(item, mode="hybrid")
            dense_results.append(dense_result)
            hybrid_results.append(hybrid_result)

        dense_overview = self._build_overview(dense_results)
        hybrid_overview = self._build_overview(hybrid_results)
        suggestions = self._build_suggestions(hybrid_overview, hybrid_results)
        return {
            "index_id": self.index_id,
            "top_k": self.top_k,
            "query_count": len(dense_results),
            "overview": {
                "dense": dense_overview,
                "hybrid_rrf": hybrid_overview,
                "delta": self._build_overview_delta(dense_overview, hybrid_overview),
            },
            "results": {
                "dense": dense_results,
                "hybrid_rrf": hybrid_results,
            },
            "suggestions": suggestions,
        }

    async def _evaluate_single_query(self, query_item: dict[str, Any], mode: str) -> dict[str, Any]:
        query = str(query_item.get("query") or "").strip()
        expected_keywords = [str(x) for x in query_item.get("expected_keywords") or [] if str(x).strip()]
        expected_sections = [str(x) for x in query_item.get("expected_sections") or [] if str(x).strip()]
        required_keywords = [str(x) for x in query_item.get("required_keywords") or [] if str(x).strip()]
        intent_keywords = [str(x) for x in query_item.get("intent_keywords") or [] if str(x).strip()]

        retrieval_results = await self._retrieve(query=query, mode=mode)

        judged_hits = []
        for rank, item in enumerate(retrieval_results, start=1):
            metadata = item.get("metadata") or {}
            content = item.get("content") or ""
            section_title = str(metadata.get("section_title") or "")
            judgment = self._judge_hit(
                query=query,
                content=content,
                section_title=section_title,
                expected_keywords=expected_keywords,
                expected_sections=expected_sections,
                required_keywords=required_keywords,
                intent_keywords=intent_keywords,
            )
            judged_hits.append(
                {
                    "rank": rank,
                    "score": item.get("score"),
                    "rerank_score": item.get("rerank_score"),
                    "rerank_bonus": item.get("rerank_bonus"),
                    "rerank_penalty": item.get("rerank_penalty"),
                    "rerank_reasons": item.get("rerank_reasons", []),
                    "retrieval_source": item.get("retrieval_source"),
                    "dense_rank": item.get("dense_rank"),
                    "sparse_rank": item.get("sparse_rank"),
                    "matched_tokens": item.get("matched_tokens", []),
                    "compression_applied": item.get("compression_applied", False),
                    "dedup_count": item.get("dedup_count", 1),
                    "duplicate_chunk_ids": item.get("duplicate_chunk_ids", []),
                    "original_snippet": str(item.get("original_content") or content)[:240],
                    "chunk_id": metadata.get("chunk_id"),
                    "section_title": section_title,
                    "page_start": metadata.get("page_start"),
                    "page_end": metadata.get("page_end"),
                    **judgment,
                    "snippet": content[:240],
                }
            )

        top1_strict_relevant = bool(judged_hits and judged_hits[0]["relevance_level"] == "strict")
        top3_strict_relevant = any(item["relevance_level"] == "strict" for item in judged_hits[:3])
        top3_usable = any(item["relevance_level"] in {"strict", "partial"} for item in judged_hits[:3])
        strict_relevant_count = sum(1 for item in judged_hits if item["relevance_level"] == "strict")
        partial_relevant_count = sum(1 for item in judged_hits if item["relevance_level"] == "partial")

        return {
            "mode": mode,
            "query": query,
            "expected_keywords": expected_keywords,
            "expected_sections": expected_sections,
            "required_keywords": required_keywords,
            "intent_keywords": intent_keywords,
            "top1_strict_relevant": top1_strict_relevant,
            "top3_strict_relevant": top3_strict_relevant,
            "top3_usable": top3_usable,
            "strict_relevant_count": strict_relevant_count,
            "partial_relevant_count": partial_relevant_count,
            "hits": judged_hits,
        }

    async def _retrieve(self, query: str, mode: str) -> list[dict[str, Any]]:
        dense_candidate_k = max(self.top_k, self.top_k * SPARSE_CANDIDATE_MULTIPLIER)
        if mode == "dense":
            dense_results = await self.embedding_service.search_index(
                index_id=self.index_id,
                query=query,
                top_k=dense_candidate_k,
            )
            return dense_results[: self.top_k]

        hybrid_results = await self.embedding_service.hybrid_search_index(
            index_id=self.index_id,
            query=query,
            top_k=self.top_k,
        )
        return hybrid_results[: self.top_k]

    def _prepare_sparse_index(self, chunks_path: Path) -> None:
        if self._sparse_ready:
            return

        payload = json.loads(chunks_path.read_text(encoding="utf-8"))
        chunks = payload.get("chunks") or []
        documents: list[dict[str, Any]] = []
        doc_freq: Counter[str] = Counter()

        for idx, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                continue
            text = str(chunk.get("text") or "").strip()
            if not text:
                continue
            section_title = str(chunk.get("section_title") or "")
            doc_text = f"{section_title}\n{text}".strip()
            tokens = self._tokenize(doc_text)
            if not tokens:
                continue
            token_counts = Counter(tokens)
            documents.append(
                {
                    "doc_id": str(chunk.get("chunk_id") or idx),
                    "text": text,
                    "metadata": {
                        "chunk_id": str(chunk.get("chunk_id") or idx),
                        "section_title": section_title,
                        "page_start": chunk.get("start_page"),
                        "page_end": chunk.get("end_page"),
                    },
                    "token_counts": token_counts,
                    "token_len": sum(token_counts.values()),
                }
            )
            doc_freq.update(set(token_counts.keys()))

        total_docs = max(len(documents), 1)
        self._idf_map = {
            token: math.log((total_docs + 1) / (freq + 1)) + 1.0
            for token, freq in doc_freq.items()
        }
        self._sparse_documents = documents
        self._sparse_ready = True

    def _sparse_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        if not self._sparse_ready:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        unique_query_tokens = list(dict.fromkeys(query_tokens))
        scored: list[dict[str, Any]] = []
        for document in self._sparse_documents:
            score = 0.0
            matched_tokens: list[str] = []
            token_counts: Counter[str] = document["token_counts"]
            for token in unique_query_tokens:
                tf = token_counts.get(token, 0)
                if tf <= 0:
                    continue
                idf = self._idf_map.get(token, 1.0)
                score += (1.0 + math.log(tf)) * idf
                matched_tokens.append(token)

            if not matched_tokens:
                continue

            normalized_score = score / (1.0 + math.log(document["token_len"] + 1))
            scored.append(
                {
                    "score": round(normalized_score, 6),
                    "content": document["text"],
                    "metadata": document["metadata"],
                    "matched_tokens": matched_tokens,
                    "retrieval_source": "sparse",
                }
            )

        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

    def _fuse_with_rrf(self, dense_results: list[dict[str, Any]], sparse_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        fused: dict[str, dict[str, Any]] = {}

        for rank, item in enumerate(dense_results, start=1):
            metadata = item.get("metadata") or {}
            doc_id = str(metadata.get("chunk_id") or rank)
            entry = fused.setdefault(
                doc_id,
                {
                    "score": 0.0,
                    "content": item.get("content", ""),
                    "metadata": metadata,
                    "dense_rank": None,
                    "sparse_rank": None,
                    "retrieval_source": "dense",
                },
            )
            entry["score"] += 1.0 / (RRF_K + rank)
            entry["dense_rank"] = rank

        for rank, item in enumerate(sparse_results, start=1):
            metadata = item.get("metadata") or {}
            doc_id = str(metadata.get("chunk_id") or rank)
            entry = fused.setdefault(
                doc_id,
                {
                    "score": 0.0,
                    "content": item.get("content", ""),
                    "metadata": metadata,
                    "dense_rank": None,
                    "sparse_rank": None,
                    "retrieval_source": "sparse",
                },
            )
            entry["score"] += 1.0 / (RRF_K + rank)
            entry["sparse_rank"] = rank
            if entry.get("retrieval_source") == "dense":
                entry["retrieval_source"] = "hybrid"

        fused_results = list(fused.values())
        fused_results.sort(key=lambda item: item["score"], reverse=True)
        return fused_results

    def _tokenize(self, text: str) -> list[str]:
        tokens = TOKEN_PATTERN.findall(text)
        expanded: list[str] = []
        for token in tokens:
            normalized = token.strip().lower()
            if not normalized:
                continue
            expanded.append(normalized)
            if len(normalized) >= 4 and all('\u4e00' <= char <= '\u9fff' for char in normalized):
                expanded.extend(normalized[i:i + 2] for i in range(len(normalized) - 1))
        return expanded

    def _judge_hit(
        self,
        query: str,
        content: str,
        section_title: str,
        expected_keywords: list[str],
        expected_sections: list[str],
        required_keywords: list[str],
        intent_keywords: list[str],
    ) -> dict[str, Any]:
        text_to_check = f"{section_title}\n{content}"
        expected_keyword_hits = [keyword for keyword in expected_keywords if keyword in text_to_check]
        required_keyword_hits = [keyword for keyword in required_keywords if keyword in text_to_check]
        section_hits = [section for section in expected_sections if section in section_title]
        intent_hits = [keyword for keyword in intent_keywords if keyword in text_to_check]
        definition_cues = [cue for cue in DEFINITION_CUES if cue in content]
        comparison_cues = [cue for cue in COMPARISON_CUES if cue in content]
        feature_cues = [cue for cue in FEATURE_CUES if cue in content]
        explanation_cues = [cue for cue in EXPLANATION_CUES if cue in content]
        low_value = self._is_low_value_hit(section_title, content)

        query_type = self._detect_query_type(query, intent_keywords)
        has_required = bool(required_keyword_hits) if required_keywords else bool(expected_keyword_hits)
        has_all_required = all(keyword in text_to_check for keyword in required_keywords) if required_keywords else bool(expected_keyword_hits)
        has_section = bool(section_hits)
        has_intent = bool(intent_hits)
        has_query_cue = self._has_query_specific_cue(
            query_type,
            definition_cues,
            comparison_cues,
            feature_cues,
            explanation_cues,
        )
        content_length = len(content.strip())
        has_substantive_content = content_length >= 60
        expected_hit_count = len(expected_keyword_hits)

        if low_value:
            relevance_level = "irrelevant"
            reason = "low_value_chunk"
        elif has_all_required and has_intent and has_query_cue and has_substantive_content:
            relevance_level = "strict"
            reason = "all_required_and_intent_matched"
        elif has_all_required and has_section and has_query_cue and has_substantive_content:
            relevance_level = "strict"
            reason = "all_required_and_section_matched"
        elif has_required and has_section and has_intent:
            relevance_level = "partial"
            reason = "required_section_intent_matched"
        elif has_required and has_intent and has_query_cue:
            relevance_level = "partial"
            reason = "required_and_intent_matched_but_not_complete"
        elif has_required and has_section:
            relevance_level = "partial"
            reason = "required_and_section_matched"
        elif expected_hit_count >= 2 and has_query_cue and has_substantive_content:
            relevance_level = "partial"
            reason = "multi_expected_keywords_with_query_cue"
        else:
            relevance_level = "irrelevant"
            reason = "insufficient_evidence"

        return {
            "expected_keyword_hits": expected_keyword_hits,
            "required_keyword_hits": required_keyword_hits,
            "section_hits": section_hits,
            "intent_hits": intent_hits,
            "query_type": query_type,
            "low_value": low_value,
            "relevance_level": relevance_level,
            "reason": reason,
        }

    def _detect_query_type(self, query: str, intent_keywords: list[str]) -> str:
        combined = f"{query} {' '.join(intent_keywords)}"
        if any(word in combined for word in COMPARISON_CUES):
            return "comparison"
        if any(word in combined for word in FEATURE_CUES):
            return "feature"
        if any(word in combined for word in EXPLANATION_CUES):
            return "explanation"
        return "definition"

    def _has_query_specific_cue(
        self,
        query_type: str,
        definition_cues: list[str],
        comparison_cues: list[str],
        feature_cues: list[str],
        explanation_cues: list[str],
    ) -> bool:
        if query_type == "comparison":
            return bool(comparison_cues)
        if query_type == "feature":
            return bool(feature_cues)
        if query_type == "explanation":
            return bool(explanation_cues or definition_cues or feature_cues)
        return bool(definition_cues)

    def _is_low_value_hit(self, section_title: str, content: str) -> bool:
        if any(keyword in section_title for keyword in LOW_VALUE_SECTION_KEYWORDS):
            return True
        snippet = content[:300]
        if sum(1 for pattern in LOW_VALUE_SNIPPET_PATTERNS if re.search(pattern, snippet)) >= 2:
            return True
        return False

    def _build_overview(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        if not results:
            return {
                "top1_strict_hit_rate": 0.0,
                "top3_strict_hit_rate": 0.0,
                "top3_usable_rate": 0.0,
                "avg_strict_relevant_per_query": 0.0,
                "avg_partial_relevant_per_query": 0.0,
            }

        return {
            "top1_strict_hit_rate": round(sum(1 for item in results if item["top1_strict_relevant"]) / len(results), 3),
            "top3_strict_hit_rate": round(sum(1 for item in results if item["top3_strict_relevant"]) / len(results), 3),
            "top3_usable_rate": round(sum(1 for item in results if item["top3_usable"]) / len(results), 3),
            "avg_strict_relevant_per_query": round(mean(item["strict_relevant_count"] for item in results), 3),
            "avg_partial_relevant_per_query": round(mean(item["partial_relevant_count"] for item in results), 3),
        }

    def _build_overview_delta(self, dense: dict[str, float], hybrid: dict[str, float]) -> dict[str, float]:
        keys = [
            "top1_strict_hit_rate",
            "top3_strict_hit_rate",
            "top3_usable_rate",
            "avg_strict_relevant_per_query",
            "avg_partial_relevant_per_query",
        ]
        return {key: round(hybrid.get(key, 0.0) - dense.get(key, 0.0), 3) for key in keys}

    def _build_suggestions(self, overview: dict[str, Any], results: list[dict[str, Any]]) -> list[str]:
        suggestions: list[str] = []
        if overview["top1_strict_hit_rate"] < 0.6:
            suggestions.append("严格口径下 Top1 偏低，说明定义句或特点句没有稳定排到最前，优先优化精确召回。")
        if overview["top3_strict_hit_rate"] < 0.8:
            suggestions.append("严格口径下 Top3 仍不稳，建议加强标题信号进入 embedding 文本，并检查术语类 chunk 是否切散。")
        if overview["top3_usable_rate"] < 0.9:
            suggestions.append("Top3 可用率不高，说明有些问题前三条都缺乏可回答证据，建议先过滤低价值 chunk。")
        if any(any(hit["low_value"] for hit in item["hits"][:3]) for item in results):
            suggestions.append("前三命中里出现低价值 chunk，建议在入库前过滤参考文献、价值导向、哲学感想等内容。")
        if not suggestions:
            suggestions.append("严格评估下检索已具备可用性，可以进入问答联调，同时继续观察具体术语类问题。")
        return suggestions


async def ensure_index(chunks_path: Path) -> str:
    embedding_service = EmbeddingService()
    result = await embedding_service.create_embeddings_from_book_rag_chunks(str(chunks_path))
    return str(result["index_id"])


async def main() -> None:
    parser = argparse.ArgumentParser(description="评估教材 RAG 检索质量")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PATH, help="book_rag_chunks.json 路径")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="评估结果输出路径")
    parser.add_argument("--top-k", type=int, default=5, help="每个问题检索的 top-k")
    parser.add_argument("--index-id", type=str, default="", help="已有索引 ID；不传则根据 chunks 自动创建")
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES_PATH, help="自定义查询配置 JSON 路径；不存在时回退到内置题集")
    parser.add_argument("--retrieval-profile", type=str, default="", help="显式指定 retrieval profile 文件路径")
    parser.add_argument("--chunking-config", type=str, default="", help="显式指定 chunking config 文件路径")
    parser.add_argument("--document-type", type=str, default="general", choices=["general", "textbook", "paper", "manual", "faq"], help="简单模式：资料类型")
    parser.add_argument("--noise-level", type=str, default="medium", choices=["low", "medium", "high"], help="简单模式：资料噪声级别")
    parser.add_argument("--answer-style", type=str, default="balanced", choices=["faithful", "balanced", "summary"], help="简单模式：回答风格")
    parser.add_argument("--retrieval-mode", type=str, default="balanced", choices=["stable", "balanced", "aggressive"], help="简单模式：检索风格")
    args = parser.parse_args()

    queries = DEFAULT_QUERIES
    if args.queries and args.queries.exists():
        queries = json.loads(args.queries.read_text(encoding="utf-8"))

    resolved_paths = ProfileResolver.resolve(
        SimpleProfileConfig(
            document_type=args.document_type,
            noise_level=args.noise_level,
            answer_style=args.answer_style,
            retrieval_mode=args.retrieval_mode,
        )
    )
    retrieval_profile_path = args.retrieval_profile.strip() or resolved_paths.retrieval_profile_path
    chunking_config_path = args.chunking_config.strip() or resolved_paths.chunking_config_path

    index_id = args.index_id.strip() or await ensure_index(args.chunks)
    evaluator = RetrievalQualityEvaluator(
        index_id=index_id,
        top_k=args.top_k,
        chunks_path=args.chunks,
        retrieval_profile_path=retrieval_profile_path,
        chunking_config_path=chunking_config_path,
    )
    result = await evaluator.evaluate_queries(queries)
    result["chunks_path"] = str(args.chunks)
    result["resolved_profiles"] = {
        "retrieval_profile_path": retrieval_profile_path,
        "chunking_config_path": chunking_config_path,
        "document_type": args.document_type,
        "noise_level": args.noise_level,
        "answer_style": args.answer_style,
        "retrieval_mode": args.retrieval_mode,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    dense_overview = result["overview"]["dense"]
    hybrid_overview = result["overview"]["hybrid_rrf"]
    delta = result["overview"]["delta"]

    print(f"检索评估完成，index_id={index_id}")
    print(f"[Dense] 严格口径 Top1 命中率: {dense_overview['top1_strict_hit_rate']}")
    print(f"[Dense] 严格口径 Top3 命中率: {dense_overview['top3_strict_hit_rate']}")
    print(f"[Dense] Top3 可用率: {dense_overview['top3_usable_rate']}")
    print(f"[Hybrid+RRF] 严格口径 Top1 命中率: {hybrid_overview['top1_strict_hit_rate']}")
    print(f"[Hybrid+RRF] 严格口径 Top3 命中率: {hybrid_overview['top3_strict_hit_rate']}")
    print(f"[Hybrid+RRF] Top3 可用率: {hybrid_overview['top3_usable_rate']}")
    print(f"[Delta] Top1: {delta['top1_strict_hit_rate']}, Top3: {delta['top3_strict_hit_rate']}, Top3可用率: {delta['top3_usable_rate']}")
    print(f"结果已写入: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())

