from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any

from app.services.book_rag.document_parser_service import DocumentParserService
from app.services.book_rag.mineru_book_rag_service import MinerUBookRAGService
from app.services.embedding_service import EmbeddingService


DEFAULT_PARSED_PATH = Path(__file__).resolve().parent / "data" / '15281823.ocr.first40.cleaned.json'
DEFAULT_CHUNKS_PATH = Path(__file__).resolve().parent / "data" / '15281823.first40.chunks.debug.json'
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "data" / '15281823.first40.eval.json'

SECTION_TITLE_PATTERN = re.compile(
    r"^(?:第[一二三四五六七八九十百千万0-9]+[章节编部分]\s*.+|[一二三四五六七八九十]+、.+|[（(][一二三四五六七八九十0-9]+[)）].+)$"
)
ENUMERATION_PATTERN = re.compile(r"^[（(]?[0-9一二三四五六七八九十]+[)）、.]")
LONG_ENGLISH_TOKEN_PATTERN = re.compile(r"\b[a-z]{10,}\b")
SUSPICIOUS_JOINED_ENGLISH_PATTERN = re.compile(
    r"(?:coordinate|coordinates|navigation|information|graphic|surface|point|chart|cloud|system|apparatus|"
    r"meteorological|aeronautical|geographic|cylindrical|conical|strapdown|inertial){2,}"
)
NATURAL_LONG_ENGLISH_WORD_PATTERN = re.compile(
    r"^(?:"
    r"information|international|navigation|electronic|automatic|communication|communications|"
    r"configuration|management|application|applications|environment|environments|"
    r"performance|technology|technologies|engineering|processing|controller|controllers|"
    r"measurement|measurements|architecture|architectures|probability|distribution|"
    r"optimization|algorithm|algorithms|simulation|simulations|instrumentation|"
    r"transmission|transportation|identification|identifications|characteristic|characteristics"
    r")$"
)
FORMULA_MARKER_PATTERN = re.compile(r"【公式】|【公式编号】|[=≈≠≤≥∑∏∫√∞±×÷∂∇]")
BAD_PUNCTUATION_PATTERN = re.compile(r"。。+|，，+|！！+|？？+|；；+|::+|，，|\.\.+")
SUSPICIOUS_TRAILING_CHAR_PATTERN = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]$")
SENTENCE_END_PATTERN = re.compile(r"[。！？；：.!?;:）)】]$")


class ParseQualityEvaluator:
    def __init__(self, parsed_result: dict[str, Any], chunk_result: dict[str, Any] | None = None):
        self.parsed_result = parsed_result
        self.chunk_result = chunk_result
        self.pages = parsed_result.get("pages", [])
        self.chunks = (chunk_result or {}).get("chunks", [])
        self._joined_token_examples: list[str] = []

    def evaluate(self) -> dict[str, Any]:
        page_metrics = [self._evaluate_page(page) for page in self.pages if (page.get("text") or "").strip()]
        chunk_metrics = [self._evaluate_chunk(chunk) for chunk in self.chunks if (chunk.get("text") or "").strip()]

        scores = {
            "integrity": self._score_integrity(page_metrics, chunk_metrics),
            "cleanliness": self._score_cleanliness(page_metrics, chunk_metrics),
            "term_readability": self._score_term_readability(page_metrics, chunk_metrics),
            "structure_fidelity": self._score_structure_fidelity(page_metrics, chunk_metrics),
            "rag_utility": self._score_rag_utility(page_metrics, chunk_metrics),
        }
        total_score = round(sum(scores.values()), 2)

        summary = {
            "pages": self._build_page_summary(page_metrics),
            "chunks": self._build_chunk_summary(chunk_metrics),
        }
        grade = self._grade(total_score)
        suggestions = self._build_suggestions(summary, scores)

        return {
            "score": {
                "total": total_score,
                **scores,
                "grade": grade,
            },
            "summary": summary,
            "suggestions": suggestions,
            "debug": self._build_debug_summary(page_metrics, chunk_metrics),
        }

    def _evaluate_page(self, page: dict[str, Any]) -> dict[str, Any]:
        text = (page.get("text") or "").strip()
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        return {
            "page": page.get("page"),
            "char_count": len(text),
            "line_count": len(lines),
            "joined_english_tokens": self._count_joined_english_tokens(text),
            "bad_punctuation_count": len(BAD_PUNCTUATION_PATTERN.findall(text)),
            "blank_line_count": text.count("\n\n"),
            "title_count": sum(1 for line in lines if self._is_section_title(line)),
            "enumeration_count": sum(1 for line in lines if ENUMERATION_PATTERN.match(line)),
            "formula_marker_count": len(FORMULA_MARKER_PATTERN.findall(text)),
            "ends_abruptly": self._ends_abruptly(text),
            "short_noise_line_count": sum(1 for line in lines if self._is_short_noise_line(line)),
            "extract_method": page.get("extract_method"),
        }

    def _evaluate_chunk(self, chunk: dict[str, Any]) -> dict[str, Any]:
        text = (chunk.get("text") or "").strip()
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        metadata = chunk.get("metadata") or {}
        trailing_line = lines[-1] if lines else ""
        return {
            "chunk_index": metadata.get("chunk_index"),
            "char_count": len(text),
            "line_count": len(lines),
            "page_span": len(metadata.get("page_numbers") or []),
            "joined_english_tokens": self._count_joined_english_tokens(text),
            "bad_punctuation_count": len(BAD_PUNCTUATION_PATTERN.findall(text)),
            "title_count": sum(1 for line in lines if self._is_section_title(line)),
            "enumeration_count": sum(1 for line in lines if ENUMERATION_PATTERN.match(line)),
            "formula_marker_count": len(FORMULA_MARKER_PATTERN.findall(text)),
            "ends_abruptly": self._ends_abruptly(text),
            "likely_catalog_chunk": self._is_catalog_like_chunk(text, lines),
            "ends_with_catalog_entry": self._is_catalog_like_line(trailing_line),
            "too_short_for_rag": len(text) < 180,
            "too_long_for_rag": len(text) > 1800,
            "has_formula": bool(metadata.get("has_formula")),
            "has_figure": bool(metadata.get("has_figure")),
        }

    def _score_integrity(self, page_metrics: list[dict[str, Any]], chunk_metrics: list[dict[str, Any]]) -> float:
        score = 20.0
        substantive_pages = [item for item in page_metrics if item["char_count"] >= 80]
        abrupt_pages = sum(1 for item in substantive_pages if item["ends_abruptly"])
        abrupt_chunks = sum(1 for item in chunk_metrics if item["ends_abruptly"] and not item.get("ends_with_catalog_entry"))
        score -= min(6.0, abrupt_pages * 0.35)
        score -= min(8.0, abrupt_chunks * 1.2)
        if chunk_metrics:
            avg_span = mean(item["page_span"] for item in chunk_metrics)
            if avg_span > 3:
                score -= min(2.0, (avg_span - 3) * 0.8)
        return round(max(0.0, score), 2)

    def _score_cleanliness(self, page_metrics: list[dict[str, Any]], chunk_metrics: list[dict[str, Any]]) -> float:
        score = 20.0
        score -= min(8.0, sum(item["bad_punctuation_count"] for item in page_metrics) * 0.8)
        score -= min(5.0, sum(item["short_noise_line_count"] for item in page_metrics) * 0.5)
        score -= min(4.0, sum(item["bad_punctuation_count"] for item in chunk_metrics) * 0.5)
        return round(max(0.0, score), 2)

    def _score_term_readability(self, page_metrics: list[dict[str, Any]], chunk_metrics: list[dict[str, Any]]) -> float:
        score = 20.0
        joined_tokens = sum(item["joined_english_tokens"] for item in page_metrics) + sum(
            item["joined_english_tokens"] for item in chunk_metrics
        )
        score -= min(14.0, joined_tokens * 0.8)
        formula_pages = [item for item in page_metrics if item["formula_marker_count"] > 0]
        if formula_pages:
            abrupt_formula_pages = sum(1 for item in formula_pages if item["ends_abruptly"])
            score -= min(4.0, abrupt_formula_pages * 1.0)
        return round(max(0.0, score), 2)

    def _score_structure_fidelity(self, page_metrics: list[dict[str, Any]], chunk_metrics: list[dict[str, Any]]) -> float:
        score = 20.0
        total_titles = sum(item["title_count"] for item in page_metrics)
        total_enums = sum(item["enumeration_count"] for item in page_metrics)
        if total_titles == 0:
            score -= 8.0
        if total_enums == 0:
            score -= 4.0
        if chunk_metrics:
            chunks_without_breaks = sum(1 for item in chunk_metrics if item["line_count"] <= 2 and item["char_count"] > 600)
            score -= min(6.0, chunks_without_breaks * 1.5)
        return round(max(0.0, score), 2)

    def _score_rag_utility(self, page_metrics: list[dict[str, Any]], chunk_metrics: list[dict[str, Any]]) -> float:
        if not chunk_metrics:
            return 0.0
        score = 20.0
        too_short = sum(1 for item in chunk_metrics if item["too_short_for_rag"])
        too_long = sum(1 for item in chunk_metrics if item["too_long_for_rag"] and not item.get("likely_catalog_chunk"))
        abrupt_chunks = sum(1 for item in chunk_metrics if item["ends_abruptly"] and not item.get("ends_with_catalog_entry"))
        score -= min(6.0, too_short * 1.2)
        score -= min(6.0, too_long * 1.2)
        score -= min(6.0, abrupt_chunks * 1.0)
        return round(max(0.0, score), 2)

    def _build_page_summary(self, page_metrics: list[dict[str, Any]]) -> dict[str, Any]:
        if not page_metrics:
            return {
                "count": 0,
                "avg_char_count": 0,
                "ocr_page_count": 0,
                "joined_english_tokens": 0,
                "abrupt_end_pages": 0,
                "bad_punctuation_count": 0,
            }

        substantive_pages = [item for item in page_metrics if item["char_count"] >= 80]
        return {
            "count": len(page_metrics),
            "avg_char_count": round(mean(item["char_count"] for item in page_metrics), 2),
            "ocr_page_count": sum(1 for item in page_metrics if item["extract_method"] == "ocr"),
            "joined_english_tokens": sum(item["joined_english_tokens"] for item in page_metrics),
            "abrupt_end_pages": sum(1 for item in substantive_pages if item["ends_abruptly"]),
            "bad_punctuation_count": sum(item["bad_punctuation_count"] for item in page_metrics),
            "short_noise_line_count": sum(item["short_noise_line_count"] for item in page_metrics),
            "title_count": sum(item["title_count"] for item in page_metrics),
            "enumeration_count": sum(item["enumeration_count"] for item in page_metrics),
        }

    def _build_chunk_summary(self, chunk_metrics: list[dict[str, Any]]) -> dict[str, Any]:
        if not chunk_metrics:
            return {
                "count": 0,
                "avg_char_count": 0,
                "avg_page_span": 0,
                "abrupt_end_chunks": 0,
                "too_short_chunks": 0,
                "too_long_chunks": 0,
            }

        return {
            "count": len(chunk_metrics),
            "avg_char_count": round(mean(item["char_count"] for item in chunk_metrics), 2),
            "avg_page_span": round(mean(item["page_span"] for item in chunk_metrics), 2),
            "joined_english_tokens": sum(item["joined_english_tokens"] for item in chunk_metrics),
            "abrupt_end_chunks": sum(1 for item in chunk_metrics if item["ends_abruptly"] and not item.get("ends_with_catalog_entry")),
            "too_short_chunks": sum(1 for item in chunk_metrics if item["too_short_for_rag"]),
            "too_long_chunks": sum(1 for item in chunk_metrics if item["too_long_for_rag"] and not item.get("likely_catalog_chunk")),
            "formula_chunks": sum(1 for item in chunk_metrics if item["has_formula"]),
            "figure_chunks": sum(1 for item in chunk_metrics if item["has_figure"]),
            "catalog_like_chunks": sum(1 for item in chunk_metrics if item.get("likely_catalog_chunk")),
        }

    def _build_suggestions(self, summary: dict[str, Any], scores: dict[str, float]) -> list[str]:
        suggestions: list[str] = []

        if scores["term_readability"] < 15:
            suggestions.append("英文术语粘连较多，建议继续增强中英文词边界修复规则。")

        if summary["chunks"].get("abrupt_end_chunks", 0) >= 2:
            suggestions.append("chunk 末尾截断偏多，建议继续优化按段落/句号切分与跨页拼接。")

        if summary["pages"].get("bad_punctuation_count", 0) >= 3:
            suggestions.append("重复标点和版面噪声仍较明显，建议增加标点归一化与噪声行过滤。")

        if summary["chunks"].get("too_short_chunks", 0) >= 2 or summary["chunks"].get("too_long_chunks", 0) >= 2:
            suggestions.append("chunk 长度分布不稳定，建议调整 chunk 长度阈值与重叠策略。")

        if not suggestions:
            suggestions.append("当前解析质量整体可用，建议继续用更多样本验证稳定性。")

        return suggestions

    def _count_joined_english_tokens(self, text: str) -> int:
        count = 0
        seen_tokens: set[str] = set()
        for token in LONG_ENGLISH_TOKEN_PATTERN.findall(text):
            normalized = token.lower()
            if normalized in seen_tokens:
                continue
            seen_tokens.add(normalized)

            if NATURAL_LONG_ENGLISH_WORD_PATTERN.fullmatch(normalized):
                continue

            if not SUSPICIOUS_JOINED_ENGLISH_PATTERN.search(normalized):
                continue

            count += 1
            if normalized not in self._joined_token_examples and len(self._joined_token_examples) < 20:
                self._joined_token_examples.append(normalized)
        return count

    def _is_catalog_like_line(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        if len(stripped) <= 18 and re.search(r"[·.•…*xXⅰⅴivxlcdm0-9]", stripped):
            return True
        if re.fullmatch(r"[.·•…xXⅰⅴivxlcdm0-9\s]+", stripped):
            return True
        return False

    def _is_catalog_like_chunk(self, text: str, lines: list[str]) -> bool:
        if len(text) < 500 or len(lines) < 12:
            return False

        catalog_lines = sum(1 for line in lines if self._is_catalog_like_line(line))
        title_lines = sum(1 for line in lines if self._is_section_title(line) or ENUMERATION_PATTERN.match(line))
        dotted_lines = sum(1 for line in lines if "·" in line or "…" in line)
        return catalog_lines >= 3 and (title_lines >= 6 or dotted_lines >= 4)

    def _build_debug_summary(self, page_metrics: list[dict[str, Any]], chunk_metrics: list[dict[str, Any]]) -> dict[str, Any]:
        substantive_pages = [item for item in page_metrics if item["char_count"] >= 80]
        abrupt_pages = [item["page"] for item in substantive_pages if item["ends_abruptly"]][:10]
        abrupt_chunks = [item["chunk_index"] for item in chunk_metrics if item["ends_abruptly"]][:10]
        too_long_chunks = [item["chunk_index"] for item in chunk_metrics if item["too_long_for_rag"]][:10]
        return {
            "joined_english_token_examples": self._joined_token_examples,
            "abrupt_page_examples": abrupt_pages,
            "abrupt_chunk_examples": abrupt_chunks,
            "too_long_chunk_examples": too_long_chunks,
        }

    def _ends_abruptly(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False

        last_line = [line.strip() for line in stripped.split("\n") if line.strip()]
        if not last_line:
            return False

        tail = last_line[-1]
        if SENTENCE_END_PATTERN.search(tail):
            return False

        if tail.startswith(("[插图]", "【公式】", "【公式编号】")):
            return False

        if len(tail) <= 24:
            return True

        if len(tail) <= 56 and not re.search(r"[。！？；：.!?;:]", tail):
            return True

        return False

    def _is_short_noise_line(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        if len(stripped) > 4:
            return False
        if self._is_section_title(stripped):
            return False
        if ENUMERATION_PATTERN.match(stripped):
            return False
        if re.search(r"[。！？；：，、]", stripped):
            return False
        return True

    def _is_section_title(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        return bool(SECTION_TITLE_PATTERN.fullmatch(stripped))

    def _grade(self, score: float) -> str:
        if score >= 90:
            return "A"
        if score >= 80:
            return "B"
        if score >= 70:
            return "C"
        if score >= 60:
            return "D"
        return "E"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_page_numbers(start_page: int | None, end_page: int | None) -> list[int]:
    if start_page is None or end_page is None:
        return []
    start = int(start_page)
    end = int(end_page)
    if end < start:
        start, end = end, start
    return list(range(start, end + 1))


def _convert_mineru_chunk_result_to_evaluator_shape(mineru_result: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    sections = mineru_result.get("sections") or []
    chunks = mineru_result.get("chunks") or []

    parsed_pages = []
    for index, section in enumerate(sections, start=1):
        text = (section.get("text") or "").strip()
        if not text:
            continue
        parsed_pages.append(
            {
                "page": index,
                "text": text,
                "raw_text": text,
                "char_count": len(text),
                "raw_char_count": len(text),
                "is_empty": False,
                "extract_method": "mineru",
                "ocr_attempted": False,
                "ocr_text_length": 0,
                "direct_text_length": len(text),
            }
        )

    evaluator_chunks = []
    for index, chunk in enumerate(chunks, start=1):
        start_page = chunk.get("start_page")
        end_page = chunk.get("end_page")
        evaluator_chunks.append(
            {
                "text": chunk.get("text") or "",
                "metadata": {
                    "chunk_index": index,
                    "page_numbers": _build_page_numbers(start_page, end_page),
                    "has_formula": "【公式】" in (chunk.get("text") or "") or "【公式编号】" in (chunk.get("text") or ""),
                    "has_figure": "[插图]" in (chunk.get("text") or "") or "图" in (chunk.get("text") or ""),
                },
            }
        )

    parsed_result = {
        "book_name": mineru_result.get("book_name") or "",
        "file_name": mineru_result.get("source_file_name") or "",
        "file_path": mineru_result.get("source_dir") or "",
        "total_pages": mineru_result.get("stats", {}).get("raw_pages", len(parsed_pages)),
        "non_empty_pages": len(parsed_pages),
        "ocr_pages": 0,
        "ocr_attempted_pages": 0,
        "pages": parsed_pages,
    }
    chunk_result = {
        "file_path": mineru_result.get("source_dir") or "",
        "parsed_output_path": mineru_result.get("source_dir") or "",
        "chunk_count": len(evaluator_chunks),
        "chunks": evaluator_chunks,
    }
    return parsed_result, chunk_result


def _resolve_default_output_paths(pdf_path: Path, max_pages: int | None) -> tuple[Path, Path, Path]:
    suffix = f"first{max_pages}" if max_pages is not None else "all"
    parsed_path = pdf_path.with_name(f"{pdf_path.stem}.ocr.{suffix}.cleaned.json")
    chunk_path = pdf_path.with_name(f"{pdf_path.stem}.{suffix}.chunks.debug.json")
    eval_path = pdf_path.with_name(f"{pdf_path.stem}.{suffix}.eval.json")
    return parsed_path, chunk_path, eval_path


def _resolve_default_mineru_output_paths(mineru_dir: Path) -> tuple[Path, Path, Path]:
    parsed_path = mineru_dir / "mineru_sections.cleaned.json"
    chunk_path = mineru_dir / "book_rag_chunks.json"
    eval_path = mineru_dir / "mineru.eval.json"
    return parsed_path, chunk_path, eval_path


def _rebuild_mineru_results(
    mineru_dir: Path,
    parsed_path: Path,
    chunk_path: Path,
    book_name: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    mineru_service = MinerUBookRAGService()
    _, mineru_result = mineru_service.save_chunks_from_mineru_result(
        mineru_dir=str(mineru_dir),
        output_path=str(chunk_path),
        book_name=book_name,
    )
    parsed_result, evaluator_chunk_result = _convert_mineru_chunk_result_to_evaluator_shape(mineru_result)
    parsed_path.parent.mkdir(parents=True, exist_ok=True)
    parsed_path.write_text(json.dumps(parsed_result, ensure_ascii=False, indent=2), encoding="utf-8")
    return parsed_result, evaluator_chunk_result


def _rebuild_results(
    pdf_path: Path,
    parsed_path: Path,
    chunk_path: Path,
    max_pages: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    parser_service = DocumentParserService(debug_ocr=True)
    _, parsed_result = parser_service.save_parsed_result(
        file_path=str(pdf_path),
        output_path=str(parsed_path),
        max_pages=max_pages,
    )

    embedding_service = EmbeddingService()
    chunk_records = embedding_service._build_chunks_from_parsed_result(parsed_result)
    chunk_result = {
        "file_path": str(pdf_path),
        "parsed_output_path": str(parsed_path),
        "chunk_count": len(chunk_records),
        "chunks": chunk_records,
    }

    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_path.write_text(json.dumps(chunk_result, ensure_ascii=False, indent=2), encoding="utf-8")
    return parsed_result, chunk_result


def _print_summary(report: dict[str, Any], parsed_path: Path, chunk_path: Path | None, output_path: Path | None) -> None:
    score = report["score"]
    page_summary = report["summary"]["pages"]
    chunk_summary = report["summary"]["chunks"]

    print("\n=== 评测摘要 ===")
    print(f"解析文件: {parsed_path}")
    print(f"分块文件: {chunk_path if chunk_path else '未提供'}")
    print(f"总分: {score['total']} / 100")
    print(f"等级: {score['grade']}")
    print(
        "分项: "
        f"完整性={score['integrity']}，"
        f"洁净度={score['cleanliness']}，"
        f"术语可读性={score['term_readability']}，"
        f"结构保真度={score['structure_fidelity']}，"
        f"RAG可用性={score['rag_utility']}"
    )
    print(
        "页统计: "
        f"有效页={page_summary['count']}，"
        f"OCR页={page_summary.get('ocr_page_count', 0)}，"
        f"英文粘连疑似={page_summary.get('joined_english_tokens', 0)}，"
        f"异常结尾页={page_summary.get('abrupt_end_pages', 0)}"
    )
    print(
        "chunk统计: "
        f"chunk数={chunk_summary['count']}，"
        f"平均长度={chunk_summary.get('avg_char_count', 0)}，"
        f"异常结尾chunk={chunk_summary.get('abrupt_end_chunks', 0)}，"
        f"过短={chunk_summary.get('too_short_chunks', 0)}，"
        f"过长={chunk_summary.get('too_long_chunks', 0)}"
    )
    if report.get("suggestions"):
        print("建议:")
        for index, suggestion in enumerate(report["suggestions"], start=1):
            print(f"  {index}. {suggestion}")
    debug = report.get("debug") or {}
    if debug:
        print("调试:")
        print(f"  英文粘连示例: {debug.get('joined_english_token_examples', [])}")
        print(f"  异常结尾页示例: {debug.get('abrupt_page_examples', [])}")
        print(f"  异常结尾chunk示例: {debug.get('abrupt_chunk_examples', [])}")
        print(f"  过长chunk示例: {debug.get('too_long_chunk_examples', [])}")
    if output_path:
        print(f"结果已保存: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF / MinerU 解析结果自动化质量评测脚本")
    parser.add_argument("--pdf", required=False, help="PDF 文件路径；提供后会先重新解析再评测")
    parser.add_argument("--mineru-dir", required=False, help="MinerU 结果目录，目录下需包含 full.md 或 content_list_v2.json")
    parser.add_argument("--book-name", required=False, help="书名，MinerU 重建时可选传入")
    parser.add_argument("--parsed", required=False, default=str(DEFAULT_PARSED_PATH), help="解析结果 JSON 路径")
    parser.add_argument("--chunks", required=False, default=str(DEFAULT_CHUNKS_PATH), help="分块调试 JSON 路径")
    parser.add_argument("--output", required=False, default=str(DEFAULT_OUTPUT_PATH), help="评测结果输出 JSON 路径")
    parser.add_argument("--max-pages", required=False, type=int, default=None, help="PDF 重建时最多解析页数，默认不限页")
    parser.add_argument("--rebuild", action="store_true", help="先重建再执行评测")
    args = parser.parse_args()

    pdf_path = Path(args.pdf) if args.pdf else None
    mineru_dir = Path(args.mineru_dir) if args.mineru_dir else None
    max_pages = args.max_pages

    if pdf_path and mineru_dir:
        raise ValueError("--pdf 与 --mineru-dir 只能二选一")

    if args.rebuild and not pdf_path and not mineru_dir:
        raise ValueError("使用 --rebuild 时必须同时提供 --pdf 或 --mineru-dir")

    if mineru_dir:
        default_parsed_path, default_chunk_path, default_output_path = _resolve_default_mineru_output_paths(mineru_dir)
        parsed_path = Path(args.parsed) if args.parsed != str(DEFAULT_PARSED_PATH) else default_parsed_path
        chunk_path = Path(args.chunks) if args.chunks != str(DEFAULT_CHUNKS_PATH) else default_chunk_path
        output_path = Path(args.output) if args.output != str(DEFAULT_OUTPUT_PATH) else default_output_path
    elif pdf_path:
        default_parsed_path, default_chunk_path, default_output_path = _resolve_default_output_paths(pdf_path, max_pages)
        parsed_path = Path(args.parsed) if args.parsed != str(DEFAULT_PARSED_PATH) else default_parsed_path
        chunk_path = Path(args.chunks) if args.chunks != str(DEFAULT_CHUNKS_PATH) else default_chunk_path
        output_path = Path(args.output) if args.output != str(DEFAULT_OUTPUT_PATH) else default_output_path
    else:
        parsed_path = Path(args.parsed)
        chunk_path = Path(args.chunks) if args.chunks else None
        output_path = Path(args.output) if args.output else None

    if mineru_dir:
        print(f"使用 MinerU 目录: {mineru_dir}")
        if args.rebuild or not parsed_path.exists() or not chunk_path or not chunk_path.exists():
            print("开始根据 MinerU 结果重建清洗结果与分块结果...")
            parsed_result, chunk_result = _rebuild_mineru_results(
                mineru_dir=mineru_dir,
                parsed_path=parsed_path,
                chunk_path=chunk_path,
                book_name=args.book_name,
            )
            print(f"清洗结果已重建: {parsed_path}")
            print(f"分块结果已重建: {chunk_path}")
        else:
            print("检测到现有 MinerU 清洗结果与分块结果，直接使用现有文件评测；如需强制重建请加 --rebuild")
            parsed_result = _load_json(parsed_path)
            chunk_result = _load_json(chunk_path) if chunk_path and chunk_path.exists() else None
    elif pdf_path:
        print(f"使用 PDF: {pdf_path}")
        if args.rebuild or not parsed_path.exists() or not chunk_path or not chunk_path.exists():
            print("开始根据 PDF 重建解析结果与分块结果...")
            parsed_result, chunk_result = _rebuild_results(
                pdf_path=pdf_path,
                parsed_path=parsed_path,
                chunk_path=chunk_path,
                max_pages=max_pages,
            )
            print(f"解析结果已重建: {parsed_path}")
            print(f"分块结果已重建: {chunk_path}")
        else:
            print("检测到现有解析结果与分块结果，直接使用现有文件评测；如需强制重建请加 --rebuild")
            parsed_result = _load_json(parsed_path)
            chunk_result = _load_json(chunk_path) if chunk_path and chunk_path.exists() else None
    else:
        parsed_result = _load_json(parsed_path)
        chunk_result = _load_json(chunk_path) if chunk_path and chunk_path.exists() else None

    evaluator = ParseQualityEvaluator(parsed_result=parsed_result, chunk_result=chunk_result)
    report = evaluator.evaluate()

    print(json.dumps(report, ensure_ascii=False, indent=2))
    _print_summary(report, parsed_path=parsed_path, chunk_path=chunk_path, output_path=output_path)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

