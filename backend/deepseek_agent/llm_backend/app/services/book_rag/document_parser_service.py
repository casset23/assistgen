from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import fitz
import numpy as np
from PyPDF2 import PdfReader
from rapidocr_onnxruntime import RapidOCR

from app.core.logger import get_logger

logger = get_logger(service="document_parser")

FIGURE_CAPTION_PATTERN = re.compile(r"^图\s*\d+(?:\s*[-—．.]\s*\d+)*\s*.*$")
PAGE_HEADER_PATTERN = re.compile(
    r"^(第[一二三四五六七八九十百千万0-9]+章.*|.*（第[一二三四五六七八九十百千万0-9]+版）)$"
)
PAGE_NUMBER_PATTERN = re.compile(r"^[·•●]?[\[(（【-]?\s*\d{1,4}\s*[\])）】.-]?[·•●]?$")
FORMULA_INDEX_PATTERN = re.compile(r"^[(（]?\d+(?:\s*[-—．.]\s*\d+)*[)）.]?$")
FORMULA_LINE_PATTERN = re.compile(
    r"(?:[=≈≠≤≥∑∏∫√∞±×÷∂∇∈∉⊂⊆→←↔]|\b(?:sin|cos|tan|cot|sec|csc|arctan|arcsin|arccos)\b|\d\s*[×xX]\s*10\s*[-−]?[0-9]+|[A-Za-z]\s*=\s*[A-Za-z0-9(])"
)
COORDINATE_DEFINITION_PATTERN = re.compile(
    r"(?:坐标系|坐标原点|X轴|Y轴|Z轴|指向|构成右手坐标系|也称为|适用于|原点在)"
)
SHORT_NOISE_PATTERN = re.compile(r"^[A-Za-z0-9_\-—=+*/\\()\[\]{}'""`~,.·:;!?<>|]{1,12}$")
FIGURE_LABEL_NOISE_PATTERN = re.compile(
    r"^(?:[A-Za-z]{1,3}|[XYZxyz](?:\([A-Za-z]{1,3}\))?|[XYZxyz][A-Za-z]{0,2}|[A-Za-z]{1,3}[A-Za-z0-9]{0,2}|[\u4e00-\u9fff]{1,6}|[A-Za-z]{0,2}[\u4e00-\u9fff]{1,4}[A-Za-z]{0,2})$"
)
FIGURE_LABEL_KEYWORDS = (
    "系统",
    "设备",
    "用户",
    "标准",
    "接口",
    "信号",
    "频率",
    "时间",
    "数据",
    "处理",
    "判断",
    "评估",
    "发送",
    "监测",
    "发播",
    "接收",
    "比对",
    "终端",
    "授时",
    "守时",
    "中心",
    "合作",
    "备份",
    "机动",
    "基准",
    "原子钟",
    "接收机",
    "产生器",
    "分配器",
    "标准源",
)



class DocumentParserService:
    """负责解析教材/书籍类 PDF 文本，支持扫描件 OCR 兜底。"""

    def __init__(self, ocr_min_text_length: int = 30, debug_ocr: bool = False):
        self.ocr_min_text_length = ocr_min_text_length
        self.debug_ocr = debug_ocr
        self.ocr_engine = RapidOCR()

    def parse_pdf(self, file_path: str, max_pages: int | None = None) -> list[dict[str, Any]]:
        """按页解析 PDF，并返回结构化文本结果。"""
        pdf_path = Path(file_path)
        self._validate_pdf(pdf_path)

        logger.info(f"Start parsing pdf: {pdf_path}, max_pages={max_pages or 'all'}")

        text_reader = PdfReader(str(pdf_path))
        pdf_document = fitz.open(str(pdf_path))
        pages: list[dict[str, Any]] = []

        try:
            for index, pdf_page in enumerate(pdf_document, start=1):
                if max_pages is not None and index > max_pages:
                    break

                direct_text = self._extract_text_from_reader_page(text_reader, index)
                use_ocr = self._should_use_ocr(direct_text)

                final_text = direct_text
                extract_method = "text"
                ocr_attempted = False
                ocr_text = ""

                if use_ocr:
                    ocr_attempted = True
                    logger.info(f"Page {index} has insufficient text, using OCR fallback")
                    ocr_text = self._extract_text_with_ocr(pdf_page)
                    if ocr_text:
                        final_text = ocr_text
                        extract_method = "ocr"
                    else:
                        extract_method = "text_empty"

                cleaned_text = self._clean_text(final_text)
                normalized_text = self._normalize_ocr_content(cleaned_text)
                page_result = {
                    "page": index,
                    "text": normalized_text,
                    "raw_text": cleaned_text,
                    "char_count": len(normalized_text),
                    "raw_char_count": len(cleaned_text),
                    "is_empty": normalized_text == "",
                    "extract_method": extract_method,
                    "ocr_attempted": ocr_attempted,
                    "ocr_text_length": len(ocr_text),
                    "direct_text_length": len(direct_text),
                }

                if self.debug_ocr and index <= 3:
                    logger.info(
                        f"OCR debug page {index} -> direct_text_length={page_result['direct_text_length']}, "
                        f"ocr_attempted={page_result['ocr_attempted']}, "
                        f"ocr_text_length={page_result['ocr_text_length']}, "
                        f"extract_method={page_result['extract_method']}, "
                        f"raw_char_count={page_result['raw_char_count']}, "
                        f"char_count={page_result['char_count']}"
                    )

                pages.append(page_result)
        finally:
            pdf_document.close()

        non_empty_pages = sum(1 for page in pages if not page["is_empty"])
        ocr_pages = sum(1 for page in pages if page["extract_method"] == "ocr")
        logger.info(
            f"Finished parsing pdf: {pdf_path.name}, total_pages={len(pages)}, non_empty_pages={non_empty_pages}, ocr_pages={ocr_pages}"
        )

        return pages

    def parse_pdf_with_metadata(
        self,
        file_path: str,
        book_name: str | None = None,
        max_pages: int | None = None,
    ) -> dict[str, Any]:
        """解析 PDF，并附带基础书籍元信息。"""
        pdf_path = Path(file_path)
        pages = self.parse_pdf(file_path, max_pages=max_pages)
        resolved_book_name = book_name or pdf_path.stem

        return {
            "book_name": resolved_book_name,
            "file_name": pdf_path.name,
            "file_path": str(pdf_path),
            "total_pages": len(pages),
            "non_empty_pages": sum(1 for page in pages if not page["is_empty"]),
            "ocr_pages": sum(1 for page in pages if page["extract_method"] == "ocr"),
            "ocr_attempted_pages": sum(1 for page in pages if page["ocr_attempted"]),
            "pages": pages,
        }

    def save_parsed_result(
        self,
        file_path: str,
        output_path: str | None = None,
        book_name: str | None = None,
        max_pages: int | None = None,
    ) -> tuple[Path, dict[str, Any]]:
        """解析 PDF 并将结果保存为 JSON 文件。"""
        parsed_result = self.parse_pdf_with_metadata(
            file_path=file_path,
            book_name=book_name,
            max_pages=max_pages,
        )

        pdf_path = Path(file_path)
        target_path = Path(output_path) if output_path else pdf_path.with_suffix(".ocr.json")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(
            json.dumps(parsed_result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        logger.info(f"Saved parsed pdf result to: {target_path}")
        return target_path, parsed_result

    def _validate_pdf(self, pdf_path: Path) -> None:
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"仅支持 PDF 文件: {pdf_path}")

    def _extract_text_from_reader_page(self, reader: PdfReader, page_number: int) -> str:
        try:
            page = reader.pages[page_number - 1]
            return self._clean_text(page.extract_text() or "")
        except Exception as exc:
            logger.warning(f"Failed to extract direct text from page {page_number}: {exc}")
            return ""

    def _should_use_ocr(self, text: str) -> bool:
        normalized_text = text.strip()
        return len(normalized_text) < self.ocr_min_text_length

    def _extract_text_with_ocr(self, pdf_page: fitz.Page) -> str:
        try:
            pix = pdf_page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            ocr_result, _ = self.ocr_engine(image)

            if not ocr_result:
                if self.debug_ocr and pdf_page.number < 3:
                    logger.warning(f"RapidOCR returned empty result on page {pdf_page.number + 1}")
                return ""

            lines: list[str] = []
            for item in ocr_result:
                if len(item) < 2:
                    continue
                text = str(item[1]).strip()
                if text:
                    lines.append(text)

            ocr_text = "\n".join(lines)

            if self.debug_ocr and pdf_page.number < 3:
                logger.info(
                    f"RapidOCR result page {pdf_page.number + 1} length={len(ocr_text.strip())} "
                    f"preview={repr(ocr_text[:200])}"
                )

            return self._clean_text(ocr_text)
        except Exception as exc:
            logger.error(f"OCR failed on page {pdf_page.number + 1}: {exc}")
            return ""

    def _normalize_ocr_content(self, text: str) -> str:
        if not text:
            return ""

        lines = [line.strip() for line in text.split("\n")]
        filtered_lines = self._remove_page_headers(lines)
        normalized_lines: list[str] = []
        figure_context_remaining = 0
        figure_buffer: list[str] = []
        pre_figure_buffer: list[str] = []

        def flush_figure_buffer() -> None:
            nonlocal figure_buffer, normalized_lines
            if not figure_buffer:
                return

            if self._should_drop_figure_block(figure_buffer):
                figure_buffer = []
                return

            normalized_lines.extend(figure_buffer)
            figure_buffer = []

        def flush_pre_figure_buffer(force_keep: bool = False) -> None:
            nonlocal pre_figure_buffer, normalized_lines
            if not pre_figure_buffer:
                return

            if not force_keep and self._should_drop_pre_figure_block(pre_figure_buffer):
                pre_figure_buffer = []
                return

            normalized_lines.extend(pre_figure_buffer)
            pre_figure_buffer = []

        for line in filtered_lines:
            if not line:
                flush_figure_buffer()
                flush_pre_figure_buffer(force_keep=True)
                if normalized_lines and normalized_lines[-1] != "":
                    normalized_lines.append("")
                if figure_context_remaining > 0:
                    figure_context_remaining -= 1
                continue

            if self._is_page_number(line):
                continue

            if self._is_figure_caption(line):
                flush_figure_buffer()
                flush_pre_figure_buffer()
                normalized_lines.append(f"[插图] {line}")
                figure_context_remaining = 10
                continue

            if figure_context_remaining > 0:
                if self._should_drop_figure_context_line(line):
                    figure_buffer.append(line)
                    figure_context_remaining -= 1
                    continue
                flush_figure_buffer()

            if self._is_formula_index(line):
                flush_pre_figure_buffer(force_keep=True)
                normalized_lines.append(f"【公式编号】{line}")
                continue

            if self._is_formula_line(line):
                flush_pre_figure_buffer(force_keep=True)
                normalized_lines.append(self._mark_formula_line(line))
                continue

            if self._should_buffer_pre_figure_line(line, pre_figure_buffer):
                pre_figure_buffer.append(line)
                continue

            flush_pre_figure_buffer(force_keep=True)
            normalized_lines.append(line)
            if figure_context_remaining > 0:
                figure_context_remaining -= 1

        flush_figure_buffer()
        flush_pre_figure_buffer(force_keep=True)

        collapsed_lines: list[str] = []
        previous_blank = False
        for line in normalized_lines:
            if not line:
                if not previous_blank:
                    collapsed_lines.append("")
                previous_blank = True
                continue

            collapsed_lines.append(line)
            previous_blank = False

        return "\n".join(collapsed_lines).strip()

    def _should_buffer_pre_figure_line(self, line: str, buffer_lines: list[str]) -> bool:
        stripped = line.strip()
        if not stripped:
            return False

        if len(buffer_lines) >= 12:
            return False

        if self._is_formula_index(stripped) or self._is_formula_line(stripped):
            return False

        if re.search(r"[。！？；：，、]", stripped):
            return False

        if len(stripped) > 18:
            return False

        if re.fullmatch(r"[一二三四五六七八九十]+、.*", stripped):
            return False

        if re.fullmatch(r"[（(][一二三四五六七八九十0-9]+[)）].*", stripped):
            return False

        if self._is_likely_figure_label_line(stripped):
            return True

        chinese_chars = re.findall(r"[\u4e00-\u9fff]", stripped)
        if 0 < len(chinese_chars) <= 10 and not re.search(r"[A-Za-z0-9]", stripped):
            return True

        return False

    def _should_drop_pre_figure_block(self, lines: list[str]) -> bool:
        effective_lines = [line.strip() for line in lines if line.strip()]
        if not effective_lines:
            return False

        if len(effective_lines) < 3:
            return False

        if any(self._is_formula_index(line) or self._is_formula_line(line) for line in effective_lines):
            return False

        if any(re.search(r"[。！？；：，、]", line) for line in effective_lines):
            return False

        label_like_count = sum(1 for line in effective_lines if self._is_likely_figure_label_line(line))
        short_line_count = sum(1 for line in effective_lines if len(line) <= 18)

        if label_like_count >= max(3, len(effective_lines) - 1) and short_line_count == len(effective_lines):
            return True

        return len(effective_lines) >= 5 and short_line_count == len(effective_lines)

    def _is_likely_figure_label_line(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False

        if len(stripped) > 18:
            return False

        if self._is_page_number(stripped) or self._is_figure_caption(stripped):
            return False

        if self._looks_like_figure_label(stripped):
            return True

        if re.search(r"[。！？；：，、]", stripped):
            return False

        if any(keyword in stripped for keyword in FIGURE_LABEL_KEYWORDS):
            return True

        chinese_chars = re.findall(r"[\u4e00-\u9fff]", stripped)
        return 0 < len(chinese_chars) <= 8 and not re.search(r"[A-Za-z0-9]", stripped)


    def _remove_page_headers(self, lines: list[str]) -> list[str]:
        if not lines:
            return lines

        filtered = list(lines)
        header_scan_limit = min(3, len(filtered))
        for index in range(header_scan_limit):
            line = filtered[index].strip()
            if line and PAGE_HEADER_PATTERN.fullmatch(line):
                filtered[index] = ""

        return filtered

    def _is_formula_index(self, line: str) -> bool:
        return bool(FORMULA_INDEX_PATTERN.fullmatch(line.strip()))

    def _is_formula_line(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False

        if self._is_figure_caption(stripped) or self._is_page_number(stripped):
            return False

        if self._looks_like_figure_label(stripped):
            return False

        if len(stripped) <= 2 and re.fullmatch(r"[A-Za-z]", stripped):
            return False

        if COORDINATE_DEFINITION_PATTERN.search(stripped) and "=" not in stripped:
            return False

        symbol_hits = len(re.findall(r"[=≈≠≤≥∑∏∫√∞±×÷∂∇∈∉⊂⊆→←↔]", stripped))
        latin_hits = len(re.findall(r"[A-Za-zα-ωΑ-Ω]", stripped))
        digit_hits = len(re.findall(r"\d", stripped))
        chinese_hits = len(re.findall(r"[\u4e00-\u9fff]", stripped))

        if FORMULA_LINE_PATTERN.search(stripped):
            return True

        if symbol_hits >= 1 and (latin_hits + digit_hits) >= 2 and chinese_hits <= max(6, len(stripped) // 3):
            return True

        if stripped.count("^") >= 1 or stripped.count("_") >= 1:
            return True

        if chinese_hits == 0 and latin_hits + digit_hits >= 4 and symbol_hits >= 1:
            return True

        return False

    def _looks_like_figure_label(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False

        if len(stripped) > 12:
            return False

        if any(token in stripped for token in ["坐标系", "原点", "方程", "公式"]):
            return False

        return bool(FIGURE_LABEL_NOISE_PATTERN.fullmatch(stripped))

    def _mark_formula_line(self, line: str) -> str:
        stripped = line.strip()
        return stripped if stripped.startswith("【公式】") else f"【公式】{stripped}"

    def _is_figure_caption(self, line: str) -> bool:
        return bool(FIGURE_CAPTION_PATTERN.match(line))

    def _is_page_number(self, line: str) -> bool:
        stripped = line.strip()
        return bool(PAGE_NUMBER_PATTERN.fullmatch(stripped))

    def _should_drop_figure_block(self, lines: list[str]) -> bool:
        effective_lines = [line.strip() for line in lines if line.strip()]
        if not effective_lines:
            return False

        if any(self._is_formula_index(line) or self._is_formula_line(line) for line in effective_lines):
            return False

        short_line_count = sum(1 for line in effective_lines if len(line) <= 14)
        label_like_count = sum(1 for line in effective_lines if self._is_likely_figure_label_line(line))
        punctuation_free_count = sum(1 for line in effective_lines if not re.search(r"[。！？；：，、]", line))

        if len(effective_lines) >= 4 and short_line_count == len(effective_lines) and punctuation_free_count == len(effective_lines):
            return True

        if len(effective_lines) >= 3 and label_like_count >= len(effective_lines) - 1:
            return True

        return False

    def _should_drop_figure_context_line(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False

        if self._is_page_number(stripped):
            return True

        if self._looks_like_figure_label(stripped):
            return True

        if self._is_formula_index(stripped):
            return False

        if self._is_probable_figure_noise(stripped):
            return True

        if self._is_formula_line(stripped):
            return False

        if self._is_likely_figure_label_line(stripped):
            return True

        if len(stripped) <= 14 and not re.search(r"[。！？；：，、]", stripped):
            if re.search(r"[A-Za-zα-ωΑ-Ωβγδεζηθικλμνξοπρστυφχψω]", stripped):
                return True
            chinese_chars = re.findall(r"[\u4e00-\u9fff]", stripped)
            if 0 < len(chinese_chars) <= 8:
                return True

        return False

    def _is_probable_figure_noise(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False

        if self._looks_like_figure_label(stripped):
            return True

        if self._is_likely_figure_label_line(stripped):
            return True

        if len(stripped) <= 2:
            return True

        if SHORT_NOISE_PATTERN.fullmatch(stripped):
            return True

        if len(stripped) <= 10 and not re.search(r"[。！？；：，、]", stripped):
            chinese_chars = re.findall(r"[\u4e00-\u9fff]", stripped)
            if 0 < len(chinese_chars) <= 4:
                return True

        return False

    def _clean_text(self, text: str) -> str:
        """做最基础的 PDF 文本清洗。"""
        text = text.replace("\u3000", " ")
        text = text.replace("\xa0", " ")
        text = text.replace("\r", "\n")
        text = re.sub(r"[\t\f\v]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ ]{2,}", " ", text)

        lines = [line.strip() for line in text.split("\n")]
        cleaned_lines: list[str] = []
        previous_blank = False

        for line in lines:
            if not line:
                if not previous_blank:
                    cleaned_lines.append("")
                previous_blank = True
                continue

            cleaned_lines.append(line)
            previous_blank = False

        return "\n".join(cleaned_lines).strip()
