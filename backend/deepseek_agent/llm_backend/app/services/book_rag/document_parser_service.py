from __future__ import annotations

import gc
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
ROMAN_PAGE_NUMBER_PATTERN = re.compile(r"^[·•●]*[ivxlcdmIVXLCDM]{1,8}[·•●]*$")
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
ENGLISH_TERM_PARTS = (
    "coordinate",
    "coordinates",
    "navigation",
    "reference",
    "resolving",
    "platform",
    "geographic",
    "electronic",
    "inertial",
    "dynamic",
    "environment",
    "information",
    "display",
    "control",
    "decision",
    "making",
    "velocity",
    "doppler",
    "correlation",
    "weather",
    "chart",
    "service",
    "frame",
    "system",
    "surface",
    "conical",
    "cylindrical",
    "point",
    "pointing",
    "southward",
    "apparatus",
    "meteorological",
    "cloud",
    "aeronautical",
    "strapdown",
    "cart",
    "rectangular",
    "cartesian",
    "earth",
    "centered",
    "center",
    "fixed",
    "tangent",
    "plane",
    "grid",
    "body",
    "gyro",
    "sun",
    "celestial",
    "quaternion",
    "euler",
    "angle",
    "direction",
    "cosine",
)
ENGLISH_TERM_COMBINATIONS = (
    ("cylindrical", "coordinate"),
    ("coordinate", "system"),
    ("coordinates", "system"),
    ("geographic", "coordinates"),
    ("geographic", "coordinate"),
    ("gcographic", "coordinate"),
    ("gcographic", "coordinates"),
    ("inertial", "coordinates"),
    ("cylindrical", "coordinates"),
    ("inertial", "coordinate"),
    ("navigation", "coordinate"),
    ("navigation", "coordinates"),
    ("grid", "coordinate"),
    ("body", "coordinate"),
    ("gyro", "coordinate"),
    ("tangent", "plane"),
    ("reference", "frame"),
    ("resolving", "frame"),
    ("direction", "cosine"),
    ("euler", "angle"),
    ("cartesian", "coordinate"),
    ("earth", "centered"),
    ("earth", "fixed"),
    ("celestial", "coordinate"),
    ("electronic", "chart"),
    ("information", "service"),
    ("weather", "chart"),
    ("doppler", "velocity"),
    ("correlation", "velocity"),
)
CATALOG_LINE_PATTERN = re.compile(r"^(?:第[一二三四五六七八九十百千万0-9]+章.*|第[一二三四五六七八九十百千万0-9]+节.*|[一二三四五六七八九十]+、.+|[（(][一二三四五六七八九十0-9]+[)）].*|[一二三四五六七八九十]+\s*$|[0-9IVXivx]+[.、]\s*.+)$")
SECTION_TITLE_PATTERN = re.compile(r"^(?:第[一二三四五六七八九十百千万0-9]+[章节编部分]\s*.+|[一二三四五六七八九十]+、.+|[（(][一二三四五六七八九十0-9]+[)）].+)$")
SENTENCE_END_PATTERN = re.compile(r"[。！？；：.!?;:）)】]$")
TRAILING_INCOMPLETE_PATTERN = re.compile(r"[\u4e00-\u9fffA-Za-z0-9）)]$")
SHORT_AXIS_LABEL_PATTERN = re.compile(r"^[A-Za-zΑ-Ωα-ω]{1,2}[a-zA-Z0-9Α-Ωα-ω]{0,2}$")
SHORT_FORMULA_BODY_PATTERN = re.compile(r"^[A-Za-zΑ-Ωα-ω][A-Za-z0-9Α-Ωα-ω_\-^(){}\[\]]{1,15}$")
LEADING_CONTINUATION_LINE_PATTERN = re.compile(r"^(?:[A-Za-z0-9Α-Ωα-ω（(]|[的地得了着并且且及与或而但并把将被于从向按对为在使由所其之仍即再又也均都则后前中内外上下里])")
CHUNK_CONTINUATION_PREFIX_PATTERN = re.compile(r"^(?:[A-Za-z0-9Α-Ωα-ω（(]|[的地得了着并且且及与或而但并把将被于从向按对为在使由所其之仍即再又也均都则后前中内外上下里]|制|统|行|测|控|导|算|姿|向|速|率|角|量|置|系|器|图|场)")



class DocumentParserService:
    """负责解析教材/书籍类 PDF 文本，支持扫描件 OCR 兜底。"""

    def __init__(
        self,
        ocr_min_text_length: int = 30,
        debug_ocr: bool = False,
        ocr_zoom_matrix: tuple[float, float] = (0.8, 0.8),
        ocr_max_pixel_count: int = 900_000,
        ocr_failure_threshold: int = 8,
    ):
        self.ocr_min_text_length = ocr_min_text_length
        self.debug_ocr = debug_ocr
        self.ocr_zoom_matrix = ocr_zoom_matrix
        self.ocr_max_pixel_count = ocr_max_pixel_count
        self.ocr_failure_threshold = ocr_failure_threshold
        self._ocr_engine: RapidOCR | None = None
        self._consecutive_ocr_failures = 0
        self._ocr_disabled_for_session = False

    def parse_pdf(self, file_path: str, max_pages: int | None = None) -> list[dict[str, Any]]:
        """按页解析 PDF，并返回结构化文本结果。"""
        pdf_path = Path(file_path)
        self._validate_pdf(pdf_path)
        self._consecutive_ocr_failures = 0
        self._ocr_disabled_for_session = False

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
        self._post_process_pages(pages)
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

    def _post_process_pages(self, pages: list[dict[str, Any]]) -> None:
        self._repair_formula_index_pairs(pages)
        self._trim_page_tail_noise(pages)
        self._merge_cross_page_continuations(pages)
        self._trim_page_tail_noise(pages)
        self._repair_abrupt_page_endings(pages)
        self._trim_catalog_front_matter(pages)
        self._refresh_page_metrics(pages)

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

    def _get_ocr_engine(self) -> RapidOCR:
        if self._ocr_engine is None:
            self._ocr_engine = RapidOCR()
        return self._ocr_engine

    def _record_ocr_failure(self, page_number: int) -> None:
        self._consecutive_ocr_failures += 1
        if self._consecutive_ocr_failures >= self.ocr_failure_threshold:
            self._ocr_disabled_for_session = True
            logger.error(
                f"OCR disabled for current parse after {self._consecutive_ocr_failures} consecutive failures, last_page={page_number}"
            )

    def _record_ocr_success(self) -> None:
        self._consecutive_ocr_failures = 0

    def _should_use_ocr(self, text: str) -> bool:
        if self._ocr_disabled_for_session:
            return False

        normalized_text = text.strip()
        if len(normalized_text) >= self.ocr_min_text_length:
            return False

        compact_text = re.sub(r"\s+", "", normalized_text)
        if len(compact_text) >= max(40, self.ocr_min_text_length // 2):
            return False

        return True

    def _build_ocr_zoom_candidates(self, pdf_page: fitz.Page) -> list[tuple[float, float]]:
        page_rect = pdf_page.rect
        page_area = max(float(page_rect.width * page_rect.height), 1.0)
        base_zoom = min(self.ocr_zoom_matrix)

        if page_area >= 1_000_000:
            candidates = [(0.5, 0.5), (0.4, 0.4), (0.35, 0.35)]
        elif page_area >= 500_000:
            candidates = [(0.6, 0.6), (0.5, 0.5), (0.4, 0.4)]
        else:
            candidates = [self.ocr_zoom_matrix, (0.6, 0.6), (0.5, 0.5), (0.4, 0.4)]
            if base_zoom < 0.8:
                candidates.insert(0, self.ocr_zoom_matrix)

        unique_candidates: list[tuple[float, float]] = []
        for candidate in candidates:
            if candidate not in unique_candidates:
                unique_candidates.append(candidate)
        return unique_candidates

    def _render_ocr_image(self, pdf_page: fitz.Page, zoom_x: float, zoom_y: float) -> np.ndarray | None:
        pix = pdf_page.get_pixmap(matrix=fitz.Matrix(zoom_x, zoom_y), alpha=False)
        try:
            if pix.width * pix.height > self.ocr_max_pixel_count:
                logger.warning(
                    f"Skip OCR render on page {pdf_page.number + 1} with zoom=({zoom_x}, {zoom_y}): "
                    f"pixel_count={pix.width * pix.height} exceeds limit={self.ocr_max_pixel_count}"
                )
                return None

            image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            return np.ascontiguousarray(image)
        finally:
            del pix
            gc.collect()

    def _extract_text_with_ocr(self, pdf_page: fitz.Page) -> str:
        zoom_candidates = self._build_ocr_zoom_candidates(pdf_page)

        for zoom_x, zoom_y in zoom_candidates:
            image: np.ndarray | None = None
            try:
                image = self._render_ocr_image(pdf_page, zoom_x, zoom_y)
                if image is None:
                    continue

                ocr_result, _ = self._get_ocr_engine()(image)

                if not ocr_result:
                    if self.debug_ocr and pdf_page.number < 3:
                        logger.warning(f"RapidOCR returned empty result on page {pdf_page.number + 1}")
                    continue

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

                self._record_ocr_success()
                return self._clean_text(ocr_text)
            except MemoryError as exc:
                logger.warning(
                    f"OCR memory pressure on page {pdf_page.number + 1} with zoom=({zoom_x}, {zoom_y}): {exc}"
                )
            except Exception as exc:
                logger.warning(
                    f"OCR failed on page {pdf_page.number + 1} with zoom=({zoom_x}, {zoom_y}): {exc}"
                )
            finally:
                if image is not None:
                    del image
                gc.collect()

        self._record_ocr_failure(pdf_page.number + 1)
        logger.error(f"OCR failed on page {pdf_page.number + 1}: all zoom attempts exhausted")
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

        normalized_text = "\n".join(collapsed_lines).strip()
        normalized_text = self._normalize_english_terms(normalized_text)
        normalized_text = self._normalize_punctuation(normalized_text)
        normalized_text = self._drop_catalog_like_noise(normalized_text)
        normalized_text = self._drop_isolated_noise_lines(normalized_text)
        normalized_text = self._repair_formula_index_text(normalized_text)
        return self._normalize_text_structure(normalized_text)

    def _drop_isolated_noise_lines(self, text: str) -> str:
        if not text:
            return ""

        lines = [line.strip() for line in text.split("\n")]
        filtered_lines: list[str] = []

        for index, line in enumerate(lines):
            if not line:
                if filtered_lines and filtered_lines[-1] != "":
                    filtered_lines.append("")
                continue

            prev_line = self._find_neighbor_line(lines, index, -1)
            next_line = self._find_neighbor_line(lines, index, 1)
            if self._is_isolated_noise_line(line, prev_line, next_line):
                continue

            filtered_lines.append(line)

        while filtered_lines and filtered_lines[-1] == "":
            filtered_lines.pop()

        return "\n".join(filtered_lines).strip()

    def _repair_formula_index_text(self, text: str) -> str:
        if not text:
            return ""

        lines = [line.strip() for line in text.split("\n")]
        for index, line in enumerate(lines):
            if not line.startswith("【公式编号】"):
                continue

            prev_index = index - 1
            while prev_index >= 0 and not lines[prev_index]:
                prev_index -= 1

            if prev_index < 0:
                continue

            previous_line = lines[prev_index].strip()
            if self._should_promote_formula_body(previous_line):
                lines[prev_index] = self._mark_formula_line(previous_line)

        return "\n".join(lines).strip()

    def _normalize_text_structure(self, text: str) -> str:
        if not text:
            return ""

        lines = [line.strip() for line in text.split("\n")]
        normalized_lines: list[str] = []

        for line in lines:
            if not line:
                if normalized_lines and normalized_lines[-1] != "":
                    normalized_lines.append("")
                continue

            if self._is_section_title(line):
                if normalized_lines and normalized_lines[-1] != "":
                    normalized_lines.append("")
                normalized_lines.append(line)
                normalized_lines.append("")
                continue

            normalized_lines.append(line)

        compact_lines: list[str] = []
        previous_blank = False
        for line in normalized_lines:
            if not line:
                if not previous_blank and compact_lines:
                    compact_lines.append("")
                previous_blank = True
                continue

            compact_lines.append(line)
            previous_blank = False

        return "\n".join(compact_lines).strip()

    def _find_neighbor_line(self, lines: list[str], start_index: int, step: int) -> str:
        index = start_index + step
        while 0 <= index < len(lines):
            candidate = lines[index].strip()
            if candidate:
                return candidate
            index += step
        return ""

    def _is_isolated_noise_line(self, line: str, prev_line: str, next_line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False

        if self._is_formula_index(stripped) or self._is_formula_line(stripped):
            return False

        if self._is_section_title(stripped) or self._is_figure_caption(stripped):
            return False

        if re.search(r"[。！？；：，、]", stripped):
            return False

        if not (SHORT_AXIS_LABEL_PATTERN.fullmatch(stripped) or stripped.isdigit() and len(stripped) <= 2):
            return False

        if prev_line and not SENTENCE_END_PATTERN.search(prev_line):
            return False

        if next_line.startswith(("[插图]", "【公式】", "【公式编号】")):
            return True

        if next_line and not re.search(r"[。！？；：，、]", next_line):
            return True

        return len(stripped) <= 2

    def _should_promote_formula_body(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped or stripped.startswith("【公式】"):
            return False

        if re.search(r"[。！？；：，、]", stripped):
            return False

        if len(stripped) > 18:
            return False

        if self._is_figure_caption(stripped) or self._is_page_number(stripped):
            return False

        return bool(SHORT_FORMULA_BODY_PATTERN.fullmatch(stripped))

    def _is_section_title(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False

        if len(stripped) < 4 or len(stripped) > 40:
            return False

        if re.search(r"[。！？；]", stripped):
            return False

        if len(re.findall(r"[\u4e00-\u9fff]", stripped)) < 2:
            return False

        return bool(SECTION_TITLE_PATTERN.fullmatch(stripped))

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

        if self._is_section_title(stripped):
            return False

        if len(stripped) <= 2 and re.fullmatch(r"[A-Za-z]", stripped):
            return False

        if re.search(r"[。！？；：，、]", stripped) and not re.search(r"[=≈≠≤≥∑∏∫√∞±×÷∂∇∈∉⊂⊆→←↔^_]", stripped):
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

    def _repair_formula_index_pairs(self, pages: list[dict[str, Any]]) -> None:
        for page in pages:
            text = (page.get("text") or "").strip()
            if not text:
                continue
            page["text"] = self._repair_formula_index_text(text)

    def _merge_cross_page_continuations(self, pages: list[dict[str, Any]]) -> None:
        for index in range(len(pages) - 1):
            current_text = (pages[index].get("text") or "").strip()
            next_text = (pages[index + 1].get("text") or "").strip()
            if not current_text or not next_text:
                continue

            current_lines = [line.strip() for line in current_text.split("\n") if line.strip()]
            next_lines = [line.strip() for line in next_text.split("\n") if line.strip()]
            if not current_lines or not next_lines:
                continue

            tail_index = self._locate_tail_line_index(current_lines)
            if tail_index < 0:
                continue

            tail = current_lines[tail_index]
            head = next_lines[0]
            if not self._should_merge_cross_page_lines(tail, head):
                continue

            merge_count = self._consume_leading_continuation_lines(next_lines)
            if merge_count == 0:
                continue

            merged_prefix = "".join(next_lines[:merge_count]).strip()
            if not merged_prefix:
                continue

            if len(merged_prefix) > 120:
                continue

            current_lines[tail_index] = f"{tail}{merged_prefix}"
            next_lines = next_lines[merge_count:]
            pages[index]["text"] = self._normalize_text_structure("\n".join(current_lines).strip())
            pages[index + 1]["text"] = self._normalize_text_structure("\n".join(next_lines).strip())

    def _repair_abrupt_page_endings(self, pages: list[dict[str, Any]]) -> None:
        for index in range(len(pages) - 1):
            current_text = (pages[index].get("text") or "").strip()
            next_text = (pages[index + 1].get("text") or "").strip()
            if not current_text or not next_text:
                continue

            current_lines = [line.strip() for line in current_text.split("\n") if line.strip()]
            next_lines = [line.strip() for line in next_text.split("\n") if line.strip()]
            if not current_lines or not next_lines:
                continue

            tail = current_lines[-1]
            if self._is_safe_page_ending(tail):
                continue

            consumed_lines = self._consume_leading_continuation_lines(next_lines)
            if consumed_lines <= 0:
                continue

            prefix = "".join(next_lines[:consumed_lines]).strip()
            if not prefix or len(prefix) > 90:
                continue

            if not self._should_attach_page_tail_with_prefix(tail, prefix):
                continue

            current_lines[-1] = f"{tail}{prefix}"
            remaining_next_lines = next_lines[consumed_lines:]
            pages[index]["text"] = self._normalize_text_structure("\n".join(current_lines).strip())
            pages[index + 1]["text"] = self._normalize_text_structure("\n".join(remaining_next_lines).strip())

    def _is_safe_page_ending(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return True

        if SENTENCE_END_PATTERN.search(stripped):
            return True

        if stripped.startswith(("[插图]", "【公式】", "【公式编号】")):
            return True

        if self._is_section_title(stripped) or self._is_formula_index(stripped) or self._is_formula_line(stripped):
            return True

        return False

    def _should_attach_page_tail_with_prefix(self, tail: str, prefix: str) -> bool:
        tail = tail.strip()
        prefix = prefix.strip()
        if not tail or not prefix:
            return False

        if len(tail) < 14:
            return False

        if self._is_catalog_entry_line(prefix) or self._is_catalog_noise_line(prefix):
            return False

        if re.search(r"[A-Za-z]$", tail) and re.match(r"^[A-Za-z]", prefix):
            return True

        if TRAILING_INCOMPLETE_PATTERN.search(tail) and CHUNK_CONTINUATION_PREFIX_PATTERN.match(prefix):
            return True

        return False

    def _trim_catalog_front_matter(self, pages: list[dict[str, Any]]) -> None:
        for page in pages:
            text = (page.get("text") or "").strip()
            if not text:
                continue

            lines = [line.strip() for line in text.split("\n") if line.strip()]
            if not lines:
                page["text"] = ""
                continue

            first_substantive_index = 0
            while first_substantive_index < len(lines):
                candidate = lines[first_substantive_index]
                if self._is_catalog_front_matter_line(candidate):
                    first_substantive_index += 1
                    continue
                break

            trimmed_lines = lines[first_substantive_index:]
            page["text"] = self._normalize_text_structure("\n".join(trimmed_lines).strip()) if trimmed_lines else ""


    def _is_catalog_front_matter_line(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False

        if self._is_page_number(stripped):
            return True

        if self._is_catalog_noise_line(stripped):
            return True

        if self._is_catalog_entry_line(stripped):
            return True

        if len(stripped) <= 24 and ("目录" in stripped or "前言" in stripped):
            return True

        return False

    def _should_keep_short_tail_line(self, line: str, previous_lines: list[str]) -> bool:
        stripped = line.strip()
        if not stripped or not previous_lines:
            return False

        previous = previous_lines[-1].strip()
        if not previous:
            return False

        if self._is_section_title(stripped):
            return True

        if self._is_formula_index(stripped) or self._is_formula_line(stripped):
            return True

        if re.search(r"[。！？；：.!?;:）)】]$", previous):
            return False

        if len(stripped) <= 18 and CHUNK_CONTINUATION_PREFIX_PATTERN.match(stripped):
            return True

        if re.match(r"^[A-Za-z0-9Α-Ωα-ω].*", stripped) and re.search(r"[A-Za-z]$", previous):
            return True

        return False

    def _trim_page_tail_noise(self, pages: list[dict[str, Any]]) -> None:
        for page in pages:
            text = (page.get("text") or "").strip()
            if not text:
                continue

            lines = [line.strip() for line in text.split("\n") if line.strip()]
            if not lines:
                page["text"] = ""
                continue

            while lines and self._should_drop_tail_line(lines[-1], lines[:-1]):
                lines.pop()

            page["text"] = self._normalize_text_structure("\n".join(lines).strip())

    def _should_drop_tail_line(self, line: str, previous_lines: list[str]) -> bool:
        stripped = line.strip()
        if not stripped:
            return True

        if self._is_page_number(stripped):
            return True

        if self._is_catalog_noise_line(stripped):
            return True

        if len(stripped) <= 20 and self._is_catalog_entry_line(stripped):
            return True

        if stripped.startswith(("[插图]", "【公式】", "【公式编号】")):
            return False

        if self._is_formula_index(stripped) or self._is_formula_line(stripped):
            return False

        if self._is_section_title(stripped):
            return False

        if previous_lines and self._should_keep_short_tail_line(stripped, previous_lines):
            return False

        if re.search(r"[。！？；：.!?;:）)】]$", stripped):
            return False

        if previous_lines:
            previous = previous_lines[-1].strip()
            if previous and not re.search(r"[。！？；：.!?;:）)】]$", previous):
                if len(stripped) <= 28 and not self._looks_like_catalog_or_noise_tail(stripped):
                    return False

        if len(stripped) <= 8 and not re.search(r"[。！？；：，、.!?;:]", stripped):
            return not self._should_keep_short_tail_line(stripped, previous_lines)

        if len(stripped) <= 18 and previous_lines:
            previous = previous_lines[-1].strip()
            if previous and not re.search(r"[。！？；：.!?;:）)】]$", previous):
                return not self._should_keep_short_tail_line(stripped, previous_lines)

        return False

    def _looks_like_catalog_or_noise_tail(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False

        if self._is_catalog_entry_line(stripped) or self._is_catalog_noise_line(stripped):
            return True

        if len(stripped) <= 4 and not re.search(r"[A-Za-z\u4e00-\u9fff]", stripped):
            return True

        if len(stripped) <= 10 and re.fullmatch(r"[A-Za-z0-9_\-./]+", stripped):
            return True

        return False

    def _locate_tail_line_index(self, lines: list[str]) -> int:
        for index in range(len(lines) - 1, -1, -1):
            line = lines[index].strip()
            if not line:
                continue
            if line.startswith(("[插图]", "【公式】", "【公式编号】")):
                continue
            return index
        return -1

    def _should_merge_cross_page_lines(self, tail: str, head: str) -> bool:
        tail = tail.strip()
        head = head.strip()
        if not tail or not head:
            return False

        if SENTENCE_END_PATTERN.search(tail):
            return False

        if self._is_section_title(head) or self._is_catalog_entry_line(head):
            return False

        if head.startswith(("[插图]", "【公式】", "【公式编号】")):
            return False

        if self._is_formula_index(head) or self._is_formula_line(head):
            return False

        if len(tail) < 8:
            return False

        if len(tail) <= 20 and self._is_catalog_entry_line(tail):
            return False

        if len(head) <= 18 and self._is_catalog_entry_line(head):
            return False

        tail_prefix = tail.split("\n")[-1][-1:] if tail else ""
        if tail_prefix and re.match(r"[A-Za-z]", tail_prefix) and re.match(r"^[A-Za-z]", head):
            return True

        if TRAILING_INCOMPLETE_PATTERN.search(tail) and CHUNK_CONTINUATION_PREFIX_PATTERN.match(head):
            return True

        return False

    def _consume_leading_continuation_lines(self, lines: list[str]) -> int:
        count = 0
        merged_length = 0
        for line in lines[:4]:
            if not line:
                break
            if count == 0:
                if not LEADING_CONTINUATION_LINE_PATTERN.match(line):
                    break
            else:
                if self._is_section_title(line) or self._is_figure_caption(line):
                    break
                if line.startswith(("[插图]", "【公式】", "【公式编号】")):
                    break
            if self._is_formula_index(line):
                break
            count += 1
            merged_length += len(line)
            if SENTENCE_END_PATTERN.search(line) or merged_length >= 90:
                break
        return count

    def _refresh_page_metrics(self, pages: list[dict[str, Any]]) -> None:
        for page in pages:
            text = (page.get("text") or "").strip()
            page["text"] = text
            page["char_count"] = len(text)
            page["is_empty"] = text == ""

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
        return bool(PAGE_NUMBER_PATTERN.fullmatch(stripped) or ROMAN_PAGE_NUMBER_PATTERN.fullmatch(stripped))

    def _normalize_english_terms(self, text: str) -> str:
        if not text:
            return ""

        known_replacements = {
            "geographicinformation": "geographic information",
            "dynamicenvironmentinformation": "dynamic environment information",
            "electronicchart": "electronic chart",
            "nauticalnavigation": "nautical navigation",
            "airbornenavigation": "airborne navigation",
            "decisionmaking": "decision-making",
            "vesselmonitoring": "vessel monitoring",
            "electronicchartanddisplayinformationsystem": "electronic chart and display information system",
            "firecontrolsystem": "fire control system",
            "hitaccuracy": "hit accuracy",
            "precisionguidance": "precision guidance",
            "aircraftcarrier": "aircraft carrier",
            "networkcentricwarfare": "network-centric warfare",
            "magneticcompass": "magnetic compass",
            "globalpositioningsystem": "global positioning system",
            "globalnavigationsatellitesystem": "global navigation satellite system",
            "weatherchartfacsimileapparatus": "weather-chart facsimile apparatus",
            "aeronauticalinformationservice": "aeronautical information service",
            "dopplervelocitylog": "Doppler velocity log",
            "correlationvelocitylog": "correlation velocity log",
            "longbaseline": "long baseline",
            "shortbaseline": "short baseline",
            "ultrashortbaseline": "ultra short baseline",
            "rectangularcartesiancoordinatesystem": "rectangular Cartesian coordinate system",
            "cylindricalcoordinatesystem": "cylindrical coordinate system",
            "conicalsurfacecoordinatesystem": "conical surface coordinate system",
            "inertialcoordinatesystem": "inertial coordinate system",
            "suncenterinertialcoordinatesystem": "sun-center inertial coordinate system",
            "earthcenterinertialcoordinatesystem": "earth-center inertial coordinate system",
            "celestialcoordinatesystem": "celestial coordinate system",
            "earthcenteredearthfixedcoordinatesystem": "earth-centered earth-fixed coordinate system",
            "coordinatesystem": "coordinate system",
            "geographiccoordinate": "geographic coordinate",
            "gcographiccoordinate": "geographic coordinate",
            "gcographiccoordinates": "geographic coordinates",
            "coordinatesystem": "coordinate system",
            "tangentplanecoordinatesystem": "tangent plane coordinate system",
            "gridcoordinatesystem": "grid coordinate system",
            "bodycoordinatesystem": "body coordinate system",
            "navigationcoordinatesystem": "navigation coordinate system",
            "computedcoordinatesystem": "computed coordinate system",
            "gyrocoordinatesystem": "gyro coordinate system",
            "referenceframe": "reference frame",
            "resolvingframe": "resolving frame",
            "directioncosine": "direction cosine",
            "eulerangle": "Euler angle",
            "quaternion": "quaternion",
            "informatization": "informatization",
            "intelligentialization": "intelligentization",
            "rotationalangularvelocity": "rotational angular velocity",
        }

        for old, new in known_replacements.items():
            text = re.sub(old, new, text, flags=re.IGNORECASE)

        text = self._repair_joined_english_phrases(text)
        text = self._split_joined_english_terms(text)
        text = self._repair_joined_english_phrases(text)
        text = re.sub(r"([a-z]{2,})(System|Service|Chart|Navigation|Control|Information)", r"\1 \2", text)
        text = re.sub(r"([A-Za-z])&([A-Za-z])", r"\1 & \2", text)
        return text

    def _repair_joined_english_phrases(self, text: str) -> str:
        repaired = text
        for left, right in ENGLISH_TERM_COMBINATIONS:
            repaired = re.sub(
                rf"(?i){left}{right}",
                f"{left} {right}",
                repaired,
            )
        return repaired

    def _split_joined_english_terms(self, text: str) -> str:
        if not text:
            return ""

        pattern = re.compile(r"\b[a-z]{10,}\b", flags=re.IGNORECASE)

        def replacer(match: re.Match[str]) -> str:
            token = match.group(0)
            if token.lower() in {"information", "international", "navigation", "electronic", "automatic"}:
                return token
            return self._split_english_token(token)

        return pattern.sub(replacer, text)

    def _split_english_token(self, token: str) -> str:
        pieces = self._segment_english_token(token.lower())
        if len(pieces) <= 1:
            return token
        return " ".join(pieces)

    def _segment_english_token(self, token: str) -> list[str]:
        if len(token) < 10:
            return [token]

        best_split: list[str] | None = None
        for part in sorted(ENGLISH_TERM_PARTS, key=len, reverse=True):
            index = token.find(part)
            while index != -1:
                prefix = token[:index]
                suffix = token[index + len(part):]
                candidate: list[str] = []

                if prefix:
                    prefix_parts = self._segment_english_token(prefix)
                    if prefix_parts == [prefix] and len(prefix) >= 8:
                        index = token.find(part, index + 1)
                        continue
                    candidate.extend(prefix_parts)

                candidate.append(part)

                if suffix:
                    suffix_parts = self._segment_english_token(suffix)
                    if suffix_parts == [suffix] and len(suffix) >= 8:
                        index = token.find(part, index + 1)
                        continue
                    candidate.extend(suffix_parts)

                if len(candidate) >= 2:
                    if best_split is None or len(" ".join(candidate)) < len(" ".join(best_split)):
                        best_split = candidate
                index = token.find(part, index + 1)

        return best_split or [token]

    def _normalize_punctuation(self, text: str) -> str:
        if not text:
            return ""

        text = re.sub(r"。。+", "。", text)
        text = re.sub(r"，，+", "，", text)
        text = re.sub(r"！！+", "！", text)
        text = re.sub(r"？？+", "？", text)
        text = re.sub(r"；；+", "；", text)
        text = re.sub(r"([。！？；：，])\1+", r"\1", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _drop_catalog_like_noise(self, text: str) -> str:
        if not text:
            return ""

        lines = [line.strip() for line in text.split("\n")]
        filtered_lines: list[str] = []
        catalog_window = 0

        for line in lines:
            if not line:
                if filtered_lines and filtered_lines[-1] != "":
                    filtered_lines.append("")
                catalog_window = 0
                continue

            is_catalog_like = self._is_catalog_noise_line(line) or self._is_catalog_entry_line(line)
            if is_catalog_like:
                catalog_window += 1
                if catalog_window >= 2 or self._is_catalog_noise_line(line):
                    continue
            else:
                catalog_window = 0

            filtered_lines.append(line)

        while filtered_lines and filtered_lines[-1] == "":
            filtered_lines.pop()

        return "\n".join(filtered_lines).strip()

    def _is_catalog_entry_line(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        if len(stripped) > 40:
            return False
        if re.search(r"[。！？；：，、]", stripped):
            return False
        if CATALOG_LINE_PATTERN.fullmatch(stripped):
            return True
        if "·" in stripped or "…" in stripped:
            return True
        return False

    def _is_catalog_noise_line(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False

        if self._is_page_number(stripped):
            return True

        if re.fullmatch(r"[·•●ivxlcdmIVXLCDM]{1,12}", stripped):
            return True

        if re.fullmatch(r"\.{2,}\d+", stripped):
            return True

        if re.fullmatch(r"第[一二三四五六七八九十百千万0-9]+章.*[·•●.]{1,}$", stripped):
            return True

        if len(stripped) <= 24 and "目录" in stripped:
            return True

        if len(stripped) <= 16 and re.search(r"思考题|参考文献|附录|前言", stripped):
            return True

        return False

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
