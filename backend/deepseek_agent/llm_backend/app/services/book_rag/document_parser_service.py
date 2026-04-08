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
                page_result = {
                    "page": index,
                    "text": cleaned_text,
                    "char_count": len(cleaned_text),
                    "is_empty": cleaned_text == "",
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
                        f"extract_method={page_result['extract_method']}"
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
