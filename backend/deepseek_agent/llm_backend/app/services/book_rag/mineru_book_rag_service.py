from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.logger import get_logger


logger = get_logger(service="mineru_book_rag")

FRONT_MATTER_KEYWORDS = (
    "版权所有",
    "内容简介",
    "图书在版编目",
    "举报电话",
    "责任编辑",
    "责任印制",
    "封面设计",
    "科学出版社出版",
    "邮政编码",
    "ISBN",
    "CIP",
    "前言",
    "序",
    "印刷",
    "定价",
    "版",
)

REAR_MATTER_HEADINGS = {
    "思考题",
    "习题",
    "练习题",
    "本章小结",
    "小结",
    "参考文献",
    "附录",
}

REAR_MATTER_PREFIXES = (
    "思考题",
    "习题",
    "练习题",
    "本章小结",
    "小结",
    "参考文献",
    "附录",
)

FRONT_MATTER_HEADINGS = {
    "序",
    "第一版前言",
    "第二版前言",
    "前言",
    "内容简介",
}
BODY_FIRST_CHAPTER_TITLES = {
    "第一章 绪论",
    "第1章 绪论",
    "第一章 绪 论",
    "第1章 绪 论",
}
CATALOG_KEYWORDS = ("目录",)
CATALOG_ENTRY_PREFIX_PATTERN = re.compile(r"^(?:第[一二三四五六七八九十百千万0-9]+[章节节篇部卷]|[一二三四五六七八九十]+、|[（(][一二三四五六七八九十0-9]+[)）])")
CATALOG_LEADER_PATTERN = re.compile(r"^(?:第[一二三四五六七八九十百千万0-9]+[章节节篇部卷]\s*.+|第[一二三四五六七八九十百千万0-9]+节\s*.+|[一二三四五六七八九十]+、.+|[（(][一二三四五六七八九十0-9]+[)）].+)$")
IMAGE_LINE_PATTERN = re.compile(r"!\[\]\([^)]*\)")
HEADING_PATTERN = re.compile(r"^(?:第[一二三四五六七八九十百千万0-9]+[章节编部分节].*|[一二三四五六七八九十]+、.+|[（(][一二三四五六七八九十0-9]+[)）].+)$")
SENTENCE_END_PATTERN = re.compile(r"[。！？；：.!?;:）)】]$")
MARKDOWN_HEADING_PATTERN = re.compile(r"^(#{1,6})\s*(.+?)\s*$")
NUMBERED_HEADING_PATTERN = re.compile(r"^(?:\d+(?:\.\d+)*[.、．]\s*.+|\d+\s+[\u4e00-\u9fffA-Za-z].+)$")
TRAILING_PAGE_NUMBER_PATTERN = re.compile(r"^(.*?)(?:\s+[·•●]?)?(\d{1,4})$")
CHAPTER_HEADING_PATTERN = re.compile(r"^第[一二三四五六七八九十百千万0-9]+章\s*.+")
SECTION_HEADING_PATTERN = re.compile(r"^第[一二三四五六七八九十百千万0-9]+节\s*.+")
SUBSECTION_HEADING_PATTERN = re.compile(r"^[一二三四五六七八九十]+、.+")
SUBSUBSECTION_HEADING_PATTERN = re.compile(r"^[（(][一二三四五六七八九十0-9]+[)）].+")
FIGURE_CAPTION_PATTERN = re.compile(r"^(?:\[插图\]\s*)?图\s*\d+(?:\s*[-—．.]\s*\d+)*\s*.*$")
TABLE_CAPTION_PATTERN = re.compile(r"^(?:\[表格\]\s*)?表\s*\d+(?:\s*[-—．.]\s*\d+)*\s*.*$")
FORMULA_ONLY_PATTERN = re.compile(r"^(?:【公式】|【公式编号】)+$")
PURE_PAGE_NUMBER_PATTERN = re.compile(r"^[·•●()（）\[\]【】\-—\s]*\d{1,4}[·•●()（）\[\]【】\-—\s]*$")
SENTENCE_CONTINUATION_ENDING_PATTERN = re.compile(r"(?:的|地|得|了|着|过|和|与|及|并|并且|或|而|而且|但|但是|则|使|使得|对|对其|把|将|被|由|于|在|按|向|从|为|作为|依据|按照|由于|为了|如果|当|其中|以及|一种|一种是|包括|例如|比如|进行|实现|采用|通过|利用|形成|得到|产生|处于|位于|关于|对于|其|该|本|此|这些|那些|各|每|上|下|前|后|中|内|外)$")
LEADING_CONTINUATION_PATTERN = re.compile(r"^(?:的|地|得|了|着|过|和|与|及|并|并且|或|而|而且|但|但是|则|使|使得|对|把|将|被|由|于|在|按|向|从|为|作为|依据|按照|由于|为了|如果|当|其中|以及|包括|例如|比如|进行|实现|采用|通过|利用|形成|得到|产生|处于|位于|关于|对于|其|该|本|此|这|那|各|每|上|下|前|后|中|内|外)")


@dataclass
class MinerUBlock:
    page: int
    block_type: str
    text: str
    level: int | None = None


class MinerUBookRAGService:
    """将 MinerU 结构化结果清洗并转换为稳定入库 chunk。"""

    def __init__(self, config_path: str | None = None):
        self._config_path = (
            Path(config_path)
            if config_path
            else Path(__file__).with_name("chunking_config.json")
        )
        self._config = self._load_chunking_config(self._config_path)

    def build_chunks_from_mineru_result(
        self,
        mineru_dir: str,
        book_name: str | None = None,
        source_file_name: str | None = None,
    ) -> dict[str, Any]:
        base_dir = Path(mineru_dir)
        if not base_dir.exists():
            raise FileNotFoundError(f"MinerU 结果目录不存在: {base_dir}")

        full_md_path = base_dir / "full.md"
        content_list_path = base_dir / "content_list_v2.json"

        if full_md_path.exists():
            cleaned_sections = self._build_sections_from_full_md(full_md_path)
            raw_pages = self._estimate_page_count_from_markdown(full_md_path)
            raw_blocks = 0
            filtered_blocks = 0
        elif content_list_path.exists():
            pages = json.loads(content_list_path.read_text(encoding="utf-8"))
            blocks = self._extract_blocks_from_pages(pages)
            filtered_block_list = self._filter_blocks(blocks)
            merged_sections = self._merge_blocks_into_sections(filtered_block_list)
            cleaned_sections = self._clean_sections(merged_sections)
            raw_pages = len(pages)
            raw_blocks = len(blocks)
            filtered_blocks = len(filtered_block_list)
        else:
            raise FileNotFoundError(
                f"缺少 full.md 或 content_list_v2.json: {base_dir}"
            )

        chunks = self._chunk_sections(cleaned_sections)
        if content_list_path.exists() and chunks:
            self._backfill_chunk_pages_from_content_list(chunks, content_list_path)

        resolved_book_name = book_name or base_dir.name
        resolved_source_name = source_file_name or self._detect_source_file_name(base_dir)

        return {
            "book_name": resolved_book_name,
            "source_type": "mineru",
            "source_dir": str(base_dir),
            "source_file_name": resolved_source_name,
            "config": self._config,
            "stats": {
                "raw_pages": raw_pages,
                "raw_blocks": raw_blocks,
                "filtered_blocks": filtered_blocks,
                "section_count": len(cleaned_sections),
                "chunk_count": len(chunks),
            },
            "sections": cleaned_sections,
            "chunks": chunks,
        }

    def save_chunks_from_mineru_result(
        self,
        mineru_dir: str,
        output_path: str | None = None,
        book_name: str | None = None,
        source_file_name: str | None = None,
    ) -> tuple[Path, dict[str, Any]]:
        result = self.build_chunks_from_mineru_result(
            mineru_dir=mineru_dir,
            book_name=book_name,
            source_file_name=source_file_name,
        )
        base_dir = Path(mineru_dir)
        target_path = Path(output_path) if output_path else base_dir / "book_rag_chunks.json"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"Saved MinerU Book RAG chunks to: {target_path}")
        return target_path, result

    def _load_chunking_config(self, config_path: Path) -> dict[str, int]:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        required_keys = [
            "target_chunk_length",
            "max_chunk_length",
            "chunk_soft_limit",
            "chunk_overlap",
            "min_chunk_length",
        ]
        missing = [key for key in required_keys if key not in raw]
        if missing:
            raise ValueError(f"chunking_config.json 缺少配置项: {missing}")
        if any(type(raw[key]) is not int for key in required_keys):
            raise ValueError("chunking_config.json 参数必须为整数")
        config = {key: raw[key] for key in required_keys}
        if not (0 < config['min_chunk_length'] <= config['target_chunk_length']
                <= config['max_chunk_length'] <= config['chunk_soft_limit']):
            raise ValueError("分块长度必须满足 0 < min <= target <= max <= soft_limit")
        if not 0 <= config['chunk_overlap'] < config['target_chunk_length']:
            raise ValueError("chunk_overlap 必须满足 0 <= overlap < target_chunk_length")
        return config

    def _extract_blocks_from_pages(self, pages: list[list[dict[str, Any]]]) -> list[MinerUBlock]:
        blocks: list[MinerUBlock] = []
        for page_idx, page_blocks in enumerate(pages, start=1):
            for block in page_blocks:
                block_type = str(block.get("type") or "").strip()
                text, level = self._extract_block_text(block_type, block.get("content") or {})
                normalized = self._normalize_block_text(text)
                if not normalized:
                    continue
                blocks.append(
                    MinerUBlock(
                        page=page_idx,
                        block_type=block_type,
                        text=normalized,
                        level=level,
                    )
                )
        return blocks

    def _extract_block_text(self, block_type: str, content: dict[str, Any]) -> tuple[str, int | None]:
        if block_type == "title":
            items = content.get("title_content") or []
            return self._join_text_items(items), content.get("level")
        if block_type == "paragraph":
            return self._join_text_items(content.get("paragraph_content") or []), None
        if block_type == "page_header":
            return self._join_text_items(content.get("page_header_content") or []), None
        if block_type == "page_footer":
            return self._join_text_items(content.get("page_footer_content") or []), None
        if block_type == "table":
            return self._join_text_items(content.get("table_caption") or content.get("table_body") or []), None
        if block_type == "image":
            caption = self._join_text_items(content.get("image_caption") or [])
            return (f"[插图] {caption}" if caption else ""), None
        return self._join_nested_text(content), None

    def _join_text_items(self, items: list[dict[str, Any]]) -> str:
        texts: list[str] = []
        for item in items:
            text = str(item.get("content") or "").strip()
            if text:
                texts.append(text)
        return "".join(texts).strip()

    def _join_nested_text(self, value: Any) -> str:
        texts: list[str] = []

        def walk(node: Any) -> None:
            if isinstance(node, dict):
                content = node.get("content")
                if isinstance(content, str) and content.strip():
                    texts.append(content.strip())
                for child in node.values():
                    walk(child)
            elif isinstance(node, list):
                for child in node:
                    walk(child)

        walk(value)
        return "".join(texts).strip()

    def _normalize_block_text(self, text: str) -> str:
        normalized = text.replace("\u3000", " ").replace("\xa0", " ")
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def _filter_blocks(self, blocks: list[MinerUBlock]) -> list[MinerUBlock]:
        filtered: list[MinerUBlock] = []
        in_front_matter = True
        catalog_page_numbers: set[int] = set()
        page_texts: dict[int, list[str]] = {}

        for block in blocks:
            text = block.text.strip()
            if not text:
                continue

            page_texts.setdefault(block.page, []).append(text)

            if block.block_type in {"page_header", "page_footer"}:
                continue

            if self._is_catalog_line(text):
                catalog_page_numbers.add(block.page)
                continue

            if in_front_matter:
                if self._is_main_content_anchor(text):
                    in_front_matter = False
                elif self._should_skip_front_matter_block(block):
                    continue

            if self._is_low_value_noise(text):
                continue

            filtered.append(block)

        for page_no, lines in page_texts.items():
            if self._is_catalog_like_page(lines, has_catalog_keyword=any(line.strip() == "目录" for line in lines)):
                catalog_page_numbers.add(page_no)

        if catalog_page_numbers:
            filtered = [block for block in filtered if block.page not in catalog_page_numbers]

        return filtered

    def _is_main_content_anchor(self, text: str) -> bool:
        if text.startswith("第一章") or text.startswith("第1章"):
            return True
        if "第一节" in text or text.startswith("一、"):
            return True
        return False

    def _should_skip_front_matter_block(self, block: MinerUBlock) -> bool:
        text = block.text
        if block.block_type == "title" and self._is_main_content_anchor(text):
            return False
        if any(keyword in text for keyword in FRONT_MATTER_KEYWORDS):
            return True
        if any(keyword == text for keyword in CATALOG_KEYWORDS):
            return True
        if len(text) <= 30 and ("出版社" in text or "邮政编码" in text):
            return True
        if re.search(r"ISBN|CIP|举报电话|定价|印张|印刷|责任编辑", text):
            return True
        return False

    def _is_catalog_line(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        if stripped == "目录":
            return True
        if "." in stripped and re.search(r"\d{1,4}\s*$", stripped):
            dotted_prefix, trailing_page = re.split(r"\d{1,4}\s*$", stripped, maxsplit=1)[0], True
            if dotted_prefix.count(".") >= 2:
                return True
        if re.fullmatch(r"[0-9IVXivx]+[.、]\s*.+", stripped):
            return True
        if re.fullmatch(r"第[一二三四五六七八九十百千万0-9]+[章节节].*\.{2,}\s*\d+", stripped):
            return True
        return False

    def _is_low_value_noise(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return True
        if len(stripped) <= 1:
            return True
        if re.fullmatch(r"[0-9IVXivx]+", stripped):
            return True
        if stripped in {"北京", "编著"}:
            return True
        if self._looks_like_figure_label_noise(stripped):
            return True
        return False

    def _looks_like_figure_label_noise(self, text: str) -> bool:
        stripped = text.strip()
        if len(stripped) < 2 or len(stripped) > 12:
            return False
        if self._looks_like_titled_line(stripped):
            return False
        if FIGURE_CAPTION_PATTERN.match(stripped) or TABLE_CAPTION_PATTERN.match(stripped):
            return False
        if re.search(r"[。！？；：.!?;:]$", stripped):
            return False
        if any(char.isdigit() for char in stripped):
            return False
        if re.fullmatch(r"[A-Za-zΑ-Ωα-ω]{1,4}", stripped):
            return True
        if re.fullmatch(r"[A-Za-zΑ-Ωα-ω]{1,3}[\u4e00-\u9fff]{1,4}", stripped):
            return True
        if re.fullmatch(r"[\u4e00-\u9fff]{2,6}", stripped):
            return True
        return False

    def _looks_like_exercise_prompt(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        if len(stripped) > 80:
            return False
        if re.fullmatch(r"(?:思考题|习题|练习题|本章小结|小结)", stripped):
            return True
        if re.fullmatch(r"(?:思考题|习题|练习题)\s*[0-9一二三四五六七八九十]+", stripped):
            return True
        if re.fullmatch(r"(?:[一二三四五六七八九十]+|\d+)[、.．].*", stripped):
            return True
        if re.fullmatch(r"[（(]\d+[)）].*", stripped):
            return True
        return False

    def _is_rear_matter_heading(self, title: str) -> bool:
        cleaned = self._clean_heading_text(title)
        return cleaned in REAR_MATTER_HEADINGS or any(
            cleaned.startswith(prefix) for prefix in REAR_MATTER_PREFIXES
        )

    def _merge_blocks_into_sections(self, blocks: list[MinerUBlock]) -> list[dict[str, Any]]:
        sections: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None

        for block in blocks:
            if block.block_type == "title":
                title_text = self._clean_heading_text(block.text)
                if not title_text:
                    continue

                if block.level and block.level >= 2:
                    if current is not None and not self._is_rear_matter_heading(title_text):
                        current["lines"].append(title_text)
                        current["end_page"] = block.page
                    continue

                if current and current["lines"]:
                    sections.append(current)
                current = {
                    "title": title_text,
                    "title_level": 1,
                    "start_page": block.page,
                    "end_page": block.page,
                    "lines": [],
                }
                continue

            if current is None:
                current = {
                    "title": "未命名章节",
                    "title_level": 1,
                    "start_page": block.page,
                    "end_page": block.page,
                    "lines": [],
                }

            current["lines"].append(block.text)
            current["end_page"] = block.page

        if current and current["lines"]:
            sections.append(current)
        return sections

    def _clean_sections(self, sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cleaned_sections: list[dict[str, Any]] = []
        for section in sections:
            text = self._clean_markdown_section_text(section["title"], section["lines"])
            if not text:
                continue
            cleaned_sections.append(
                {
                    "title": section["title"],
                    "title_level": section["title_level"],
                    "start_page": section["start_page"],
                    "end_page": section["end_page"],
                    "text": text,
                }
            )
        return cleaned_sections

    def _combine_section_lines(self, title: str, lines: list[str]) -> str:
        merged_lines = [title.strip(), ""]
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if IMAGE_LINE_PATTERN.search(stripped):
                continue
            merged_lines.append(stripped)
        return "\n".join(merged_lines).strip()

    def _clean_markdown_section_text(self, title: str, lines: list[str]) -> str:
        normalized_lines = self._normalize_markdown_section_lines(lines)
        body_lines = [line for line in normalized_lines if line]
        if self._is_rear_matter_heading(title):
            body_lines = [line for line in body_lines if not self._looks_like_exercise_prompt(line)]
        body_lines = self._compact_section_body_lines(body_lines)
        if not body_lines and not self._is_substantive_title_only_section(title):
            return ""

        parts = [title.strip()] if title.strip() else []
        if body_lines:
            parts.append("\n\n".join(body_lines))

        text = "\n\n".join(parts).strip()
        text = self._normalize_markdown_section_spacing(text)
        if len(text) < 20 and not self._is_substantive_title_only_section(title):
            return ""
        return text

    def _compact_section_body_lines(self, lines: list[str]) -> list[str]:
        compacted: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if self._should_drop_fragment_line(stripped):
                continue
            if compacted and self._should_merge_paragraph_into_previous(compacted[-1], stripped):
                merged = self._build_chunk_text([compacted[-1], stripped])
                if len(merged) <= self._config["max_chunk_length"] + 120:
                    compacted[-1] = merged
                    continue
            compacted.append(stripped)
        return compacted

    def _normalize_markdown_section_lines(self, lines: list[str]) -> list[str]:
        normalized_lines: list[str] = []
        current_paragraph: list[str] = []

        def flush_paragraph() -> None:
            if not current_paragraph:
                return
            paragraph = self._merge_markdown_paragraph(current_paragraph)
            current_paragraph.clear()
            if paragraph:
                normalized_lines.append(paragraph)

        for raw_line in lines:
            stripped = self._normalize_markdown_line(raw_line)
            if not stripped:
                flush_paragraph()
                continue
            if self._should_skip_markdown_line(stripped):
                flush_paragraph()
                continue
            if self._is_markdown_block_boundary(stripped):
                flush_paragraph()
                normalized_lines.append(stripped)
                continue
            current_paragraph.append(stripped)

        flush_paragraph()
        return normalized_lines

    def _merge_markdown_paragraph(self, lines: list[str]) -> str:
        cleaned: list[str] = []
        previous = ""
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if cleaned and stripped == previous:
                continue
            cleaned.append(stripped)
            previous = stripped

        if not cleaned:
            return ""
        if len(cleaned) == 1:
            return cleaned[0]

        merged = cleaned[0]
        for line in cleaned[1:]:
            merged = self._append_markdown_line(merged, line)
        return merged.strip()

    def _append_markdown_line(self, current: str, line: str) -> str:
        if not current:
            return line
        if self._should_keep_markdown_line_separate(current, line):
            return f"{current}\n{line}"
        if self._should_join_without_space(current, line):
            return f"{current}{line}"
        return f"{current} {line}"

    def _should_keep_markdown_line_separate(self, current: str, line: str) -> bool:
        return bool(
            FIGURE_CAPTION_PATTERN.match(line)
            or TABLE_CAPTION_PATTERN.match(line)
            or self._is_strong_markdown_heading(line)
            or current.endswith("：")
            or current.endswith(":")
            or current.endswith("；")
            or current.endswith(";")
        )

    def _should_join_without_space(self, current: str, line: str) -> bool:
        if current.endswith(("-", "—", "/", "（", "(", "“", '"')):
            return True
        if line.startswith(("，", "。", "；", "：", "）", ")", "、", "！", "？", ",", ".", ";", ":", "!", "?")):
            return True
        if re.search(r"[A-Za-z0-9]$", current) and re.match(r"^[A-Za-z0-9]", line):
            return False
        return True

    def _is_markdown_block_boundary(self, line: str) -> bool:
        return bool(
            FIGURE_CAPTION_PATTERN.match(line)
            or TABLE_CAPTION_PATTERN.match(line)
            or self._is_strong_markdown_heading(line)
        )

    def _is_strong_markdown_heading(self, line: str) -> bool:
        return bool(
            CHAPTER_HEADING_PATTERN.match(line)
            or SECTION_HEADING_PATTERN.match(line)
        )

    def _normalize_markdown_section_spacing(self, text: str) -> str:
        normalized = text.replace("\r\n", "\n")
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        normalized = re.sub(r"[ \t]+", " ", normalized)
        normalized = re.sub(r" ?\n ?", "\n", normalized)
        return normalized.strip()

    def _build_sections_from_full_md(self, full_md_path: Path) -> list[dict[str, Any]]:
        lines = full_md_path.read_text(encoding="utf-8").splitlines()
        pages = self._split_markdown_pages(lines)
        sections: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None
        body_started = False

        for page_no, page_lines in pages:
            if self._is_full_md_catalog_page(page_lines):
                continue

            for raw_line in page_lines:
                line = raw_line.strip()
                if not line:
                    if body_started and current and current["lines"] and current["lines"][-1] != "":
                        current["lines"].append("")
                    continue

                if line.startswith("<!--") and line.endswith("-->"):
                    continue
                if IMAGE_LINE_PATTERN.search(line):
                    continue

                normalized_line = self._normalize_markdown_line(line)
                if not normalized_line:
                    continue

                heading_info = self._parse_markdown_heading(normalized_line)
                if heading_info:
                    cleaned_title = self._clean_heading_text(heading_info["title"])
                    if not cleaned_title or self._should_skip_markdown_heading(cleaned_title):
                        continue

                    normalized_level = self._resolve_title_level(cleaned_title, heading_info["level"])
                    if normalized_level == 1 and self._is_body_entry_heading(cleaned_title):
                        body_started = True

                    if not body_started:
                        continue

                    if normalized_level >= 2:
                        if current is not None and not self._is_rear_matter_heading(cleaned_title):
                            current["lines"].append(cleaned_title)
                            current["end_page"] = page_no
                        continue

                    if current and self._section_has_meaningful_content(current):
                        sections.append(self._finalize_markdown_section(current))
                    current = {
                        "title": cleaned_title,
                        "title_level": normalized_level,
                        "start_page": page_no,
                        "end_page": page_no,
                        "lines": [],
                    }
                    continue

                if not body_started or self._should_skip_markdown_line(normalized_line):
                    continue

                if current is None:
                    continue

                current["lines"].append(normalized_line)
                current["end_page"] = page_no

        if current and self._section_has_meaningful_content(current):
            sections.append(self._finalize_markdown_section(current))

        return [
            section
            for section in self._clean_sections(
                [section for section in sections if not self._is_catalog_only_section(section)]
            )
            if section["title_level"] == 1
        ]

    def _split_markdown_pages(self, lines: list[str]) -> list[tuple[int, list[str]]]:
        pages: list[tuple[int, list[str]]] = []
        current_page = 1
        current_lines: list[str] = []

        for raw_line in lines:
            stripped = raw_line.strip()
            if stripped.startswith("<!-- Page") and stripped.endswith("-->"):
                if current_lines:
                    pages.append((current_page, current_lines))
                    current_lines = []
                current_page += 1
                continue
            current_lines.append(raw_line)

        if current_lines:
            pages.append((current_page, current_lines))

        if not pages:
            return [(1, lines)]
        return pages

    def _is_full_md_catalog_page(self, page_lines: list[str]) -> bool:
        meaningful_lines: list[str] = []
        has_catalog_keyword = False

        for raw_line in page_lines:
            line = raw_line.strip()
            if not line or (line.startswith("<!--") and line.endswith("-->")):
                continue
            if IMAGE_LINE_PATTERN.search(line):
                continue

            normalized_line = self._normalize_markdown_line(line)
            if not normalized_line:
                continue

            heading_info = self._parse_markdown_heading(normalized_line)
            text = self._clean_heading_text(heading_info["title"]) if heading_info else normalized_line
            if not text:
                continue

            if text == "目录":
                has_catalog_keyword = True
                continue

            meaningful_lines.append(text)

        return self._is_catalog_like_page(meaningful_lines, has_catalog_keyword=has_catalog_keyword)

    def _is_catalog_like_page(self, lines: list[str], has_catalog_keyword: bool = False) -> bool:
        meaningful_lines = [line.strip() for line in lines if line and line.strip()]
        if not meaningful_lines:
            return has_catalog_keyword

        catalog_like_count = sum(1 for line in meaningful_lines if self._looks_like_catalog_entry(line))
        titled_count = sum(1 for line in meaningful_lines if self._looks_like_titled_line(line))
        short_count = sum(1 for line in meaningful_lines if len(line) <= 40)
        chapter_or_section_count = sum(
            1 for line in meaningful_lines if CHAPTER_HEADING_PATTERN.match(line) or SECTION_HEADING_PATTERN.match(line)
        )
        entry_prefix_count = sum(1 for line in meaningful_lines if CATALOG_ENTRY_PREFIX_PATTERN.match(line))
        trailing_page_count = sum(1 for line in meaningful_lines if re.search(r"\d{1,4}\s*$", line))
        sentence_like_count = sum(1 for line in meaningful_lines if SENTENCE_END_PATTERN.search(line))
        max_len = max(len(line) for line in meaningful_lines)

        if has_catalog_keyword and catalog_like_count >= max(2, len(meaningful_lines) // 2):
            return True
        if catalog_like_count >= max(3, len(meaningful_lines) - 1) and short_count == len(meaningful_lines):
            return True
        if titled_count >= 4 and short_count == len(meaningful_lines) and max_len <= 60:
            return True
        if chapter_or_section_count >= 3 and entry_prefix_count >= 3 and sentence_like_count <= 1 and max_len <= 80:
            return True
        if trailing_page_count >= 3 and entry_prefix_count >= 3 and sentence_like_count == 0:
            return True
        if len(meaningful_lines) <= 12 and catalog_like_count >= max(4, len(meaningful_lines) - 2) and sentence_like_count <= 1:
            return True
        return False

    def _is_body_entry_heading(self, title: str) -> bool:
        cleaned = self._clean_heading_text(title)
        if cleaned in BODY_FIRST_CHAPTER_TITLES:
            return True
        return bool(CHAPTER_HEADING_PATTERN.match(cleaned))

    def _estimate_page_count_from_markdown(self, full_md_path: Path) -> int:
        content = full_md_path.read_text(encoding="utf-8")
        page_markers = len(re.findall(r"<!--\s*Page", content))
        if page_markers > 0:
            return page_markers
        return max(1, len(content.splitlines()) // 30)

    def _normalize_markdown_line(self, line: str) -> str:
        normalized = line.replace("\u3000", " ").replace("\xa0", " ").strip()
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()

    def _parse_markdown_heading(self, line: str) -> dict[str, Any] | None:
        match = MARKDOWN_HEADING_PATTERN.match(line)
        if not match:
            return None
        return {
            "level": len(match.group(1)),
            "title": match.group(2).strip(),
        }

    def _clean_heading_text(self, text: str) -> str:
        stripped = self._normalize_markdown_line(text)
        if not stripped:
            return ""
        if stripped == "目录":
            return stripped
        match = TRAILING_PAGE_NUMBER_PATTERN.match(stripped)
        if match:
            candidate = match.group(1).strip()
            if self._looks_like_titled_line(candidate):
                stripped = candidate
        stripped = stripped.rstrip("·•●.-— ").strip()
        return stripped

    def _looks_like_titled_line(self, text: str) -> bool:
        return any(
            pattern.match(text)
            for pattern in (
                CHAPTER_HEADING_PATTERN,
                SECTION_HEADING_PATTERN,
                SUBSECTION_HEADING_PATTERN,
                SUBSUBSECTION_HEADING_PATTERN,
                NUMBERED_HEADING_PATTERN,
            )
        )

    def _resolve_title_level(self, title: str, markdown_level: int) -> int:
        if CHAPTER_HEADING_PATTERN.match(title):
            return 1
        if SECTION_HEADING_PATTERN.match(title):
            return 2
        if SUBSECTION_HEADING_PATTERN.match(title):
            return 3
        if SUBSUBSECTION_HEADING_PATTERN.match(title):
            return 4
        if NUMBERED_HEADING_PATTERN.match(title):
            return 4
        return markdown_level

    def _is_body_start_heading(self, title: str) -> bool:
        cleaned = self._clean_heading_text(title)
        return bool(
            CHAPTER_HEADING_PATTERN.match(cleaned)
            or cleaned in {"序", "第一版前言", "第二版前言", "前言"}
        )

    def _is_catalog_terminator(self, title: str) -> bool:
        cleaned = self._clean_heading_text(title)
        return cleaned in FRONT_MATTER_HEADINGS or CHAPTER_HEADING_PATTERN.match(cleaned) is not None

    def _should_skip_markdown_heading(self, title: str) -> bool:
        if not title:
            return True
        if title in CATALOG_KEYWORDS:
            return True
        if self._is_low_value_noise(title):
            return True
        if title in {"导航概论（第二版）", "导航概论 (第二版)", "导航概论", "版权所有，侵权必究"}:
            return True
        if any(keyword in title for keyword in FRONT_MATTER_KEYWORDS) and not self._is_body_start_heading(title):
            return True
        return False

    def _should_skip_markdown_line(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return True
        if PURE_PAGE_NUMBER_PATTERN.fullmatch(stripped):
            return True
        if FORMULA_ONLY_PATTERN.fullmatch(stripped):
            return True
        if self._is_catalog_line(stripped):
            return True
        if self._looks_like_catalog_entry(stripped) and not self._looks_like_titled_line(stripped):
            return True
        if stripped.startswith("![]("):
            return True
        if self._looks_like_figure_label_noise(stripped):
            return True
        return False

    def _looks_like_catalog_entry(self, line: str) -> bool:
        if not line:
            return False
        stripped = line.strip()
        if not stripped:
            return False
        if re.search(r"\s\d{1,4}$", stripped) and self._looks_like_titled_line(re.sub(r"\s\d{1,4}$", "", stripped).strip()):
            return True
        heading_like_count = len(re.findall(r"(?:第[一二三四五六七八九十百千万0-9]+[章节节]|[一二三四五六七八九十]+、|[（(][一二三四五六七八九十0-9]+[)）])", stripped))
        if heading_like_count >= 2 and len(stripped) <= 120:
            return True
        if re.search(r"\.{2,}|·{2,}|•{2,}|…{2,}", stripped) and re.search(r"\d{1,4}\s*$", stripped):
            return True
        compact = re.sub(r"\s+", "", stripped)
        compact_heading_count = len(CATALOG_LEADER_PATTERN.findall(compact))
        if compact_heading_count >= 2 and len(compact) <= 140:
            return True
        if compact_heading_count >= 1 and re.search(r"\d{2,4}", compact) and len(compact) <= 100:
            return True
        return False

    def _is_catalog_only_section(self, section: dict[str, Any]) -> bool:
        title = (section.get("title") or "").strip()
        lines = [line.strip() for line in section.get("lines", []) if line.strip()]
        if not lines:
            return True
        if title and self._looks_like_chapter_index_page(title, lines):
            return True
        if self._is_catalog_like_page(([title] if title else []) + lines):
            return True
        if len(lines) <= 3 and all(self._looks_like_catalog_entry(line) or self._clean_heading_text(line) == line and self._looks_like_titled_line(line) for line in lines):
            return True
        if len(lines) <= 2 and all(len(line) <= 30 for line in lines):
            return True
        return False

    def _looks_like_chapter_index_page(self, title: str, lines: list[str]) -> bool:
        if not CHAPTER_HEADING_PATTERN.match(title):
            return False
        meaningful_lines = [line for line in lines if line.strip()]
        if not meaningful_lines:
            return True
        if len(meaningful_lines) > 10:
            return False
        title_like_lines = sum(
            1
            for line in meaningful_lines
            if self._looks_like_titled_line(line) or line.startswith("思考题")
        )
        if title_like_lines < max(3, len(meaningful_lines) - 1):
            return False
        if any(len(line) > 45 and not line.startswith("思考题") for line in meaningful_lines):
            return False
        return True

    def _section_has_meaningful_content(self, section: dict[str, Any]) -> bool:
        text = "\n".join(line for line in section["lines"] if line.strip()).strip()
        if len(text) < 20 and not self._is_substantive_title_only_section(section["title"]):
            return False
        return True

    def _is_substantive_title_only_section(self, title: str) -> bool:
        return title in {"序", "第一版前言", "第二版前言", "前言"} or CHAPTER_HEADING_PATTERN.match(title) is not None

    def _finalize_markdown_section(self, section: dict[str, Any]) -> dict[str, Any]:
        return {
            "title": section["title"],
            "title_level": section["title_level"],
            "start_page": section["start_page"],
            "end_page": max(section["start_page"], section["end_page"]),
            "lines": self._normalize_section_lines(section["lines"]),
        }

    def _normalize_section_lines(self, lines: list[str]) -> list[str]:
        normalized_lines: list[str] = []
        pending_short_line = ""

        for line in lines:
            stripped = line.strip()
            if not stripped:
                pending_short_line = ""
                if normalized_lines and normalized_lines[-1] != "":
                    normalized_lines.append("")
                continue
            if FIGURE_CAPTION_PATTERN.match(stripped) or TABLE_CAPTION_PATTERN.match(stripped):
                pending_short_line = ""
                normalized_lines.append(stripped)
                continue
            if self._should_drop_fragment_line(stripped):
                continue
            if pending_short_line:
                stripped = self._append_markdown_line(pending_short_line, stripped)
                pending_short_line = ""
            if self._looks_like_orphan_fragment(stripped):
                if normalized_lines and self._can_attach_fragment_to_previous(normalized_lines[-1], stripped):
                    normalized_lines[-1] = self._append_markdown_line(normalized_lines[-1], stripped)
                else:
                    pending_short_line = stripped
                continue
            normalized_lines.append(stripped)

        if pending_short_line:
            if normalized_lines and self._can_attach_fragment_to_previous(normalized_lines[-1], pending_short_line):
                normalized_lines[-1] = self._append_markdown_line(normalized_lines[-1], pending_short_line)
            else:
                normalized_lines.append(pending_short_line)

        while normalized_lines and normalized_lines[-1] == "":
            normalized_lines.pop()
        return normalized_lines

    def _should_drop_fragment_line(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return True
        if self._looks_like_figure_label_noise(stripped):
            return True
        if len(stripped) <= 6 and not self._looks_like_titled_line(stripped) and not re.search(r"[。！？；：.!?;:]", stripped):
            return True
        if self._is_short_figure_adjacent_noise(stripped):
            return True
        return False

    def _looks_like_orphan_fragment(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        # 完整段落可能以“并发”“其中”等词开头，不能仅凭首字当作残句。
        if len(stripped) > 28 and SENTENCE_END_PATTERN.search(stripped):
            return False
        if self._looks_like_titled_line(stripped):
            return False
        if FIGURE_CAPTION_PATTERN.match(stripped) or TABLE_CAPTION_PATTERN.match(stripped):
            return False
        if len(stripped) <= 28 and not SENTENCE_END_PATTERN.search(stripped):
            return True
        if self._starts_like_continuation(stripped):
            return True
        if self._ends_like_incomplete_sentence(stripped) and len(stripped) <= 42:
            return True
        return False

    def _can_attach_fragment_to_previous(self, previous: str, fragment: str) -> bool:
        prev = previous.strip()
        frag = fragment.strip()
        if not prev or not frag:
            return False
        if prev == "" or prev in {"：", ":"}:
            return False
        if self._looks_like_titled_line(prev):
            return True
        if FIGURE_CAPTION_PATTERN.match(prev) or TABLE_CAPTION_PATTERN.match(prev):
            return False
        if self._ends_like_incomplete_sentence(prev):
            return True
        if len(prev) <= 36:
            return True
        return self._starts_like_continuation(frag)

    def _is_short_figure_adjacent_noise(self, text: str) -> bool:
        stripped = text.strip()
        if len(stripped) > 18:
            return False
        if FIGURE_CAPTION_PATTERN.match(stripped) or TABLE_CAPTION_PATTERN.match(stripped):
            return False
        if self._looks_like_titled_line(stripped):
            return False
        if re.search(r"[。！？；：.!?;:]", stripped):
            return False
        return bool(re.fullmatch(r"(?:[A-Za-zΑ-Ωα-ω]{1,6}|[\u4e00-\u9fff]{2,8}|[A-Za-zΑ-Ωα-ω]{1,4}[\u4e00-\u9fff]{1,4})", stripped))

    def _chunk_sections(self, sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        chunk_id = 1

        for section in sections:
            paragraphs = self._split_section_into_paragraphs(section["text"])
            current_start_page = section["start_page"]

            for paragraph in paragraphs:
                paragraph_text = paragraph["text"]
                paragraph_page = paragraph.get("page", section["start_page"])
                split_parts = self._recursive_split_text(
                    paragraph_text,
                    ["\n\n", "\n", "sentence", "clause"],
                    self._config["max_chunk_length"],
                )
                split_parts = [part for part in split_parts if part and not self._should_drop_paragraph(part)]
                if not split_parts:
                    continue

                current_parts: list[str] = []
                last_page = paragraph_page
                for part in split_parts:
                    if not current_parts:
                        current_start_page = paragraph_page
                    should_flush = self._should_flush_before_append(current_parts, part)
                    if should_flush:
                        chunk_text = self._build_chunk_text(current_parts)
                        chunk_record = self._build_chunk_record(
                            chunk_id,
                            section,
                            chunk_text,
                            current_start_page,
                            last_page,
                        )
                        if chunk_record:
                            chunks.append(chunk_record)
                            chunk_id += 1
                        overlap_text = self._tail_overlap_text(chunk_text)
                        current_parts = [overlap_text] if overlap_text else []
                        current_start_page = last_page

                    current_parts.append(part)
                    last_page = paragraph_page

                if current_parts:
                    chunk_text = self._build_chunk_text(current_parts)
                    if (
                        chunks
                        and len(chunk_text) < self._config["min_chunk_length"]
                        and chunks[-1]["section_title"] == section["title"]
                        and chunks[-1]["char_count"] + 2 + len(chunk_text) <= self._config["chunk_soft_limit"]
                    ):
                        merged_text = self._build_chunk_text([chunks[-1]["text"], chunk_text])
                        chunks[-1]["text"] = merged_text
                        chunks[-1]["char_count"] = len(merged_text)
                        chunks[-1]["end_page"] = last_page
                    else:
                        chunk_record = self._build_chunk_record(
                            chunk_id,
                            section,
                            chunk_text,
                            current_start_page,
                            last_page,
                        )
                        if chunk_record:
                            chunks.append(chunk_record)
                            chunk_id += 1

        return self._merge_small_or_title_only_chunks(chunks)

    def _split_section_into_paragraphs(self, text: str) -> list[dict[str, Any]]:
        raw_parts = [part.strip() for part in text.split("\n\n") if part.strip()]
        paragraphs: list[dict[str, Any]] = []
        current_page = 1

        for part in raw_parts:
            current_page = self._extract_page_hint(part, current_page)
            normalized_part = self._strip_page_marker(part).strip()
            if not normalized_part:
                continue
            if self._should_drop_paragraph(normalized_part):
                continue
            if paragraphs and self._should_merge_paragraph_into_previous(paragraphs[-1]["text"], normalized_part):
                merged_text = self._build_chunk_text([paragraphs[-1]["text"], normalized_part])
                if len(merged_text) <= self._config["max_chunk_length"] + 80:
                    paragraphs[-1]["text"] = merged_text
                    continue
            paragraphs.append({"text": normalized_part, "page": current_page})
        return paragraphs

    def _recursive_split_text(self, text: str, separators: list[str], max_length: int) -> list[str]:
        stripped = text.strip()
        if not stripped:
            return []
        if len(stripped) <= max_length:
            return [stripped]
        if not separators:
            return self._split_by_length(stripped)

        separator = separators[0]
        parts = self._split_text_by_separator(stripped, separator)
        if len(parts) <= 1:
            return self._recursive_split_text(stripped, separators[1:], max_length)

        result: list[str] = []
        current = ""
        for part in parts:
            candidate = self._build_chunk_text([current, part]) if current else part
            if len(candidate) <= max_length:
                current = candidate
                continue
            if current:
                result.extend(self._recursive_split_text(current, separators[1:], max_length))
            current = part
        if current:
            result.extend(self._recursive_split_text(current, separators[1:], max_length))
        return self._merge_recursive_split_parts(result, max_length)

    def _split_text_by_separator(self, text: str, separator: str) -> list[str]:
        if separator == "\n\n":
            return [part.strip() for part in text.split("\n\n") if part.strip()]
        if separator == "\n":
            return [part.strip() for part in text.split("\n") if part.strip()]
        if separator == "sentence":
            return [part.strip() for part in re.split(r"(?<=[。！？；：!?;:])", text) if part.strip()]
        if separator == "clause":
            return [part.strip() for part in re.split(r"(?<=[，、,])", text) if part.strip()]
        return [text.strip()]

    def _merge_recursive_split_parts(self, parts: list[str], max_length: int) -> list[str]:
        merged: list[str] = []
        for part in parts:
            stripped = part.strip()
            if not stripped:
                continue
            if merged:
                candidate = self._build_chunk_text([merged[-1], stripped])
                if len(stripped) <= max(80, self._config["min_chunk_length"] // 2) and len(candidate) <= max_length:
                    merged[-1] = candidate
                    continue
                if self._starts_like_continuation(stripped) and len(candidate) <= max_length:
                    merged[-1] = candidate
                    continue
            merged.append(stripped)
        return merged

    def _should_merge_paragraph_into_previous(self, previous: str, current: str) -> bool:
        prev = previous.strip()
        curr = current.strip()
        if not prev or not curr:
            return False
        if self._looks_like_titled_line(curr):
            return False
        if FIGURE_CAPTION_PATTERN.match(curr) or TABLE_CAPTION_PATTERN.match(curr):
            return False
        if self._starts_like_continuation(curr):
            return True
        if self._ends_like_incomplete_sentence(prev):
            return True
        if len(curr) <= 40 and not SENTENCE_END_PATTERN.search(curr):
            return True
        if len(prev) <= 36:
            return True
        return False

    def _should_drop_paragraph(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return True
        if stripped in {"作者", "编著"}:
            return True
        if PURE_PAGE_NUMBER_PATTERN.fullmatch(stripped):
            return True
        if FORMULA_ONLY_PATTERN.fullmatch(stripped):
            return True
        if self._looks_like_catalog_entry(stripped):
            return True
        if self._looks_like_figure_or_table_noise(stripped):
            return True
        if self._looks_like_too_short_noise(stripped):
            return True
        return False

    def _looks_like_figure_or_table_noise(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        if FIGURE_CAPTION_PATTERN.match(stripped) or TABLE_CAPTION_PATTERN.match(stripped):
            return True
        if len(stripped) <= 28 and self._looks_like_figure_label_noise(stripped):
            return True
        if re.fullmatch(r"(?:\[插图\]\s*)?图\s*\d+(?:\s*[-—．.]\s*\d+)*", stripped):
            return True
        if re.fullmatch(r"(?:\[表格\]\s*)?表\s*\d+(?:\s*[-—．.]\s*\d+)*", stripped):
            return True
        return False

    def _looks_like_too_short_noise(self, text: str) -> bool:
        stripped = text.strip()
        if len(stripped) >= 12:
            return False
        if self._looks_like_titled_line(stripped):
            return False
        if re.search(r"[A-Za-z\u4e00-\u9fff]", stripped) is None:
            return True
        if len(stripped) <= 4 and stripped not in {"序", "前言"}:
            return True
        return False

    def _merge_small_or_title_only_chunks(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not chunks:
            return chunks
        merged: list[dict[str, Any]] = []
        for chunk in chunks:
            if self._should_merge_chunk_into_previous(chunk, merged):
                previous = merged[-1]
                previous["text"] = self._build_chunk_text([previous["text"], chunk["text"]])
                previous["char_count"] = len(previous["text"])
                previous["end_page"] = max(previous["end_page"], chunk["end_page"])
                continue
            merged.append(chunk)

        for idx, chunk in enumerate(merged, start=1):
            chunk["chunk_id"] = idx
        return merged

    def _should_merge_chunk_into_previous(self, chunk: dict[str, Any], merged: list[dict[str, Any]]) -> bool:
        if not merged:
            return False
        text = chunk["text"].strip()
        previous = merged[-1]
        if chunk["section_title"] != previous["section_title"]:
            return False
        conservative_merge_limit = min(
            self._config["chunk_soft_limit"],
            self._config["max_chunk_length"] + 60,
        )
        merged_length = previous["char_count"] + 2 + len(text)
        if merged_length > conservative_merge_limit:
            return False
        if len(text) < max(self._config["min_chunk_length"], 520):
            return True
        if text == chunk["section_title"].strip():
            return True
        if len(text.splitlines()) <= 3 and self._looks_like_titled_line(text):
            return True
        if self._starts_like_continuation(text) and len(text) <= max(180, self._config["chunk_overlap"] * 2):
            return True
        if self._ends_like_incomplete_sentence(previous["text"]) and len(text) <= max(260, self._config["min_chunk_length"]):
            return True
        if previous["char_count"] < max(self._config["target_chunk_length"] - 200, self._config["min_chunk_length"]):
            return True
        return False

    def _build_chunk_record(
        self,
        chunk_id: int,
        section: dict[str, Any],
        text: str,
        start_page: int | None = None,
        end_page: int | None = None,
    ) -> dict[str, Any] | None:
        cleaned_text = text.strip()
        if not cleaned_text:
            return None
        if self._is_catalog_like_page(cleaned_text.splitlines()):
            return None
        resolved_start_page = start_page if start_page is not None else section["start_page"]
        resolved_end_page = end_page if end_page is not None else section["end_page"]
        return {
            "chunk_id": chunk_id,
            "section_title": section["title"],
            "title_level": section["title_level"],
            "start_page": resolved_start_page,
            "end_page": max(resolved_start_page, resolved_end_page),
            "char_count": len(cleaned_text),
            "text": cleaned_text,
        }

    def _backfill_chunk_pages_from_content_list(
        self,
        chunks: list[dict[str, Any]],
        content_list_path: Path,
    ) -> None:
        try:
            pages = json.loads(content_list_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"读取 content_list_v2.json 失败，跳过 chunk 页码回填: {exc}")
            return

        page_texts = self._build_page_texts_from_content_list(pages)
        if not page_texts:
            return

        for chunk in chunks:
            matched_range = self._locate_chunk_page_range(chunk["text"], page_texts)
            if matched_range is None:
                continue
            chunk["start_page"], chunk["end_page"] = matched_range

    def _build_page_texts_from_content_list(self, pages: list[list[dict[str, Any]]]) -> list[str]:
        page_texts: list[str] = []
        for page_blocks in pages:
            lines: list[str] = []
            for block in page_blocks:
                block_type = str(block.get("type") or "").strip()
                if block_type in {"page_header", "page_footer"}:
                    continue
                text, _ = self._extract_block_text(block_type, block.get("content") or {})
                normalized = self._normalize_block_text(text)
                if not normalized or self._should_skip_page_alignment_text(normalized):
                    continue
                lines.append(normalized)
            page_texts.append("\n".join(lines))
        return page_texts

    def _should_skip_page_alignment_text(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return True
        if self._is_catalog_line(stripped):
            return True
        if self._looks_like_catalog_entry(stripped):
            return True
        if self._is_low_value_noise(stripped):
            return True
        if self._looks_like_figure_or_table_noise(stripped):
            return True
        if PURE_PAGE_NUMBER_PATTERN.fullmatch(stripped):
            return True
        return False

    def _locate_chunk_page_range(
        self,
        chunk_text: str,
        page_texts: list[str],
    ) -> tuple[int, int] | None:
        chunk_compact = self._compact_alignment_text(chunk_text)
        if len(chunk_compact) < 12:
            return None

        best_start: int | None = None
        best_end: int | None = None
        best_score = -1

        for start_idx in range(len(page_texts)):
            combined = ""
            for end_idx in range(start_idx, min(len(page_texts), start_idx + 4)):
                combined += page_texts[end_idx]
                combined_compact = self._compact_alignment_text(combined)
                if not combined_compact:
                    continue
                score = self._score_chunk_page_match(chunk_compact, combined_compact)
                if score > best_score:
                    best_score = score
                    best_start = start_idx + 1
                    best_end = end_idx + 1
                    if score >= 3:
                        break
            if best_score >= 3:
                break

        if best_start is None or best_end is None or best_score <= 0:
            return None
        return best_start, best_end

    def _score_chunk_page_match(self, chunk_text: str, page_text: str) -> int:
        if not chunk_text or not page_text:
            return 0

        score = 0
        anchors = self._build_alignment_anchors(chunk_text)
        for anchor in anchors:
            if anchor and anchor in page_text:
                score += 1

        if chunk_text[: min(len(chunk_text), 80)] in page_text:
            score += 2
        if len(chunk_text) >= 160 and chunk_text[-80:] in page_text:
            score += 2
        if len(chunk_text) >= 80 and chunk_text in page_text:
            score += 4
        return score

    def _build_alignment_anchors(self, text: str) -> list[str]:
        length = len(text)
        anchor_sizes = [24, 40, 64]
        anchors: list[str] = []
        for size in anchor_sizes:
            if length >= size:
                anchors.append(text[:size])
                anchors.append(text[-size:])
        if length >= 120:
            mid = length // 2
            half = 20
            anchors.append(text[max(0, mid - half): min(length, mid + half)])
        return [anchor for anchor in anchors if len(anchor) >= 16]

    def _compact_alignment_text(self, text: str) -> str:
        return re.sub(r"\s+", "", text or "")

    def _build_chunk_text(self, parts: list[str]) -> str:
        return "\n\n".join(part.strip() for part in parts if part.strip()).strip()

    def _should_flush_before_append(self, current_parts: list[str], paragraph: str) -> bool:
        if not current_parts:
            return False
        current_text = self._build_chunk_text(current_parts)
        projected = len(self._build_chunk_text([*current_parts, paragraph]))
        current_length = len(current_text)
        paragraph_length = len(paragraph.strip())
        hard_flush_threshold = self._config["max_chunk_length"] + max(80, self._config["chunk_overlap"] // 2)
        preferred_flush_threshold = max(
            self._config["target_chunk_length"] + max(160, self._config["chunk_overlap"]),
            self._config["max_chunk_length"] - 80,
        )
        continuation_guard_limit = min(
            self._config["chunk_soft_limit"],
            self._config["max_chunk_length"] + max(60, self._config["chunk_overlap"] // 2),
        )
        if projected > self._config["chunk_soft_limit"]:
            return True
        if current_length < self._config["min_chunk_length"]:
            return False
        if self._ends_like_incomplete_sentence(current_text) or self._starts_like_continuation(paragraph):
            return False
        if projected > continuation_guard_limit:
            return True
        if projected >= hard_flush_threshold:
            return True
        if projected >= preferred_flush_threshold:
            if paragraph_length <= max(180, self._config["chunk_overlap"] + 40):
                return False
            if self._looks_like_strong_boundary(current_parts[-1]):
                return True
            if self._looks_like_good_boundary(current_text) and self._looks_like_good_boundary(paragraph):
                return True
        return False

    def _extract_page_hint(self, text: str, fallback_page: int) -> int:
        match = re.search(r"<!--\s*Page\s*(\d+)\s*-->", text)
        if match:
            return int(match.group(1))
        return fallback_page

    def _strip_page_marker(self, text: str) -> str:
        return re.sub(r"<!--\s*Page\s*\d+\s*-->", "", text).strip()

    def _split_by_length(self, text: str) -> list[str]:
        text = text.strip()
        if len(text) <= self._config["max_chunk_length"]:
            return [text]

        segments: list[str] = []
        start = 0
        max_length = self._config["max_chunk_length"]
        min_length = max(1, self._config["min_chunk_length"])

        while start < len(text):
            end = min(len(text), start + max_length)
            if end < len(text):
                window = text[start:end]
                boundary = max(
                    window.rfind(mark)
                    for mark in ("。", "！", "？", "；", "：", ".", "!", "?", ";", ":", "\n")
                )
                if boundary >= min_length:
                    end = start + boundary + 1
            segment = text[start:end].strip()
            if segment:
                segments.append(segment)
            if end <= start:
                break
            start = end

        return segments or [text]

    def _tail_overlap_text(self, text: str) -> str:
        overlap = self._config["chunk_overlap"]
        if overlap <= 0 or len(text) <= overlap:
            return ""
        tail = text[-overlap:]
        split_index = tail.find("\n\n")
        if split_index != -1:
            candidate = tail[split_index + 2 :].strip()
            if len(candidate) >= 40 and not self._starts_like_continuation(candidate) and not self._looks_like_figure_or_table_noise(candidate):
                return candidate
        sentence_boundaries = [tail.rfind(mark) for mark in ("。", "！", "？", "；", ":", "!", "?", ";")]
        split_index = max(sentence_boundaries)
        if split_index != -1 and split_index + 1 < len(tail):
            candidate = tail[split_index + 1 :].strip()
            if len(candidate) >= 40 and not self._starts_like_continuation(candidate) and not self._looks_like_figure_or_table_noise(candidate):
                return candidate
        candidate = tail.strip()
        if len(candidate) < 40 or self._starts_like_continuation(candidate) or self._looks_like_figure_or_table_noise(candidate):
            return ""
        return candidate

    def _ends_like_incomplete_sentence(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        last_line = next((line.strip() for line in reversed(stripped.splitlines()) if line.strip()), "")
        if not last_line:
            return False
        if self._looks_like_titled_line(last_line):
            return False
        if FIGURE_CAPTION_PATTERN.match(last_line) or TABLE_CAPTION_PATTERN.match(last_line):
            return False
        if SENTENCE_END_PATTERN.search(last_line):
            return False
        if len(last_line) <= 18:
            return True
        return SENTENCE_CONTINUATION_ENDING_PATTERN.search(last_line) is not None

    def _starts_like_continuation(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        if self._looks_like_titled_line(stripped):
            return False
        if FIGURE_CAPTION_PATTERN.match(stripped) or TABLE_CAPTION_PATTERN.match(stripped):
            return False
        if LEADING_CONTINUATION_PATTERN.match(stripped):
            return True
        return bool(re.match(r"^[）)、，；：,:]", stripped))

    def _looks_like_good_boundary(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        if HEADING_PATTERN.match(stripped):
            return True
        return bool(SENTENCE_END_PATTERN.search(stripped))

    def _looks_like_strong_boundary(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        if CHAPTER_HEADING_PATTERN.match(stripped) or SECTION_HEADING_PATTERN.match(stripped):
            return True
        if FIGURE_CAPTION_PATTERN.match(stripped) or TABLE_CAPTION_PATTERN.match(stripped):
            return True
        if len(stripped) >= 80 and SENTENCE_END_PATTERN.search(stripped):
            return True
        return False

    def _detect_source_file_name(self, base_dir: Path) -> str:
        pdf_files = sorted(base_dir.glob("*_origin.pdf"))
        if pdf_files:
            return pdf_files[0].name
        return ""

