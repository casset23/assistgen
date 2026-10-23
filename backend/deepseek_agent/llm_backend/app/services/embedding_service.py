from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from collections import Counter
from difflib import SequenceMatcher
from threading import Lock
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
import faiss
import json
from pathlib import Path
import hashlib
import math
import re

from app.services.book_rag.document_parser_service import DocumentParserService


class _RerankerLoadLogger:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def __enter__(self):
        print(f"[EmbeddingService] 开始加载 reranker 模型: {self.model_name}", flush=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            print(f"[EmbeddingService] reranker 模型加载完成: {self.model_name}", flush=True)
        else:
            print(f"[EmbeddingService] reranker 模型加载失败: {self.model_name}，原因: {exc}", flush=True)
        return False


SENTENCE_ENDINGS = ("。", "！", "？", "；", "：", ".", "!", "?", ";", ":", "）", ")", "】")
CONTINUATION_PREFIX_PATTERN = re.compile(r"^(?:[A-Za-z0-9Α-Ωα-ω（(]|[的地得了着并且且及与或而但并把将被于从向按对为在使由所其之仍即再又也均都则后前中内外上一下里]|制|统|行|测|控|导|算|姿|向|速|率|角|量|置|系|器|图|场)")
TITLE_OR_ENUM_PATTERN = re.compile(r"^(?:第[一二三四五六七八九十百千万0-9]+[章节编部分]|[一二三四五六七八九十]+、|[（(][一二三四五六七八九十0-9]+[)）])")
TRAILING_INCOMPLETE_PATTERN = re.compile(r"[\u4e00-\u9fffA-Za-z0-9）)]$")
CATALOGISH_LINE_PATTERN = re.compile(r"^(?:第[一二三四五六七八九十百千万0-9]+[章节节].*|[一二三四五六七八九十]+、.+|[（(][一二三四五六七八九十0-9]+[)）].*|[0-9IVXivx]+[.、].*)$")
PUNCTUATION_SPLIT_PATTERN = re.compile(r"(?<=[。！？；：.!?;:])")
TRAILING_FRAGMENT_PATTERN = re.compile(r"(?:[\u4e00-\u9fffA-Za-z0-9]{1,18}|[，、：；,.]{0,2}[\u4e00-\u9fffA-Za-z0-9]{1,12})$")
REPEATED_PUNCTUATION_PATTERN = re.compile(r"([。！？；：，、,.!?;:])\1+")
LOW_VALUE_LINE_PATTERN = re.compile(r"^(?:[A-Za-z0-9]{1,4}|[·•●\-_=+*/\\|]{1,10}|[A-Za-z0-9._-]{1,14}|[ivxlcdmIVXLCDM]{1,8}|\d{1,4})$")
NOISE_LINE_PATTERN = re.compile(r"^(?:作考|作者|录I|录Ⅱ|录Ⅲ|图书在版编目|CIP|ISBN|开本|定价|责任编辑|责任校对|责任印制|封面设计|中国国家版本馆|数据核字|年\s*月\s*日.*|[A-Za-z]{1,3}\s*[:：]\s*[A-Za-z0-9-]{1,12})")
HEADER_LIKE_PATTERN = re.compile(r"^(?:[国是]?导航.{0,6}[论版].*(?:第二版)?|.*导航概论.*(?:第二版)?|.*普通高等教育本科规划教材.*)$")
FRONT_MATTER_KEYWORDS = ("版权所有", "侵权必究", "图书在版编目", "CIP", "中国国家版本馆", "定价", "开本", "编著", "出版社", "责任编辑")
TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_.-]+")
RRF_K = 60
SPARSE_CANDIDATE_MULTIPLIER = 4
EXACT_PHRASE_MIN_LENGTH = 4


@dataclass(frozen=True)
class RetrievalProfile:
    profile_id: str = "default"
    low_value_section_keywords: tuple[str, ...] = ("参考文献", "价值导向", "哲学", "科学思想")
    low_value_weak_hints: tuple[str, ...] = ("附录", "思考题", "习题", "练习题", "本章小结")
    low_value_snippet_patterns: tuple[str, ...] = (
        r"\[[0-9]+\]\s*",
        r"出版社",
        r"译\.\s*北京",
        r"参考文献",
    )
    query_stopwords: frozenset[str] = frozenset({"什么", "多少", "哪些", "怎么", "如何", "为什么", "吗", "呢", "呀", "是", "的", "和", "与", "及", "有", "有哪些", "有什么"})
    definition_cues: tuple[str, ...] = ("定义", "是指", "概念", "称为", "叫作", "可定义为")
    feature_cues: tuple[str, ...] = ("特点", "特征", "优点", "缺点", "性质", "基本要求")
    comparison_cues: tuple[str, ...] = ("区别", "不同", "联系", "比较", "相同点", "不同点")
    comparison_patterns: tuple[str, ...] = ("区别在于", "不同在于", "相比", "相比之下", "前者", "后者", "二者", "两者", "一方面", "另一方面")
    feature_patterns: tuple[str, ...] = ("其特点是", "主要特点", "具有", "表现为", "优点是", "缺点是")
    definition_patterns: tuple[str, ...] = ("是", "是指", "可定义为", "称为", "叫作", "所谓")
    subtype_definition_prefixes: tuple[str, ...] = ("气象", "综合", "组合", "惯性", "无线电", "卫星", "水声", "光学", "数据库匹配")
    exact_phrase_bonus_per_hit: float = 0.04
    exact_phrase_bonus_max: float = 0.12
    primary_phrase_hit_bonus: float = 0.05
    title_phrase_hit_bonus: float = 0.03
    definition_pattern_bonus: float = 0.08
    comparison_pattern_bonus: float = 0.04
    comparison_pattern_bonus_max: float = 0.1
    feature_pattern_bonus: float = 0.03
    feature_pattern_bonus_max: float = 0.08
    phrase_structure_bonus_max: float = 0.18
    main_subject_bonus: float = 0.08
    subtype_definition_penalty: float = 0.06
    min_content_chars_for_compression: int = 140
    max_compressed_chars: int = 360
    max_selected_sentences: int = 3
    dedup_similarity_threshold: float = 0.9
    dedup_overlap_threshold: float = 0.78
    cross_encoder_enabled: bool = False
    cross_encoder_model_name: str = "BAAI/bge-reranker-base"
    cross_encoder_candidate_top_n: int = 12
    cross_encoder_score_weight: float = 0.15

    @classmethod
    def from_file(cls, profile_path: str | Path | None = None) -> "RetrievalProfile":
        default_path = Path(__file__).with_name("book_rag") / "retrieval_profile.default.json"
        resolved_path = Path(profile_path) if profile_path else default_path
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
        if payload.get("extends"):
            base_name = str(payload.get("extends") or "default").strip()
            base_path = default_path.with_name(f"retrieval_profile.{base_name}.json") if base_name != "default" else default_path
            base_payload = json.loads(base_path.read_text(encoding="utf-8"))
            payload = cls._deep_merge(base_payload, payload)

        compression = payload.get("compression") or {}
        dedup = payload.get("dedup") or {}
        rerank_weights = payload.get("rerank_weights") or {}
        cross_encoder = payload.get("cross_encoder") or {}

        return cls(
            profile_id=str(payload.get("profile_id") or resolved_path.stem),
            low_value_section_keywords=tuple(payload.get("low_value_section_keywords") or cls.low_value_section_keywords),
            low_value_weak_hints=tuple(payload.get("low_value_weak_hints") or cls.low_value_weak_hints),
            low_value_snippet_patterns=tuple(payload.get("low_value_snippet_patterns") or cls.low_value_snippet_patterns),
            query_stopwords=frozenset(payload.get("query_stopwords") or cls.query_stopwords),
            definition_cues=tuple(payload.get("definition_cues") or cls.definition_cues),
            feature_cues=tuple(payload.get("feature_cues") or cls.feature_cues),
            comparison_cues=tuple(payload.get("comparison_cues") or cls.comparison_cues),
            comparison_patterns=tuple(payload.get("comparison_patterns") or cls.comparison_patterns),
            feature_patterns=tuple(payload.get("feature_patterns") or cls.feature_patterns),
            definition_patterns=tuple(payload.get("definition_patterns") or cls.definition_patterns),
            subtype_definition_prefixes=tuple(payload.get("subtype_definition_prefixes") or cls.subtype_definition_prefixes),
            exact_phrase_bonus_per_hit=float(rerank_weights.get("exact_phrase_bonus_per_hit", cls.exact_phrase_bonus_per_hit)),
            exact_phrase_bonus_max=float(rerank_weights.get("exact_phrase_bonus_max", cls.exact_phrase_bonus_max)),
            primary_phrase_hit_bonus=float(rerank_weights.get("primary_phrase_hit_bonus", cls.primary_phrase_hit_bonus)),
            title_phrase_hit_bonus=float(rerank_weights.get("title_phrase_hit_bonus", cls.title_phrase_hit_bonus)),
            definition_pattern_bonus=float(rerank_weights.get("definition_pattern_bonus", cls.definition_pattern_bonus)),
            comparison_pattern_bonus=float(rerank_weights.get("comparison_pattern_bonus", cls.comparison_pattern_bonus)),
            comparison_pattern_bonus_max=float(rerank_weights.get("comparison_pattern_bonus_max", cls.comparison_pattern_bonus_max)),
            feature_pattern_bonus=float(rerank_weights.get("feature_pattern_bonus", cls.feature_pattern_bonus)),
            feature_pattern_bonus_max=float(rerank_weights.get("feature_pattern_bonus_max", cls.feature_pattern_bonus_max)),
            phrase_structure_bonus_max=float(rerank_weights.get("phrase_structure_bonus_max", cls.phrase_structure_bonus_max)),
            main_subject_bonus=float(rerank_weights.get("main_subject_bonus", cls.main_subject_bonus)),
            subtype_definition_penalty=float(rerank_weights.get("subtype_definition_penalty", cls.subtype_definition_penalty)),
            min_content_chars_for_compression=int(compression.get("min_content_chars", cls.min_content_chars_for_compression)),
            max_compressed_chars=int(compression.get("max_compressed_chars", cls.max_compressed_chars)),
            max_selected_sentences=int(compression.get("max_selected_sentences", cls.max_selected_sentences)),
            dedup_similarity_threshold=float(dedup.get("similarity_threshold", cls.dedup_similarity_threshold)),
            dedup_overlap_threshold=float(dedup.get("overlap_threshold", cls.dedup_overlap_threshold)),
            cross_encoder_enabled=bool(cross_encoder.get("enabled", cls.cross_encoder_enabled)),
            cross_encoder_model_name=str(cross_encoder.get("model_name", cls.cross_encoder_model_name)),
            cross_encoder_candidate_top_n=int(cross_encoder.get("candidate_top_n", cls.cross_encoder_candidate_top_n)),
            cross_encoder_score_weight=float(cross_encoder.get("score_weight", cls.cross_encoder_score_weight)),
        )

    @staticmethod
    def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        merged = dict(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = RetrievalProfile._deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged


@dataclass(frozen=True)
class ChunkingConfig:
    profile_id: str = "default"
    target_chunk_length: int = 1100
    max_chunk_length: int = 1500
    chunk_soft_limit: int = 1620
    chunk_overlap: int = 120
    min_chunk_length: int = 320

    @classmethod
    def from_file(cls, config_path: str | Path | None = None) -> "ChunkingConfig":
        resolved_path = Path(config_path) if config_path else Path(__file__).with_name("book_rag") / "chunking_config.json"
        config_data = json.loads(resolved_path.read_text(encoding="utf-8"))
        return cls(
            profile_id=str(config_data.get("profile_id") or "default"),
            target_chunk_length=int(config_data.get("target_chunk_length", cls.target_chunk_length)),
            max_chunk_length=int(config_data.get("max_chunk_length", cls.max_chunk_length)),
            chunk_soft_limit=int(config_data.get("chunk_soft_limit", cls.chunk_soft_limit)),
            chunk_overlap=int(config_data.get("chunk_overlap", cls.chunk_overlap)),
            min_chunk_length=int(config_data.get("min_chunk_length", cls.min_chunk_length)),
        )


class EmbeddingService:
    _shared_model: SentenceTransformer | None = None
    _shared_model_lock = Lock()
    _shared_reranker: AutoModelForSequenceClassification | None = None
    _shared_reranker_tokenizer: AutoTokenizer | None = None
    _shared_reranker_model_name: str | None = None
    _shared_reranker_lock = Lock()
    _index_cache: dict[str, dict[str, Any]] = {}
    _index_cache_lock = Lock()

    def __init__(self, chunking_config_path: str | None = None, retrieval_profile_path: str | None = None):
        self.model = None
        self.model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
        self.index_dir = Path("indexes")
        self.index_dir.mkdir(exist_ok=True)
        self.reranker = None
        self.reranker_tokenizer = None
        
        # 初始化空索引和文档存储
        self.dimension = 384  # 修改为与模型输出维度一致
        self.current_index = None
        self.current_documents = {}
        self.sparse_documents: list[Dict[str, Any]] = []
        self.sparse_idf_map: dict[str, float] = {}
        self.document_parser = DocumentParserService()

        # 检索策略配置：后续新增资料时优先通过 profile 调整，而不是改代码
        self.retrieval_profile = RetrievalProfile.from_file(retrieval_profile_path)

        # 从配置文件加载教材通用分块参数，后续换教材时优先只调这些值
        self.chunking_config = ChunkingConfig.from_file(chunking_config_path)
        self.target_chunk_length = self.chunking_config.target_chunk_length
        self.max_chunk_length = self.chunking_config.max_chunk_length
        self.chunk_soft_limit = self.chunking_config.chunk_soft_limit
        self.chunk_overlap = self.chunking_config.chunk_overlap
        self.min_chunk_length = self.chunking_config.min_chunk_length
        
    def _create_index(self) -> faiss.IndexFlatL2:
        """创建新的 FAISS 索引"""
        return faiss.IndexFlatL2(self.dimension)
    
    def _get_model(self) -> SentenceTransformer:
        """延迟加载 embedding 模型，加快仅调试分块时的启动速度。"""
        if EmbeddingService._shared_model is None:
            with EmbeddingService._shared_model_lock:
                if EmbeddingService._shared_model is None:
                    EmbeddingService._shared_model = SentenceTransformer(self.model_name)
        self.model = EmbeddingService._shared_model
        return self.model

    def _get_reranker(self) -> tuple[AutoTokenizer, AutoModelForSequenceClassification] | None:
        if not self.retrieval_profile.cross_encoder_enabled:
            return None

        model_name = self.retrieval_profile.cross_encoder_model_name
        shared_model_ready = (
            EmbeddingService._shared_reranker is not None
            and EmbeddingService._shared_reranker_tokenizer is not None
            and EmbeddingService._shared_reranker_model_name == model_name
        )
        if not shared_model_ready:
            with EmbeddingService._shared_reranker_lock:
                shared_model_ready = (
                    EmbeddingService._shared_reranker is not None
                    and EmbeddingService._shared_reranker_tokenizer is not None
                    and EmbeddingService._shared_reranker_model_name == model_name
                )
                if not shared_model_ready:
                    with _RerankerLoadLogger(model_name):
                        EmbeddingService._shared_reranker_tokenizer = AutoTokenizer.from_pretrained(model_name)
                        EmbeddingService._shared_reranker = AutoModelForSequenceClassification.from_pretrained(model_name)
                        EmbeddingService._shared_reranker.eval()
                        EmbeddingService._shared_reranker_model_name = model_name

        self.reranker_tokenizer = EmbeddingService._shared_reranker_tokenizer
        self.reranker = EmbeddingService._shared_reranker
        return self.reranker_tokenizer, self.reranker

    async def create_embeddings(self, file_path: str, index_dir: Optional[str] = None) -> Dict:
        """从文件创建向量索引"""
        try:
            parsed_result = self.document_parser.parse_pdf_with_metadata(file_path=file_path)
            chunk_records = self._build_chunks_from_parsed_result(parsed_result)

            if not chunk_records:
                raise ValueError("未生成有效文本分块，无法创建向量索引")

            file_hash = hashlib.md5(file_path.encode()).hexdigest()
            index_id = self._build_and_save_embeddings(
                file_id=file_hash,
                chunk_records=chunk_records,
            )

            return {
                "status": "success",
                "index_id": index_id,
                "chunks": len(chunk_records),
                "pages": parsed_result.get("total_pages", 0),
                "non_empty_pages": parsed_result.get("non_empty_pages", 0),
            }

        except Exception as e:
            raise Exception(f"创建向量失败: {str(e)}")

    async def create_embeddings_from_book_rag_chunks(self, json_path: str) -> Dict[str, Any]:
        """从 book_rag_chunks.json 创建向量索引。"""
        try:
            json_file = Path(json_path)
            payload = json.loads(json_file.read_text(encoding="utf-8"))
            chunk_records = self._build_chunks_from_book_rag_payload(payload)
            if not chunk_records:
                raise ValueError("book_rag_chunks.json 中没有可入库的 chunk")

            file_hash = hashlib.md5(str(json_file.resolve()).encode()).hexdigest()
            index_id = self._build_and_save_embeddings(
                file_id=file_hash,
                chunk_records=chunk_records,
            )
            
            return {
                "status": "success",
                "index_id": index_id,
                "chunks": len(chunk_records),
                "book_name": payload.get("book_name", ""),
                "source_file_name": payload.get("source_file_name", ""),
                "source_path": str(json_file),
            }
        except Exception as e:
            raise Exception(f"从 book_rag chunks 创建向量失败: {str(e)}")

    def _build_and_save_embeddings(self, file_id: str, chunk_records: List[Dict[str, Any]]) -> str:
        texts = [chunk["text"] for chunk in chunk_records]
        index = self._create_index()

        model = self._get_model()
        vectors = model.encode(texts)
        vectors = np.asarray(vectors, dtype="float32")
        index.add(vectors)

        documents = {
            str(i): {
                "text": chunk["text"],
                "metadata": chunk["metadata"],
            }
            for i, chunk in enumerate(chunk_records)
        }

        self._save_index(file_id, index, documents)
        return f"index_{file_id}"

    def _build_chunks_from_book_rag_payload(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        chunks = payload.get("chunks") or []
        if not isinstance(chunks, list):
            raise ValueError("book_rag_chunks.json 的 chunks 字段格式不正确")

        book_name = payload.get("book_name", "")
        source_file_name = payload.get("source_file_name", "")
        source_dir = payload.get("source_dir", "")
        normalized_chunks: List[Dict[str, Any]] = []

        for idx, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                continue

            text = str(chunk.get("text") or "").strip()
            if not text:
                continue

            chunk_id = str(chunk.get("chunk_id") or f"book_chunk_{idx:04d}")
            start_page = chunk.get("start_page")
            end_page = chunk.get("end_page")
            metadata = {
                "chunk_id": chunk_id,
                "chunk_index": idx,
                "source": source_dir,
                "file_name": source_file_name,
                "book_name": book_name,
                "section_title": chunk.get("section_title", ""),
                "title_level": chunk.get("title_level"),
                "page_start": start_page,
                "page_end": end_page,
                "page_numbers": self._expand_page_numbers(start_page, end_page),
                "char_count": int(chunk.get("char_count") or len(text)),
                "has_formula": "【公式】" in text or "【公式编号】" in text,
                "has_figure": "[插图]" in text,
            }
            normalized_chunks.append({
                "text": text,
                "metadata": metadata,
            })

        return normalized_chunks

    def _expand_page_numbers(self, start_page: Any, end_page: Any) -> List[int]:
        if not isinstance(start_page, int) or not isinstance(end_page, int):
            return []
        if end_page < start_page:
            return [start_page]
        return list(range(start_page, end_page + 1))

    def _profile_patterns(self, patterns: tuple[str, ...]) -> tuple[re.Pattern[str], ...]:
        return tuple(re.compile(pattern) for pattern in patterns)

    def _build_chunks_from_parsed_result(self, parsed_result: Dict) -> List[Dict]:
        """基于清洗后的页级结果构建分块。"""
        page_entries = []
        for page in parsed_result.get("pages", []):
            if self._should_skip_page_for_chunking(page):
                continue

            text = (page.get("text") or "").strip()
            if not text:
                continue

            normalized_text = self._prepare_page_text_for_chunking(text)
            if not normalized_text:
                continue

            page_entries.append({
                "page": page["page"],
                "text": normalized_text,
            })

        if not page_entries:
            return []

        chunks: List[Dict] = []
        current_text = ""
        current_pages: List[int] = []
        chunk_index = 0

        for page_index, page_entry in enumerate(page_entries):
            page_number = page_entry["page"]
            segments = self._split_page_into_segments(page_entry["text"])
            next_page_text = page_entries[page_index + 1]["text"] if page_index + 1 < len(page_entries) else ""

            if segments and next_page_text:
                segments = self._merge_cross_page_segments(segments, next_page_text)

            for segment in segments:
                if not segment:
                    continue

                candidate_text = segment if not current_text else f"{current_text}\n\n{segment}"

                if current_text and len(candidate_text) > self.max_chunk_length:
                    if self._should_hold_for_more_context(current_text, segment):
                        current_text = candidate_text
                        current_pages.append(page_number)
                        continue

                    chunks.append(
                        self._make_chunk_record(
                            parsed_result=parsed_result,
                            chunk_index=chunk_index,
                            text=self._normalize_chunk_edges(current_text),
                            page_numbers=current_pages,
                        )
                    )
                    chunk_index += 1

                    overlap_text = self._build_overlap_text(current_text)
                    overlap_pages = current_pages[-1:] if overlap_text else []
                    current_text = f"{overlap_text}\n\n{segment}" if overlap_text else segment
                    current_pages = overlap_pages + [page_number]
                    continue

                current_text = candidate_text
                current_pages.append(page_number)

                if len(current_text) >= self.target_chunk_length and self._is_good_chunk_boundary(current_text):
                    chunks.append(
                        self._make_chunk_record(
                            parsed_result=parsed_result,
                            chunk_index=chunk_index,
                            text=self._normalize_chunk_edges(current_text),
                            page_numbers=current_pages,
                        )
                    )
                    chunk_index += 1
                    overlap_text = self._build_overlap_text(current_text)
                    current_text = overlap_text
                    current_pages = current_pages[-1:] if overlap_text else []

        return self._finalize_chunks(parsed_result, chunks, self._normalize_chunk_edges(current_text), current_pages, chunk_index)

    def _prepare_page_text_for_chunking(self, text: str) -> str:
        """为分块做轻量整理，尽量合并页内断裂行，减少半句切分。"""
        lines = [line.strip() for line in text.split("\n")]
        cleaned_lines: List[str] = []
        previous_blank = False

        for line in lines:
            if not line:
                if not previous_blank:
                    cleaned_lines.append("")
                previous_blank = True
                continue

            previous_blank = False
            if self._is_low_value_line(line) or self._is_obvious_noise_line(line) or self._is_header_like_line(line):
                continue
            cleaned_lines.append(line)

        while cleaned_lines and self._is_front_matter_or_catalog_line(cleaned_lines[0]):
            cleaned_lines.pop(0)

        while cleaned_lines and (
            self._is_obvious_noise_line(cleaned_lines[-1])
            or self._is_header_like_line(cleaned_lines[-1])
            or self._is_catalogish_page(cleaned_lines)
        ):
            if self._is_catalogish_page(cleaned_lines):
                return ""
            cleaned_lines.pop()

        merged_lines: List[str] = []
        for line in cleaned_lines:
            if not line:
                if merged_lines and merged_lines[-1] != "":
                    merged_lines.append("")
                continue

            if self._should_attach_to_previous_line(merged_lines, line):
                merged_lines[-1] = f"{merged_lines[-1]}{line}"
            else:
                merged_lines.append(line)

        return "\n".join(merged_lines).strip()

    def _is_front_matter_or_catalog_line(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False

        if len(stripped) <= 24 and ("目录" in stripped or "前言" in stripped):
            return True

        if CATALOGISH_LINE_PATTERN.match(stripped):
            return True

        if "·" in stripped or "…" in stripped:
            return True

        if re.fullmatch(r"[0-9IVXivx.·•●]+", stripped):
            return True

        return False

    def _split_page_into_segments(self, text: str) -> List[str]:
        """优先按段落切分，过长段落再按句子回退切分。"""
        paragraphs = [paragraph.strip() for paragraph in re.split(r"\n{2,}", text) if paragraph.strip()]
        segments: List[str] = []

        for paragraph in paragraphs:
            if len(paragraph) <= self.max_chunk_length:
                segments.append(paragraph)
                continue

            segments.extend(self._split_long_paragraph(paragraph))

        return segments

    def _split_long_paragraph(self, paragraph: str) -> List[str]:
        parts = PUNCTUATION_SPLIT_PATTERN.split(paragraph)
        parts = [part.strip() for part in parts if part.strip()]
        if not parts:
            return [paragraph]

        segments: List[str] = []
        current = ""
        for part in parts:
            candidate = part if not current else f"{current}{part}"
            if current and len(candidate) > self.max_chunk_length:
                if len(current) < 320 and segments:
                    segments[-1] = f"{segments[-1]}{current}".strip()
                else:
                    segments.append(current.strip())
                current = part
                continue

            if len(part) > self.max_chunk_length:
                hard_splits = self._hard_split_text(part, self.max_chunk_length)
                if current:
                    segments.append(current.strip())
                    current = ""
                segments.extend(hard_splits)
                continue

            current = candidate

        if current.strip():
            if len(current) < 320 and segments:
                segments[-1] = f"{segments[-1]}{current}".strip()
            else:
                segments.append(current.strip())

        return segments

    def _merge_cross_page_segments(self, segments: List[str], next_page_text: str) -> List[str]:
        if not segments or not next_page_text:
            return segments

        last_segment = segments[-1].strip()
        next_text = next_page_text.strip()
        if not last_segment or not next_text:
            return segments

        next_page_segments = self._split_page_into_segments(next_text)
        if not next_page_segments:
            return segments

        next_prefix = "".join(next_page_segments[:2]).strip()
        if not next_prefix:
            return segments

        if not self._should_merge_with_next_page(last_segment, next_prefix):
            return segments

        merged_last_segment = self._normalize_chunk_edges(f"{last_segment}{next_prefix}")
        return [*segments[:-1], merged_last_segment]

    def _should_merge_with_next_page(self, last_segment: str, next_segment: str) -> bool:
        tail = last_segment.rstrip()
        head = next_segment.lstrip()
        if not tail or not head:
            return False

        if tail.endswith(SENTENCE_ENDINGS):
            return False

        if TITLE_OR_ENUM_PATTERN.match(head) or CATALOGISH_LINE_PATTERN.match(head):
            return False

        if head.startswith(("[插图]", "【公式】", "【公式编号】")):
            return False

        if len(tail) < 12 or len(head) < 2:
            return False

        if CATALOGISH_LINE_PATTERN.match(tail) and len(tail) <= 24:
            return False

        if re.search(r"[A-Za-z]$", tail) and re.match(r"^[A-Za-z]", head):
            return True

        if CONTINUATION_PREFIX_PATTERN.match(head):
            return bool(TRAILING_INCOMPLETE_PATTERN.search(tail)) and len(head) <= 36

        if len(head) <= 28 and not re.search(r"[。！？；：.!?;:]", head):
            return bool(TRAILING_INCOMPLETE_PATTERN.search(tail))

        return False

    def _should_hold_for_more_context(self, current_text: str, next_segment: str) -> bool:
        if len(current_text) >= self.chunk_soft_limit:
            return False

        tail = current_text.strip()
        head = next_segment.strip()
        if not tail or not head:
            return False

        if self._is_good_chunk_boundary(tail):
            return False

        if head.startswith(("[插图]", "【公式】", "【公式编号】")):
            return False

        if TITLE_OR_ENUM_PATTERN.match(head):
            return False

        if len(head) > 220:
            return False

        if len(tail) < 120:
            return False

        if CONTINUATION_PREFIX_PATTERN.match(head):
            return bool(TRAILING_INCOMPLETE_PATTERN.search(tail))

        if len(head) <= 72 and not re.search(r"[。！？；：.!?;:]", head):
            return bool(TRAILING_INCOMPLETE_PATTERN.search(tail))

        return False

    def _hard_split_text(self, text: str, max_length: int) -> List[str]:
        return [text[i:i + max_length].strip() for i in range(0, len(text), max_length) if text[i:i + max_length].strip()]

    def _build_overlap_text(self, text: str) -> str:
        stripped = text.strip()
        if len(stripped) <= self.chunk_overlap:
            return ""

        start = max(0, len(stripped) - self.chunk_overlap)
        overlap_text = stripped[start:].strip()
        sentence_break_match = re.search(r"[。！？；：.!?;:]\s*([^。！？；：.!?;:].*)$", overlap_text)
        if sentence_break_match:
            overlap_text = sentence_break_match.group(1).strip()

        overlap_text = re.sub(r"^[，、；：,.\s]+", "", overlap_text)
        overlap_text = TRAILING_FRAGMENT_PATTERN.sub("", overlap_text).strip()

        if not overlap_text:
            overlap_text = stripped[-min(len(stripped), max(48, self.chunk_overlap // 2)):].strip()
            overlap_text = re.sub(r"^[，、；：,.\s]+", "", overlap_text)

        if CATALOGISH_LINE_PATTERN.match(overlap_text) or self._is_obvious_noise_line(overlap_text):
            return ""

        if len(overlap_text) < 24:
            return ""

        return overlap_text

    def _is_good_chunk_boundary(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        return bool(re.search(r"[。！？；：.!?;:）)】]$", stripped))

    def _normalize_chunk_edges(self, text: str) -> str:
        stripped = text.strip()
        stripped = re.sub(r"\n{3,}", "\n\n", stripped)
        stripped = REPEATED_PUNCTUATION_PATTERN.sub(r"\1", stripped)
        return stripped

    def _finalize_chunks(self, parsed_result: Dict, chunks: List[Dict], current_text: str, current_pages: List[int], chunk_index: int) -> List[Dict]:
        if current_text:
            pending_chunk = self._make_chunk_record(
                parsed_result=parsed_result,
                chunk_index=chunk_index,
                text=current_text,
                page_numbers=current_pages,
            )

            should_merge_tail = chunks and (
                pending_chunk["metadata"]["char_count"] < self.min_chunk_length
                or (
                    pending_chunk["metadata"]["char_count"] < 760
                    and not self._is_good_chunk_boundary(pending_chunk["text"])
                )
                or self._has_abrupt_chunk_tail(pending_chunk["text"])
                or self._is_low_quality_tail_chunk(pending_chunk["text"])
            )

            if should_merge_tail:
                previous_chunk = chunks[-1]
                merged_text = f"{previous_chunk['text']}\n\n{pending_chunk['text']}".strip()
                merged_pages = previous_chunk["metadata"]["page_numbers"] + pending_chunk["metadata"]["page_numbers"]
                chunks[-1] = self._make_chunk_record(
                    parsed_result=parsed_result,
                    chunk_index=previous_chunk["metadata"]["chunk_index"],
                    text=self._normalize_chunk_edges(merged_text),
                    page_numbers=merged_pages,
                )
            else:
                chunks.append(pending_chunk)

        return chunks

    def _is_low_quality_tail_chunk(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False

        lines = [line.strip() for line in stripped.split("\n") if line.strip()]
        if not lines:
            return False

        tail = lines[-1]
        if self._is_good_chunk_boundary(tail):
            return False

        if len(lines) <= 2 and len(stripped) < 900:
            return True

        if len(tail) <= 56 and not re.search(r"[。！？；：.!?;:]", tail):
            return True

        return False

    def _has_abrupt_chunk_tail(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False

        if self._is_good_chunk_boundary(stripped):
            return False

        last_line = stripped.split("\n")[-1].strip()
        if not last_line:
            return False

        if TITLE_OR_ENUM_PATTERN.match(last_line) or CATALOGISH_LINE_PATTERN.match(last_line):
            return False

        if len(last_line) <= 24:
            return True

        if len(last_line) <= 56 and not re.search(r"[。！？；：.!?;:]", last_line):
            return True

        return bool(re.search(r"[\u4e00-\u9fffA-Za-z0-9）)]$", last_line)) and len(last_line) <= 140

    def _should_attach_to_previous_line(self, merged_lines: List[str], line: str) -> bool:
        if not merged_lines:
            return False

        previous_line = merged_lines[-1]
        if not previous_line:
            return False

        if previous_line.endswith(SENTENCE_ENDINGS):
            return False

        if line.startswith(("（", "(", "[插图]", "【公式】", "【公式编号】")):
            return False

        if TITLE_OR_ENUM_PATTERN.match(line):
            return False

        return True

    def _make_chunk_record(self, parsed_result: Dict, chunk_index: int, text: str, page_numbers: List[int]) -> Dict:
        unique_pages = sorted(set(page_numbers))
        return {
            "text": text,
            "metadata": {
                "chunk_id": f"{Path(parsed_result.get('file_name', 'document')).stem}_chunk_{chunk_index:04d}",
                "chunk_index": chunk_index,
                "source": parsed_result.get("file_path", ""),
                "file_name": parsed_result.get("file_name", ""),
                "book_name": parsed_result.get("book_name", ""),
                "page_start": unique_pages[0] if unique_pages else None,
                "page_end": unique_pages[-1] if unique_pages else None,
                "page_numbers": unique_pages,
                "char_count": len(text),
                "has_formula": "【公式】" in text or "【公式编号】" in text,
                "has_figure": "[插图]" in text,
            },
        }

    def _is_low_value_line(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False

        if len(stripped) <= 1:
            return True

        if LOW_VALUE_LINE_PATTERN.fullmatch(stripped):
            return True

        return False

    def _is_obvious_noise_line(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False

        if NOISE_LINE_PATTERN.match(stripped):
            return True

        if len(stripped) <= 24 and re.search(r"(?:ISBN|CIP|定价|开本|责任编辑|责任校对|责任印制|封面设计)", stripped, re.IGNORECASE):
            return True

        if len(stripped) <= 28 and re.fullmatch(r"[A-Za-z0-9\s:：,，.。;；()（）/_-]+", stripped) and re.search(r"\d", stripped):
            return True

        return False
    
    def _save_index(self, file_id: str, index: faiss.Index, documents: dict):
        """保存索引和文档数据"""
        try:
            index_path = self.index_dir / f"index_{file_id}.bin"
            docs_path = self.index_dir / f"docs_{file_id}.json"
            
            faiss.write_index(index, str(index_path))
            
            with open(docs_path, 'w', encoding='utf-8') as f:
                json.dump(documents, f, ensure_ascii=False, indent=2)

            normalized_index_id = f"index_{file_id}"
            cached_payload = {
                "index": index,
                "documents": documents,
            }
            with EmbeddingService._index_cache_lock:
                EmbeddingService._index_cache[normalized_index_id] = cached_payload
                EmbeddingService._index_cache[file_id] = cached_payload
                
        except Exception as e:
            raise Exception(f"保存索引失败: {str(e)}")
    
    def _set_current_index_state(self, index: faiss.Index, documents: dict[str, Any]) -> None:
        self.current_index = index
        self.current_documents = documents
        if not self.current_documents:
            raise ValueError("文档数据为空")
        self._prepare_sparse_documents()

    def _read_index_from_disk(self, normalized_index_id: str) -> tuple[faiss.Index, dict[str, Any]]:
        file_id = normalized_index_id.removeprefix("index_")
        index_path = self.index_dir / f"{normalized_index_id}.bin"
        docs_path = self.index_dir / f"docs_{file_id}.json"

        if not index_path.exists() or not docs_path.exists():
            raise FileNotFoundError(f"找不到索引文件: {normalized_index_id}")

        loaded_index = faiss.read_index(str(index_path))
        if loaded_index.d != self.dimension:
            raise ValueError(f"索引维度不匹配: 期望 {self.dimension}, 实际 {loaded_index.d}")

        with open(docs_path, 'r', encoding='utf-8') as f:
            loaded_documents = json.load(f)

        return loaded_index, loaded_documents
    
    def _load_index(self, index_id: str):
        """加载索引和文档数据"""
        try:
            normalized_index_id = index_id if index_id.startswith("index_") else f"index_{index_id}"
            file_id = normalized_index_id.removeprefix("index_")

            cached_payload = EmbeddingService._index_cache.get(normalized_index_id)
            if cached_payload is None:
                cached_payload = EmbeddingService._index_cache.get(file_id)

            if cached_payload is None:
                with EmbeddingService._index_cache_lock:
                    cached_payload = EmbeddingService._index_cache.get(normalized_index_id)
                    if cached_payload is None:
                        cached_payload = EmbeddingService._index_cache.get(file_id)
                    if cached_payload is None:
                        print(f"[EmbeddingService] 首次加载索引到内存: {normalized_index_id}", flush=True)
                        loaded_index, loaded_documents = self._read_index_from_disk(normalized_index_id)
                        cached_payload = {
                            "index": loaded_index,
                            "documents": loaded_documents,
                        }
                        EmbeddingService._index_cache[normalized_index_id] = cached_payload
                        EmbeddingService._index_cache[file_id] = cached_payload

            self._set_current_index_state(
                index=cached_payload["index"],
                documents=cached_payload["documents"],
            )
            
        except Exception as e:
            self.current_index = None
            self.current_documents = {}
            raise Exception(f"加载索引失败: {str(e)}")
    
    def load_index(self, index_id: str) -> Dict[str, Any]:
        self._load_index(index_id)
        return {
            "status": "success",
            "index_id": index_id if index_id.startswith("index_") else f"index_{index_id}",
            "document_count": len(self.current_documents),
            "vector_count": int(self.current_index.ntotal) if self.current_index is not None else 0,
        }
    
    async def search(self, query: str, top_k: int = 3) -> List[dict]:
        """搜索最相关的文档片段"""
        try:
            if not self.current_index:
                raise Exception("未加载索引")
            
            model = self._get_model()
            query_vector = model.encode(query, convert_to_tensor=False)
            query_vector = np.asarray([query_vector], dtype="float32")

            distances, indices = self.current_index.search(query_vector, top_k)
            
            results = []
            for i in range(len(indices[0])):
                doc_idx = int(indices[0][i])
                if doc_idx < 0:
                    continue
                idx_str = str(doc_idx)
                if idx_str not in self.current_documents:
                    continue

                    results.append({
                        "score": float(distances[0][i]),
                        "content": self.current_documents[idx_str]["text"],
                        "metadata": self.current_documents[idx_str]["metadata"]
                    })
            
            return results
                
        except Exception as e:
            raise Exception(f"搜索失败: {str(e)}") 

    async def search_index(self, index_id: str, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        self._load_index(index_id)
        return await self.search(query=query, top_k=top_k)

    async def hybrid_search_index(self, index_id: str, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        self._load_index(index_id)
        return await self.hybrid_search(query=query, top_k=top_k)

    async def hybrid_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self.current_index:
            raise Exception("未加载索引")

        dense_candidate_k = max(top_k, top_k * SPARSE_CANDIDATE_MULTIPLIER)
        dense_results = await self.search(query=query, top_k=dense_candidate_k)
        sparse_results = self._sparse_search(query=query, top_k=dense_candidate_k)
        fused_results = self._fuse_with_rrf(dense_results=dense_results, sparse_results=sparse_results)
        reranked_results = self._rerank_results(query=query, results=fused_results)
        cross_encoded_results = self._cross_encode_rerank(query=query, results=reranked_results)
        deduplicated_results = self._deduplicate_results(cross_encoded_results)
        compressed_results = self._compress_results(query=query, results=deduplicated_results)
        return compressed_results[:top_k]

    def _prepare_sparse_documents(self) -> None:
        documents: list[Dict[str, Any]] = []
        doc_freq: Counter[str] = Counter()

        for entry in self.current_documents.values():
            text = str(entry.get("text") or "").strip()
            metadata = entry.get("metadata") or {}
            section_title = str(metadata.get("section_title") or "")
            doc_text = f"{section_title}\n{text}".strip()
            tokens = self._tokenize(doc_text)
            if not tokens:
                continue
            token_counts = Counter(tokens)
            documents.append(
                {
                    "text": text,
                    "metadata": metadata,
                    "token_counts": token_counts,
                    "token_len": sum(token_counts.values()),
                }
            )
            doc_freq.update(set(token_counts.keys()))

        total_docs = max(len(documents), 1)
        self.sparse_idf_map = {
            token: math.log((total_docs + 1) / (freq + 1)) + 1.0
            for token, freq in doc_freq.items()
        }
        self.sparse_documents = documents

    def _sparse_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        if not self.sparse_documents:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        unique_query_tokens = list(dict.fromkeys(query_tokens))
        scored: list[Dict[str, Any]] = []
        for document in self.sparse_documents:
            metadata = document.get("metadata") or {}
            if self._is_low_value_chunk(metadata, document.get("text", "")):
                continue

            score = 0.0
            matched_tokens: list[str] = []
            token_counts: Counter[str] = document["token_counts"]
            for token in unique_query_tokens:
                tf = token_counts.get(token, 0)
                if tf <= 0:
                    continue
                idf = self.sparse_idf_map.get(token, 1.0)
                score += (1.0 + math.log(tf)) * idf
                matched_tokens.append(token)

            if not matched_tokens:
                continue

            normalized_score = score / (1.0 + math.log(document["token_len"] + 1))
            scored.append(
                {
                    "score": round(normalized_score, 6),
                    "content": document["text"],
                    "metadata": metadata,
                    "matched_tokens": matched_tokens,
                    "retrieval_source": "sparse",
                }
            )

        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

    def _fuse_with_rrf(self, dense_results: List[Dict[str, Any]], sparse_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        fused: dict[str, Dict[str, Any]] = {}

        for rank, item in enumerate(dense_results, start=1):
            metadata = item.get("metadata") or {}
            if self._is_low_value_chunk(metadata, item.get("content", "")):
                continue
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
                    "matched_tokens": [],
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
                    "matched_tokens": [],
                },
            )
            entry["score"] += 1.0 / (RRF_K + rank)
            entry["sparse_rank"] = rank
            if entry.get("retrieval_source") == "dense":
                entry["retrieval_source"] = "hybrid"
            matched_tokens = item.get("matched_tokens") or []
            if matched_tokens:
                existing_tokens = entry.get("matched_tokens") or []
                entry["matched_tokens"] = list(dict.fromkeys([*existing_tokens, *matched_tokens]))

        fused_results = list(fused.values())
        fused_results.sort(key=lambda item: item["score"], reverse=True)
        return fused_results

    def _rerank_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not results:
            return []

        query_type = self._detect_query_type(query)
        query_terms = self._extract_query_terms(query)
        exact_phrases = self._extract_exact_phrases(query)
        reranked: List[Dict[str, Any]] = []

        for item in results:
            metadata = item.get("metadata") or {}
            content = str(item.get("content") or "")
            section_title = str(metadata.get("section_title") or "")
            base_score = float(item.get("score") or 0.0)
            bonuses: list[tuple[str, float]] = []
            penalties: list[tuple[str, float]] = []

            low_value_penalty = self._get_low_value_penalty(section_title=section_title, content=content)
            if low_value_penalty > 0:
                penalties.append(("low_value_penalty", low_value_penalty))

            exact_phrase_bonus = self._get_exact_phrase_bonus(exact_phrases=exact_phrases, section_title=section_title, content=content)
            if exact_phrase_bonus > 0:
                bonuses.append(("exact_phrase_bonus", exact_phrase_bonus))

            phrase_structure_bonus = self._get_phrase_structure_bonus(
                query=query,
                query_type=query_type,
                exact_phrases=exact_phrases,
                section_title=section_title,
                content=content,
            )
            if phrase_structure_bonus > 0:
                bonuses.append(("phrase_structure_bonus", phrase_structure_bonus))

            intent_bonus = self._get_intent_bonus(query_type=query_type, query_terms=query_terms, section_title=section_title, content=content)
            if intent_bonus > 0:
                bonuses.append(("intent_bonus", intent_bonus))

            section_bonus = self._get_section_bonus(query_terms=query_terms, section_title=section_title)
            if section_bonus > 0:
                bonuses.append(("section_bonus", section_bonus))

            token_bonus = self._get_matched_token_bonus(query_terms=query_terms, matched_tokens=item.get("matched_tokens") or [], content=content)
            if token_bonus > 0:
                bonuses.append(("matched_token_bonus", token_bonus))

            hybrid_bonus = self._get_hybrid_bonus(item)
            if hybrid_bonus > 0:
                bonuses.append(("hybrid_bonus", hybrid_bonus))

            weak_fragment_penalty = self._get_weak_fragment_penalty(content=content, query_type=query_type)
            if weak_fragment_penalty > 0:
                penalties.append(("weak_fragment_penalty", weak_fragment_penalty))

            total_bonus = sum(value for _, value in bonuses)
            total_penalty = sum(value for _, value in penalties)
            rerank_score = round(base_score + total_bonus - total_penalty, 6)

            reranked.append(
                {
                    **item,
                    "rerank_score": rerank_score,
                    "rerank_bonus": round(total_bonus, 6),
                    "rerank_penalty": round(total_penalty, 6),
                    "rerank_reasons": [name for name, _ in bonuses] + [name for name, _ in penalties],
                    "query_type": query_type,
                }
            )

        reranked.sort(key=lambda item: item.get("rerank_score", item.get("score", 0.0)), reverse=True)
        return reranked

    def _cross_encode_rerank(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        reranker_bundle = self._get_reranker()
        if reranker_bundle is None or not results:
            return results

        tokenizer, reranker = reranker_bundle
        candidate_top_n = min(len(results), max(1, self.retrieval_profile.cross_encoder_candidate_top_n))
        top_candidates = results[:candidate_top_n]
        tail_candidates = results[candidate_top_n:]
        pair_queries = [query for _ in top_candidates]
        pair_docs = [self._build_rerank_text(item) for item in top_candidates]

        try:
            print(
                f"[EmbeddingService] 开始进行 cross-encoder 重排，候选数: {candidate_top_n}，模型: {self.retrieval_profile.cross_encoder_model_name}",
                flush=True,
            )
            inputs = tokenizer(
                pair_queries,
                pair_docs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = reranker(**inputs).logits
            scores = logits.view(-1).detach().cpu().tolist()
            print("[EmbeddingService] cross-encoder 重排完成", flush=True)
        except Exception:
            return results

        weighted_candidates: List[Dict[str, Any]] = []
        for item, raw_score in zip(top_candidates, scores):
            base_score = float(item.get("rerank_score", item.get("score", 0.0)) or 0.0)
            cross_encoder_score = float(raw_score)
            final_score = round(base_score + cross_encoder_score * self.retrieval_profile.cross_encoder_score_weight, 6)
            reasons = list(item.get("rerank_reasons") or [])
            reasons.append("cross_encoder_rerank")
            weighted_candidates.append(
                {
                    **item,
                    "cross_encoder_score": round(cross_encoder_score, 6),
                    "final_rerank_score": final_score,
                    "rerank_reasons": reasons,
                }
            )

        weighted_candidates.sort(
            key=lambda item: item.get("final_rerank_score", item.get("rerank_score", item.get("score", 0.0))),
            reverse=True,
        )
        return weighted_candidates + tail_candidates

    def _build_rerank_text(self, item: Dict[str, Any]) -> str:
        metadata = item.get("metadata") or {}
        section_title = str(metadata.get("section_title") or "").strip()
        content = str(item.get("content") or "").strip()
        if section_title:
            return f"标题：{section_title}\n内容：{content}"
        return content

    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        deduplicated: List[Dict[str, Any]] = []

        for item in results:
            content = str(item.get("content") or "").strip()
            metadata = item.get("metadata") or {}
            if not content:
                continue

            duplicate_of = None
            for kept_item in deduplicated:
                kept_content = str(kept_item.get("content") or "").strip()
                kept_metadata = kept_item.get("metadata") or {}
                if self._is_duplicate_chunk(content, metadata, kept_content, kept_metadata):
                    duplicate_of = kept_item
                    break

            if duplicate_of is None:
                deduplicated.append(item)
                continue

            duplicate_meta = item.get("metadata") or {}
            duplicate_group = duplicate_of.setdefault("duplicate_chunk_ids", [])
            duplicate_group.append(duplicate_meta.get("chunk_id"))
            duplicate_of["duplicate_chunk_ids"] = [chunk_id for chunk_id in dict.fromkeys(duplicate_group) if chunk_id]
            duplicate_of["dedup_count"] = int(duplicate_of.get("dedup_count") or 1) + 1

        return deduplicated

    def _is_duplicate_chunk(
        self,
        content: str,
        metadata: Dict[str, Any],
        kept_content: str,
        kept_metadata: Dict[str, Any],
    ) -> bool:
        chunk_id = str(metadata.get("chunk_id") or "")
        kept_chunk_id = str(kept_metadata.get("chunk_id") or "")
        if chunk_id and kept_chunk_id and chunk_id == kept_chunk_id:
            return True

        normalized = self._normalize_text_for_dedup(content)
        kept_normalized = self._normalize_text_for_dedup(kept_content)
        if not normalized or not kept_normalized:
            return False

        if normalized == kept_normalized:
            return True

        if normalized in kept_normalized or kept_normalized in normalized:
            overlap_ratio = min(len(normalized), len(kept_normalized)) / max(len(normalized), len(kept_normalized))
            if overlap_ratio >= self.retrieval_profile.dedup_overlap_threshold:
                return True

        if self._share_same_source_window(metadata, kept_metadata):
            similarity = SequenceMatcher(None, normalized, kept_normalized).ratio()
            if similarity >= self.retrieval_profile.dedup_similarity_threshold:
                return True

        return False

    def _share_same_source_window(self, metadata: Dict[str, Any], kept_metadata: Dict[str, Any]) -> bool:
        file_name = str(metadata.get("file_name") or "")
        kept_file_name = str(kept_metadata.get("file_name") or "")
        if file_name and kept_file_name and file_name != kept_file_name:
            return False

        current_pages = set(metadata.get("page_numbers") or [])
        kept_pages = set(kept_metadata.get("page_numbers") or [])
        if current_pages and kept_pages and current_pages.intersection(kept_pages):
            return True

        start_page = metadata.get("page_start")
        end_page = metadata.get("page_end")
        kept_start_page = kept_metadata.get("page_start")
        kept_end_page = kept_metadata.get("page_end")
        if all(isinstance(value, int) for value in (start_page, end_page, kept_start_page, kept_end_page)):
            return not (end_page < kept_start_page or kept_end_page < start_page)

        return file_name == kept_file_name and bool(file_name)

    def _normalize_text_for_dedup(self, text: str) -> str:
        normalized = re.sub(r"\s+", "", text).lower()
        return normalized[:1200]

    def _compress_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not results:
            return []

        query_terms = self._extract_query_terms(query)
        exact_phrases = self._extract_exact_phrases(query)
        compression_focus_terms = list(dict.fromkeys([query.strip().lower(), *exact_phrases, *query_terms]))
        compressed_results: List[Dict[str, Any]] = []

        for item in results:
            content = str(item.get("content") or "")
            compressed_content = self._compress_content(
                content=content,
                query_terms=compression_focus_terms,
                exact_phrases=exact_phrases,
            )
            compressed_results.append(
                {
                    **item,
                    "content": compressed_content,
                    "original_content": content,
                    "compression_applied": compressed_content != content,
                }
            )

        return compressed_results

    def _compress_content(self, content: str, query_terms: List[str], exact_phrases: List[str]) -> str:
        stripped = content.strip()
        if len(stripped) <= self.retrieval_profile.min_content_chars_for_compression:
            return stripped

        sentences = self._split_into_sentences(stripped)
        if len(sentences) <= 2:
            return stripped

        scored_sentences: List[tuple[int, float, str]] = []
        for idx, sentence in enumerate(sentences):
            score = self._score_sentence_for_compression(sentence, query_terms, exact_phrases)
            scored_sentences.append((idx, score, sentence))

        scored_sentences.sort(key=lambda item: item[1], reverse=True)
        selected: List[tuple[int, float, str]] = []
        current_length = 0

        for sentence_item in scored_sentences:
            _, score, sentence = sentence_item
            if score <= 0 and selected:
                continue
            sentence_length = len(sentence)
            if selected and current_length + sentence_length > self.retrieval_profile.max_compressed_chars:
                continue
            selected.append(sentence_item)
            current_length += sentence_length
            if current_length >= self.retrieval_profile.max_compressed_chars:
                break
            if len(selected) >= self.retrieval_profile.max_selected_sentences:
                break

        if not selected:
            return stripped[:self.retrieval_profile.max_compressed_chars]

        selected.sort(key=lambda item: item[0])
        merged = "".join(sentence for _, _, sentence in selected).strip()
        return merged if merged else stripped[:self.retrieval_profile.max_compressed_chars]

    def _split_into_sentences(self, content: str) -> List[str]:
        sentences = [sentence.strip() for sentence in PUNCTUATION_SPLIT_PATTERN.split(content) if sentence.strip()]
        return sentences

    def _score_sentence_for_compression(self, sentence: str, query_terms: List[str], exact_phrases: List[str]) -> float:
        lowered = sentence.lower()
        score = 0.0

        term_hits = sum(1 for term in query_terms if term in lowered)
        score += term_hits * 1.4

        phrase_hits = sum(1 for phrase in exact_phrases if phrase in lowered)
        score += phrase_hits * 1.8

        cue_hits = sum(1 for cue in (*self.retrieval_profile.definition_cues, *self.retrieval_profile.feature_cues, *self.retrieval_profile.comparison_cues) if cue in sentence)
        score += min(cue_hits * 0.6, 1.2)

        if len(sentence) < 18:
            score -= 0.8
        elif len(sentence) <= 120:
            score += 0.4
        elif len(sentence) >= 220:
            score -= 0.3

        if sentence[:1] in {"（", "(", "["}:
            score -= 0.2

        return score

    def _detect_query_type(self, query: str) -> str:
        query_text = query.strip()
        if any(cue in query_text for cue in self.retrieval_profile.comparison_cues):
            return "comparison"
        if any(cue in query_text for cue in self.retrieval_profile.feature_cues):
            return "feature"
        return "definition"

    def _extract_query_terms(self, query: str) -> List[str]:
        tokens = self._tokenize(query)
        normalized_tokens: List[str] = []
        for token in tokens:
            if token in self.retrieval_profile.query_stopwords:
                continue
            if len(token) <= 1:
                continue
            normalized_tokens.append(token)
        return list(dict.fromkeys(normalized_tokens))

    def _extract_exact_phrases(self, query: str) -> List[str]:
        phrases = [match.strip().lower() for match in TOKEN_PATTERN.findall(query) if len(match.strip()) >= EXACT_PHRASE_MIN_LENGTH]
        filtered_phrases = [phrase for phrase in phrases if phrase not in self.retrieval_profile.query_stopwords]
        return list(dict.fromkeys(filtered_phrases))

    def _get_low_value_penalty(self, section_title: str, content: str) -> float:
        if self._is_low_value_chunk({"section_title": section_title}, content):
            return 0.18
        if any(keyword in section_title for keyword in self.retrieval_profile.low_value_weak_hints):
            return 0.08
        return 0.0

    def _get_exact_phrase_bonus(self, exact_phrases: List[str], section_title: str, content: str) -> float:
        if not exact_phrases:
            return 0.0
        combined = f"{section_title}\n{content}".lower()
        hits = sum(1 for phrase in exact_phrases if phrase in combined)
        if hits <= 0:
            return 0.0
        return min(0.12, 0.04 * hits)

    def _get_phrase_structure_bonus(
        self,
        query: str,
        query_type: str,
        exact_phrases: List[str],
        section_title: str,
        content: str,
    ) -> float:
        combined = f"{section_title}\n{content}"
        lowered_combined = combined.lower()
        query_text = query.strip().lower()
        bonus = 0.0

        primary_phrases = [phrase for phrase in exact_phrases if phrase not in self.retrieval_profile.query_stopwords and len(phrase) >= 3]
        if query_text and len(query_text) >= 3:
            primary_phrases = list(dict.fromkeys([query_text, *primary_phrases]))

        for phrase in primary_phrases:
            if phrase in lowered_combined:
                bonus += self.retrieval_profile.primary_phrase_hit_bonus
                if phrase in section_title.lower():
                    bonus += self.retrieval_profile.title_phrase_hit_bonus

        main_subject_bonus = self._get_main_subject_bonus(
            query=query,
            query_type=query_type,
            primary_phrases=primary_phrases,
            section_title=section_title,
            content=content,
        )
        if main_subject_bonus != 0:
            bonus += main_subject_bonus

        if query_type == "definition":
            for phrase in primary_phrases[:2]:
                if self._has_definition_pattern(phrase=phrase, text=lowered_combined):
                    bonus += self.retrieval_profile.definition_pattern_bonus
                    break
        elif query_type == "comparison":
            comparison_hits = sum(1 for pattern in self.retrieval_profile.comparison_patterns if pattern in combined)
            if comparison_hits > 0:
                bonus += min(self.retrieval_profile.comparison_pattern_bonus_max, self.retrieval_profile.comparison_pattern_bonus * comparison_hits)
        elif query_type == "feature":
            feature_hits = sum(1 for pattern in self.retrieval_profile.feature_patterns if pattern in combined)
            if feature_hits > 0:
                bonus += min(self.retrieval_profile.feature_pattern_bonus_max, self.retrieval_profile.feature_pattern_bonus * feature_hits)

        return min(self.retrieval_profile.phrase_structure_bonus_max, bonus)

    def _get_main_subject_bonus(
        self,
        query: str,
        query_type: str,
        primary_phrases: List[str],
        section_title: str,
        content: str,
    ) -> float:
        if query_type != "definition":
            return 0.0

        lowered_query = query.strip().lower()
        lowered_title = section_title.lower()
        lowered_content = content.lower()
        subject_candidates = [phrase for phrase in primary_phrases if len(phrase) >= 2]
        if lowered_query and len(lowered_query) >= 2:
            subject_candidates = list(dict.fromkeys([lowered_query, *subject_candidates]))

        for subject in subject_candidates:
            normalized_subject = self._normalize_definition_subject(subject)
            if not normalized_subject:
                continue
            if self._has_definition_pattern(normalized_subject, lowered_content) or self._has_definition_pattern(normalized_subject, lowered_title):
                return self.retrieval_profile.main_subject_bonus
            if self._is_subtype_definition(normalized_subject, lowered_content) or self._is_subtype_definition(normalized_subject, lowered_title):
                return -self.retrieval_profile.subtype_definition_penalty

        return 0.0

    def _normalize_definition_subject(self, phrase: str) -> str:
        normalized = phrase.strip().lower()
        normalized = normalized.removesuffix("？").removesuffix("?")
        for prefix in ("什么是", "什么叫", "什么叫做", "什么是指"):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
        return normalized.strip()

    def _is_subtype_definition(self, subject: str, text: str) -> bool:
        if not subject or not text:
            return False
        return any(
            self._has_definition_pattern(f"{prefix}{subject}", text)
            for prefix in self.retrieval_profile.subtype_definition_prefixes
        )

    def _has_definition_pattern(self, phrase: str, text: str) -> bool:
        if not phrase:
            return False
        return any(f"{phrase}{pattern}" in text for pattern in self.retrieval_profile.definition_patterns)

    def _get_intent_bonus(self, query_type: str, query_terms: List[str], section_title: str, content: str) -> float:
        combined = f"{section_title}\n{content}"
        if query_type == "comparison":
            cue_hits = sum(1 for cue in self.retrieval_profile.comparison_cues if cue in combined)
            return min(0.1, 0.04 * cue_hits) if cue_hits else 0.0
        if query_type == "feature":
            cue_hits = sum(1 for cue in self.retrieval_profile.feature_cues if cue in combined)
            return min(0.1, 0.04 * cue_hits) if cue_hits else 0.0

        cue_hits = sum(1 for cue in self.retrieval_profile.definition_cues if cue in combined)
        if cue_hits <= 0:
            return 0.0

        if query_terms:
            strong_patterns = []
            for term in query_terms:
                strong_patterns.extend([
                    f"{term}是",
                    f"{term}是指",
                    f"{term}的定义",
                    f"所谓{term}",
                ])
            if any(pattern in combined for pattern in strong_patterns):
                return 0.1
        return min(0.08, 0.03 * cue_hits)

    def _get_section_bonus(self, query_terms: List[str], section_title: str) -> float:
        if not query_terms or not section_title:
            return 0.0
        hits = sum(1 for term in query_terms if len(term) >= 2 and term in section_title.lower())
        if hits <= 0:
            return 0.0
        return min(0.08, 0.03 * hits)

    def _get_matched_token_bonus(self, query_terms: List[str], matched_tokens: List[str], content: str) -> float:
        if not query_terms:
            return 0.0
        matched = set(matched_tokens)
        coverage_hits = sum(1 for term in query_terms if term in matched or term in content.lower())
        if coverage_hits <= 0:
            return 0.0
        coverage_ratio = coverage_hits / max(len(query_terms), 1)
        if coverage_ratio >= 0.8:
            return 0.06
        if coverage_ratio >= 0.5:
            return 0.03
        return 0.0

    def _get_hybrid_bonus(self, item: Dict[str, Any]) -> float:
        if item.get("retrieval_source") == "hybrid":
            return 0.02
        return 0.0

    def _get_weak_fragment_penalty(self, content: str, query_type: str) -> float:
        stripped = content.strip()
        if not stripped:
            return 0.08
        if len(stripped) >= 120:
            return 0.0

        if query_type == "definition" and any(cue in stripped for cue in self.retrieval_profile.definition_cues):
            return 0.0
        if query_type == "feature" and any(cue in stripped for cue in self.retrieval_profile.feature_cues):
            return 0.0
        if query_type == "comparison" and any(cue in stripped for cue in self.retrieval_profile.comparison_cues):
            return 0.0
        return 0.05

    def _tokenize(self, text: str) -> List[str]:
        tokens = TOKEN_PATTERN.findall(text)
        expanded: List[str] = []
        for token in tokens:
            normalized = token.strip().lower()
            if not normalized:
                continue
            expanded.append(normalized)
            if len(normalized) >= 4 and all('\u4e00' <= char <= '\u9fff' for char in normalized):
                expanded.extend(normalized[i:i + 2] for i in range(len(normalized) - 1))
        return expanded

    def _is_low_value_chunk(self, metadata: Dict[str, Any], content: str) -> bool:
        section_title = str(metadata.get("section_title") or "")
        if any(keyword in section_title for keyword in self.retrieval_profile.low_value_section_keywords):
            return True

        snippet = content[:300]
        if sum(1 for pattern in self._profile_patterns(self.retrieval_profile.low_value_snippet_patterns) if pattern.search(snippet)) >= 2:
            return True

        return False
