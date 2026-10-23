from __future__ import annotations

import json
import re
from typing import Any, AsyncGenerator, Dict, List

from openai import AsyncOpenAI

from app.core.config import settings
from app.core.logger import get_logger
from app.services.embedding_service import EmbeddingService
from app.services.profile_resolver_service import ProfileResolver, SimpleProfileConfig

logger = get_logger(service="book_rag_chat")

DEFINITION_CUES = ("定义", "是什么", "什么是", "概念", "是指", "称为")
FEATURE_CUES = ("特点", "特征", "优点", "缺点", "性质", "要求")
COMPARISON_CUES = ("区别", "不同", "联系", "比较", "相同点", "不同点")


class BookRAGChatService:
    def __init__(
        self,
        document_type: str = "textbook",
        noise_level: str = "medium",
        answer_style: str = "balanced",
        retrieval_mode: str = "balanced",
    ):
        self.client = AsyncOpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL,
        )
        self.model = settings.DEEPSEEK_MODEL
        self.profile_config = SimpleProfileConfig(
            document_type=document_type,  # type: ignore[arg-type]
            noise_level=noise_level,  # type: ignore[arg-type]
            answer_style=answer_style,  # type: ignore[arg-type]
            retrieval_mode=retrieval_mode,  # type: ignore[arg-type]
        )
        resolved_paths = ProfileResolver.resolve(self.profile_config)
        self.embedding_service = EmbeddingService(
            chunking_config_path=resolved_paths.chunking_config_path,
            retrieval_profile_path=resolved_paths.retrieval_profile_path,
        )
        self.answer_style = answer_style

    async def retrieve(self, index_id: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return await self.embedding_service.hybrid_search_index(index_id=index_id, query=query, top_k=top_k)

    async def generate(self, query: str, index_id: str, top_k: int = 5) -> Dict[str, Any]:
        retrieval_results = await self.retrieve(index_id=index_id, query=query, top_k=top_k)
        references = [self._build_reference_item(item) for item in retrieval_results]
        messages = self._build_messages(query=query, retrieval_results=retrieval_results)
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=False,
        )
        answer = response.choices[0].message.content or ""
        return {
            "query": query,
            "answer": answer,
            "references": references,
            "profile": {
                "document_type": self.profile_config.document_type,
                "noise_level": self.profile_config.noise_level,
                "answer_style": self.profile_config.answer_style,
                "retrieval_mode": self.profile_config.retrieval_mode,
            },
        }

    async def generate_stream(
        self,
        query: str,
        index_id: str,
        top_k: int = 5,
    ) -> AsyncGenerator[str, None]:
        try:
            retrieval_results = await self.retrieve(index_id=index_id, query=query, top_k=top_k)
            references = [self._build_reference_item(item) for item in retrieval_results]
            messages = self._build_messages(query=query, retrieval_results=retrieval_results)

            yield f"data: {json.dumps({'type': 'references', 'references': references, 'profile': {'document_type': self.profile_config.document_type, 'noise_level': self.profile_config.noise_level, 'answer_style': self.profile_config.answer_style, 'retrieval_mode': self.profile_config.retrieval_mode}}, ensure_ascii=False)}\n\n"

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
            )

            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    yield f"data: {json.dumps({'type': 'answer', 'content': content}, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error(f"Book RAG generate_stream error: {str(e)}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    def _build_system_prompt(self, prompt_hint: str) -> str:
        style_instruction = {
            "faithful": "尽量贴近教材原文表达，不额外延伸。",
            "balanced": "先直接回答结论，再用 1 到 3 句补充必要解释。",
            "summary": "优先用精炼总结作答，控制篇幅。",
        }.get(self.answer_style, "先直接回答结论，再补充必要解释。")

        return (
            "你是教材知识库问答助手。你必须严格依据提供的参考片段回答问题，"
            "不要使用参考片段之外的知识进行扩写、脑补或发挥。"
            "如果参考片段里没有标准定义句或完整特点句，但存在足够相关的信息，"
            "你应该优先基于相关片段做保守概括，不要轻易回答依据不足。"
            "只要参考片段中出现了与问题核心概念直接相关的定义、性质、用途、分类、图示说明或描述性语句，"
            "就先综合这些片段给出简洁回答，再说明“根据教材片段可概括为”。"
            "只有在片段确实与问题核心概念无关时，才回答“根据当前检索到的教材片段，暂无充分依据回答该问题”。"
            "如果多个片段都相关，优先采用最直接回答问题的片段，并保持表述一致。"
            f"{style_instruction}"
            f"当前问题的回答组织要求：{prompt_hint}"
        )

    def _build_messages(self, query: str, retrieval_results: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        context = self._build_context(retrieval_results)
        prompt_hint = self._build_prompt_hint(query)
        return [
            {
                "role": "system",
                "content": self._build_system_prompt(prompt_hint),
            },
            {
                "role": "user",
                "content": (
                    f"问题：{query}\n\n"
                    "请按以下优先级作答：\n"
                    "1. 优先直接回答问题，不要先说背景。\n"
                    "2. 若片段能支持简要概括，就先给出概括，再补 1 到 3 个依据点。\n"
                    "3. 定义题优先给一句定义；特点题优先列要点；比较题优先分别说明差异。\n"
                    "4. 只有在片段无法支持任何有效回答时，才明确说依据不足。\n\n"
                    f"回答要求：{prompt_hint}\n\n"
                    f"参考片段：\n{context}"
                ),
            },
        ]

    def _build_prompt_hint(self, query: str) -> str:
        if any(cue in query for cue in COMPARISON_CUES):
            return "先概括两者关系，再分点说明主要区别或联系。"
        if any(cue in query for cue in FEATURE_CUES):
            return "先给出核心特点，再按要点分条说明，不要展开无关背景。"
        if any(cue in query for cue in DEFINITION_CUES):
            return "先给出教材中的标准定义，再补充一句必要解释。"
        return "先直接回答问题，再补充与问题最相关的依据，不要发散。"

    def _build_context(self, retrieval_results: List[Dict[str, Any]]) -> str:
        if not retrieval_results:
            return "未检索到可用参考片段。"

        blocks = []
        for idx, item in enumerate(retrieval_results, start=1):
            metadata = item.get("metadata") or {}
            section_title = metadata.get("section_title") or metadata.get("book_name") or "未命名章节"
            content = self._normalize_content(str(item.get("content") or ""))
            blocks.append(
                "\n".join(
                    [
                        f"[{idx}] 章节：{section_title}",
                        f"页码：{metadata.get('page_start')} - {metadata.get('page_end')}",
                        f"chunk_id：{metadata.get('chunk_id', '')}",
                        f"内容：{content}",
                    ]
                )
            )
        return "\n\n---\n\n".join(blocks)

    def _normalize_content(self, content: str) -> str:
        normalized = re.sub(r"\s+", " ", content).strip()
        if len(normalized) <= 560:
            return normalized
        return normalized[:560].rstrip("，、；： ") + "……"

    def _build_reference_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        metadata = item.get("metadata") or {}
        content = item.get("content", "")
        original_content = item.get("original_content", content)
        return {
            "chunk_id": metadata.get("chunk_id", ""),
            "book_name": metadata.get("book_name", ""),
            "section_title": metadata.get("section_title", ""),
            "start_page": metadata.get("page_start"),
            "end_page": metadata.get("page_end"),
            "score": item.get("score"),
            "retrieval_source": item.get("retrieval_source", "dense"),
            "dense_rank": item.get("dense_rank"),
            "sparse_rank": item.get("sparse_rank"),
            "matched_tokens": item.get("matched_tokens", []),
            "compression_applied": item.get("compression_applied", False),
            "dedup_count": item.get("dedup_count", 1),
            "duplicate_chunk_ids": item.get("duplicate_chunk_ids", []),
            "snippet": content[:280],
            "original_snippet": original_content[:280],
        }

