from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


DocumentType = Literal["general", "textbook", "paper", "manual", "faq"]
NoiseLevel = Literal["low", "medium", "high"]
AnswerStyle = Literal["faithful", "balanced", "summary"]
RetrievalMode = Literal["stable", "balanced", "aggressive"]


@dataclass(frozen=True)
class SimpleProfileConfig:
    document_type: DocumentType = "general"
    noise_level: NoiseLevel = "medium"
    answer_style: AnswerStyle = "balanced"
    retrieval_mode: RetrievalMode = "balanced"


@dataclass(frozen=True)
class ResolvedProfilePaths:
    retrieval_profile_path: str | None
    chunking_config_path: str | None


class ProfileResolver:
    @staticmethod
    def resolve(config: SimpleProfileConfig) -> ResolvedProfilePaths:
        base_dir = Path(__file__).resolve().parent / "book_rag"

        if config.document_type == "textbook":
            retrieval_profile = base_dir / "retrieval_profile.educate_rag_book.json"
            chunking_config = base_dir / "chunking_config.json"
        else:
            retrieval_profile = base_dir / "retrieval_profile.default.json"
            chunking_config = base_dir / "chunking_config.json"

        return ResolvedProfilePaths(
            retrieval_profile_path=str(retrieval_profile),
            chunking_config_path=str(chunking_config),
        )

