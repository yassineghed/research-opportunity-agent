"""Centralized pipeline configuration.

Loads defaults from environment variables and provides a single
``PipelineConfig`` dataclass consumed by the agent, main.py,
and evaluation scripts.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


def _env(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value.strip())


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class PipelineConfig:
    # Embedding
    embedding_model: str = field(
        default_factory=lambda: _env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    )

    # Retrieval
    top_k_retrieval: int = field(
        default_factory=lambda: _env_int("TOP_K_RETRIEVAL", 20)
    )
    top_k_final: int = field(
        default_factory=lambda: _env_int("TOP_K_FINAL", 3)
    )

    # LLM providers (primary + secondary for dual reranking)
    llm_providers: list[str] = field(
        default_factory=lambda: _env("LLM_PROVIDERS", "gemini,qwen").split(",")
    )
    llm_allow_fallback: bool = field(
        default_factory=lambda: _env_bool("LLM_ALLOW_FALLBACK", True)
    )

    # Data paths (relative to PROJECT_ROOT)
    researchers_path: str = field(
        default_factory=lambda: _env(
            "RESEARCHERS_PATH", "data/mock/researchers.json"
        )
    )
    opportunities_path: str = field(
        default_factory=lambda: _env(
            "OPPORTUNITIES_PATH", "data/processed/opportunities.json"
        )
    )

    # Persistence
    index_persist_dir: str = field(
        default_factory=lambda: _env(
            "INDEX_PERSIST_DIR", "vector_store/index"
        )
    )

    @property
    def researchers_file(self) -> Path:
        return PROJECT_ROOT / self.researchers_path

    @property
    def opportunities_file(self) -> Path:
        return PROJECT_ROOT / self.opportunities_path

    @property
    def index_persist_path(self) -> Path:
        return PROJECT_ROOT / self.index_persist_dir
