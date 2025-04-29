"""
Provider-agnostic configuration
───────────────────────────────
The code no longer imports Azure/OpenAI classes anywhere; each part of the
pipeline calls `load_provider()` when it actually needs an LLM or an
embedding model.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.config.prompts import DECISION_SYSTEM_PROMPT
from app.config.constants import AgentName


# --------------------------------------------------------------------------- #
# 1.  RAG-specific knobs (no direct LLM reference here any more)
# --------------------------------------------------------------------------- #
class RAGSettings(BaseSettings):
    # vector store
    vector_db_type: Literal["qdrant"] = "qdrant"
    embedding_dim: int = 1536

    # retrieval
    min_retrieval_confidence: float = 0.80
    context_limit: int = 20
    insufficient_info_keywords: list[str] = [
        "don't have enough information",
        "insufficient information",
        "cannot answer",
        "unable to answer",
    ]

    # local persistence
    local_path: Path = Path("./data/qdrant_db")
    processed_docs_dir: Path = Path("./data/processed")

    @field_validator("local_path", "processed_docs_dir")
    @classmethod
    def _ensure_dirs(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v


# --------------------------------------------------------------------------- #
# 2.  Top-level agent settings
# --------------------------------------------------------------------------- #
class AgentSettings(BaseSettings):
    # nested
    rag: RAGSettings = Field(default_factory=RAGSettings)

    # orchestrator knobs
    web_search_context_limit: int = 20
    max_conversation_history: int = 40
    DECISION_SYSTEM_PROMPT: str = DECISION_SYSTEM_PROMPT
    CONFIDENCE_THRESHOLD: float = 0.85

    # dynamic provider – “ollama”, “openai”, “azure”, …
    provider: str = Field(default="ollama", validation_alias="LLM_PROVIDER")

    # allowed agents
    ALL_AGENTS: set[str] = {
        AgentName.CONVERSATION.value,
        AgentName.RAG.value,
        AgentName.WEB_SEARCH.value,
    }

    ROLE_TO_ALLOWED: dict[str, set[str]] = {
        "doctor": {
            AgentName.CONVERSATION.value,
            AgentName.RAG.value,
            AgentName.WEB_SEARCH.value,
        },
        "patient": {
            AgentName.CONVERSATION.value,
            AgentName.RAG.value,
        },
    }

    model_config = SettingsConfigDict(
        env_prefix="AGENT_",
        env_file=f".env.{os.getenv('APP_ENV', 'development')}",
        env_file_encoding="utf-8",
        extra="ignore"
    )


# single global instance
settings = AgentSettings()
