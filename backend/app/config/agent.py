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

from pydantic import Field, field_validator, FieldValidationInfo
from pydantic_settings import BaseSettings, SettingsConfigDict


# --------------------------------------------------------------------------- #
# 1.  RAG-specific knobs (no direct LLM reference here any more)
# --------------------------------------------------------------------------- #
class RAGSettings(BaseSettings):
    # vector store
    vector_db_type: Literal["pgvector"] = "pgvector"
    embedding_dim: int = 768

    # retrieval
    context_limit: int = 20
    insufficient_info_keywords: list[str] = [
        "don't have enough information",
        "insufficient information",
        "cannot answer",
        "unable to answer",
    ]

    # new: fine‑tune your chunking & embeddings
    embedding_model_name: str = Field(
        "models/text-embedding-004", env="RAG_EMBEDDING_MODEL_NAME"
    )
    chunk_size: int = Field(1500, env="RAG_CHUNK_SIZE")
    chunk_overlap: int = Field(200, env="RAG_CHUNK_OVERLAP")
    vector_collection_name: str = Field(
        "medical_documents", env="RAG_VECTOR_COLLECTION_NAME"
    )

    # new: reranker settings
    reranker: str = Field("rerank-v3.5", env="RAG_RERANKER_MODEL")
    reranker_top_k: int = Field(3, env="RAG_RERANKER_TOP_K")

    # local persistence
    processed_docs_dir: Path = Path("./data/processed")

    @field_validator("chunk_overlap")
    @classmethod
    def check_overlap(cls, v: int, info: FieldValidationInfo) -> int:
        if "chunk_size" in info.data and v >= info.data["chunk_size"]:
            raise ValueError("chunk_overlap must be less than chunk_size")
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
    # CONFIDENCE_THRESHOLD: float = 0.85

    rag_fallback_confidence_threshold: float = Field(
        default=0.75, env="AGENT_RAG_FALLBACK_CONFIDENCE"
    )

    model_config = SettingsConfigDict(
        env_prefix="AGENT_",
        env_file=f".env.{os.getenv('APP_ENV', 'development')}",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# single global instance
settings = AgentSettings()
