"""
Minimal Retrieval‑Augmented Generation for internal medical KB
───────────────────────────────────────────────────────────────
* PGVector for similarity search
* (optional) Cohere‑ReRank for quality boost
* Gemini for final synthesis
"""

from __future__ import annotations
import logging
from typing import List, Dict, Any, Optional

from langchain_core.messages import AIMessage
from langchain_core.documents import Document

from app.config.agent import settings as agent_settings
from app.core.models import get_llm, get_reranker
from .vector_store import get_vector_store, search_vector_store

logger = logging.getLogger(__name__)


class MedicalRAG:
    def __init__(self) -> None:
        logger.info("Attempting to initialize MedicalRAG...")
        self.vector_store = get_vector_store()
        if not self.vector_store:
            logger.error(
                "MedicalRAG Initialization Error: get_vector_store() returned None."
            )
            # Raise error immediately if critical dependency is missing
            raise RuntimeError("RAG initialisation failed – vector store is missing")

        self.llm = get_llm("rag_generator")
        if not self.llm:
            logger.error(
                "MedicalRAG Initialization Error: get_llm('rag_generator') returned None."
            )
            # Raise error immediately
            raise RuntimeError(
                "RAG initialisation failed – rag_generator LLM is missing"
            )

        self.reranker = get_reranker()  # may be None

        if not self.vector_store or not self.llm:
            raise RuntimeError(
                "RAG initialisation failed – vector store or LLM missing"
            )

        logger.info(
            "MedicalRAG ready%s",
            " (reranker ON)" if self.reranker else " (no reranker)",
        )

    # ──────────────────────────────────────────────────────────────────
    async def process_query(
        self, query: str, chat_history_str: Optional[str] = None
    ) -> Dict[str, Any]:
        """Return answer + confidence + source metadata."""
        if not self.vector_store or not self.llm:
            logger.error(
                "MedicalRAG.process_query called but instance is not properly initialized."
            )
            return {
                "response": AIMessage(content="Internal RAG Error: Component missing."),
                "sources": [],
                "confidence": 0.0,
            }

        k = agent_settings.rag.reranker_top_k * 3

        # 1. similarity search
        logger.debug(f"RAG: Performing vector search for query: '{query[:50]}...'")
        docs_scores = await search_vector_store(query, k=k * 3)
        logger.debug(f"RAG: Vector search returned {len(docs_scores)} results.")
        
        if not docs_scores:
            return {
                "response": AIMessage(content="I found no relevant information."),
                "sources": [],
                "confidence": 0.0,
            }

        docs, scores = zip(*docs_scores)

        # 2. (optional) rerank
        if self.reranker:
            docs = self.reranker.compress_documents(list(docs), query)[:k]

        context = "\n\n---\n\n".join(d.page_content for d in docs)

        # 3. confidence = mean of similarity scores for the docs actually used
        confidence = round(sum(scores[: len(docs)]) / len(docs), 3)

        # 4. generate answer
        prompt = (
            f"Answer strictly based on the context below.\n\n"
            f"Context:\n{context}\n\n"
            f"User question: {query}\n\nAnswer:"
        )
        answer = await self.llm.ainvoke(prompt)

        # 5. collect metadata
        sources = [d.metadata for d in docs]

        return {"response": answer, "sources": sources, "confidence": confidence}
