# backend/app/agents/rag/core.py
import logging
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
import logging
import random
from langchain_core.messages import AIMessage
from langchain_cohere import CohereRerank  # Assuming we use this

from app.core.models import get_llm, get_reranker
from .vector_store import search_vector_store, get_vector_store
from app.config.agent import settings as agent_settings
# Removed compression retriever import for now, using reranker more directly

logger = logging.getLogger(__name__)


class MedicalRAG:
    """
    Core RAG Agent: retrieves documents, reranks, generates response, calculates confidence.
    """

    def __init__(self, config=None):
        self.config = config
        self.reranker: Optional[CohereRerank] = (
            get_reranker()
        )  # Type hint clarifies it's CohereRerank
        self.llm = get_llm("rag_generator")
        self.vector_store = get_vector_store()  # Get store instance once

        if not self.vector_store:
            logger.error("RAG init failed: Vector store unavailable.")
            # Should probably raise an error here to prevent agent use
            raise ConnectionError("MedicalRAG cannot function without a vector store.")
        if not self.llm:
            logger.error("RAG LLM failed to initialize.")
            # raise ValueError("MedicalRAG cannot function without an LLM.")

        log_msg = "Initialized MedicalRAG"
        if self.reranker:
            log_msg += " with Reranker."
        else:
            log_msg += " without Reranker."
        logger.info(log_msg)

    async def process_query(
        self, query: str, chat_history_str: Optional[str] = None
    ) -> dict:
        logger.info(f"RAG processing query: '{query}'")

        # --- 1. Initial Retrieval ---
        # Retrieve more documents than finally needed if reranking is active
        k_initial_retrieval = (
            agent_settings.rag.reranker_top_k * 3
            if self.reranker
            else agent_settings.rag.reranker_top_k
        )
        logger.debug(
            f"Performing initial retrieval for top {k_initial_retrieval} documents."
        )

        try:
            results_with_scores = await search_vector_store(
                query=query,
                k=k_initial_retrieval,
                store_instance=self.vector_store,  # Pass the instance
            )
            if not results_with_scores:
                logger.warning("No documents found during initial retrieval.")
                return {
                    "response": AIMessage(
                        content="I couldn't find relevant information."
                    ),
                    "sources": [],
                    "confidence": 0.0,
                }

        except Exception as e:
            logger.error(f"Error during vector store search: {e}", exc_info=True)
            return {
                "response": AIMessage(
                    content="An error occurred during information retrieval."
                ),
                "sources": [],
                "confidence": 0.0,
            }

        initial_docs = [doc for doc, score in results_with_scores]
        # Store initial scores associated with doc id for later confidence calculation
        initial_scores_map = {
            doc.metadata.get("doc_id", f"doc_{i}"): score
            for i, (doc, score) in enumerate(results_with_scores)
        }

        # --- 2. Reranking ---
        final_docs_for_context: List[Document] = []
        if self.reranker and initial_docs:
            logger.debug(f"Applying reranker to {len(initial_docs)} documents.")
            try:
                # CohereRerank expects documents and query
                reranked_docs = self.reranker.compress_documents(
                    documents=initial_docs, query=query
                )
                # Ensure we don't exceed the desired final number
                final_docs_for_context = reranked_docs[
                    : agent_settings.rag.reranker_top_k
                ]
                logger.info(
                    f"Reranked down to {len(final_docs_for_context)} documents."
                )
                if not final_docs_for_context:
                    logger.warning(
                        "Reranker eliminated all documents, falling back to top initial results."
                    )
                    final_docs_for_context = initial_docs[
                        : agent_settings.rag.reranker_top_k
                    ]

            except Exception as e:
                logger.error(
                    f"Error during reranking: {e}. Falling back to initial retrieval order.",
                    exc_info=True,
                )
                final_docs_for_context = initial_docs[
                    : agent_settings.rag.reranker_top_k
                ]
        else:
            # No reranker or no initial docs, just take the top K from initial retrieval
            final_docs_for_context = initial_docs[: agent_settings.rag.reranker_top_k]

        if not final_docs_for_context:
            logger.warning("No documents available for generation context.")
            return {
                "response": AIMessage(
                    content="No relevant information found to generate an answer."
                ),
                "sources": [],
                "confidence": 0.0,
            }

        # --- 3. Calculate Confidence Score ---
        # Base confidence on the *initial retrieval scores* of the documents *actually used* in the final context
        confidence = 0.0
        scores_for_confidence = []
        doc_ids_used = {
            doc.metadata.get("doc_id", f"doc_{i}")
            for i, doc in enumerate(final_docs_for_context)
        }

        for doc_id, score in initial_scores_map.items():
            if doc_id in doc_ids_used:
                scores_for_confidence.append(score)

        if scores_for_confidence:
            confidence = sum(scores_for_confidence) / len(scores_for_confidence)
        logger.info(f"Calculated RAG confidence score: {confidence:.4f}")

        # --- 4. Generation ---
        context = "\n\n---\n\n".join(
            [doc.page_content for doc in final_docs_for_context]
        )
        sources_dict = {
            doc.metadata.get("source", f"Unknown_{i}"): doc.metadata
            for i, doc in enumerate(final_docs_for_context)
        }
        sources = list(sources_dict.values())

        prompt = f"""**Task:** Answer the clinical query based *strictly* on the provided context from medical literature/documents. Assume the user is a healthcare professional.

**Retrieved Context:**
---
{context}
---

**Query:** {query}

**Instructions:**
1.  Synthesize a direct and informative answer using **only** the information present in the "Retrieved Context".
2.  Prioritize accuracy and relevance to the clinical query.
3.  Use appropriate medical terminology as found in the context.
4.  Structure the answer logically (e.g., bullet points for lists, paragraphs for explanations).
5.  If the context provides relevant data (e.g., dosages, statistics, diagnostic criteria), include it accurately.
6.  If the context does not contain sufficient information to fully answer the query, state clearly what information is missing or cannot be confirmed from the provided text. Example: "The provided context discusses mechanism X but does not detail outcome Y."
7.  **Do not** add information, interpretations, or conclusions not explicitly supported by the context. Avoid making recommendations not present in the source text.

**Answer:**"""

        if not self.llm:
            # Handle missing LLM (already logged in init)
            return {
                "response": AIMessage(content="Error: Response generator unavailable."),
                "sources": sources,
                "confidence": 0.0,
            }

        try:
            logger.debug("Generating final response from context...")
            ai_response = await self.llm.ainvoke(prompt)
            logger.info("RAG response generated.")
        except Exception as e:
            logger.error(f"Error during response generation: {e}", exc_info=True)
            return {
                "response": AIMessage(
                    content="An error occurred generating the answer."
                ),
                "sources": sources,
                "confidence": confidence,
            }

        # --- 5. Format Output ---
        return {
            "response": ai_response,
            "sources": sources,
            "confidence": confidence,
        }
