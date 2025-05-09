import logging
from langchain_core.tools import tool
from typing import Optional, List, Dict, Any
from langchain_core.messages import AIMessage

# ─── RAG & search helpers ──────────────────────────────────────────────────
from app.tools.research.core import MedicalRAG          # trimmed‑down version below
from langchain_community.tools.tavily_search import TavilySearchResults

from app.config.settings import settings

logger = logging.getLogger(__name__)

_RAG: Optional[MedicalRAG] = None  # singleton so we don’t re‑load every call


def _get_rag() -> MedicalRAG | None:
    global _RAG
    if _RAG is None:
        logger.info("_get_rag: RAG instance is None, attempting to create.")
        try:
            _RAG = MedicalRAG()
        except RuntimeError as e:
            logger.error(f"_get_rag: Failed to create MedicalRAG instance: {e}")
            _RAG = None  # Ensure it remains None on failure
        except Exception as e:
            logger.error(
                f"_get_rag: Unexpected error creating MedicalRAG: {e}", exc_info=True
            )
            _RAG = None
    logger.debug(f"_get_rag: Returning RAG instance: {'Exists' if _RAG else 'None'}")
    return _RAG


@tool("run_rag")
async def run_rag(
    query: str, chat_history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Search the **internal** medical knowledge‑base.

    Returns
    -------
    dict  –  { "answer": str, "confidence": float, "sources": list }
    """
    logger.info(f"run_rag tool invoked for query: '{query[:50]}...'")

    rag_instance = _get_rag()

    if not rag_instance:
        logger.error("run_rag tool: Failed to get valid MedicalRAG instance.")
        return {
            "answer": "Error: The internal knowledge base (RAG) is currently unavailable.",
            "confidence": 0.0,
            "sources": [],
        }
    try:
        chat_history_as_string = None
        if chat_history:
            logger.debug(f"Chat history received by tool: {chat_history}")
            pass

        result = await rag_instance.process_query(
            query, chat_history_str=chat_history_as_string
        )

        if "error" in result.get("answer", "").lower():
            logger.warning(
                f"run_rag: process_query returned an error message: {result.get('answer')}"
            )
            result["confidence"] = 0.0

            # Ensure response is AIMessage before accessing .content
        response_content = "Error processing RAG response."
        if isinstance(result.get("response"), AIMessage):
            response_content = result["response"].content
        elif isinstance(
            result.get("answer"), str
        ):  # If process_query returns string directly in 'answer'
            response_content = result["answer"]

        return {
            "answer": response_content,
            "confidence": round(result.get("confidence", 0.0), 3),
            "sources": result.get("sources", []),
        }

    except TypeError as te:  # Catch TypeError specifically
        logger.error(
            f"run_rag tool: TypeError during rag.process_query: {te}", exc_info=True
        )
        return {
            "answer": f"Internal error calling knowledge base (TypeError): {te}",
            "confidence": 0.0,
            "sources": [],
        }

    except Exception as e:
        logger.error(
            f"run_rag tool: Error during rag.process_query: {e}", exc_info=True
        )
        return {
            "answer": f"Error processing query with internal knowledge base: {e}",
            "confidence": 0.0,
            "sources": [],
        }


@tool("run_web_search")
async def run_web_search(query: str, k: int = 5) -> List[Dict[str, str]]:
    """
    Search the public web using Tavily for up-to-date information.

    Returns:
    -------
    List[Dict[str, str]]
        A list of dictionaries, where each dictionary contains a 'snippet'
        of text and its corresponding 'url'. Returns an empty list or list
        with error message if search fails.
    """
    logger.info(f"Running web search for query: '{query}' (k={k})")
    if not settings.tavily_api_key:
        logger.error("TAVILY_API_KEY not configured")
        return [{"snippet": "web search is not configured.", "url": ""}]

    try:
        tavily_tool = TavilySearchResults(max_results=k)
        results = await tavily_tool.ainvoke(query)

        formatted_results = []

        if isinstance(results, list):
            for item in results:
                snippet = ""
                url = ""
                if hasattr(item, "page_content") and hasattr(item, "metadata"):
                    snippet = item.page_content
                    url = item.metadata.get("source", "") or item.metadata.get(
                        "url", ""
                    )
                elif isinstance(item, dict):
                    snippet = item.get("content", "") or item.get("snippet", "")
                    url = item.get("url", "") or item.get("source", "")

                if snippet:
                    formatted_results.append(
                        {
                            "snippet": snippet,
                            "url": url
                            or "Source URL not available",  # Provide fallback text if URL missing
                        }
                    )

        elif isinstance(results, str):
            logger.warning("Tavily tool returned a single string, expected list")
            formatted_results.append(
                {"snippet": results, "url": "Source url not available"}
            )

        logger.info(f"Web search formatted {len(formatted_results)} results")
        if not formatted_results:
            return [{"snippet": "No relevant information found on the web", "url": ""}]

        return formatted_results

    except Exception as e:
        logger.error(f"Error during web search for '{query}': '{e}'", exc_info=True)
        return [{"snippet": "An error occured during web search ", "url": ""}]
