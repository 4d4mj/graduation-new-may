import logging
from langchain_core.tools import tool
from typing import Optional, List, Dict
from langchain_core.messages import AIMessage

# ─── RAG & search helpers ──────────────────────────────────────────────────
from app.agents.rag.core import MedicalRAG  # trimmed‑down version below
from langchain_community.tools.tavily_search import TavilySearchResults

from app.config.settings import settings

logger = logging.getLogger(__name__)

_RAG: Optional[MedicalRAG] = None  # singleton so we don’t re‑load every call


def _get_rag() -> MedicalRAG:
    global _RAG
    if _RAG is None:
        _RAG = MedicalRAG()  # very light‑weight object now
    return _RAG


@tool("run_rag")
async def run_rag(query: str, chat_history: str | None = None) -> dict:
    """
    Search the **internal** medical knowledge‑base.

    Returns
    -------
    dict  –  { "answer": str, "confidence": float, "sources": list }
    """
    rag = _get_rag()
    result = await rag.process_query(query, chat_history)
    answer_msg: AIMessage = result["response"]
    return {
        "answer": answer_msg.content,
        "confidence": round(result.get("confidence", 0.0), 3),
        "sources": result.get("sources", []),
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
                if isinstance(item, dict):
                    snippet = item.get("content", "") or item.get("snippet", "")
                    url = item.get("url", "") or item.get("source", "")
                elif hasattr(item, "page_content") or hasattr(item, "metadata"):
                    snippet = item.page_content
                    url = item.metadata.get("source", "") or item.metada.get("url", "")

                if snippet and url:
                    formatted_results.append({"snippet": snippet, "url": url})
                elif snippet:
                    formatted_results.append(
                        {"snippet": snippet, "url": "Source URL not available"}
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
