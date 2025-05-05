import logging
from langchain_core.tools import tool
from typing import Optional
from langchain_core.messages import AIMessage
# ─── RAG & search helpers ──────────────────────────────────────────────────
from app.agents.rag.core import MedicalRAG          # trimmed‑down version below
from langchain_community.tools.tavily_search import TavilySearchResults

logger = logging.getLogger(__name__)

# @tool("small_talk", return_direct=True)
# def small_talk(user_message: str) -> str:
#     """
#     Handle general conversation, greetings, and non-medical chat.
#     Use this for casual conversation or when the patient is making small talk.
#     """
#     return "I'm here to help with any medical questions or concerns. Is there something specific about your health you'd like to discuss?"

# ═══════════════════════════════════════════════════════════════════════════
# 2.  Knowledge‑retrieval tools
# ═══════════════════════════════════════════════════════════════════════════
_RAG: Optional[MedicalRAG] = None      # singleton so we don’t re‑load every call


def _get_rag() -> MedicalRAG:
    global _RAG
    if _RAG is None:
        _RAG = MedicalRAG()           # very light‑weight object now
    return _RAG


@tool("run_rag", return_direct=False)
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
        "answer":     answer_msg.content,
        "confidence": round(result.get("confidence", 0.0), 3),
        "sources":    result.get("sources", []),
    }


@tool("run_web_search", return_direct=False)
async def run_web_search(query: str, k: int = 5) -> str:
    """
    Lightweight public‑web fallback (Tavily).

    Returns the first `k` snippets concatenated.
    """
    tavily = TavilySearchResults(k=k)
    snippets = tavily.run(query)
    return "\n".join([item["snippet"] for item in snippets])
