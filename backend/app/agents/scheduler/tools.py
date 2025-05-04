"""
Database‑powered appointment scheduler + knowledge‑retrieval tools
──────────────────────────────────────────────────────────────────
Everything here is exposed to the LLM as a *LangChain tool*.

✔ list_free_slots     – see open half‑hour slots for a doctor
✔ book_appointment    – create a new appointment
✔ cancel_appointment  – cancel an existing appointment
✔ run_rag             – search the internal medical KB and return an answer + confidence
✔ run_web_search      – fallback: search the public web for recent info
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, date, time, timezone
from typing import Optional, Iterable, Dict, Any, List

from langchain_core.tools import tool
from langchain_core.messages import AIMessage

# ─── DB crud helpers (unchanged) ────────────────────────────────────────────
from sqlalchemy.ext.asyncio import AsyncSession
from typing_extensions import Annotated
from langgraph.prebuilt import InjectedState

from app.db.crud.appointment import (
    get_available_slots_for_day,
    create_appointment,
    delete_appointment,
)
from app.db.crud.doctor import get_doctor_by_name
from app.db.session import get_db_session
from app.config.settings import settings

# ─── RAG & search helpers ──────────────────────────────────────────────────
from app.agents.rag.core import MedicalRAG          # trimmed‑down version below
from langchain_community.tools.tavily_search import TavilySearchResults

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Basic date/‑time utilities (unchanged)
# ═══════════════════════════════════════════════════════════════════════════
def _parse_iso_date(day_str: str | None) -> date:
    try:
        return date.fromisoformat(day_str) if day_str else (
            datetime.now(timezone.utc) + timedelta(days=1)
        ).date()
    except ValueError:
        logger.warning("Invalid ISO date '%s' – defaulting to tomorrow", day_str)
        return (datetime.now(timezone.utc) + timedelta(days=1)).date()


def _parse_iso_datetime(dt_str: str) -> datetime | None:
    """Very small wrapper around datetime.fromisoformat + UTC fallback."""
    try:
        dt = datetime.fromisoformat(dt_str.strip())
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        logger.warning("Invalid ISO datetime '%s'", dt_str)
        return None


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Appointment‑related tools  (kept exactly as before)
# ═══════════════════════════════════════════════════════════════════════════
@tool("list_free_slots", return_direct=False)
async def list_free_slots(doctor_name: str, day: str | None = None) -> str:
    """
    Human readable list of 30‑minute free slots for a doctor on a given day.

    Parameters
    ----------
    doctor_name : str  – Which doctor's calendar to check.
    day         : str  – ISO date (YYYY‑MM‑DD).  Tomorrow by default.
    """
    target_day = _parse_iso_date(day)
    async for db in get_db_session(str(settings.database_url)):
        if not (doc := await get_doctor_by_name(db, doctor_name)):
            return f"Sorry, I don't know any doctor named '{doctor_name}'."

        slots = await get_available_slots_for_day(db, doc.user_id, target_day)
        break                     # we only need the first successful session

    if not slots:
        return f"{doctor_name} has no free slots on {target_day}."
    return (
        f"Here are {doctor_name}'s free 30‑minute slots on {target_day}:\n"
        + " • ".join(slots)
    )


@tool("book_appointment", return_direct=False)
async def book_appointment(
    doctor_name: str,
    starts_at: str,
    patient_id: Annotated[int, InjectedState("user_id")],
    duration_minutes: int = 30,
    location: str = "Main Clinic",
    notes: str | None = None,
) -> str:
    """Create an appointment and return the DB confirmation / error text."""
    if not patient_id:
        return "I couldn't identify you – please log in again."

    start_dt = _parse_iso_datetime(starts_at)
    if not start_dt:
        return "Please give the start time in ISO format `YYYY‑MM‑DDTHH:MM:SS`."

    async for db in get_db_session(str(settings.database_url)):
        doc = await get_doctor_by_name(db, doctor_name)
        if not doc:
            return f"No doctor named '{doctor_name}'."

        result = await create_appointment(
            db,
            patient_id=patient_id,
            doctor_id=doc.user_id,
            starts_at=start_dt,
            ends_at=start_dt + timedelta(minutes=duration_minutes),
            location=location,
            notes=notes,
        )
        break

    if not result or result["status"] != "confirmed":
        return result.get("message", "Could not book – please try another slot.")
    return (
        f"Your appointment (ID {result['id']}) with Dr. {doctor_name} "
        f"on {start_dt:%Y‑%m‑%d at %H:%M UTC} is confirmed 🎉."
    )


@tool("cancel_appointment", return_direct=False)
async def cancel_appointment(
    appointment_id: int,
    patient_id: Annotated[int, InjectedState("user_id")],
) -> str:
    """Cancel an existing appointment owned by the current user."""
    if not patient_id:
        return "I couldn't identify you – please log in again."
    async for db in get_db_session(str(settings.database_url)):
        result = await delete_appointment(db, appointment_id, patient_id)
        break
    return result.get("message", "Sorry – I couldn't cancel that appointment.")


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
