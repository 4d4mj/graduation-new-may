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
from app.db.session import tool_db_session
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
@tool("list_free_slots", return_direct=True)
async def list_free_slots(doctor_name: str, day: str | None = None) -> str:
    """
    Human readable list of 30‑minute free slots for a doctor on a given day.

    Parameters
    ----------
    doctor_name : str  – Which doctor's calendar to check.
    day         : str  – ISO date (YYYY‑MM‑DD).  Tomorrow by default.
    """
    target_day = _parse_iso_date(day)
    logger.info(f"Tool 'list_free_slots' called for Dr. {doctor_name} on {target_day}")

    try:
        async with tool_db_session() as db:
            if not (doc := await get_doctor_by_name(db, doctor_name)):
                logger.warning(f"Doctor '{doctor_name}' not found in tool.")
                error_msg = f"Sorry, I don't know any doctor named '{doctor_name}'."
                logger.info(f"Tool list_free_slots returning: '{error_msg}'")
                return error_msg

            logger.debug(f"Found doctor {doc.user_id}. Checking slots...")
            slots = await get_available_slots_for_day(db, doc.user_id, target_day)
            logger.debug(f"Found slots: {slots}")

        if not slots:
            no_slots_msg = f"{doctor_name} has no free slots on {target_day}."
            logger.info(f"Tool list_free_slots returning: '{no_slots_msg}'")
            return no_slots_msg

        result_string = f"Here are {doctor_name}'s free 30‑minute slots on {target_day}:\n" + " • ".join(slots)
        logger.info(f"Tool list_free_slots returning: '{result_string}'")
        return result_string
    except Exception as e:
        logger.error(f"Error executing list_free_slots tool: {e}", exc_info=True)
        error_msg = "I encountered an error while trying to check the schedule. Please try again later."
        logger.info(f"Tool list_free_slots returning error: '{error_msg}'")
        return error_msg


@tool("book_appointment", return_direct=True)
async def book_appointment(
    doctor_name: str,
    starts_at: str,
    patient_id: Annotated[int, InjectedState("user_id")],
    duration_minutes: int = 30,
    location: str = "Main Clinic",
    notes: str | None = None,
) -> str:
    """Create an appointment and return the DB confirmation / error text."""
    logger.info(f"Tool 'book_appointment' called by user {patient_id} for Dr. {doctor_name} at {starts_at}")
    if not patient_id:
        logger.warning("Booking tool called without patient_id.")
        return "I couldn't identify you – please log in again."

    start_dt = _parse_iso_datetime(starts_at)
    if not start_dt:
        logger.warning(f"Invalid starts_at format received: {starts_at}")
        return "Please give the start time in ISO format `YYYY‑MM‑DDTHH:MM:SS`."

    try:
        async with tool_db_session() as db:
            doc = await get_doctor_by_name(db, doctor_name)
            if not doc:
                logger.warning(f"Doctor '{doctor_name}' not found during booking.")
                return f"No doctor named '{doctor_name}'."

            logger.debug(f"Attempting to create appointment entry in DB for user {patient_id} with doctor {doc.user_id}")
            result = await create_appointment(
                db,
                patient_id=patient_id,
                doctor_id=doc.user_id,
                starts_at=start_dt,
                ends_at=start_dt + timedelta(minutes=duration_minutes),
                location=location,
                notes=notes,
            )

        if not result or result.get("status") != "confirmed":
            logger.warning(f"Booking failed for Dr. {doctor_name} at {starts_at}. Reason: {result.get('message', 'Unknown') if result else 'None'}")
            return result.get("message", "Could not book – please try another slot.") if result else "Booking failed for an unknown reason."

        logger.info(f"Booking successful: ID {result.get('id')}")
        return (
            f"Your appointment (ID {result['id']}) with Dr. {doctor_name} "
            f"on {start_dt:%Y‑%m‑%d at %H:%M %Z} is confirmed 🎉." # Use %Z for timezone if available
        )
    except Exception as e:
        logger.error(f"Error executing book_appointment tool: {e}", exc_info=True)
        return "I encountered an error while trying to book the appointment. Please try again later."


@tool("cancel_appointment", return_direct=True)
async def cancel_appointment(
    appointment_id: int,
    patient_id: Annotated[int, InjectedState("user_id")],
) -> str:
    """Cancel an existing appointment owned by the current user."""
    logger.info(f"Tool 'cancel_appointment' called by user {patient_id} for appointment ID {appointment_id}")
    if not patient_id:
        logger.warning("Cancel tool called without patient_id.")
        return "I couldn't identify you – please log in again."

    try:
        async with tool_db_session() as db:
            logger.debug(f"Attempting to delete appointment {appointment_id} for user {patient_id}")
            result = await delete_appointment(db, appointment_id, patient_id)

        log_func = logger.info if result.get("status") == "cancelled" else logger.warning
        log_func(f"Cancellation result for appt {appointment_id} / user {patient_id}: {result}")

        return result.get("message", "Sorry – I couldn't cancel that appointment.")
    except Exception as e:
        logger.error(f"Error executing cancel_appointment tool: {e}", exc_info=True)
        return "I encountered an error while trying to cancel the appointment. Please try again later."


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
