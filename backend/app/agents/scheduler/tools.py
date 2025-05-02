"""
Local (non‑MCP) appointment‑scheduler tools.

They wrap your existing DummyScheduler *today*; swap the implementation
for real DB calls whenever you're ready – the agent code will not change.
"""
from __future__ import annotations
import logging
from datetime import datetime, timedelta
from langchain_core.tools import tool
from langchain_core.messages import AIMessage
from .core import DummyScheduler               # the stub you already have

logger = logging.getLogger(__name__)
_scheduler = DummyScheduler()                  # single instance – keeps "previous offer" in memory


# ------------------------------------------------------------------ helpers
def _as_ai(msg: str) -> AIMessage:
    return AIMessage(content=msg)


# ------------------------------------------------------------------ tools
@tool("list_free_slots", return_direct=False)
def list_free_slots(doctor_id: int = 1, day: str | None = None) -> AIMessage:
    """
    Return a *human‑readable* list of 30‑minute free slots for a doctor on a given ISO date.

    Arguments
    ----------
    doctor_id : int   – internal doctor identifier (default 1)
    day       : str   – date in YYYY‑MM‑DD; defaults to tomorrow
    """
    if day is None:
        day = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")

    # With DummyScheduler we cheat and just ask it to "list slots …"
    resp = _scheduler.process_schedule(f"list slots for doctor {doctor_id} on {day}")

    if isinstance(resp, AIMessage):
        return resp                            # DummyScheduler already gave us text

    # Real DB version should build a nice string here
    return _as_ai("Here are the available slots …")


@tool("book_appointment", return_direct=False)
def book_appointment(patient_id: int,
                     doctor_id:  int,
                     starts_at:  str,
                     ends_at:    str,
                     location:   str = "Main Clinic",
                     notes:      str | None = None) -> AIMessage:
    """
    Persist a new appointment and return a confirmation.

    For now this just proxies DummyScheduler; swap out with a DB insert later.
    """
    # DummyScheduler only needs a natural‑language message:
    msg = (
        f"schedule appointment for patient {patient_id} with doctor {doctor_id} "
        f"at {starts_at}‑{ends_at} in {location}. {f'Notes: {notes}' if notes else ''}"
    )
    resp = _scheduler.process_schedule(msg)
    return _as_ai(resp["response"] if isinstance(resp, dict) else str(resp))


@tool("cancel_appointment", return_direct=False)
def cancel_appointment(appointment_id: int,
                       patient_id:     int) -> AIMessage:
    """
    Cancel an appointment and confirm to the user.
    """
    # Again, DummyScheduler only simulates; replace with DELETE … WHERE id = …
    msg  = f"cancel appointment {appointment_id} for patient {patient_id}"
    resp = _scheduler.process_schedule(msg)
    return _as_ai("Your appointment has been cancelled." if isinstance(resp, dict) else str(resp))

