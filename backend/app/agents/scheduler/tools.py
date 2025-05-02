"""
Local (non‑MCP) appointment‑scheduler tools.

They wrap your existing DummyScheduler *today*; swap the implementation
for real DB calls whenever you're ready – the agent code will not change.
"""
from __future__ import annotations
import logging
from datetime import datetime, timedelta
from langchain_core.tools import tool
from .core import DummyScheduler               # the stub you already have

logger = logging.getLogger(__name__)
_scheduler = DummyScheduler()                  # single instance – keeps "previous offer" in memory


# ------------------------------------------------------------------ tools
@tool("list_free_slots", return_direct=False)
def list_free_slots(doctor_id: int = 1, day: str | None = None) -> str:
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

    if hasattr(resp, "content"):
        return resp.content  # Extract the content if it's an AIMessage

    # Generate a more detailed response with formatted slots
    slots = _scheduler.available_slots.get(day, ["09:00", "10:30", "14:00", "16:30"])
    slot_list = "\n • ".join(slots)
    return f"Here are the available slots for Dr. Smith on {day}:\n • {slot_list}"


@tool("book_appointment", return_direct=False)  # Keep return_direct=False as this usually needs LLM interpretation
def book_appointment(patient_id: int,
                     doctor_id:  int,
                     starts_at:  str,
                     ends_at:    str,
                     location:   str = "Main Clinic",
                     notes:      str | None = None) -> str:
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

    # Extract the content if it's an AIMessage or dict, otherwise use as is
    if hasattr(resp, "content"):
        return resp.content
    elif isinstance(resp, dict) and "response" in resp:
        return resp["response"] if isinstance(resp["response"], str) else resp["response"].content
    else:
        return str(resp)


@tool("cancel_appointment", return_direct=False)  # Changed to return_direct=True as this is usually a simple confirmation
def cancel_appointment(appointment_id: int,
                       patient_id:     int) -> str:
    """
    Cancel an appointment and confirm to the user.
    """
    # Again, DummyScheduler only simulates; replace with DELETE … WHERE id = …
    msg  = f"cancel appointment {appointment_id} for patient {patient_id}"
    resp = _scheduler.process_schedule(msg)

    # Extract the content if it's an AIMessage, otherwise use a simple confirmation
    if hasattr(resp, "content"):
        return resp.content
    elif isinstance(resp, dict):
        return "Your appointment has been successfully cancelled."
    else:
        return str(resp)

