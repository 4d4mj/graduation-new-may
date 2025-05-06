from langgraph.types import interrupt
from app.agents.states import SchedulerState


def confirm_booking(state: SchedulerState):
    slot = state["pending_booking"]
    # 1️⃣ interrupt until the user says YES / NO
    answer = interrupt({
        "type": "confirm_booking",
        "doctor": slot["doctor"],
        "starts_at": slot["starts_at"].isoformat()
    })
    # answer is only returned *after* the resume step
    if str(answer).lower().startswith("y"):
        # proceed to actually call book_appointment tool here
        return {"messages": "✅ booked!"}
    else:
        return {"messages": "Okay, not booking anything."}
