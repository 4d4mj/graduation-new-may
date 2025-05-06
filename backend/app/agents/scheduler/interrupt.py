from langgraph.types import interrupt
import logging

from app.agents.states import SchedulerState

logger = logging.getLogger(__name__)


def confirm_booking(state: SchedulerState):
    """
    Interrupt handler for booking confirmation.
    Pauses execution and returns control to user for confirmation.
    When resumed, processes the user's confirmation response.

    Args:
        state (dict): The current state of the conversation

    Returns:
        dict: Updated state with next actions based on user confirmation
    """
    payload = state["pending_booking"]
    logger.info(f"Interrupting for booking confirmation: {payload}")

    # This raises a GraphInterrupt that bubbles to the caller
    answer = interrupt(payload)

    # Execution resumes here after the user replies
    logger.info(f"Resumed with user answer: {answer}")

    if str(answer).lower().startswith("y"):
        # User confirmed - replay the tool call with the real booking
        logger.info("User confirmed booking, proceeding with appointment creation")
        return {
            "messages": [],
            "extra_calls": [
                ("book_appointment", payload)  # This will be executed by LangGraph after resume
            ]
        }
    else:
        # User declined - exit with a message
        logger.info("User declined booking, canceling operation")
        return {"final_output": "Understood. No appointment has been booked."}
