from langgraph.types import interrupt
import logging
from langchain_core.messages import ToolMessage  # new import

logger = logging.getLogger(__name__)


def _find_last_proposal(messages):
    """
    Find the last propose_booking tool message in the conversation history.

    Args:
        messages (list): List of conversation messages

    Returns:
        dict or None: The content of the last propose_booking tool call, or None if not found
    """
    for m in reversed(messages):
        if isinstance(m, ToolMessage) and m.name == "propose_booking":
            return m.content
    return None


def confirm_booking(state):
    """
    Interrupt handler for booking confirmation.
    Pauses execution and returns control to user for confirmation.
    When resumed, processes the user's confirmation response.

    Args:
        state (dict): The current state of the conversation

    Returns:
        dict: Updated state with next actions based on user confirmation
    """
    # ❶ Get the proposal either from saved state or by scanning messages
    payload = state.get("pending_booking") or _find_last_proposal(state["messages"])

    if not payload:
        # Handle case where pending_booking is missing
        logger.error("confirm_booking called but no proposal found")
        return {
            "final_output": "I'm sorry, there was an error processing your booking request. Please try again.",
            "agent_name": "Scheduler"
        }

    logger.info(f"Interrupting for booking confirmation: {payload}")

    # This raises a GraphInterrupt that bubbles to the caller
    answer = interrupt(payload)

    # Execution resumes here after the user replies
    logger.info(f"Resumed with user answer: {answer}")

    if str(answer).lower().startswith("y"):
        # User confirmed - replay the tool call with the real booking
        logger.info("User confirmed booking, proceeding with appointment creation")

        # Store the extra_calls first before clearing pending_booking
        extra_calls = [
            ("book_appointment", payload)  # This will be executed by LangGraph after resume
        ]

        # Only clear pending_booking after we've used it
        if "pending_booking" in state:
            del state["pending_booking"]

        return {
            "messages": [],
            "extra_calls": extra_calls,
            "agent_name": "Scheduler"
        }
    else:
        # User declined - exit with a message
        logger.info("User declined booking, canceling operation")

        # Clear the pending booking since it was rejected
        if "pending_booking" in state:
            del state["pending_booking"]

        return {
            "final_output": "Understood. No appointment has been booked.",
            "agent_name": "Scheduler"
        }
