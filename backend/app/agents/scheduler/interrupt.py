from langgraph.types import interrupt
import logging

logger = logging.getLogger(__name__)


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
    # Use state.get() instead of direct dict access to avoid KeyError
    payload = state.get("pending_booking")

    if not payload:
        # Handle case where pending_booking is missing
        logger.error("confirm_booking called but no pending_booking found in state")
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
        return {
            "messages": [],
            "extra_calls": [
                ("book_appointment", payload)  # This will be executed by LangGraph after resume
            ],
            "agent_name": "Scheduler"
        }
    else:
        # User declined - exit with a message
        logger.info("User declined booking, canceling operation")
        return {
            "final_output": "Understood. No appointment has been booked.",
            "agent_name": "Scheduler"
        }
