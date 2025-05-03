import logging
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

@tool("small_talk", return_direct=True)
def small_talk(user_message: str) -> str:
    """
    Handle general conversation, greetings, and non-medical chat.
    Use this for casual conversation or when the patient is making small talk.
    """
    return "I'm here to help with any medical questions or concerns. Is there something specific about your health you'd like to discuss?"
