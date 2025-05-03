import logging
from typing import Dict, Any, Union
from langchain_core.tools import tool, Tool
from langchain_core.messages import AIMessage

from app.agents.rag.core import DummyRAG
from app.agents.web_search.core import DummyWebSearch
# Import the scheduler tools instead of using DummyScheduler
from app.agents.scheduler.tools import list_free_slots, book_appointment, cancel_appointment
from app.config.agent import settings as agentSettings

logger = logging.getLogger(__name__)

@tool("rag_query", return_direct=False)
def rag_query(query: str) -> str:
    """
    Retrieve medical knowledge for a patient query from a trusted medical database.
    Use this when the user asks about medical conditions, treatments, or general health information.
    """
    rag = DummyRAG()
    resp = rag.process_query(query)
    return resp["response"].content if hasattr(resp["response"], "content") else str(resp["response"])

@tool("web_search", return_direct=False)
def web_search(query: str) -> str:
    """
    Perform a web search for current medical information that might not be in our database.
    Use this for recent medical news, emerging treatments, or when you need supplementary information.
    """
    web = DummyWebSearch()
    result = web.process_web_search_results(query)
    return result.content if hasattr(result, "content") else str(result)

# Properly configure the scheduler tools with Tool.from_function
# This is the correct way to expose async tools to LangChain
list_slots_tool = Tool.from_function(
    func=list_free_slots,
    name="list_free_slots",
    description=list_free_slots.__doc__,
    coroutine=list_free_slots,
    return_direct=False
)

book_appointment_tool = Tool.from_function(
    func=book_appointment,
    name="book_appointment",
    description=book_appointment.__doc__,
    coroutine=book_appointment,
    return_direct=False
)

cancel_appointment_tool = Tool.from_function(
    func=cancel_appointment,
    name="cancel_appointment",
    description=cancel_appointment.__doc__,
    coroutine=cancel_appointment,
    return_direct=False
)

@tool("small_talk", return_direct=True)
def small_talk(user_message: str) -> str:
    """
    Handle general conversation, greetings, and non-medical chat.
    Use this for casual conversation or when the patient is making small talk.
    """
    return "I'm here to help with any medical questions or concerns. Is there something specific about your health you'd like to discuss?"
