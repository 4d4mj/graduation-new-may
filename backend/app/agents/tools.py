import logging
from typing import Dict, Any, Union
from langchain_core.tools import tool
from langchain_core.messages import AIMessage

from app.agents.rag.core import DummyRAG
from app.agents.web_search.core import DummyWebSearch
from app.agents.scheduler.core import DummyScheduler
from app.config.agent import settings as agentSettings

logger = logging.getLogger(__name__)

@tool("rag_query", return_direct=False)
def rag_query(query: str) -> AIMessage:
    """
    Retrieve medical knowledge for a patient query from a trusted medical database.
    Use this when the user asks about medical conditions, treatments, or general health information.
    """
    rag = DummyRAG()
    resp = rag.process_query(query)
    return resp["response"]

@tool("web_search", return_direct=False)
def web_search(query: str) -> AIMessage:
    """
    Perform a web search for current medical information that might not be in our database.
    Use this for recent medical news, emerging treatments, or when you need supplementary information.
    """
    web = DummyWebSearch()
    return web.process_web_search_results(query)

@tool("schedule_appointment", return_direct=False)
def schedule_appointment(query: str) -> Union[AIMessage, Dict[str, Any]]:
    """
    Book or suggest appointments based on patient symptoms or direct requests.
    Use this when the patient wants to schedule a visit or when their symptoms suggest they should see a doctor.
    """
    sched = DummyScheduler()
    result = sched.process_schedule(query)

    # Handle both AIMessage and dict return types
    if isinstance(result, dict) and "response" in result:
        if isinstance(result["response"], AIMessage):
            return result["response"]
        else:
            return AIMessage(content=str(result["response"]))
    else:
        return result

@tool("small_talk", return_direct=True)
def small_talk(user_message: str) -> str:
    """
    Handle general conversation, greetings, and non-medical chat.
    Use this for casual conversation or when the patient is making small talk.
    """
    return f"I'm here to help with any medical questions or concerns. Is there something specific about your health you'd like to discuss?"
