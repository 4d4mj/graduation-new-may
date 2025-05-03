from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from app.config.settings import settings
# ── built‑in tools
from app.agents.tools import (
    small_talk
)

from app.agents.states import DoctorState

from app.agents.scheduler.tools import run_rag, run_web_search
from typing import Sequence
import logging

logger = logging.getLogger(__name__)

BASE_TOOLS = [
    small_talk,
    run_rag,
    run_web_search
]

ASSISTANT_SYSTEM_PROMPT = """
"""

def build_medical_agent(extra_tools: Sequence[BaseTool] = ()):
    """
    Build a React agent for medical assistance using LangGraph prebuilt components.

    Args:
        extra_tools (Sequence[BaseTool]): Additional tools to include, typically MCP tools

    Returns:
        A compiled agent that can be used as a node in the doctor graph
    """
    try:
        # Initialize the LLM with the Google Generative AI
        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-04-17",
            api_key=settings.google_api_key,
            temperature=0.7
        )

        # Combine base tools with extra tools
        tools = list(BASE_TOOLS) + list(extra_tools)

        # Log the tools being used
        tool_names = [getattr(t, "name", str(t)) for t in tools]
        logger.info(f"Building medical agent with tools: {tool_names}")

        # Create the React agent using the updated parameter names
        agent = create_react_agent(
            model=model,
            tools=tools,
            prompt=ASSISTANT_SYSTEM_PROMPT,
            state_schema=DoctorState,
            debug=True,
            version="v1"
        )

        logger.info("Medical react agent created successfully")
        return agent

    except Exception as e:
        logger.error(f"Error creating medical agent: {str(e)}", exc_info=True)
        raise

# Create a placeholder that will be replaced in the application lifecycle
medical_agent = None
