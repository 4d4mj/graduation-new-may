from langchain_core.tools import StructuredTool, BaseTool
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from app.config.settings import settings
# ── built‑in tools
from app.agents.tools import rag_query, web_search, small_talk
# ── new scheduler tools
from app.agents.scheduler.tools import (
    list_free_slots,
    book_appointment,
    cancel_appointment,
)
from typing import Sequence
import logging

logger = logging.getLogger(__name__)

# Define base tools that are always available
BASE_TOOLS = [
    rag_query,
    web_search,
    small_talk,
    list_free_slots,
    book_appointment,
    cancel_appointment,
]

ASSISTANT_SYSTEM_PROMPT = """You are a professional, empathetic medical assistant AI.

YOUR CAPABILITIES:
1. Answer medical questions using trusted medical databases (rag_query)
2. Search the web for recent medical information (web_search)
3. Help patients schedule appointments with scheduling tools
4. Engage in general conversation (small_talk)

GUIDELINES:
- For medical questions, prioritize using rag_query for reliable information
- If you need recent or supplementary information, use web_search
- For any symptoms described as severe or concerning, suggest scheduling an appointment
- Always be respectful, clear, and empathetic
- Keep responses concise and focused on the patient's needs
- Do NOT diagnose or prescribe medications
- NEVER tell the patient you're going to use a specific tool - just use it naturally

SPECIAL INSTRUCTIONS FOR FOLLOW-UPS:
- If you have just offered to schedule an appointment and the user responds with a short affirmative like "yes", "sure", "okay", or "please", use the scheduling tools with their last reported symptoms
- Maintain context between conversation turns - if a user mentioned a symptom in a previous message, remember it when they ask follow-up questions

SCHEDULING TOOLS:
- If the patient asks to book, cancel, or see available slots, use the appropriate scheduling tool:
  - Use list_free_slots to find available appointment times for a doctor on a specific day
  - Use book_appointment to create a new appointment
  - Use cancel_appointment to cancel an existing appointment

Decide which tool to use based on the patient's needs, and provide helpful, accurate information.
"""

def build_medical_agent(extra_tools: Sequence[BaseTool] = ()):
    """
    Build a React agent for medical assistance using LangGraph prebuilt components.

    Args:
        extra_tools (Sequence[BaseTool]): Additional tools to include, typically MCP tools

    Returns:
        A compiled agent that can be used as a node in the patient graph
    """
    try:
        # Initialize the LLM with the Google Generative AI
        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-04-17",
            api_key=settings.google_api_key,
            temperature=0.7
        )

        # Combine base tools with extra tools (typically MCP scheduling tools)
        tools = list(BASE_TOOLS) + list(extra_tools)

        # Log the tools being used
        tool_names = [getattr(t, "name", str(t)) for t in tools]
        logger.info(f"Building medical agent with tools: {tool_names}")

        # Create the React agent using the updated parameter names
        agent = create_react_agent(
            model=model,
            tools=tools,
            prompt=ASSISTANT_SYSTEM_PROMPT,
            debug=False,
            version="v1"
        )

        logger.info("Medical react agent created successfully")
        return agent

    except Exception as e:
        logger.error(f"Error creating medical agent: {str(e)}", exc_info=True)
        raise

# Create a placeholder that will be replaced in the application lifecycle
medical_agent = None
