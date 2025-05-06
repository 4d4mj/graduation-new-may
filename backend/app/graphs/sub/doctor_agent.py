from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from app.config.settings import settings
from app.agents.states import DoctorState

from app.agents.tools import run_rag, run_web_search
from typing import Sequence
import logging

logger = logging.getLogger(__name__)

BASE_TOOLS = [
    run_rag,
    run_web_search
]

ASSISTANT_SYSTEM_PROMPT = """You are an AI assistant for healthcare professionals. You help doctors with information retrieval, scheduling, and administrative tasks.

AVAILABLE TOOLS:
1. run_rag - Primary knowledge retrieval tool that searches internal medical knowledge base
2. run_web_search - Secondary search tool for when internal knowledge is insufficient

INSTRUCTIONS:
- Always introduce yourself as a medical office assistant
- For medical questions, use the run_rag tool first to search our internal knowledge base
- Only use run_web_search when:
  * The confidence score from run_rag is below 0.7
  * The information needed is not found in our internal knowledge base
  * You need to supplement internal knowledge with recent medical developments
- Prioritize using internal knowledge (run_rag) whenever possible
- When responding to questions, cite sources if available
- For appointment or scheduling questions, acknowledge and offer to help with scheduling
- Be professional, clear, and concise in your responses
- Do not diagnose or provide medical advice - always clarify you're retrieving information only
- If you're unsure about any medical information, clearly state the limitations of your knowledge

Remember that your primary role is to assist healthcare professionals with information retrieval, not to replace medical judgment.
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
