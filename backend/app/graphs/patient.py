from langgraph.graph import StateGraph, END # type: ignore
from app.tools.guardrails import guard_in, guard_out
from app.graphs.states import PatientState
from app.graphs.agents import patient_agent
import logging
from typing import Literal, Optional
from langchain_core.messages import ToolMessage

# Set up logging
logger = logging.getLogger(__name__)

# Define tools for direct output
DIRECT_TO_UI_PATIENT_TOOLS = {"list_free_slots", "list_doctors", "book_appointment", "propose_booking"}


# --- ADD THIS ROUTING FUNCTION ---
def route_after_guard_in(state: dict) -> Literal["agent", "__end__"]:
    """Routes to agent if input is safe, otherwise ends the graph."""
    if state.get("final_output"):
        # final_output was set by guard_in, meaning input was blocked
        logger.warning("Input guardrail triggered routing to END.")
        return "__end__"  # Special node name for LangGraph's end
    else:
        # Input is safe, proceed to the agent
        logger.info("Input guardrail passed, routing to agent.")
        return "agent"
# --- END ADDITION ---

# --- ADD THIS HELPER FUNCTION ---
# Helper to get the last tool invocation
def get_last_tool_invocation(state: dict) -> Optional[ToolMessage]:
    for m in reversed(state.get("messages", [])):
        if isinstance(m, ToolMessage):
            return m
    return None
# --- END ADDITION ---

# --- ADD THIS ROUTING FUNCTION ---
def route_after_agent(state: dict) -> Literal["structured_output_patient", "guard_out"]:
    last_tool_message = get_last_tool_invocation(state)

    if last_tool_message:
        if last_tool_message.name in DIRECT_TO_UI_PATIENT_TOOLS:
            state["raw_tool_output"] = last_tool_message.content
            state["is_direct_tool_response"] = True
            logger.info(f"Detected direct UI tool call: {last_tool_message.name}, routing to structured_output_patient")
            return "structured_output_patient"

    logger.info("No specific tool route after agent, proceeding to output guardrail.")
    return "guard_out"
# --- END ADDITION ---

# --- ADD THIS NODE FUNCTION ---
# New node for structured output
def structured_output_patient_node(state: dict) -> dict:
    if state.get("is_direct_tool_response") and state.get("raw_tool_output") is not None:
        state["final_output"] = state["raw_tool_output"]
        if isinstance(state["raw_tool_output"], dict):
            state["agent_name"] = state["raw_tool_output"].get("agent", "System Tool")
        else:
            state["agent_name"] = "System Tool"
        logger.info(f"Passing raw tool output as final_output for tool: {state.get('messages',[-1]).name if state.get('messages') else 'unknown'}")
    else:
        logger.warning("structured_output_patient_node called unexpectedly. Falling back to last reply.")
    return state
# --- END ADDITION ---

def create_patient_graph() -> StateGraph:
    """
    Create a streamlined patient orchestrator graph using the prebuilt React agent approach.

    Flow:
    1. Apply input guardrails
    2. If safe, format message history & run medical agent
    3. If unsafe, END
    4. Apply output guardrails

    Returns:
        A compiled patient StateGraph
    """
    g = StateGraph(PatientState)

    # Add nodes
    g.add_node("guard_in", guard_in)
    g.add_node("agent", patient_agent.medical_agent.ainvoke)
    g.add_node("guard_out", guard_out)
    g.add_node("structured_output_patient", structured_output_patient_node)

    # Set entry point
    g.set_entry_point("guard_in")

    # --- REPLACE SIMPLE EDGE WITH CONDITIONAL EDGE ---
    g.add_conditional_edges(
        "guard_in",
        route_after_guard_in,
        {
            "agent": "agent",       # If route_after_guard_in returns "agent"
            "__end__": END,         # If route_after_guard_in returns "__end__"
        }
    )
    # --- END REPLACEMENT ---

    # --- ADD CONDITIONAL EDGE FROM AGENT TO EITHER STRUCTURED_OUTPUT OR GUARD_OUT ---
    g.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "structured_output_patient": "structured_output_patient",  # If direct tool response, go to structured output
            "guard_out": "guard_out"  # Otherwise proceed to output guardrail
        }
    )
    # --- END ADDITION ---

    g.add_edge("guard_out", END)
    g.add_edge("structured_output_patient", END)

    logger.info("Patient graph created with direct tool output flow.")
    return g
