from langgraph.graph import StateGraph, END
from app.agents.guardrails import guard_in, guard_out
from app.agents.states import PatientState
from app.graphs.agents import patient_agent
from app.agents.scheduler.interrupt import confirm_booking
import logging
from typing import Literal
from langchain_core.messages import ToolMessage

# Set up logging
logger = logging.getLogger(__name__)


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

# --- ADD THIS ROUTING FUNCTION ---
def route_after_agent(state: dict) -> Literal["confirm", "guard_out"]:
    """Routes to confirmation if booking is detected, otherwise to output guardrail."""
    # Look for propose_booking tool messages in the state
    for m in reversed(state["messages"]):
        if isinstance(m, ToolMessage) and m.name == "propose_booking":
            # Found a propose_booking message, set the pending_booking state and route to confirm
            state["pending_booking"] = m.content
            # Set the agent_name to "Scheduler" for better UI display
            state["agent_name"] = "Scheduler"
            logger.info(f"Detected propose_booking: {m.content}, routing to confirmation step")
            return "confirm"

    # No propose_booking detected, route to output guardrail
    logger.info("No booking proposal detected, routing to output guardrail.")
    return "guard_out"
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
    g.add_node("confirm", confirm_booking)
    g.add_node("guard_out", guard_out)

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

    # --- ADD CONDITIONAL EDGE FROM AGENT TO EITHER CONFIRM OR GUARD_OUT ---
    g.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "confirm": "confirm",   # If pending_booking exists, go to confirmation
            "guard_out": "guard_out"  # Otherwise proceed to output guardrail
        }
    )
    # --- END ADDITION ---

    # Add edge from confirm to guard_out
    g.add_edge("confirm", "guard_out")
    g.add_edge("guard_out", END)

    logger.info("Patient graph created with booking confirmation flow.")
    return g
