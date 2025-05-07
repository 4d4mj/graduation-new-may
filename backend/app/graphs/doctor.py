from langgraph.graph import StateGraph, END
from app.tools.guardrails import guard_in, guard_out
from app.graphs.states import DoctorState
from app.graphs.agents import doctor_agent
import logging
from typing import Literal  # <-- Import Literal

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


def create_doctor_graph() -> StateGraph:
    """
    Create a streamlined doctor orchestrator graph using the prebuilt React agent approach.

    Flow:
    1. Apply input guardrails
    2. If safe, format message history & run medical agent
    3. If unsafe, END
    4. Apply output guardrails

    Returns:
        A compiled doctor StateGraph
    """
    g = StateGraph(DoctorState)

    # Add nodes
    g.add_node("guard_in", guard_in)
    g.add_node("agent", doctor_agent.medical_agent.ainvoke)
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

    # Remaining edges
    g.add_edge("agent", "guard_out")
    g.add_edge("guard_out", END)

    logger.info("Doctor graph created with conditional routing after input guardrail.")
    return g
