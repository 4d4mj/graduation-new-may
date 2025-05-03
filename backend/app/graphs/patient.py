from langgraph.graph import StateGraph, END
from app.agents.guardrails.nodes import apply_input_guardrails, apply_output_guardrails
from app.agents.states import PatientState
from app.graphs.sub import agent_node
import logging

# Set up logging
logger = logging.getLogger(__name__)


def create_patient_graph() -> StateGraph:
    """
    Create a streamlined patient orchestrator graph using the prebuilt React agent approach.

    Flow:
    1. Apply input guardrails
    2. Format message history for Gemini compatibility
    3. Run medical agent (handles intent recognition, tool selection, and execution automatically)
    4. Apply output guardrails

    Returns:
        A compiled patient StateGraph
    """
    g = StateGraph(PatientState)

    # Add nodes for guardrails and the unified agent
    g.add_node("guard_in", apply_input_guardrails)
    g.add_node("agent", agent_node.medical_agent.ainvoke)
    g.add_node("apply_out", apply_output_guardrails)

    # Updated flow to include message formatting
    g.set_entry_point("guard_in")
    g.add_edge("agent", "apply_out")
    g.add_edge("apply_out", END)

    logger.info("Patient graph created with unified medical agent architecture and Gemini compatibility")
    return g
