from langgraph.graph import StateGraph, END
from app.agents.guardrails.nodes import apply_input_guardrails, apply_output_guardrails
from app.agents.states import PatientState
from app.graphs.sub.wrappers import (
    run_patient_supervisor_subgraph,
    run_conversation_subgraph,
    run_medical_qa_subgraph,
    run_scheduler_subgraph,
)
import logging

# Set up logging
logger = logging.getLogger(__name__)

def create_patient_graph() -> StateGraph:
    """
    Create a simplified patient orchestrator graph with a unified supervisor approach.
    The supervisor node replaces both the keyword classifier and LLM analysis.

    Flow:
    1. Apply input guardrails
    2. Run patient supervisor (LLM that handles intent, response, and scheduling decisions)
    3. Route based on scheduling flag and intent
    4. Apply output guardrails

    Returns:
        A compiled patient StateGraph
    """
    g = StateGraph(PatientState)

    # Add nodes for guardrails and unified supervisor
    g.add_node("guard_in", apply_input_guardrails)
    g.add_node("supervisor", run_patient_supervisor_subgraph)

    # Add task nodes for the different capabilities
    g.add_node("conversation", run_conversation_subgraph)
    g.add_node("medical_qa", run_medical_qa_subgraph)
    g.add_node("scheduler", run_scheduler_subgraph)
    g.add_node("apply_out", apply_output_guardrails)

    # Set up the flow of the graph
    g.set_entry_point("guard_in")
    g.add_edge("guard_in", "supervisor")

    # Combined routing logic - handle both scheduling and intent in a single conditional
    g.add_conditional_edges(
        "supervisor",
        lambda s: (
            "scheduler" if s.get("request_scheduling", False)
            else s.get("intent", "conversation")  # Default to conversation if no intent
        ),
        {
            "scheduler": "scheduler",
            "conversation": "conversation",
            "medical_qa": "medical_qa"
        }
    )

    # All paths converge to output guardrails
    g.add_edge("conversation", "apply_out")
    g.add_edge("medical_qa", "apply_out")
    g.add_edge("scheduler", "apply_out")
    g.add_edge("apply_out", END)

    return g
