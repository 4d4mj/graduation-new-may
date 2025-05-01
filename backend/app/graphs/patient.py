from langgraph.graph import StateGraph, END
from app.graphs.sub.intents import Task, classify_patient_intent
from app.agents.guardrails.nodes import apply_input_guardrails, apply_output_guardrails
from app.agents.states import PatientState
from app.graphs.sub.wrappers import (
    run_patient_analysis_subgraph,
    run_conversation_subgraph,
    run_medical_qa_subgraph,
    run_scheduler_subgraph,
)
import logging

# Set up logging
logger = logging.getLogger(__name__)

def create_patient_graph() -> StateGraph:
    """
    Create a patient orchestrator graph that uses patient analysis before routing to task-specific subgraphs.

    Returns:
        A compiled patient StateGraph
    """
    g = StateGraph(PatientState)

    # Add nodes for guardrails, intent classification, and patient analysis
    g.add_node("guard_in", apply_input_guardrails)
    g.add_node("classify", classify_patient_intent)
    g.add_node("patient_analysis", run_patient_analysis_subgraph)

    # Add task nodes for the different capabilities
    g.add_node("conversation", run_conversation_subgraph)
    g.add_node("medical_qa", run_medical_qa_subgraph)
    g.add_node("scheduler", run_scheduler_subgraph)

    # Set up the flow of the graph
    g.set_entry_point("guard_in")
    g.add_edge("guard_in", "classify")

    # First route based on basic classifier results
    g.add_conditional_edges(
        "classify",
        lambda s: {
            Task.CONVERSATION.value: "patient_analysis",
            Task.MEDICAL_QA.value: "medical_qa",
            Task.SCHEDULING.value: "scheduler"
        }.get(s.get("intent")),
        {"patient_analysis": "patient_analysis", "medical_qa": "medical_qa", "scheduler": "scheduler"}
    )

    # From patient_analysis, decide whether to go to conversation or scheduler
    g.add_conditional_edges(
        "patient_analysis",
        lambda s: "scheduler" if s.get("request_scheduling", False) else "conversation",
        {"scheduler": "scheduler", "conversation": "conversation"}
    )

    # Add output guardrails after any task
    g.add_node("apply_out", apply_output_guardrails)
    g.add_edge("conversation", "apply_out")
    g.add_edge("medical_qa", "apply_out")
    g.add_edge("scheduler", "apply_out")
    g.add_edge("apply_out", END)

    return g
