from langgraph.graph import StateGraph, END
from app.graphs.sub.intents import Task, classify_patient_intent
from app.agents.guardrails.nodes import apply_input_guardrails, apply_output_guardrails
from app.agents.states import PatientState
from app.graphs.sub.wrappers import (
    conversation_graph,
    medical_qa_graph,
    scheduler_graph,
)

async def run_conversation_subgraph(state, config):
    """Run the conversation subgraph on the given state."""
    # Pass the config to ensure thread_id is maintained
    return await conversation_graph.ainvoke(state, config=config)


async def run_medical_qa_subgraph(state, config):
    """Run the medical Q&A subgraph on the given state."""
    # Pass the config to ensure thread_id is maintained
    return await medical_qa_graph.ainvoke(state, config=config)


async def run_scheduler_subgraph(state, config):
    """Run the scheduler subgraph on the given state."""
    # Pass the config to ensure thread_id is maintained
    return await scheduler_graph.ainvoke(state, config=config)


def create_patient_graph() -> StateGraph:
    """
    Create a patient orchestrator graph that routes to different task functions based on intent.

    Returns:
        A compiled patient StateGraph
    """
    g = StateGraph(PatientState)

    # Add nodes for guardrails and intent classification
    g.add_node("guard_in", apply_input_guardrails)
    g.add_node("classify", classify_patient_intent)

    # Add task nodes directly instead of using subgraphs
    g.add_node("conversation", run_conversation_subgraph)
    g.add_node("medical_qa", run_medical_qa_subgraph)
    g.add_node("scheduler", run_scheduler_subgraph)

    # Set up the flow of the graph
    g.set_entry_point("guard_in")
    g.add_edge("guard_in", "classify")

    # Route based on classifier results
    g.add_conditional_edges(
        "classify",
        lambda s: {
            Task.CONVERSATION.value: "conversation",
            Task.MEDICAL_QA.value: "medical_qa",
            Task.SCHEDULING.value: "scheduler"
        }.get(s.get("intent")),
        {"conversation": "conversation", "medical_qa": "medical_qa", "scheduler": "scheduler"}
    )

    # Add output guardrails after any task
    g.add_node("apply_out", apply_output_guardrails)
    g.add_edge("conversation", "apply_out")
    g.add_edge("medical_qa", "apply_out")
    g.add_edge("scheduler", "apply_out")
    g.add_edge("apply_out", END)

    return g
