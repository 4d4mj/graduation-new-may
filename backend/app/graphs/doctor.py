from langgraph.graph import StateGraph, END
from app.graphs.sub.intents import Task, classify_doctor_intent
from app.agents.guardrails.nodes import apply_input_guardrails, apply_output_guardrails
from app.agents.states import DoctorState
from app.graphs.sub.wrappers import (
    conversation_graph,
    medical_qa_graph,
    scheduler_graph,
)


def build_summary_graph():
    """Placeholder for the doctor-only summary graph function."""
    # This is a placeholder that will be implemented later
    g = StateGraph(DoctorState)
    g.add_node("summary", lambda s: {**s, "final_output": "This is a placeholder for patient summary functionality."})
    g.set_entry_point("summary")
    g.add_edge("summary", END)
    return g


def build_db_query_graph():
    """Placeholder for the doctor-only database query graph function."""
    # This is a placeholder that will be implemented later
    g = StateGraph(DoctorState)
    g.add_node("db_query", lambda s: {**s, "final_output": "This is a placeholder for database query functionality."})
    g.set_entry_point("db_query")
    g.add_edge("db_query", END)
    return g


def build_image_analysis_graph():
    """Placeholder for the doctor-only image analysis graph function."""
    # This is a placeholder that will be implemented later
    g = StateGraph(DoctorState)
    g.add_node("image_analysis", lambda s: {**s, "final_output": "This is a placeholder for image analysis functionality."})
    g.set_entry_point("image_analysis")
    g.add_edge("image_analysis", END)
    return g


# Pre-compile doctor-only sub-graphs at module load time for better performance
_summary_graph = build_summary_graph().compile(checkpointer=None)
_db_query_graph = build_db_query_graph().compile(checkpointer=None)
_image_analysis_graph = build_image_analysis_graph().compile(checkpointer=None)


def create_doctor_graph() -> StateGraph:
    """
    Create a doctor orchestrator graph that routes to different task functions based on intent.
    Includes both shared task nodes and doctor-only task nodes.

    Returns:
        A compiled doctor StateGraph
    """
    g = StateGraph(DoctorState)

    # Add nodes for guardrails and intent classification
    g.add_node("guard_in", apply_input_guardrails)
    g.add_node("classify", classify_doctor_intent)

    # Add task nodes directly instead of using subgraphs
    g.add_node("conversation", run_conversation_subgraph)
    g.add_node("medical_qa", run_medical_qa_subgraph)
    g.add_node("scheduler", run_scheduler_subgraph)
    g.add_node("summary", run_summary_subgraph)
    g.add_node("db_query", run_db_query_subgraph)
    g.add_node("image_analysis", run_image_analysis_subgraph)

    # Set up the flow of the graph
    g.set_entry_point("guard_in")
    g.add_edge("guard_in", "classify")

    # Route based on classifier results
    g.add_conditional_edges(
        "classify",
        lambda s: s.get("intent"),
        {
            Task.CONVERSATION.value: "conversation",
            Task.MEDICAL_QA.value: "medical_qa",
            Task.SCHEDULING.value: "scheduler",
            Task.SUMMARY.value: "summary",
            Task.DB_QUERY.value: "db_query",
            Task.IMAGE_ANALYSIS.value: "image_analysis"
        }
    )

    # Add output guardrails after any task node
    g.add_node("apply_out", apply_output_guardrails)
    for node in ["conversation", "medical_qa", "scheduler", "summary", "db_query", "image_analysis"]:
        g.add_edge(node, "apply_out")
    g.add_edge("apply_out", END)

    return g


# Wrapper functions to run the pre-compiled subgraphs
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


async def run_summary_subgraph(state, config):
    """Run the summary subgraph on the given state."""
    # Pass the config to ensure thread_id is maintained
    return await _summary_graph.ainvoke(state, config=config)


async def run_db_query_subgraph(state, config):
    """Run the database query subgraph on the given state."""
    # Pass the config to ensure thread_id is maintained
    return await _db_query_graph.ainvoke(state, config=config)


async def run_image_analysis_subgraph(state, config):
    """Run the image analysis subgraph on the given state."""
    # Pass the config to ensure thread_id is maintained
    return await _image_analysis_graph.ainvoke(state, config=config)
