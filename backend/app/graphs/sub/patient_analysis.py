from langgraph.graph import StateGraph, END
from app.agents.states import PatientState
from app.agents.patient.nodes import analyze_patient_query, prepare_for_scheduling

def build_patient_analysis_graph() -> StateGraph:
    """
    Build a patient analysis graph that understands symptoms and makes scheduling decisions.

    Returns:
        A StateGraph for patient analysis
    """
    g = StateGraph(PatientState)

    # Add the analyze_patient_query node from the original graph
    g.add_node("analyze", analyze_patient_query)
    g.add_node("prepare_scheduling", prepare_for_scheduling)

    g.set_entry_point("analyze")

    # Conditionally route to scheduling preparation if needed
    g.add_conditional_edges(
        "analyze",
        lambda s: "prepare_scheduling" if s.get("request_scheduling", False) else END,
        {"prepare_scheduling": "prepare_scheduling", END: END}
    )

    g.add_edge("prepare_scheduling", END)
    return g

# Async wrapper for the patient analysis subgraph
async def async_patient_analysis_wrapper(state, config):
    """
    Async wrapper for the patient analysis subgraph.

    Args:
        state: The current state
        config: Configuration including thread_id

    Returns:
        Updated state with symptom understanding and scheduling decisions
    """
    # Pass the config to ensure thread_id is maintained
    return await patient_analysis_graph.ainvoke(state, config=config)
