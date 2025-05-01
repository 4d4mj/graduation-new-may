import logging
from langgraph.graph import StateGraph, END

from .nodes import analyze_patient_query, prepare_for_scheduling
from app.agents.states import PatientState
from app.config.constants import AgentName
from app.agents.runners import run_conversation_agent, run_rag_agent, run_scheduler_agent
from app.agents.guardrails.nodes import perform_human_validation, apply_output_guardrails
from app.agents.routing import route_to_agent

logger = logging.getLogger(__name__)

def create_patient_graph() -> StateGraph:
    """
    Build (but do not compile) the patient-care StateGraph.
    The `prune_to` parameter isn't needed here - patients always get the same three nodes.
    """
    g = StateGraph(PatientState)

    g.add_node("analyze_patient_query", analyze_patient_query)
    g.add_node("prepare_for_scheduling", prepare_for_scheduling)
    g.add_node("route_to_agent", route_to_agent)

    # atomic agents
    g.add_node(AgentName.CONVERSATION, run_conversation_agent)
    g.add_node(AgentName.RAG, run_rag_agent)
    g.add_node(AgentName.SCHEDULER, run_scheduler_agent)

    # guardrails & validation
    g.add_node("perform_human_validation", perform_human_validation)
    g.add_node("apply_output_guardrails", apply_output_guardrails)

    g.set_entry_point("analyze_patient_query")

    # Modified edge: If the analysis already produced a patient-centric response
    # with a scheduling flag, go directly to scheduling. If not scheduling and the
    # response is satisfactory, go directly to END instead of additional processing
    g.add_conditional_edges(
        "analyze_patient_query",
        lambda s: (
            "prepare_for_scheduling" if s.get("request_scheduling", False)
            else (END if s.get("patient_response_text") else "route_to_agent")
        ),
        {
            "prepare_for_scheduling": "prepare_for_scheduling",
            "route_to_agent": "route_to_agent",
            END: END
        }
    )

    # once prepped, hand off to scheduler
    g.add_edge("prepare_for_scheduling", AgentName.SCHEDULER)

    # generic routing for non-scheduling
    g.add_conditional_edges(
        "route_to_agent",
        lambda s: s.get("next_agent"),
        {
            AgentName.CONVERSATION: AgentName.CONVERSATION,
            AgentName.RAG: AgentName.RAG,
            AgentName.SCHEDULER: AgentName.SCHEDULER
        }
    )

    # common post-agent path
    for node in [AgentName.CONVERSATION, AgentName.RAG, AgentName.SCHEDULER]:
        g.add_edge(node, "perform_human_validation")
    g.add_edge("perform_human_validation", "apply_output_guardrails")
    g.add_edge("apply_output_guardrails", END)

    logger.info("Patient graph built (uncompiled)")
    return g
