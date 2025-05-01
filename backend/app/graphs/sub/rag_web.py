from langgraph.graph import StateGraph, END
from app.agents.states import BaseAgentState
from app.agents.runners import run_rag_agent, run_web_search_processor_agent
from app.agents.routing import confidence_based_routing

def build_medical_qa_graph() -> StateGraph:
    """
    Build a medical Q&A graph that uses RAG and web search.
    RAG is used first, and if confidence is low, web search is used.

    Returns:
        A StateGraph for medical Q&A
    """
    g = StateGraph(BaseAgentState)
    g.add_node("rag", async_rag_wrapper)
    g.add_node("web", async_web_search_wrapper)
    g.set_entry_point("rag")

    # If low confidence -> web search
    g.add_conditional_edges(
        "rag",
        lambda s: "web" if confidence_based_routing(s) == "WEB_SEARCH_AGENT" else END,
        {"web": "web", END: END}
    )
    g.add_edge("web", END)
    return g

# Async wrapper for the RAG agent
async def async_rag_wrapper(state, config):
    """
    Async wrapper for the synchronous RAG agent.

    Args:
        state: The current state
        config: Configuration including thread_id

    Returns:
        Updated state with RAG response
    """
    return run_rag_agent(state)

# Async wrapper for the web search agent
async def async_web_search_wrapper(state, config):
    """
    Async wrapper for the synchronous web search agent.

    Args:
        state: The current state
        config: Configuration including thread_id

    Returns:
        Updated state with web search response
    """
    return run_web_search_processor_agent(state)
