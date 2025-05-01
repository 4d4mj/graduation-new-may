from langgraph.graph import StateGraph, END
from app.agents.states import BaseAgentState
from app.agents.runners import run_conversation_agent

def build_conversation_graph() -> StateGraph:
    """
    Build a simple conversation graph that uses the conversation agent.

    Returns:
        A StateGraph with a single conversation node
    """
    g = StateGraph(BaseAgentState)
    # Wrap the synchronous runner in an async-compatible node
    g.add_node("run_conversation", async_conversation_wrapper)
    g.set_entry_point("run_conversation")
    g.add_edge("run_conversation", END)
    return g

# Add an async wrapper for the synchronous conversation agent
async def async_conversation_wrapper(state, config):
    """
    Async wrapper for the synchronous conversation agent.

    Args:
        state: The current state
        config: Configuration including thread_id

    Returns:
        Updated state with conversation response
    """
    # Call the synchronous conversation agent
    updated_state = run_conversation_agent(state)
    return updated_state
