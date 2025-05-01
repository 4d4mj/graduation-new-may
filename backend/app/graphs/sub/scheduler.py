from langgraph.graph import StateGraph, END
from app.agents.states import BaseAgentState
from app.agents.runners import run_scheduler_agent

def build_scheduler_graph() -> StateGraph:
    """
    Build a simple scheduler graph that uses the scheduler agent.

    Returns:
        A StateGraph with a single scheduler node
    """
    g = StateGraph(BaseAgentState)
    g.add_node("schedule", async_scheduler_wrapper)
    g.set_entry_point("schedule")
    g.add_edge("schedule", END)
    return g

# Async wrapper for the scheduler agent
async def async_scheduler_wrapper(state, config):
    """
    Async wrapper for the synchronous scheduler agent.

    Args:
        state: The current state
        config: Configuration including thread_id

    Returns:
        Updated state with scheduler response
    """
    return run_scheduler_agent(state)
