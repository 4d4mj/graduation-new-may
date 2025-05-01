from langgraph.graph import StateGraph, END
from app.agents.states import BaseAgentState
from app.agents.runners import run_scheduler_agent
from langchain_core.messages import AIMessage

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
    updated_state = run_scheduler_agent(state)

    # Extract content from AIMessage if needed
    if "output" in updated_state and isinstance(updated_state["output"], AIMessage):
        updated_state["final_output"] = updated_state["output"].content

    # Also handle if final_output is directly an AIMessage
    if "final_output" in updated_state and hasattr(updated_state["final_output"], "content"):
        updated_state["final_output"] = updated_state["final_output"].content

    return updated_state
