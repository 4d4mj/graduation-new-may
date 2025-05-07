from langgraph.graph import StateGraph, END
from app.tools.guardrails import guard_in, guard_out
from app.graphs.states import PatientState
from app.graphs.agents import patient_agent
from app.tools.scheduler.interrupt import confirm_booking
import logging
from typing import Literal, Dict, Any, List
from langchain_core.messages import ToolMessage

# Set up logging
logger = logging.getLogger(__name__)

# Define direct response tools that should bypass agent reformulation
DIRECT_TO_UI = {"list_free_slots", "list_doctors"}

def route_after_guard_in(state: dict) -> Literal["agent", "__end__"]:
    """Routes to agent if input is safe, otherwise ends the graph."""
    if state.get("final_output"):
        # final_output was set by guard_in, meaning input was blocked
        logger.warning("Input guardrail triggered routing to END.")
        return "__end__"  # Special node name for LangGraph's end
    else:
        # Input is safe, proceed to the agent
        logger.info("Input guardrail passed, routing to agent.")
        return "agent"

def route_after_agent(state: dict) -> str:
    """Routes based on the agent's output."""
    last = state["messages"][-1]
    if isinstance(last, ToolMessage) and last.name == "propose_booking":
        # Don't touch the state here – just pick the edge
        return "confirm"
    if getattr(last, "tool_calls", None):
        return "tools"
    return "guard_out"

def route_after_tools(state: dict) -> str:
    """Routes based on the tool output after tool execution.

    If the last message is a direct response tool, route to structured_output.
    Otherwise, return to the agent for further processing.
    """
    last = state["messages"][-1]  # after ToolNode runs this is a ToolMessage
    if isinstance(last, ToolMessage) and last.name in DIRECT_TO_UI:
        # Do NOT mutate the state here – just choose the edge
        logger.info(f"Routing direct response tool {last.name} to structured_output")
        return "structured_output"
    return "agent"

def structured_output(state: dict) -> dict:
    """
    Direct passthrough of structured data from tool response to final output.

    This ensures that direct response tools bypass the agent and go straight to the frontend.
    """
    tool_msg = state["messages"][-1]
    tool_name = tool_msg.name
    tool_content = tool_msg.content

    # Parse the tool content (most tools already return JSON)
    import json
    if isinstance(tool_content, str):
        try:
            structured_output = json.loads(tool_content)
        except json.JSONDecodeError:
            # Not valid JSON, use as is
            structured_output = {"type": "error", "message": "Error processing tool response"}
            logger.error(f"Tool {tool_name} returned non-JSON content: {tool_content}")
    else:
        # Already a dict/object
        structured_output = tool_content

    # Set the structured output directly as the final response
    state["final_output"] = structured_output
    state["agent_name"] = structured_output.get("agent", "Scheduler")

    logger.info(f"Bypassing agent reformulation for {tool_name} with structured output type: {structured_output.get('type', 'unknown')}")

    return state

def create_patient_graph() -> StateGraph:
    """
    Create a streamlined patient orchestrator graph.

    Flow:
    1. Apply input guardrails
    2. If safe, format message history & run medical agent
    3. Route after agent to either confirm, tools, or guard_out
    4. Route after tools to either structured_output for UI components or agent for reformulation
    5. If booking proposal detected, confirm with user
    6. Apply output guardrails

    Returns:
        A compiled patient StateGraph
    """
    g = StateGraph(PatientState)

    # Add nodes
    g.add_node("guard_in", guard_in)
    g.add_node("agent", patient_agent.medical_agent.ainvoke)
    g.add_node("tools", lambda state: state)  # LangGraph will fill this with tool execution
    g.add_node("confirm", confirm_booking)
    g.add_node("guard_out", guard_out)
    g.add_node("structured_output", structured_output)  # Direct structured output node

    # Set entry point
    g.set_entry_point("guard_in")

    # Conditional edge from input guardrail
    g.add_conditional_edges(
        "guard_in",
        route_after_guard_in,
        {
            "agent": "agent",
            "__end__": END,
        }
    )

    # Conditional edge from agent
    g.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tools": "tools",
            "confirm": "confirm",
            "guard_out": "guard_out",
        },
    )

    # Route after tools based on message type
    g.add_conditional_edges(
        "tools",
        route_after_tools,
        {
            "structured_output": "structured_output",  # Direct structured output to frontend
            "agent": "agent"                          # Continue processing with agent
        }
    )

    # Add edge from structured_output directly to END (bypassing guard_out)
    g.add_edge("structured_output", END)

    # Add edge from confirm to guard_out
    g.add_edge("confirm", "guard_out")
    g.add_edge("guard_out", END)

    logger.info("Patient graph created with direct structured output for UI components.")
    return g
