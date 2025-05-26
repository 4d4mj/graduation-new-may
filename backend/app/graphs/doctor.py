from langgraph.graph import StateGraph, END
from app.tools.guardrails import guard_in, guard_out
from app.graphs.states import DoctorState
from app.graphs.agents import doctor_agent
import logging
from typing import Literal  # <-- Import Literal
from langchain_core.messages import ToolMessage

# Set up logging
logger = logging.getLogger(__name__)


# --- ADD THIS ROUTING FUNCTION ---
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


# --- END ADDITION ---


def route_after_agent(state: dict) -> str:
    """Routes based on the agent's output."""
    last = state["messages"][-1]
    if isinstance(last, ToolMessage) and last.name == "propose_booking":
        return "confirm"
    if getattr(last, "tool_calls", None):
        return "tools"
    return "guard_out"


# Define direct response tools that should bypass agent reformulation
DIRECT_TO_UI = {"list_free_slots", "list_patients"}


# New node to process tool outputs and update the graph structure to handle direct tool responses, ensuring raw outputs are returned when necessary.
def process_tool_output_node(state: DoctorState):
    """
    Inspects the last message (ToolMessage). If from a DIRECT_OUTPUT_TOOL,
    prepares raw output. Otherwise, prepares for LLM to process.
    """
    last_message = state.messages[-1]
    state.is_direct_tool_response = False  # Default

    if isinstance(last_message, ToolMessage):
        if last_message.name in DIRECT_TO_UI:
            tool_content = last_message.content
            state.raw_tool_output = tool_content  # Store raw output
            state.is_direct_tool_response = True


# Routing function after processing tool output
def route_after_tool_processing(state: DoctorState) -> str:
    """Decides where to go after processing the tool's output."""
    if state.is_direct_tool_response:
        return "structured_output"  # End graph execution, raw tool output is ready
    return "agent"  # Continue processing with agent


# Add structured output node
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
            structured_output = {
                "type": "error",
                "message": "Error processing tool response",
            }
            logger.error(f"Tool {tool_name} returned non-JSON content: {tool_content}")
    else:
        # Already a dict/object
        structured_output = tool_content

    # Set the structured output directly as the final response
    state["final_output"] = structured_output
    state["agent_name"] = structured_output.get("agent", "Doctor")

    logger.info(
        f"Bypassing agent reformulation for {tool_name} with structured output type: {structured_output.get('type', 'unknown')}"
    )

    return state


def create_doctor_graph() -> StateGraph:
    """
    Create a streamlined doctor orchestrator graph with direct tool output handling.

    Flow:
    1. Apply input guardrails
    2. If safe, format message history & run medical agent
    3. Route after agent to either tools or guard_out
    4. Route after tools to either structured_output for UI components or agent for reformulation
    5. Apply output guardrails

    Returns:
        A compiled doctor StateGraph
    """
    g = StateGraph(DoctorState)

    # Add nodes
    g.add_node("guard_in", guard_in)
    g.add_node("agent", doctor_agent.medical_agent.ainvoke)
    g.add_node(
        "tools", lambda state: state
    )  # LangGraph will fill this with tool execution
    g.add_node("process_tool_output", process_tool_output_node)
    g.add_node("guard_out", guard_out)
    g.add_node("structured_output", structured_output)  # Direct structured output node

    # Set entry point
    g.set_entry_point("guard_in")

    # --- REPLACE SIMPLE EDGE WITH CONDITIONAL EDGE ---
    g.add_conditional_edges(
        "guard_in",
        route_after_guard_in,
        {
            "agent": "agent",  # If route_after_guard_in returns "agent"
            "__end__": END,  # If route_after_guard_in returns "__end__"
        },
    )
    # --- END REPLACEMENT ---

    # Conditional edge from agent
    g.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tools": "tools",
            "guard_out": "guard_out",
        },
    )

    # Add edge from tools to process_tool_output
    g.add_edge("tools", "process_tool_output")

    # Conditional edge from process_tool_output
    g.add_conditional_edges(
        "process_tool_output",
        route_after_tool_processing,
        {
            "structured_output": "structured_output",  # Direct structured output to frontend
            "agent": "agent",  # Continue processing with agent
        },
    )

    # Add edge from structured_output directly to END (bypassing guard_out)
    g.add_edge("structured_output", END)

    # Add edge from guard_out to END
    g.add_edge("guard_out", END)

    logger.info("Doctor graph created with direct structured output for UI components.")
    return g
