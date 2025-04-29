from typing import Dict, Union, List, Optional, Set
import functools

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, BaseMessage

from app.config.agent import settings
from app.config.constants import AgentName
from .state import AgentState, init_agent_state
from .routing import analyze_input, route_to_agent, confidence_based_routing
from .nodes import (
    run_conversation_agent, run_rag_agent, run_web_search_processor_agent,
    perform_human_validation, apply_output_guardrails
)

# thread_config moved here from settings
THREAD_CONFIG = {"configurable": {"thread_id": "1"}}
memory = MemorySaver()

def create_agent_graph(prune_to: Optional[Set[str]] = None) -> StateGraph:
     """Stitch nodes + edges → compiled StateGraph."""
     workflow = StateGraph(AgentState)

     ALL_AGENTS = settings.ALL_AGENTS  # set of AgentName.value strings


     allowed = prune_to if prune_to is not None else ALL_AGENTS

     # dynamic mapping of agent names to runner functions
     AGENT_RUNNERS = {
         AgentName.CONVERSATION.value: run_conversation_agent,
         AgentName.RAG.value: run_rag_agent,
         AgentName.WEB_SEARCH.value: run_web_search_processor_agent,
     }

     # Add nodes
     workflow.add_node("analyze_input", analyze_input)
     workflow.add_node("route_to_agent", route_to_agent)
     for agent in ALL_AGENTS:
         if agent in allowed:
             runner = AGENT_RUNNERS[agent]
             workflow.add_node(agent, runner)
     workflow.add_node("check_validation", check_validation_wrapper)
     workflow.add_node("human_validation", perform_human_validation)
     workflow.add_node("apply_guardrails", apply_output_guardrails)

     # Entry and edges
     workflow.set_entry_point("analyze_input")
     workflow.add_conditional_edges(
         "analyze_input",
         lambda state: "apply_guardrails" if state.get("bypass_routing") else "route_to_agent",
         {"apply_guardrails": "apply_guardrails", "route_to_agent": "route_to_agent"}
     )
     # automatically map each allowed agent to itself; add needs_validation for RAG
     route_map = { a: a for a in allowed }
     if AgentName.RAG.value in allowed:
         route_map["needs_validation"] = AgentName.RAG.value
     workflow.add_conditional_edges(
         "route_to_agent",
         lambda x: x["next"],
         route_map
     )

     # Attach edges for each allowed agent
     if AgentName.CONVERSATION.value in allowed:
         workflow.add_edge(AgentName.CONVERSATION.value, "check_validation")
     if AgentName.WEB_SEARCH.value in allowed:
         workflow.add_edge(AgentName.WEB_SEARCH.value, "check_validation")
     if AgentName.RAG.value in allowed:
         workflow.add_conditional_edges(AgentName.RAG.value, confidence_based_routing)

     # Common edges
     workflow.add_edge("human_validation", "apply_guardrails")
     workflow.add_edge("apply_guardrails", END)

     workflow.add_conditional_edges(
         "check_validation",
         lambda x: x.get("next", END),
         {"human_validation": "human_validation", END: "apply_guardrails"}
     )

     return workflow.compile(checkpointer=memory)


@functools.cache
def get_graph_for_role(role: str) -> StateGraph:
   # default to all agents if role not found
   allowed = settings.ROLE_TO_ALLOWED.get(role.lower(), settings.ALL_AGENTS)
   return create_agent_graph(prune_to=allowed)


def process_query(query: Union[str, Dict], role, conversation_history: List[BaseMessage] = None) -> StateGraph:
     """Public façade used by FastAPI route."""
     graph = get_graph_for_role(role)
     state = init_agent_state()
     state["current_input"] = query
     if conversation_history:
      state["messages"] = conversation_history + [HumanMessage(content=query)]
     else:
      state["messages"] = [HumanMessage(content=query)]
     result = graph.invoke(state, THREAD_CONFIG)
     # Trim history
     max_hist = settings.max_conversation_history
     if len(result.get("messages", [])) > max_hist:
         result["messages"] = result["messages"][-max_hist:]
     return result


 # add a lightweight validation check wrapper
def check_validation_wrapper(state: AgentState) -> Dict:
     """Decide whether human validation is needed."""
     if state.get("needs_human_validation"):
         return {"agent_state": state, "next": "human_validation"}
     return {"agent_state": state, "next": END}
