from typing import Dict, List, Optional, TypedDict, Union
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    messages: List[BaseMessage]
    agent_name: Optional[str]
    current_input: Optional[Union[str, Dict]]
    output: Optional[str]
    needs_human_validation: bool
    retrieval_confidence: float
    bypass_routing: bool
    insufficient_info: bool


class AgentDecision(TypedDict):
    """Output structure for the decision agent."""
    agent: str
    reasoning: str
    confidence: float


def init_agent_state() -> AgentState:
    """Allocate the vanilla AgentState dict."""
    return {
        "messages": [],
        "agent_name": None,
        "current_input": None,
        "output": None,
        "needs_human_validation": False,
        "retrieval_confidence": 0.0,
        "bypass_routing": False,
        "insufficient_info": False
    }
