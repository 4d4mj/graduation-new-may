from typing import TypedDict

class PatientAnalysisOutput(TypedDict):
    """output schema for the patient analysis LLM call"""

    response_text: str
    request_scheduling: bool

class AgentDecision(TypedDict):
    """output schema for the agent routing LLM call"""
    agent: str
    confidence: float | None = None
    reason: str | None = None
