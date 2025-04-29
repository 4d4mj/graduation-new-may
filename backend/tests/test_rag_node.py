# tests/test_rag_node.py

import pytest
from app.agents.orchestrator.nodes import run_rag_agent
from app.agents.orchestrator.state import init_agent_state

# monkey-patch the real MedicalRAG to your DummyRAG
import app.agents.rag as rag_mod
from tests._stubs import DummyRAG
rag_mod.MedicalRAG = DummyRAG

def test_run_rag_agent_returns_stub_answer():
    # build a minimal state
    state = init_agent_state()
    state["current_input"] = "anything"
    state["messages"] = []
    out = run_rag_agent(state)
    assert out["output"].content == "(stub RAG answer)"
    assert out["agent_name"] == "rag"
    assert out["retrieval_confidence"] == 0.99
