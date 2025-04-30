# tests/test_orchestrator.py

###############
# 0) Early stubs
###############
import app.agents.guardrails.nodes as nodes
from app.config.constants import AgentName

# stub analyze_input to bypass guard-rails entirely
nodes.apply_input_guardrails = lambda state: state

# stub the decision step so it never hits LangChain
def fake_route_to_agent(state):
    # pretend the model always picks "rag"
    new_state = {**state, "agent_name": AgentName.RAG.value}
    return {"agent_state": new_state, "next": AgentName.RAG.value}

nodes.route_to_agent = fake_route_to_agent

# confidence-based-rerouting won't be used in this simple test
nodes.confidence_based_routing = lambda state: AgentName.RAG.value

###############
# 1) Stub out RAG/Web agents
###############
import app.agents.rag as rag_mod
import app.agents.web_search as web_mod
from tests._stubs import DummyRAG, DummyWebSearch

rag_mod.MedicalRAG = DummyRAG
web_mod.WebSearchProcessorAgent = DummyWebSearch

###############
# 2) Stub out BOTH guardrails
###############
import app.agents.guardrails.core as guard_mod
class DummyGuard:
    def __init__(self, *a, **kw): pass
    def check_input(self, t):   return (True, "")
    def check_output(self, o, i): return o
guard_mod.Guardrails = DummyGuard
# also stub the second guard inside nodes.py
import app.agents.runners as nodes_mod
nodes_mod._get_output_guard = lambda: DummyGuard()

###############
# 3) Now import core and test
###############
from app.agents.orchestrator import core

def test_only_rag_runs_when_confident():
    result = core.process_query(
        query="anything",
        role="patient",
        conversation_history=[],
    )
    assert result["output"].content == "(stub RAG answer)"
    assert result["agent_name"] == AgentName.RAG.value
