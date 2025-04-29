# tests/_stubs.py
from langchain_core.messages import AIMessage
from typing import Any

class DummyRAG:
    def __init__(self, *a: Any, **kw: Any): ...
    def process_query(self, query, *_, **__) -> dict:
        return {
            "response": AIMessage(content="(stub RAG answer)"),
            "confidence": 0.99,  # high so orchestrator accepts it
        }
