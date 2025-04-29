# tests/_stubs.py
from langchain_core.messages import AIMessage
from typing import Any

class DummyWebSearch:
    def __init__(self, *a: Any, **kw: Any): ...
    def process_web_search_results(self, *, query, **__) -> AIMessage:
        return AIMessage(content="(stub WEB answer)")
