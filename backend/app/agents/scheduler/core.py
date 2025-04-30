# tests/_stubs.py
from langchain_core.messages import AIMessage
from typing import Any

class DummyScheduler:
    def __init__(self, *a: Any, **kw: Any): ...
    def process_schedule(self, *, query, **__) -> AIMessage:
        return AIMessage(content="(stub SCHEDULER answer)")
