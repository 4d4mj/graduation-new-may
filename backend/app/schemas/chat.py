# app/schemas/chat.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# 1.  A single turn in the conversation
# --------------------------------------------------------------------------
class ChatMessage(BaseModel):
    """
    A generic chat message exchanged between the user and the assistant.
    Extend this later with `image_url`, `attachments`, etc.
    """
    role: Literal["user", "assistant"]          # or "system" if you need it
    content: str
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# --------------------------------------------------------------------------
# 2.  Incoming payload  ➜  /chat  (POST)
# --------------------------------------------------------------------------
class ChatRequest(BaseModel):
    """
    What the *frontend* sends to /chat.

    * `message`      - the user's current turn
    * `history`      - the last N messages the browser still has. Optional,
                       because you may reconstruct it server-side instead.
    * `image_url`    - future-proof: if you later re-enable vision, you can
                       simply change `Optional[str]` into a richer object.
    """
    message: str = Field(..., min_length=1, description="Current user utterance")
    history: Optional[List[ChatMessage]] = None
    image_url: Optional[str] = None
    user_tz: Optional[str] = Field(None, description="User IANA timezone, e.g. 'Asia/Beirut'")


# --------------------------------------------------------------------------
# 3.  Outgoing payload  ⇦  /chat  (200 OK)
# --------------------------------------------------------------------------
class ChatResponse(BaseModel):
    """
    What your endpoint returns.

    * `reply`        - the assistant's answer (already sanitized /
                       guard-railed by LangGraph).
    * `agent`        - which specialised agent inside the orchestrator
                       produced that reply (conversation, RAG, web-search…).
    * `messages`     - the *new* canonical history,
                       i.e. `history + [user_msg] + [assistant_msg]`.
                       Returning it lets the browser keep perfect state even
                       if you later trim / summarise server-side.
    """
    reply: str
    agent: str
    messages: List[ChatMessage]
