# app/schemas/chat.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Literal, Optional, Any

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
    * `user_tz`      - the user's timezone (IANA format, e.g. 'Asia/Beirut').
                       This is used to parse dates and times in the conversation.
                       If not provided, UTC is assumed.
    * `interrupt_id` - ID of the interrupt to resume, if applicable
    * `resume_value` - Value to resume with (e.g., "yes" or "no" for confirmations)
    """
    message: str = Field(..., min_length=1, description="Current user utterance")
    user_tz: Optional[str] = Field(None, description="User IANA timezone, e.g. 'Asia/Beirut'")
    interrupt_id: Optional[str] = Field(None, description="ID of the interrupt to resume")
    resume_value: Optional[Any] = Field(None, description="Value to resume with")


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
    * `session`      - the session ID (cookie) of the user who asked this.
    * `interrupt_id` - ID of the interrupt if this response is a pause for user confirmation
    * `session_id`   - Alias for session to maintain backward compatibility
    * `messages`     - List of messages in the conversation history (optional)
    """
    reply: Any  # Can be string or structured data for special bubbles
    agent: str
    session: str
    interrupt_id: Optional[str] = None
    session_id: Optional[str] = None
    messages: Optional[List[ChatMessage]] = None
