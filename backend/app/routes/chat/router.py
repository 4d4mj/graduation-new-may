# backend/app/routes/chat/router.py

import logging
from fastapi import APIRouter, HTTPException, Cookie, Request, Depends
from app.config.settings import env
from langchain_core.messages import HumanMessage, BaseMessage, ToolMessage
from typing import List
from app.core.middleware import get_current_user
from app.schemas.chat import ChatRequest, ChatResponse
from langgraph.errors import GraphInterrupt
from langgraph.types import Command
import json
import random  # Keep for /test endpoint
from app.schemas.chat import ChatMessage

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)  # Corrected logger name

secure_cookie = env == "production"

# 1️⃣ Add a helper function to find the most recent tool message
def _find_last_tool_or_ai_message(messages):
    """
    Find the most recent ToolMessage or AIMessage in the current turn.
    ToolMessage takes precedence, but we never look beyond the most recent HumanMessage.

    Args:
        messages: List of conversation messages

    Returns:
        The content of the most relevant message for display
    """
    for m in reversed(messages):
        if isinstance(m, ToolMessage):
            return m.content  # ToolMessage (structured JSON) wins
        if isinstance(m, HumanMessage):
            break  # Stop at previous user turn - don't look across turns

    # If we get here, there was no ToolMessage in the current turn
    return messages[-1].content if messages else None  # Fallback to last message

@router.post("/", response_model=ChatResponse, status_code=200)
async def chat(
    payload: ChatRequest,
    request: Request,
    current_user: dict = Depends(get_current_user),
    session: str | None = Cookie(default=None, alias="session")
):
    # Use the user data from middleware instead of decoding the token again
    user_id = current_user["user_id"]
    role = current_user["role"]

    # 2. Graph Retrieval
    if not hasattr(request.app.state, "graphs") or not request.app.state.graphs:
        logger.critical(
            "Chat request: app.state.graphs not found or is empty! Graphs not initialized properly in lifespan."
        )
        raise HTTPException(500, "Chat service not properly initialized.")

    graph = request.app.state.graphs.get(role)  # Get graph based on role
    if not graph:
        # Log available graphs if lookup fails
        available_graphs = (
            list(request.app.state.graphs.keys()) if request.app.state.graphs else []
        )
        logger.error(
            f"Chat request: Graph for role '{role}' not found in app.state.graphs! Available graphs: {available_graphs}"
        )
        raise HTTPException(500, f"Invalid role '{role}' or graph not initialized.")
    else:
        logger.info(
            f"Chat request: Successfully retrieved graph for role '{role}'."
        )  # Log successful retrieval

    # 3. Graph Invocation
    config = {"configurable": {"thread_id": session}}
    final_state = None
    try:
        # NOTE: Interrupt handling is kept, assuming it might be needed for complex flows later,
        # but the doctor agent currently isn't expected to trigger it.
        if payload.interrupt_id:
            logger.info(
                f"Resuming from interrupt: {payload.interrupt_id} for role {role}"
            )
            cmd = Command(resume=payload.resume_value)
            final_state = await graph.ainvoke(cmd, config=config)
        else:
            # Normal invocation
            graph_input = {
                "messages": [HumanMessage(content=payload.message)],
                "current_input": payload.message,
                "final_output": None,
                "agent_name": None,
                "user_id": user_id,
                "role": role,
                "user_tz": payload.user_tz,
            }
            logger.debug(
                f"Invoking graph for role '{role}' with input keys: {list(graph_input.keys())} and config: {config}"
            )
            final_state = await graph.ainvoke(graph_input, config=config)

    except GraphInterrupt as gi:
        # Handle interrupt - might be relevant for patient flow later
        logger.warning(f"Graph interrupted unexpectedly for role '{role}': {gi.value}")
        # Decide how to respond - maybe just return the interrupt value as string?
        # Build history for context even on interrupt
        interrupted_messages_state = gi.args[0] if gi.args else {}
        interrupted_history = interrupted_messages_state.get("messages", [])
        response_messages_interrupt = []
        for msg in interrupted_history:
            msg_role_i = "user" if isinstance(msg, HumanMessage) else "assistant"
            content_i = str(getattr(msg, "content", ""))
            if content_i:
                response_messages_interrupt.append(
                    ChatMessage(role=msg_role_i, content=content_i)
                )

        return ChatResponse(
            reply=f"Action required: {str(gi.value)}",  # Provide context
            agent="System",  # Indicate it's an interrupt state
            interrupt_id=gi.ns[0],
            messages=response_messages_interrupt,
        )
    except Exception as e:
        logger.exception(f"Error running graph for role '{role}'")
        raise HTTPException(500, f"Processing error: {e}")

    # --- Process the final state ---
    if final_state is None:
        logger.error(
            f"Graph execution finished for role '{role}' but final_state is None."
        )
        raise HTTPException(500, "Internal processing error: No final state.")

    all_messages: List[BaseMessage] = final_state.get("messages", [])

    # 2️⃣ Use the helper function instead of just looking at the last message
    if all_messages:
        reply = _find_last_tool_or_ai_message(all_messages)
        logger.info(f"Selected message as reply using improved logic")
    else:
        reply = None

    # 3️⃣ Prioritize guardrail output (final_output) if it exists
    if final_state.get("final_output") is not None:
        reply = final_state.get("final_output")
        logger.info("Using final_output from guardrails as reply")

    # Fallback Reply
    if reply is None:
        reply = "I apologize, but I encountered an issue processing your request. Please try again."
        logger.error(
            f"Graph finished for role '{role}' but no reply content found in final state."
        )

    # Ensure reply is always a string
    if not isinstance(reply, str):
        logger.warning(
            f"Final reply content was not a string ({type(reply)}), coercing."
        )
        reply = str(reply)

    # Get agent name
    agent_name = final_state.get("agent_name")
    if agent_name is None:
        # Use a role-specific default
        agent_name = "Doctor Assistant" if role == "doctor" else "Patient Assistant"
        logger.warning(
            f"No agent_name set in final state for role '{role}', using default: '{agent_name}'"
        )

    # Build response history
    response_messages = []
    for msg in all_messages:
        msg_role = "user" if isinstance(msg, HumanMessage) else "assistant"
        # Ensure content is string, skip if empty
        content = str(getattr(msg, "content", ""))
        if content:
            response_messages.append(ChatMessage(role=msg_role, content=content))

    logger.info(f"Final reply string for role '{role}': '{reply[:100]}...'")

    # --- Return Response (matching ChatResponse schema) ---
    return ChatResponse(
        reply=reply,
        agent=agent_name,
        messages=response_messages,
        session=session,
        session_id=session,                    # Include history
        interrupt_id=None # Explicitly null unless set by GraphInterrupt block
    )


# Keep the /test endpoint for UI testing if needed
@router.post("/test", response_model=ChatResponse, status_code=200)
async def testChat(
    current_user: dict = Depends(get_current_user),
    session: str | None = Cookie(default=None, alias="session")
):
    """
    Simulates a more realistic chat interaction for UI testing.
    Includes mock history and varied, longer responses.
    """
    # We're now using the middleware for authentication, no need to decode the token again

    # --- Simulate Realistic Interaction ---
    # Select a random realistic reply
    reply = random.choice([
        "That's an interesting point. Could you elaborate a bit more on that?",
        "I understand. Based on what you've said, perhaps we could explore options like [Option A] or [Option B]. What are your thoughts?",
        "Thank you for sharing that. Let me process this information. One moment please...",
        "Okay, I've noted that down. Is there anything else you'd like to add or discuss regarding this topic?",
        "Processing your request... This might take a few moments. In the meantime, have you considered [Related Topic]?",
        "Acknowledged. I'm accessing the relevant information now. This is a complex area, so accuracy is key.",
        "Let's break that down. The first aspect to consider is [...], followed by [...]. Does that make sense so far?",
    ])

    reply = '{"type": "slots", "doctor": "Chen", "date": "May 6, 2025", "options": ["09:00", "09:30", "10:00", "10:30", "11:00", "11:30", "12:00", "12:30", "13:00", "13:30", "14:30", "15:00", "15:30", "16:00", "16:30"]}'

    agent_name = "Medical Assistant"

    # --- Return Response ---
    return ChatResponse(
        reply=reply,
        agent=agent_name,
        messages=[ChatMessage(role="assistant", content=reply)],
    )
