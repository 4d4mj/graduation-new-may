import logging
from fastapi import APIRouter, HTTPException, Cookie, Request, Depends
from app.config.settings import env
from langchain_core.messages import HumanMessage, BaseMessage, ToolMessage
from typing import List
from app.core.middleware import get_current_user
from app.schemas.chat import ChatRequest, ChatResponse
from langgraph.errors import GraphInterrupt
from langgraph.types import Command

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)

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

    # Get the graph from app.state
    graph = request.app.state.graphs.get(role)
    if not graph:
        logger.error(f"Graph for role '{role}' not found!")
        raise HTTPException(500, f"Invalid role '{role}' or graph not initialized.")

    # Define config for the graph invocation (needed for both paths)
    config = {
        "configurable": {
            "thread_id": session
        }
    }

    # --- Handle resume with interrupt_id if present ---
    if payload.interrupt_id:
        logger.info(f"Resuming from interrupt: {payload.interrupt_id}")
        # Don't include interrupt_id in the Command - it will be matched internally
        cmd = Command(resume=payload.resume_value)

        try:
            final_state = await graph.ainvoke(cmd, config=config)
        except Exception as e:
            logger.exception(f"Error resuming graph execution: {e}")
            raise HTTPException(500, f"Error resuming: {e}")
    else:
        # --- Prepare input for normal graph.ainvoke with checkpointer ---
        graph_input = {
            "messages": [HumanMessage(content=payload.message)],
            "current_input": payload.message,  # Still useful for guard_in
            "final_output": None,
            "agent_name": None,
            "user_id": user_id,
            "user_tz": payload.user_tz,
        }

        try:
            final_state = await graph.ainvoke(graph_input, config=config)
        except GraphInterrupt as gi:
            # Surface the interrupt payload to the UI - this contains raw tool data
            logger.info(f"Graph interrupted: {gi.value}")
            return ChatResponse(
                reply=gi.value,
                agent="Scheduler",
                interrupt_id=gi.ns[0],  # Pass the interrupt ID to the client
                session=session
            )
        except Exception as e:
            logger.exception("Error running graph")
            raise HTTPException(500, f"Processing error: {e}")

    # --- Process the final state ---
    # Extract the response from the final state
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

    # Fallback reply
    if reply is None:  # Use 'is None' to handle potential empty string replies
        reply = "I apologize, but I couldn't process your request. Please try again later."
        logger.error("Graph finished but no reply content found in final state.")

    # Get agent name
    agent_name = final_state.get("agent_name")
    if agent_name is None:
        agent_name = "Medical Assistant"  # Provide a default value to pass validation
        logger.warning("No agent_name was set in final state, using default: %s", agent_name)

    return ChatResponse(
        reply=reply,
        agent=agent_name,
        session=session,
    )

import random
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
        session=session, # Return the session token received
    )
