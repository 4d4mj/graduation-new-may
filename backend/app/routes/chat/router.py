import logging
from fastapi import APIRouter, HTTPException, Cookie, Request
from app.config.settings import env
from langchain_core.messages import HumanMessage, BaseMessage, ToolMessage
from typing import List
from app.core.auth import decode_access_token
from app.schemas.chat import ChatRequest, ChatResponse
from langgraph.errors import GraphInterrupt
from langgraph.types import Command

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)

secure_cookie = env == "production"

@router.post("/", response_model=ChatResponse, status_code=200)
async def chat(
    payload: ChatRequest,
    request: Request,
    session: str | None = Cookie(default=None, alias="session")
):

    # authentication check
    if session is None:
        raise HTTPException(401, "Not authenticated")
    try:
        token = decode_access_token(session)
        user_id = token.get("sub")
        role = token["role"]
    except Exception:
        raise HTTPException(401, "Invalid session token")

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

    # Check for tool messages - these contain raw structured data
    for message in reversed(all_messages):
        if isinstance(message, ToolMessage):
            # If the last message is from a tool, use its raw content directly
            reply = message.content
            logger.info(f"Using raw tool output as reply: {reply}")
            break
    else:
        # Prioritize final_output if set (likely by guardrails)
        reply = final_state.get("final_output")

        # If guardrails didn't set final_output, get reply from the last message
        if reply is None and all_messages:
            last_message = all_messages[-1]
            # Check if it's an AIMessage, not a ToolMessage or HumanMessage
            if hasattr(last_message, 'content') and not isinstance(last_message, HumanMessage):
                reply = str(last_message.content)
            else:
                logger.warning("Last message was not AI content, state: %s", final_state)

    logger.info("Using reply: %s", reply)

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
    session: str | None = Cookie(default=None, alias="session")
):
    """
    Simulates a more realistic chat interaction for UI testing.
    Includes mock history and varied, longer responses.
    """
    # --- Authentication Check ---
    if session is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        # Replace with your actual token validation logic
        user_info = decode_access_token(session)
        # You might use user_info later if needed
    except Exception as e:
        # Log the error e for debugging if necessary
        raise HTTPException(status_code=401, detail="Invalid session token")

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
    # Add a bit more context if needed, e.g., mentioning the user's message
    # reply = f"Regarding your message about '{payload.message[:30]}...': {reply}" # Optional: Add context

    agent_name = "Medical Assistant" # Or make this dynamic if needed

    # --- Return Response ---
    return ChatResponse(
        reply=reply,
        agent=agent_name,
        session=session, # Return the session token received
    )
