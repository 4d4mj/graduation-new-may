import logging
from fastapi import APIRouter, HTTPException, Cookie, Request
from app.config.settings import env
from langchain_core.messages import HumanMessage, BaseMessage
from typing import List
from app.core.auth import decode_access_token
from app.schemas.chat import ChatRequest, ChatResponse, ChatMessage

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

    # --- Prepare input for graph.ainvoke with checkpointer ---
    graph_input = {
        "messages": [HumanMessage(content=payload.message)],
        "current_input": payload.message,  # Still useful for guard_in
        "final_output": None,
        "agent_name": None,
        "user_id": user_id,
    }

    # Define config for the graph invocation
    config = {
        "configurable": {
            "thread_id": session
        }
    }

    try:
        final_state = await graph.ainvoke(graph_input, config=config)

    except Exception as e:
        logger.exception("Error running graph")
        raise HTTPException(500, f"Processing error: {e}")

    # --- Process the final state ---
    # Extract the response from the final state
    all_messages: List[BaseMessage] = final_state.get("messages", [])

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
        agent_name = "medical_assistant"  # Provide a default value to pass validation
        logger.warning("No agent_name was set in final state, using default: %s", agent_name)

    # --- Build response history ---
    # Use the history from the FINAL state managed by LangGraph/Checkpointer
    response_messages = []
    for msg in all_messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        content = str(getattr(msg, 'content', ''))
        # Basic filtering: Don't show empty messages
        if content:
            response_messages.append(ChatMessage(role=role, content=content))

    return ChatResponse(
        reply=reply,
        agent=agent_name,
        messages=response_messages,
        session_id=session,
    )
