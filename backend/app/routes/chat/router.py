import uuid
import logging
from fastapi import APIRouter, HTTPException, Cookie, Request
from langchain_core.messages import HumanMessage, AIMessage

from app.config.settings import env
from app.agents.states import init_state_for_role
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
    # 1) AUTH
    if session is None:
        raise HTTPException(401, "Not authenticated")
    try:
        token = decode_access_token(session)
        role = token["role"]
    except Exception:
        raise HTTPException(401, "Invalid session token")

    # 2) INITIALIZATION CHECK
    tm    = request.app.state.tool_manager
    graphs = request.app.state.graphs
    if tm is None or not graphs:
        raise HTTPException(500, "Not initialized")

    graph = graphs.get(role)
    if not graph:
        raise HTTPException(400, f"Unknown role {role}")

    # 3) PREPARE STATE
    state = init_state_for_role(role)
    logger.info("state is a %r", state)
    state["current_input"] = payload.message

    # Convert history from payload into LangChain message format
    messages = []
    if payload.history:
        for msg in payload.history:
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                messages.append(AIMessage(content=msg.content))

    # Add the current message
    messages.append(HumanMessage(content=payload.message))

    # Set the messages in state
    state["messages"] = messages
    state["user_role"] = role

    # 4) INVOKE YOUR GRAPH/PROCESSOR
    # Use the same session cookie as thread_id
    # use a cookie-based thread id, or fall back to a new one
    thread_id = session or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # <-- THIS is the key: await ainvoke, not a blocking call
        final_state = await graph.ainvoke(state, config=config)
        logger.info("Final state keys: %s", list(final_state.keys()))

        # Log actual values for debugging
        logger.info("final_output: %r", final_state.get("final_output"))
        logger.info("patient_response_text: %r", final_state.get("patient_response_text"))
        logger.info("output: %r", final_state.get("output"))
    except Exception as e:
        logger.exception("Error running graph")
        raise HTTPException(500, f"Processing error: {e}")

    # Enhanced reply extraction with more robust fallbacks
    reply = None

    # Try final_output first
    if final_state.get("final_output") is not None:
        reply = str(final_state.get("final_output"))
        logger.info("Using final_output for reply: %s", reply)

    # Try output if final_output is empty or None
    elif final_state.get("output") is not None:
        output = final_state.get("output")
        if isinstance(output, AIMessage):
            reply = output.content
        else:
            reply = str(output)
        logger.info("Using output for reply: %s", reply)

    # Try patient_response_text as fallback
    elif final_state.get("patient_response_text") is not None:
        reply = str(final_state.get("patient_response_text"))
        logger.info("Using patient_response_text for reply: %s", reply)

    # Final fallback
    if reply is None or reply == "None":
        logger.warning("Empty reply detected or value is 'None'. Available keys: %s", list(final_state.keys()))
        reply = "I apologize, but I couldn't process your request. Please try again."

    # build history (or persist in state if you want)
    history = payload.history or []
    history.append(ChatMessage(role="user", content=payload.message))
    history.append(ChatMessage(role="assistant", content=reply))

    return ChatResponse(
        reply=reply,
        agent=final_state.get("agent_name"),
        messages=history,
        session_id=thread_id,
    )
