import uuid
import logging
from fastapi import APIRouter, HTTPException, Cookie, Request
from langchain_core.messages import HumanMessage, AIMessage

from app.config.settings import env
from app.agents.states import init_state_for_role
from app.core.auth import decode_access_token
from app.schemas.chat import ChatRequest, ChatResponse, ChatMessage
from app.graphs.patient import create_patient_graph
# from app.graphs.doctor import create_doctor_graph
from langgraph.checkpoint.memory import MemorySaver

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)

# Increase logging level for conversation memory diagnosis
MEMORY_DEBUG = True

secure_cookie = env == "production"

# Initialize the graphs at module level, to be used in the FastAPI app startup event
role_graphs = {
    "patient": None,
    "doctor": None
}

def init_graphs():
    """Initialize the graphs for each role with checkpointers."""
    global role_graphs
    role_graphs = {
        "patient": create_patient_graph().compile(checkpointer=MemorySaver()),
        # "doctor": create_doctor_graph().compile(checkpointer=MemorySaver()),
    }
    return role_graphs

@router.post("/", response_model=ChatResponse, status_code=200)
async def chat(
    payload: ChatRequest,
    request: Request,
    session: str | None = Cookie(default=None, alias="session")
):
    # MEMORY DIAGNOSTIC: Log the thread ID and payload
    thread_id = session or str(uuid.uuid4())
    if MEMORY_DEBUG:
        logger.info(f"[MEMORY] SESSION/THREAD ID: {thread_id}")
        logger.info(f"[MEMORY] INCOMING MESSAGE: {payload.message}")
        logger.info(f"[MEMORY] HISTORY LENGTH: {len(payload.history or [])}")

    # 1) AUTH
    if session is None:
        raise HTTPException(401, "Not authenticated")
    try:
        token = decode_access_token(session)
        role = token["role"]
    except Exception:
        raise HTTPException(401, "Invalid session token")

    # 2) INITIALIZATION CHECK
    if not role_graphs or not role_graphs.get(role):
        # Initialize graphs if they don't exist
        if not role_graphs.get(role):
            init_graphs()
            logger.info(f"Initialized graphs for roles: {list(role_graphs.keys())}")
        # If still not available, raise an error
        if not role_graphs.get(role):
            raise HTTPException(400, f"Unknown role {role}")

    graph = role_graphs.get(role)

    # 3) PREPARE INPUT STATE
    # Create a fresh input state with just what we need for this turn
    input_state = init_state_for_role(role)

    # Set the current input from the user
    input_state["current_input"] = payload.message

    # Convert current message from payload into LangChain message format
    current_message = HumanMessage(content=payload.message)

    # We only need to set minimal context for LangGraph - it will
    # restore the rest from its checkpointer
    input_state["messages"] = [current_message]
    input_state["user_role"] = role

    # MEMORY DIAGNOSTIC: Check if we can retrieve previous state for this thread
    if MEMORY_DEBUG:
        try:
            checkpointer = graph.checkpointer
            if hasattr(checkpointer, 'get'):
                saved = await checkpointer.get(thread_id)
                if saved:
                    logger.info(f"[MEMORY] PREVIOUS STATE EXISTS: {thread_id}")
                    # Log messages from previous state
                    prev_messages = saved.get("messages", [])
                    logger.info(f"[MEMORY] PREVIOUS MESSAGES COUNT: {len(prev_messages)}")
                    for i, msg in enumerate(prev_messages[:3]):  # Log first 3 messages
                        content = msg.content if hasattr(msg, 'content') else str(msg)
                        logger.info(f"[MEMORY] PREV MSG {i}: {content[:100]}...")
                else:
                    logger.info(f"[MEMORY] NO PREVIOUS STATE FOR: {thread_id}")
        except Exception as e:
            logger.warning(f"[MEMORY] Error checking previous state: {e}")

    # 4) INVOKE GRAPH/PROCESSOR WITH THREAD ID FOR STATE MANAGEMENT
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # MEMORY DIAGNOSTIC: Log the moment of graph invocation
        if MEMORY_DEBUG:
            logger.info(f"[MEMORY] INVOKING GRAPH with thread_id={thread_id}")

        # Let LangGraph handle loading state from the thread_id
        final_state = await graph.ainvoke(input_state, config=config)

        logger.info("Final state keys: %s", list(final_state.keys()))

        # MEMORY DIAGNOSTIC: Log messages after invocation
        if MEMORY_DEBUG:
            messages_after = final_state.get("messages", [])
            logger.info(f"[MEMORY] MESSAGES AFTER INVOCATION: {len(messages_after)}")
            for i, msg in enumerate(messages_after[-3:]):  # Log last 3 messages
                content = msg.content if hasattr(msg, 'content') else str(msg)
                logger.info(f"[MEMORY] AFTER MSG {i}: {content[:100]}...")

        # Log actual values for debugging
        logger.info("final_output: %r", final_state.get("final_output"))
        logger.info("patient_response_text: %r", final_state.get("patient_response_text"))
        logger.info("output: %r", final_state.get("output"))
    except Exception as e:
        logger.exception("Error running graph")
        raise HTTPException(500, f"Processing error: {e}")

    # 5) EXTRACT REPLY WITH ROBUST FALLBACKS
    reply = None

    # Try final_output first
    if final_state.get("final_output") is not None:
        final_output = final_state.get("final_output")
        # Handle AIMessage objects or objects with content attribute
        if hasattr(final_output, 'content'):
            reply = final_output.content
        else:
            # Extract text content from string representations of AIMessage
            text_repr = str(final_output)
            if "content='" in text_repr and "'" in text_repr.split("content='", 1)[1]:
                # Extract content from string representation of AIMessage
                content_part = text_repr.split("content='", 1)[1]
                reply = content_part.split("'", 1)[0]
            else:
                reply = text_repr

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

    # Get messages from final state for history (should include the new ones)
    messages = final_state.get("messages", [])

    # Build history for response (keeping client-side synchronized)
    history = payload.history or []
    history.append(ChatMessage(role="user", content=payload.message))
    history.append(ChatMessage(role="assistant", content=reply))

    # Ensure agent is a string, use a default if it's None
    agent_name = final_state.get("agent_name")
    if agent_name is None:
        # If we don't have an agent_name but have intent, use that
        if final_state.get("intent") is not None:
            agent_name = final_state.get("intent")
        # Otherwise use a generic fallback
        else:
            agent_name = "conversation"

    # MEMORY DIAGNOSTIC: Verify state was saved properly
    if MEMORY_DEBUG:
        try:
            checkpointer = graph.checkpointer
            if hasattr(checkpointer, 'get'):
                saved = await checkpointer.get(thread_id)
                if saved:
                    logger.info(f"[MEMORY] STATE SAVED SUCCESSFULLY: {thread_id}")
                    saved_messages = saved.get("messages", [])
                    logger.info(f"[MEMORY] SAVED MESSAGES COUNT: {len(saved_messages)}")
                else:
                    logger.warning(f"[MEMORY] STATE NOT SAVED AFTER INVOCATION: {thread_id}")
        except Exception as e:
            logger.warning(f"[MEMORY] Error checking saved state: {e}")

    return ChatResponse(
        reply=reply,
        agent=agent_name,
        messages=history,
        session_id=thread_id,
    )
