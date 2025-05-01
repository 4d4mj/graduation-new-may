import uuid
import logging
from fastapi import APIRouter, HTTPException, Cookie, Request
from langchain_core.messages import HumanMessage, AIMessage

from app.config.settings import env
from app.agents.states import init_state_for_role
from app.core.auth import decode_access_token
from app.schemas.chat import ChatRequest, ChatResponse, ChatMessage
from app.graphs.patient import create_patient_graph
from app.graphs.doctor import create_doctor_graph
from langgraph.checkpoint.memory import MemorySaver

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)

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
        "doctor": create_doctor_graph().compile(checkpointer=MemorySaver()),
    }
    return role_graphs

async def rehydrate_state(thread_id: str, graph, role: str):
    """
    Pull stored state from the checkpointer if it exists.

    Args:
        thread_id: The unique identifier for the conversation thread
        graph: The LangGraph instance to use for retrieving state
        role: The user role (for initializing a new state if needed)

    Returns:
        Rehydrated state or a fresh state if none exists
    """
    try:
        # Try to read previous state from the checkpointer
        # Different checkers have different APIs
        checkpointer = graph.checkpointer

        if hasattr(checkpointer, 'get'):  # StandardMemorySaver/DBSaver pattern
            saved = await checkpointer.get(thread_id)
        elif hasattr(checkpointer, 'load'):  # Some implementations use load
            saved = await checkpointer.load(thread_id)
        elif hasattr(checkpointer, 'read_config'):  # Others might use read_config
            saved = await checkpointer.read_config(thread_id)
        else:
            logger.warning(f"Unknown checkpointer type: {type(checkpointer)}")
            saved = None

        if saved:
            logger.info(f"Found existing state for thread_id: {thread_id}")
            return saved
    except Exception as e:
        logger.warning(f"Error reading state from checkpointer: {e}")

    # Fall back to a fresh state
    logger.info(f"Creating new state for thread_id: {thread_id}")
    return init_state_for_role(role)

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
    if not role_graphs or not role_graphs.get(role):
        # Initialize graphs if they don't exist
        if not role_graphs.get(role):
            init_graphs()
            logger.info(f"Initialized graphs for roles: {list(role_graphs.keys())}")
        # If still not available, raise an error
        if not role_graphs.get(role):
            raise HTTPException(400, f"Unknown role {role}")

    graph = role_graphs.get(role)

    # 3) PREPARE STATE WITH REHYDRATION
    # Use the same session cookie as thread_id, or fall back to a new one
    thread_id = session or str(uuid.uuid4())

    # Rehydrate state from previous conversation if available
    state = await rehydrate_state(thread_id, graph, role)
    logger.info("State rehydrated or created for thread_id: %s", thread_id)

    # Set the current input
    state["current_input"] = payload.message

    # Preserve existing messages and append the new one
    messages = state.get("messages", []) or []

    # Convert current message from payload into LangChain message format and append
    current_message = HumanMessage(content=payload.message)
    messages.append(current_message)

    # Set the updated messages in state
    state["messages"] = messages
    state["user_role"] = role

    # 4) INVOKE YOUR GRAPH/PROCESSOR
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

    return ChatResponse(
        reply=reply,
        agent=agent_name,
        messages=history,
        session_id=thread_id,
    )
