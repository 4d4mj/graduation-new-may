import uuid
import logging
from fastapi import APIRouter, HTTPException, Cookie, Request
from langchain_core.messages import HumanMessage, AIMessage

from app.config.settings import env
from app.agents.states import init_state_for_role
from app.core.auth import decode_access_token
from app.schemas.chat import ChatRequest, ChatResponse, ChatMessage
from app.graphs.patient import create_patient_graph
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

    # Create memory saver for conversation persistence
    patient_checkpointer = MemorySaver()

    # Initialize the graph for patient role with the checkpointer
    patient_graph = create_patient_graph().compile(checkpointer=patient_checkpointer)

    role_graphs = {
        "patient": patient_graph,
    }

    logger.info("Initialized graph with memory persistence for roles: %s", list(role_graphs.keys()))
    return role_graphs

@router.post("/", response_model=ChatResponse, status_code=200)
async def chat(
    payload: ChatRequest,
    session: str | None = Cookie(default=None, alias="session")
):
    # Get thread ID from session token
    thread_id = session or str(uuid.uuid4())

    # Basic logging if needed
    if MEMORY_DEBUG:
        logger.info(f"[MEMORY] Thread ID: {thread_id}")
        logger.info(f"[MEMORY] User message: {payload.message}")

    # 1) AUTH
    if session is None:
        raise HTTPException(401, "Not authenticated")
    try:
        token = decode_access_token(session)
        role = token["role"]
    except Exception:
        raise HTTPException(401, "Invalid session token")

    # 2) INITIALIZATION CHECK - only if needed
    if not role_graphs or not role_graphs.get(role):
        init_graphs()
        if not role_graphs.get(role):
            raise HTTPException(400, f"Unknown role {role}")

    # 3) PREPARE MINIMAL INPUT
    input_state = init_state_for_role(role)
    input_state["current_input"] = payload.message
    input_state["messages"] = [HumanMessage(content=payload.message)]
    input_state["user_role"] = role

    # 4) RUN GRAPH - LangGraph handles state restoration and persistence
    try:
        final_state = await role_graphs[role].ainvoke(
            input_state,
            config={"configurable": {"thread_id": thread_id}}
        )
    except Exception as e:
        logger.exception("Error running graph")
        raise HTTPException(500, f"Processing error: {e}")

    # 5) EXTRACT REPLY - simplified approach
    messages = final_state.get("messages", [])
    if messages and isinstance(messages[-1], AIMessage) and messages[-1].content.strip():
        reply = messages[-1].content
    else:
        reply = "I apologize, but I couldn't process your request properly. Please try again."

    # Determine which agent or tool was used (if needed)
    agent_name = "medical_assistant"
    tool_calls = final_state.get("tool_calls", [])
    if tool_calls and isinstance(tool_calls[-1], dict) and "name" in tool_calls[-1]:
        tool_to_agent = {
            "rag_query": "medical_knowledge",
            "web_search": "web_search",
            "schedule_appointment": "scheduler",
            "small_talk": "conversation"
        }
        agent_name = tool_to_agent.get(tool_calls[-1]["name"], agent_name)

    # Return the response with minimal history construction
    return ChatResponse(
        reply=reply,
        agent=agent_name,
        messages=[
            *(payload.history or []),
            ChatMessage(role="user", content=payload.message),
            ChatMessage(role="assistant", content=reply),
        ],
    )
