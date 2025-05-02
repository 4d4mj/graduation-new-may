from langchain.callbacks import StdOutCallbackHandler
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
    input_state["messages"] = [HumanMessage(content=payload.message)]  # Only the new message
    input_state["current_input"] = payload.message  # Add back for guardrails to access original text
    input_state["user_role"] = role
    # Removed: input_state["callbacks"] = [StdOutCallbackHandler()]  # This causes serialization issues

    # 4) RUN GRAPH - LangGraph handles state restoration and persistence
    try:
        final_state = await role_graphs[role].ainvoke(
            input_state,
            config={
                # thread-local settings (persisted)
                "configurable": {"thread_id": thread_id},

                # run-only settings (NOT persisted)
                "callbacks": [StdOutCallbackHandler()],  # Add callbacks here, where they won't be serialized
                "recursion_limit": 15,          # safety-net
                "run_kwargs": {"stream_mode": "updates"}  # Enable streaming updates
            }
        )
    except Exception as e:
        logger.exception("Error running graph")
        raise HTTPException(500, f"Processing error: {e}")

    # Log final state keys for debugging
    if MEMORY_DEBUG:
        logger.debug(f"[MEMORY] final_state keys: {list(final_state.keys())}")

    # 5) EXTRACT REPLY - robust version
    reply: str | None = None

    # ① prefer the cleaned / canonical answer
    for key in ("final_output", "output"):
        if (txt := final_state.get(key)) and str(txt).strip():
            reply = txt.content if hasattr(txt, "content") else str(txt)
            break

    # ② otherwise fall back to the last assistant / tool message
    if reply is None:
        for msg in reversed(final_state.get("messages", [])):
            if hasattr(msg, "content") and str(msg.content).strip():
                reply = msg.content
                break

    # ③ last resort
    if reply is None:
        reply = (
            "I'm sorry, I couldn't process your request right now. "
            "Please try again in a moment."
        )

    # Determine which agent or tool was used (if needed)
    agent_name = "medical_assistant"
    tool_calls = final_state.get("tool_calls", [])
    if tool_calls and isinstance(tool_calls[-1], dict) and "name" in tool_calls[-1]:
        tool_name = tool_calls[-1]["name"]
        tool_to_agent = {
            "rag_query": "medical_knowledge",
            "web_search": "web_search",
            "schedule_appointment": "scheduler",
            "small_talk": "conversation",
            # Add mappings for the new scheduler tools
            "list_free_slots": "scheduler",
            "book_appointment": "scheduler",
            "cancel_appointment": "scheduler"
        }
        agent_name = tool_to_agent.get(tool_name, agent_name)

        # Log which tool was chosen - helpful for diagnosing conversation flow issues
        logger.debug(f"[FLOW] Tool chosen: {tool_name}")

        # For appointment scheduling specifically, log additional details
        if tool_name == "schedule_appointment" and "output" in final_state:
            output = final_state["output"]
            content = output.content if hasattr(output, "content") else str(output)
            logger.debug(f"[FLOW] Scheduler output: {content[:100]}...")

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
