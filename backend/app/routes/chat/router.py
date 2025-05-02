from langchain.callbacks import StdOutCallbackHandler
import uuid
import logging
from fastapi import APIRouter, HTTPException, Cookie, Request
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from app.config.settings import env
from app.agents.states import init_state_for_role
from app.core.auth import decode_access_token
from app.schemas.chat import ChatRequest, ChatResponse, ChatMessage
from app.graphs.patient import create_patient_graph
from langgraph.checkpoint.memory import MemorySaver
from typing import List, Dict, Any

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)

# Increase logging level for conversation memory diagnosis
MEMORY_DEBUG = True

secure_cookie = env == "production"

# Initialize the graphs at module level, to be used in the FastAPI app startup event
role_graphs: Dict[str, Any] = {
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
        # Add doctor graph initialization here if/when needed
    }

    logger.info("Initialized graph with memory persistence for roles: %s", list(role_graphs.keys()))
    return role_graphs

@router.post("/", response_model=ChatResponse, status_code=200)
async def chat(
    payload: ChatRequest,
    request: Request,
    session: str | None = Cookie(default=None, alias="session")
):
    # Get thread ID from session token or create new one
    # Use a consistent prefix for guest sessions if needed
    is_guest = session is None
    thread_id = session if session else f"guest_{uuid.uuid4()}"

    # Basic logging
    if MEMORY_DEBUG:
        logger.info(f"[MEMORY] Thread ID: {thread_id} (Guest: {is_guest})")
        logger.info(f"[MEMORY] User message: {payload.message}")

    # 1) AUTH / ROLE DETERMINATION
    role = "patient" # Default role, adjust if needed based on auth
    if not is_guest:
        try:
            token = decode_access_token(session)
            role = token.get("role", "patient") # Safely get role
            user_id = token.get("sub")
            logger.info(f"Authenticated user: ID={user_id}, Role={role}")
        except Exception as e:
            logger.warning(f"Invalid session token for thread {thread_id}: {e}")
            # Decide how to handle invalid token - treat as guest or raise error?
            # For now, let's treat as guest and assign default role
            thread_id = f"guest_{uuid.uuid4()}" # Generate new guest ID
            role = "patient"
            # Or raise HTTPException(401, "Invalid session token") if strict auth is needed
    else:
         logger.info(f"Guest user session: {thread_id}")
         raise HTTPException(401, "Not authenticated")

    # 2) GET GRAPH FOR ROLE
    # Access graphs from app state where they were initialized
    app_graphs = getattr(request.app.state, "graphs", None)
    if not app_graphs or role not in app_graphs or app_graphs[role] is None:
         # Attempt to initialize if missing (e.g., during development hot reload)
         logger.warning(f"Graph for role '{role}' not found or not initialized. Attempting re-initialization.")
         request.app.state.graphs = init_graphs() # Re-initialize
         app_graphs = request.app.state.graphs
         if not app_graphs or role not in app_graphs or app_graphs[role] is None:
             logger.error(f"Failed to initialize graph for role {role}")
             raise HTTPException(500, f"Chat service not available for role {role}")

    graph_to_run = app_graphs[role]

    # 3) PREPARE INPUT STATE
    input_state = init_state_for_role(role)
    input_state["messages"] = [HumanMessage(content=payload.message)]  # Only the new message
    input_state["current_input"] = payload.message  # Add back for guardrails to access original text
    input_state["user_role"] = role

    # 4) RUN GRAPH
    try:
        final_state = await graph_to_run.ainvoke(
            input_state,
            config={
                # thread-local settings (persisted)
                "configurable": {"thread_id": thread_id},

                # run-only settings (NOT persisted)
                "callbacks": [StdOutCallbackHandler()] if MEMORY_DEBUG else [],
                "recursion_limit": 15,          # safety-net
                "run_kwargs": {"stream_mode": "updates"}  # Enable streaming updates
            }
        )

        if final_state is None:
             raise ValueError("Graph execution returned None state.")

    except Exception as e:
        logger.exception(f"Error running graph for thread {thread_id}")
        raise HTTPException(500, f"Processing error: {str(e)}")

    # Log final state keys for debugging
    if MEMORY_DEBUG:
        logger.debug(f"[MEMORY] Final state keys for thread {thread_id}: {list(final_state.keys())}")
        # Log specific potentially relevant keys
        if "messages" in final_state:
            logger.debug(f"[MEMORY] Final 'messages' count: {len(final_state['messages'])}")
        logger.debug(f"[MEMORY] Final 'final_output': {final_state.get('final_output')}")
        if "output" in final_state:
            output_val = final_state["output"]
            output_content = output_val.content if hasattr(output_val, "content") else str(output_val)
            logger.debug(f"[MEMORY] Final 'output': {output_content[:100]}...")

    # 5) EXTRACT REPLY - Refined Logic
    reply: str | None = None

    # ① Trust final_output first (should be set by apply_output_guardrails)
    final_output_val = final_state.get("final_output")
    if isinstance(final_output_val, str) and final_output_val.strip():
        reply = final_output_val
        logger.info(f"Extracted reply from 'final_output' for thread {thread_id}")
    else:
        # ② Fallback: Check the 'output' field if it's an AIMessage (also set by guardrails)
        output_val = final_state.get("output")
        if isinstance(output_val, AIMessage) and output_val.content.strip():
            reply = output_val.content
            logger.info(f"Extracted reply from 'output' (AIMessage) for thread {thread_id}")
        else:
            # ③ Fallback: Check the very last message in the 'messages' list
            messages_val = final_state.get("messages", [])
            if messages_val:
                last_message = messages_val[-1]
                if isinstance(last_message, AIMessage) and last_message.content.strip():
                    reply = last_message.content
                    logger.info(f"Extracted reply from last 'messages' list item for thread {thread_id}")

    # ④ Last resort
    if reply is None or not reply.strip():
        reply = (
            "I'm sorry, I couldn't process your request right now. "
            "Please try again in a moment."
        )
        logger.warning(f"Could not extract valid reply for thread {thread_id}. Using default.")

    # 6) DETERMINE AGENT/TOOL USED (for UI feedback)
    agent_name = final_state.get("agent_name", "medical_assistant") # Get agent name if set by nodes

    # Check tool calls for agent determination if agent_name not set by nodes
    if agent_name == "medical_assistant":
        tool_calls = final_state.get("tool_calls", [])
        if tool_calls and isinstance(tool_calls[-1], dict) and "name" in tool_calls[-1]:
            tool_name = tool_calls[-1]["name"]
            tool_to_agent = {
                "rag_query": "medical_knowledge",
                "web_search": "web_search",
                "schedule_appointment": "scheduler",
                "small_talk": "conversation",
                "list_free_slots": "scheduler",
                "book_appointment": "scheduler",
                "cancel_appointment": "scheduler"
            }
            agent_name = tool_to_agent.get(tool_name, agent_name)

            # Log which tool was chosen
            logger.debug(f"[FLOW] Tool chosen: {tool_name}")

            # For scheduler tools, log additional details
            if tool_name in ["schedule_appointment", "list_free_slots", "book_appointment", "cancel_appointment"] and "output" in final_state:
                output = final_state["output"]
                content = output.content if hasattr(output, "content") else str(output)
                logger.debug(f"[FLOW] Scheduler output: {content[:100]}...")

    # 7) CONSTRUCT RESPONSE
    # For simplicity, we're using the approach that worked previously
    # You could enhance this to use the full messages list from final_state if needed
    return ChatResponse(
        reply=reply,
        agent=agent_name,
        messages=[
            *(payload.history or []),
            ChatMessage(role="user", content=payload.message),
            ChatMessage(role="assistant", content=reply),
        ],
    )
