from langchain.callbacks import StdOutCallbackHandler
import uuid
import logging
from fastapi import APIRouter, HTTPException, Cookie, Request, status
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from app.config.settings import env
from app.agents.states import init_state_for_role, BaseAgentState
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
    is_guest = session is None
    thread_id = session if session else f"guest_{uuid.uuid4()}"

    # Basic logging
    if MEMORY_DEBUG:
        logger.info(f"[MEMORY] Thread ID: {thread_id} (Guest: {is_guest})")
        logger.info(f"[MEMORY] User message: {payload.message}")

    # 1) AUTH / ROLE DETERMINATION
    role = "patient" # Default
    if not is_guest:
        try:
            token = decode_access_token(session)
            role = token.get("role", "patient")
            user_id = token.get("sub")
            logger.info(f"Authenticated user: ID={user_id}, Role={role}, Thread={thread_id}")
        except Exception as e:
            # Raise error for invalid tokens on authenticated routes
            logger.error(f"Invalid session token provided: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid session token")
    else:
         # Block guest access if session cookie is required
         logger.warning(f"Guest access denied for thread_id attempt (session required).")
         raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required.")

    # 2) GET GRAPH FOR ROLE
    app_graphs = getattr(request.app.state, "graphs", None)
    if not app_graphs or role not in app_graphs or app_graphs[role] is None:
         logger.error(f"Graph for role '{role}' not found or not initialized.")
         raise HTTPException(500, f"Chat service not available for role '{role}'")
    graph_to_run = app_graphs[role]

    # 3) PREPARE INPUT FOR ainvoke
    #    LangGraph with MemorySaver loads the history.
    #    We only provide the *new* message(s).
    #    For MessagesState, the key should be 'messages' and the value a list.
    current_message = HumanMessage(content=payload.message)

    # Fix for Gemini API message sequencing
    # Before invoking, check if we need to ensure alternating message pattern
    try:
        # First, attempt to load the existing conversation
        config_for_load = {
            "configurable": {"thread_id": thread_id},
        }
        # Use a try/except because this will fail on first message (no history yet)
        existing_state = await graph_to_run.checkpointer.get(thread_id)

        if existing_state and "messages" in existing_state:
            existing_messages = existing_state.get("messages", [])
            # Check if the last message is from the user (HumanMessage)
            # This would break alternating pattern required by Gemini
            if existing_messages and isinstance(existing_messages[-1], HumanMessage):
                logger.warning(f"[MEMORY] Detected consecutive user messages. Adding an empty AI response to maintain message sequencing.")
                # Insert an empty AI message to maintain alternating pattern
                graph_input = {"messages": [AIMessage(content=""), current_message]}
            else:
                # Normal case - just add the new message
                graph_input = {"messages": [current_message]}
        else:
            # First message in conversation - no special handling needed
            graph_input = {"messages": [current_message]}
    except Exception as e:
        # Something went wrong or it's a new conversation - use standard behavior
        logger.info(f"[MEMORY] No existing conversation found or error retrieving it: {str(e)}")
        graph_input = {"messages": [current_message]}

    logger.debug(f"Passing input to graph for thread {thread_id}: {graph_input}")

    # 4) RUN GRAPH
    try:
        # Note: 'streaming_mode="updates"' might cause issues if not all nodes
        # support it well or if state isn't handled correctly during streaming.
        # Consider removing it if problems persist, and use standard ainvoke.
        config = {
            "configurable": {"thread_id": thread_id},
            "callbacks": [StdOutCallbackHandler()] if MEMORY_DEBUG else [],
            "recursion_limit": 15,
            # "run_kwargs": {"stream_mode": "updates"} # Temporarily disable if causing issues
        }

        # Use standard ainvoke for robustness first
        final_state: BaseAgentState = await graph_to_run.ainvoke(graph_input, config=config)

        if final_state is None:
             raise ValueError("Graph execution returned None state.")

    except Exception as e:
        logger.exception(f"Error running graph for thread {thread_id}")
        raise HTTPException(500, f"Processing error: {str(e)}")

    # Log final state keys for debugging
    if MEMORY_DEBUG:
        logger.debug(f"[MEMORY] Final state keys for thread {thread_id}: {list(final_state.keys())}")
        if "messages" in final_state:
            logger.debug(f"[MEMORY] Final 'messages' count: {len(final_state['messages'])}")
            # Log last few messages
            for i, msg in enumerate(final_state['messages'][-5:]):
                logger.debug(f"[MEMORY] Final msg [-{5-i}]: Type={type(msg).__name__}, Content='{str(getattr(msg, 'content', 'N/A'))[:100]}...'")
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

    # Check tool calls for agent determination if agent_name not set by nodes or is generic
    # React agent might not explicitly set agent_name state key, rely on tool calls
    tool_calls = final_state.get("tool_calls", []) # React agent adds tool_calls key
    last_tool_name = None
    if tool_calls and isinstance(tool_calls, list) and tool_calls:
         # Find the name of the last tool called
         # Structure might vary slightly, check typical React agent output
         # Often it's a list of dictionaries or ToolCall objects
         last_call = tool_calls[-1]
         if isinstance(last_call, dict) and 'name' in last_call:
             last_tool_name = last_call['name']
         elif hasattr(last_call, 'name'): # If it's an object like ToolCall
             last_tool_name = last_call.name

    if last_tool_name:
        tool_to_agent = {
            "rag_query": "medical_knowledge",
            "web_search": "web_search",
            "small_talk": "conversation",
            "list_free_slots": "scheduler",
            "book_appointment": "scheduler",
            "cancel_appointment": "scheduler"
        }
        agent_name = tool_to_agent.get(last_tool_name, "medical_assistant") # Fallback to default
        logger.debug(f"[FLOW] Last tool chosen: {last_tool_name}, mapped to agent: {agent_name}")

    # 7) CONSTRUCT RESPONSE HISTORY FOR FRONTEND
    #    Send back the *full* history from the final state, converted to schema.
    response_messages: List[ChatMessage] = []
    final_messages_from_state = final_state.get("messages", [])

    for msg in final_messages_from_state:
         if isinstance(msg, HumanMessage):
             response_messages.append(ChatMessage(role="user", content=msg.content))
         elif isinstance(msg, AIMessage):
             # Use the final extracted reply for the *very last* assistant message in history
             # This ensures the UI shows the guardrail-checked response correctly
             if msg is final_messages_from_state[-1] and isinstance(msg, AIMessage):
                 response_messages.append(ChatMessage(role="assistant", content=reply))
             else:
                  # Keep previous assistant messages as they were
                  response_messages.append(ChatMessage(role="assistant", content=msg.content))
         # Ignore ToolMessages, SystemMessages etc. for the frontend history display

    # Optional: Limit history length sent back to frontend
    MAX_HISTORY_LEN = 20 # Example limit
    if len(response_messages) > MAX_HISTORY_LEN:
        response_messages = response_messages[-MAX_HISTORY_LEN:]

    return ChatResponse(
        reply=reply,
        agent=agent_name,
        messages=response_messages, # Send the potentially truncated, converted history
    )
