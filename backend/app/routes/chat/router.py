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
# from app.graphs.doctor import create_doctor_graph
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
    # Create a more robust checkpointer with proper dict handling
    patient_checkpointer = MemorySaver()  # Explicitly set state_class to dict
    patient_graph = create_patient_graph().compile(checkpointer=patient_checkpointer)
    role_graphs = {"patient": patient_graph}
    logger.info("Initialized graph with memory persistence for roles: %s", list(role_graphs.keys()))
    return role_graphs

@router.post("/", response_model=ChatResponse, status_code=200)
async def chat(
    payload: ChatRequest,
    request: Request,
    session: str | None = Cookie(default=None, alias="session")
):
    is_guest = session is None
    thread_id = session if session else f"guest_{uuid.uuid4()}"

    if MEMORY_DEBUG:
        logger.info(f"[MEMORY] Thread ID: {thread_id} (Guest: {is_guest})")
        logger.info(f"[MEMORY] User message: {payload.message}")

    # 1) AUTH / ROLE DETERMINATION
    role = "patient"
    user_id = None # Initialize user_id
    if not is_guest:
        try:
            token = decode_access_token(session)
            role = token.get("role", "patient")
            user_id = token.get("sub") # Get user ID from token
            logger.info(f"Authenticated user: ID={user_id}, Role={role}, Thread={thread_id}")
        except Exception as e:
            logger.error(f"Invalid session token provided: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid session token")
    else:
         logger.warning(f"Guest access denied for thread_id attempt (session required).")
         raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required.")

    # 2) GET GRAPH FOR ROLE
    if not role_graphs or role not in role_graphs or role_graphs[role] is None:
        # Initialize graphs if they don't exist
        if not role_graphs.get(role):
            init_graphs()
            logger.info(f"Initialized graphs for roles: {list(role_graphs.keys())}")
        # If still not available, raise an error
        if not role_graphs.get(role):
            raise HTTPException(400, f"Unknown role {role}")

    graph_to_run = role_graphs.get(role)

    # 3) PREPARE INPUT FOR ainvoke
    #    Only pass the new message. LangGraph handles loading/merging history via checkpointer.
    current_message = HumanMessage(content=payload.message)
    graph_input: Dict[str, Any] = {"messages": [current_message]}

    # Add user_id directly to the input dict for potential use by tools/nodes
    if user_id:
        graph_input["user_id"] = int(user_id) # Ensure it's an int if needed by tools

    graph_input["user_role"] = role

    logger.debug(f"Passing input to graph for thread {thread_id}: {graph_input}")

    # --- Log Checkpoint Retrieval ---
    if MEMORY_DEBUG:
        try:
            retrieved_checkpoint = await graph_to_run.checkpointer.aget(
                {"configurable": {"thread_id": thread_id}}
            )
            logger.debug(f"[MEMORY RETRIEVAL] Raw checkpoint for thread {thread_id}: {retrieved_checkpoint}")
            if isinstance(retrieved_checkpoint, dict):
                retrieved_state_values = retrieved_checkpoint.get("channel_values", {})
                logger.debug(f"[MEMORY RETRIEVAL] Checkpoint channel_values: {retrieved_state_values}")
                retrieved_messages = retrieved_state_values.get("messages", [])
                logger.debug(f"[MEMORY RETRIEVAL] Messages count in loaded checkpoint: {len(retrieved_messages)}")
                for i, msg in enumerate(retrieved_messages[-5:]):
                    logger.debug(f"[MEMORY RETRIEVAL] Loaded msg [-{5-i}]: Type={type(msg).__name__}, Content='{str(getattr(msg, 'content', 'N/A'))[:100]}...'")
            else:
                logger.debug(f"[MEMORY RETRIEVAL] Checkpoint was not a dictionary or was None.")

        except Exception as chkpt_err:
            logger.error(f"[MEMORY RETRIEVAL] Error retrieving checkpoint for thread {thread_id}: {chkpt_err}", exc_info=True)
    # --- End Log Checkpoint ---

    logger.debug(f"Passing input to graph for thread {thread_id}: {graph_input}") # Keep this


    # 4) RUN GRAPH
    try:
        config = {
            "configurable": {"thread_id": thread_id},
            "callbacks": [StdOutCallbackHandler()] if MEMORY_DEBUG else [],
            "recursion_limit": 15,
        }
        final_state = await graph_to_run.ainvoke(graph_input, config=config)

        if final_state is None:
             raise ValueError("Graph execution returned None state.")

    except Exception as e:
        # Check if the error is the Gemini sequencing error
        if "ensure that single turn requests end with a user role" in str(e):
             logger.error(f"Gemini sequencing error detected for thread {thread_id}. This might indicate an issue with state merging or consecutive tool calls without an intermediate AI response.", exc_info=True)
             # You could potentially retry or return a specific error message
             raise HTTPException(500, "Chat processing error related to conversation flow. Please try rephrasing.")
        else:
             logger.exception(f"Error running graph for thread {thread_id}")
             raise HTTPException(500, f"Processing error: {str(e)}")

    # Log final state keys for debugging
    if MEMORY_DEBUG:
        # Safely access keys
        final_state_keys = list(final_state.keys()) if isinstance(final_state, dict) else []
        logger.debug(f"[MEMORY] Final state keys for thread {thread_id}: {final_state_keys}")
        if isinstance(final_state, dict) and "messages" in final_state:
            messages_list = final_state['messages']
            logger.debug(f"[MEMORY] Final 'messages' count: {len(messages_list)}")
            # Log last few messages
            for i, msg in enumerate(messages_list[-5:]):
                logger.debug(f"[MEMORY] Final msg [-{5-i}]: Type={type(msg).__name__}, Content='{str(getattr(msg, 'content', 'N/A'))[:100]}...'")
        else:
            logger.debug(f"[MEMORY] Final state is not a dict or 'messages' key is missing.")

        final_output_val = final_state.get("final_output", "N/A") if isinstance(final_state, dict) else "N/A (Not a dict)"
        logger.debug(f"[MEMORY] Final 'final_output': {final_output_val}")

        if isinstance(final_state, dict) and "output" in final_state:
            output_val = final_state["output"]
            output_content = output_val.content if hasattr(output_val, "content") else str(output_val)
            logger.debug(f"[MEMORY] Final 'output': {output_content[:100]}...")
        else:
            logger.debug(f"[MEMORY] Final state is not a dict or 'output' key is missing.")

    # 5) EXTRACT REPLY - Refined Logic
    reply: str | None = None

    # Ensure final_state is a dictionary before proceeding
    if not isinstance(final_state, dict):
        logger.error(f"Final state is not a dictionary for thread {thread_id}. State: {final_state}")
        reply = "An internal error occurred while processing the response."
    else:
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
                if messages_val and isinstance(messages_val, list):
                    last_message = messages_val[-1]
                    # Ensure last message is AIMessage before extracting content
                    if isinstance(last_message, AIMessage) and last_message.content and last_message.content.strip():
                        reply = last_message.content
                        logger.info(f"Extracted reply from last 'messages' list item for thread {thread_id}")

        # ④ Last resort
        if reply is None or not reply.strip():
            default_error = "I'm sorry, I couldn't process your request right now. Please try again in a moment."
            # Check if a more specific error message was already set (e.g., by a failing tool)
            if final_output_val and isinstance(final_output_val, str):
                 reply = final_output_val # Propagate error from final_output if it exists
            elif final_state.get("messages"):
                 last_msg = final_state["messages"][-1]
                 if isinstance(last_msg, AIMessage) and last_msg.content: # Check if last msg has content
                     reply = last_msg.content # Could be an error message set by agent/guardrail
                 else:
                     reply = default_error
            else:
                reply = default_error
            if reply == default_error: # Only log warning if we ended up using the generic default
                logger.warning(f"Could not extract valid reply for thread {thread_id}. Using default.")

    # 6) DETERMINE AGENT/TOOL USED (for UI feedback)
    agent_name = "medical_assistant" # Default
    last_tool_name = None
    if isinstance(final_state, dict):
        agent_name = final_state.get("agent_name", "medical_assistant")
        tool_calls = final_state.get("tool_calls", [])
        if tool_calls and isinstance(tool_calls, list) and tool_calls:
             last_call = tool_calls[-1]
             if isinstance(last_call, dict) and 'name' in last_call:
                 last_tool_name = last_call['name']
             elif hasattr(last_call, 'name'):
                 last_tool_name = last_call.name

    if last_tool_name:
        tool_to_agent = {
            "rag_query": "medical_knowledge", "web_search": "web_search",
            "small_talk": "conversation", "list_free_slots": "scheduler",
            "book_appointment": "scheduler", "cancel_appointment": "scheduler"
        }
        agent_name = tool_to_agent.get(last_tool_name, agent_name) # Use mapped name or existing agent_name
        logger.debug(f"[FLOW] Last tool chosen: {last_tool_name}, mapped to agent: {agent_name}")

    # 7) CONSTRUCT RESPONSE HISTORY FOR FRONTEND
    response_messages: List[ChatMessage] = []
    if isinstance(final_state, dict):
        final_messages_from_state = final_state.get("messages", [])
        if isinstance(final_messages_from_state, list):
            for msg in final_messages_from_state:
                 if isinstance(msg, HumanMessage):
                     response_messages.append(ChatMessage(role="user", content=msg.content))
                 elif isinstance(msg, AIMessage):
                     # Use the final extracted reply for the *very last* assistant message
                     if msg is final_messages_from_state[-1]:
                         response_messages.append(ChatMessage(role="assistant", content=reply))
                     # Only include non-empty previous assistant messages
                     elif msg.content and msg.content.strip():
                          response_messages.append(ChatMessage(role="assistant", content=msg.content))
        else:
            logger.warning(f"Final 'messages' in state is not a list for thread {thread_id}")
            # Fallback: use original payload history + new messages
            response_messages = [
                *(payload.history or []),
                ChatMessage(role="user", content=payload.message),
                ChatMessage(role="assistant", content=reply),
            ]
    else:
         logger.error(f"Final state was not a dictionary, cannot construct history for thread {thread_id}")
         # Fallback: use original payload history + new messages
         response_messages = [
             *(payload.history or []),
             ChatMessage(role="user", content=payload.message),
             ChatMessage(role="assistant", content=reply),
         ]

    MAX_HISTORY_LEN = 20
    if len(response_messages) > MAX_HISTORY_LEN:
        response_messages = response_messages[-MAX_HISTORY_LEN:]

    return ChatResponse(
        reply=reply,
        agent=agent_name,
        messages=response_messages,
        session_id=thread_id,
    )
