import logging
from fastapi import APIRouter, HTTPException, Cookie, Request
from langchain.callbacks import StdOutCallbackHandler
from app.config.settings import env
from langchain_core.messages import HumanMessage
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

    # Create a fresh input state with what we need for this turn
    input_state = init_state_for_role(role)

    # Set the current input from the user
    input_state["user_id"] = int(user_id)
    input_state["current_input"] = payload.message
    input_state["messages"] = [HumanMessage(content=payload.message)]

    try:
        # invoke the graph with the input state
        final_state = await graph.ainvoke(input_state, config={"configurable": {"thread_id": session},
                                                               "callbacks": [StdOutCallbackHandler()]
                                                               })

    except Exception as e:
        logger.exception("Error running graph")
        raise HTTPException(500, f"Processing error: {e}")

    # extract the final output from the state
    reply = final_state.get("final_output")
    logger.info("Using final_output for reply: %s", reply)

    if not reply:
        reply = "I apologize, but I couldn't process your request. Please try again later."

    agent_name = final_state.get("agent_name", "Unknown Agent")

    # Build history for response (keeping client-side synchronized)
    response_messages = []
    if payload.history:
        response_messages = payload.history

    # Add the most recent exchange to the history
    response_messages.append(ChatMessage(role="user", content=payload.message))
    response_messages.append(ChatMessage(role="assistant", content=reply))

    return ChatResponse(
        reply=reply,
        agent=agent_name,
        messages=response_messages,
        session_id=session,
    )
