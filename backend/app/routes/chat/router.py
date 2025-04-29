from fastapi import APIRouter, HTTPException, Response, Cookie
from app.core.auth import decode_access_token
from app.config.settings import env
from app.agents.orchestrator.core import process_query
from app.schemas.chat import ChatRequest, ChatResponse, ChatMessage
from langchain_core.messages import HumanMessage, AIMessage

router = APIRouter(prefix="/chat", tags=["chat"])

secure_cookie = env == "production"

@router.post("/", response_model=ChatResponse, status_code=200)
async def chat(
    payload: ChatRequest,
    session: str | None = Cookie(default=None, alias="session")
):
    # ------------------------------------------------------------------ JWT
    if session is None:
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        token_data = decode_access_token(session)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid session token")

    user_role = token_data["role"]          # "patient" | "doctor" | …

    # ------------------------------------------------------------------ LangGraph
    # Convert inbound message into ChatMessage so history is uniform
    user_msg = ChatMessage(role="user", content=payload.message)

    # Convert incoming ChatMessage history to LLM messages
    server_history: list[HumanMessage|AIMessage] = []
    for msg in (payload.history or []):
        if msg.role == "user":
            server_history.append(HumanMessage(content=msg.content))
        else:
            server_history.append(AIMessage(content=msg.content))
    # Orchestrate with role-based agents
    result = process_query(
        query=user_msg.content,
        role=user_role,
        conversation_history=server_history
    )

    raw_output = result.get("output")
    reply_text = raw_output.content if hasattr(raw_output, "content") else str(raw_output)
    assistant_msg = ChatMessage(role="assistant", content=reply_text)

    full_history = (payload.history or []) + [user_msg, assistant_msg]

    return ChatResponse(
        reply    = assistant_msg.content,
        agent    = result["agent_name"],
        messages = full_history,
    )


# =================================================================================================================

@router.post("/validate", response_model=ChatResponse, status_code=200)
async def validate(
    payload: ChatRequest,
    session: str | None = Cookie(default=None, alias="session")
):
    # auth check
    if session is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        token_data = decode_access_token(session)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid session token")
    user_role = token_data["role"]

    # Build validation query and convert history
    validation_query = f"Validation result: {payload.message}"
    server_history: list[HumanMessage|AIMessage] = []
    for msg in (payload.history or []):
        if msg.role == "user":
            server_history.append(HumanMessage(content=msg.content))
        else:
            server_history.append(AIMessage(content=msg.content))

    # invoke orchestrator
    result = process_query(
        query=validation_query,
        role=user_role,
        conversation_history=server_history
    )

    raw_output = result.get("output")
    reply_text = raw_output.content if hasattr(raw_output, "content") else str(raw_output)
    assistant_msg = ChatMessage(role="assistant", content=reply_text)
    user_msg = ChatMessage(role="user", content=validation_query)
    full_history = (payload.history or []) + [user_msg, assistant_msg]

    return ChatResponse(
        reply=assistant_msg.content,
        agent=result["agent_name"],
        messages=full_history,
    )

