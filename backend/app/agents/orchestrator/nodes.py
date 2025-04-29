from langchain_core.messages import HumanMessage, AIMessage
from app.config.agent import settings
from app.config.provider import PROVIDER
from .state import AgentState
# TODO:
from app.agents.rag.core import DummyRAG as MedicalRAG
from app.agents.web_search.core import DummyWebSearch as WebSearchProcessorAgent
from app.agents.guardrails.local_guardrails import LocalGuardrails
from app.config.constants import AgentName
from app.config.prompts import CONVERSATION_SYSTEM_PROMPT
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder

# Initialize the conversation prompt template with roles and dynamic history
conversation_chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(CONVERSATION_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="messages"),
    HumanMessagePromptTemplate.from_template("{input}")
])

provider = PROVIDER

# lazy initialize output guard to avoid import-time instantiation
_output_guard = None

def _get_output_guard():
    global _output_guard
    if _output_guard is None:
        _output_guard = LocalGuardrails(provider.chat())
    return _output_guard


def run_conversation_agent(state: AgentState) -> AgentState:
    """Wrapper around conversation agent: engage in general chat."""
    current_input = state.get("current_input", "")
    input_text = current_input if isinstance(current_input, str) else current_input.get("text", "")

    recent_msgs = state.get("messages", [])[-settings.max_conversation_history:]
    # Format the prompt using ChatPromptTemplate with dynamic history and user input
    prompt_value = conversation_chat_prompt.format_prompt(messages=recent_msgs, input=input_text)
    llm = provider.chat(temperature=0.7)
    # Invoke the model with the assembled message list
    response = llm.invoke(prompt_value.to_messages())
    output_msg = response if isinstance(response, AIMessage) else AIMessage(content=str(response))
    return {**state, "output": output_msg, "agent_name": AgentName.CONVERSATION.value}


def run_rag_agent(state: AgentState) -> AgentState:
    """Wrapper around RAG agent: handle medical knowledge queries."""
    rag_agent = MedicalRAG(settings)
    messages = state.get("messages", [])
    recent_context = "".join(
        f"User: {msg.content}\n" if isinstance(msg, HumanMessage)
        else f"Assistant: {msg.content}\n" for msg in messages[-settings.rag.context_limit:]
    )
    response = rag_agent.process_query(state.get("current_input"), chat_history=recent_context)
    confidence = response.get("confidence", 0.0)
    content = response.get("response")
    text = getattr(content, "content", content)
    insufficient = any(substr in text.lower() for substr in settings.rag.insufficient_info_keywords)
    raw = response.get("response")
    # only reveal answer if above confidence threshold
    if confidence >= settings.rag.min_retrieval_confidence:
        output_msg = raw if isinstance(raw, AIMessage) else AIMessage(content=str(raw))
    else:
        output_msg = AIMessage(content="")
    return {
        **state,
        "output": output_msg,
        "needs_human_validation": False,
        "retrieval_confidence": confidence,
        "agent_name": AgentName.RAG.value,
        "insufficient_info": insufficient
    }


def run_web_search_processor_agent(state: AgentState) -> AgentState:
    """Wrapper around web search processor agent: get and refine current info."""
    web_agent = WebSearchProcessorAgent(settings)
    context = "".join(
        f"User: {msg.content}\n" if isinstance(msg, HumanMessage)
        else f"Assistant: {msg.content}\n" for msg in state.get("messages", [])[-settings.web_search_context_limit:]
    )
    processed = web_agent.process_web_search_results(query=state.get("current_input"), chat_history=context)
    agents = state.get("agent_name")
    combined = f"{agents}, {AgentName.WEB_SEARCH.value}" if agents else AgentName.WEB_SEARCH.value
    raw = processed
    output_msg = raw if isinstance(raw, AIMessage) else AIMessage(content=str(raw))
    return {**state, "output": output_msg, "agent_name": combined}


def perform_human_validation(state: AgentState) -> AgentState:
    """Handle human validation node: append and check user's validation response."""
    # create validation prompt
    content = state.get("output")
    text = getattr(content, "content", content)
    validation_prompt = f"{text}\n\n**Human Validation Required:**\n- If you're a healthcare professional: Please validate the output. Select **Yes** or **No**. If No, provide comments.\n- If you're a patient: Simply click Yes to confirm."
    return {**state, "output": AIMessage(content=validation_prompt), "agent_name": f"{state.get('agent_name')}, HUMAN_VALIDATION"}


def apply_output_guardrails(state: AgentState) -> AgentState:
    """Sanitize and finalize output using guardrails."""
    output = state.get("output")
    input_text = state.get("current_input") if isinstance(state.get("current_input"), str) else state.get("current_input", {}).get("text", "")
    text = output.content if isinstance(output, AIMessage) else output or ""
    guard = _get_output_guard()
    sanitized = guard.check_output(text, input_text)
    sanitized_msg = AIMessage(content=sanitized)
    # return {**state, "messages": state.get("messages", []) + [sanitized_msg], "output": sanitized_msg}
    # coerce .messages to a list if needed
    msgs = state.get("messages")
    if not isinstance(msgs, list):
        msgs = []
    return {**state, "messages": msgs + [sanitized_msg], "output": sanitized_msg}
