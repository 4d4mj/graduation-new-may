from langchain_core.messages import HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder

from app.agents.states import BaseAgentState
from app.config.agent import settings as agentSettings
from app.config.settings import settings
from app.config.prompts import CONVERSATION_SYSTEM_PROMPT
from langchain_google_genai import ChatGoogleGenerativeAI

# TODO: implement real agents
from app.agents.rag.core import DummyRAG as MedicalRAG
from app.agents.web_search.core import DummyWebSearch as WebSearchProcessorAgent
from app.agents.scheduler.core import DummyScheduler as SchedulerAgent
from app.config.constants import AgentName  # ensure import

# Initialize the conversation prompt template with roles and dynamic history
conversation_chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(CONVERSATION_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="messages"),
    HumanMessagePromptTemplate.from_template("{input}")
])


def run_conversation_agent(state: BaseAgentState) -> BaseAgentState:
    """Wrapper around conversation agent: engage in general chat."""
    current_input = state.get("current_input")
    input_text = current_input if isinstance(current_input, str) else current_input.get("text", "") if current_input else ""

    messages = state.get("messages", [])
    recent_msgs = messages[-agentSettings.max_conversation_history:]

    # Build context string from recent messages for the system prompt
    context = "".join(
        f"User: {msg.content}\n" if isinstance(msg, HumanMessage)
        else f"Assistant: {msg.content}\n" for msg in recent_msgs
    )

    # Format the prompt using ChatPromptTemplate with dynamic history, context, and user input
    prompt_value = conversation_chat_prompt.format_prompt(messages=recent_msgs, input=input_text, context=context)

    # directly use Google Generative AI
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-preview-04-17",
        api_key=settings.google_api_key,
        temperature=0.7
    )
    # Invoke the model with the assembled message list
    response = llm.invoke(prompt_value.to_messages())
    output_msg = response if isinstance(response, AIMessage) else AIMessage(content=str(response))

    # Set both output and final_output to ensure consistency
    state["output"] = output_msg
    state["final_output"] = output_msg.content if hasattr(output_msg, "content") else str(output_msg)
    state["agent_name"] = AgentName.CONVERSATION.value
    return state


def run_rag_agent(state: BaseAgentState) -> BaseAgentState:
    """Wrapper around RAG agent: handle medical knowledge queries."""
    rag_agent = MedicalRAG(agentSettings)
    messages = state.get("messages", [])
    recent_context = "".join(
        f"User: {msg.content}\n" if isinstance(msg, HumanMessage)
        else f"Assistant: {msg.content}\n" for msg in messages[-agentSettings.rag.context_limit:]
    )
    response = rag_agent.process_query(state.get("current_input", ""), chat_history=recent_context)
    confidence = response.get("confidence", 0.0)
    content = response.get("response")
    text = getattr(content, "content", content)
    insufficient = any(substr in text.lower() for substr in agentSettings.rag.insufficient_info_keywords)
    raw = response.get("response")
    # only reveal answer if above confidence threshold
    if confidence >= agentSettings.rag.min_retrieval_confidence:
        output_msg = raw if isinstance(raw, AIMessage) else AIMessage(content=str(raw))
    else:
        output_msg = AIMessage(content="")

    state["output"] = output_msg
    # Set final_output for consistency
    state["final_output"] = output_msg.content if hasattr(output_msg, "content") else str(output_msg)
    state["needs_human_validation"] = False
    state["retrieval_confidence"] = confidence
    state["agent_name"] = AgentName.RAG.value
    state["insufficient_info"] = insufficient
    return state


def run_web_search_processor_agent(state: BaseAgentState) -> BaseAgentState:
    """Wrapper around web search processor agent: get and refine current info."""
    web_agent = WebSearchProcessorAgent(agentSettings)
    messages = state.get("messages", [])
    context = "".join(
        f"User: {msg.content}\n" if isinstance(msg, HumanMessage)
        else f"Assistant: {msg.content}\n" for msg in messages[-agentSettings.web_search_context_limit:]
    )
    # 1) Pull out the user's query
    query = state.get("query", state.get("current_input", ""))
    processed = web_agent.process_web_search_results(query, chat_history=context)
    agents = state.get("agent_name", "")
    combined = f"{agents}, {AgentName.WEB_SEARCH.value}" if agents else AgentName.WEB_SEARCH.value
    raw = processed
    output_msg = raw if isinstance(raw, AIMessage) else AIMessage(content=str(raw))

    state["output"] = output_msg
    # Set final_output for consistency
    state["final_output"] = output_msg.content if hasattr(output_msg, "content") else str(output_msg)
    state["agent_name"] = combined
    return state


def run_scheduler_agent(state: BaseAgentState) -> BaseAgentState:
    """Wrapper around the scheduling agent: handle appointment / scheduling requests."""
    # 1) Pull out the user's query
    query = state.get("current_input", "")

    # Get the patient response text if it exists (from the patient analysis node)
    patient_response = state.get("patient_response_text", "")

    # 2) Build a little chat-history context
    messages = state.get("messages", [])
    recent = messages[-agentSettings.max_conversation_history:]
    chat_history = "".join(
        f"User: {m.content}\n" if isinstance(m, HumanMessage)
        else f"Assistant: {m.content}\n"
        for m in recent
    )

    # 3) Call your scheduler "LLM" or tool
    scheduler = SchedulerAgent(settings)
    result = scheduler.process_schedule(query, chat_history=chat_history)

    # 4) Normalize into an AIMessage
    if isinstance(result, dict):
        resp = result.get("response", "")
    else:
        resp = result

    # Ensure we have a valid response - use patient_response as fallback if needed
    if not resp or resp == "None":
        if patient_response:
            resp = patient_response
        else:
            resp = "I can help you schedule an appointment with a healthcare professional. Would you like to do that now?"

    output = AIMessage(content=str(resp))

    # 5) Package back into state
    state["output"] = output
    state["final_output"] = output.content if hasattr(output, "content") else str(output)
    state["agent_name"] = AgentName.SCHEDULER.value

    # 6) (Optional) If your scheduler returns structured details, stash them too:
    if isinstance(result, dict) and "appointment_details" in result:
        state["appointment_details"] = result["appointment_details"]

    return state
