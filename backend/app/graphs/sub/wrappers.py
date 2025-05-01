from app.graphs.sub.conversation import build_conversation_graph
from app.graphs.sub.rag_web import build_medical_qa_graph
from app.graphs.sub.scheduler import build_scheduler_graph
from app.graphs.sub.patient_supervisor import build_patient_supervisor_graph
from langgraph.checkpoint.memory import MemorySaver
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Memory debugging flag (same as router)
MEMORY_DEBUG = True

# Pre-compile sub-graphs with proper checkpointers
patient_supervisor_graph = build_patient_supervisor_graph().compile(checkpointer=MemorySaver())
conversation_graph = build_conversation_graph().compile(checkpointer=MemorySaver())
medical_qa_graph = build_medical_qa_graph().compile(checkpointer=MemorySaver())
scheduler_graph = build_scheduler_graph().compile(checkpointer=MemorySaver())

# Subgraph wrapper functions updated to propagate thread_id properly
async def run_patient_supervisor_subgraph(state, config):
    """Run the patient supervisor subgraph on the given state with logging."""
    # Ensure we propagate the thread_id to maintain conversation memory
    thread_id = config.get("configurable", {}).get("thread_id")

    if MEMORY_DEBUG and thread_id:
        logger.info(f"[MEMORY] Supervisor subgraph using thread_id: {thread_id}")

    # Log the incoming state for debugging
    logger.info("↪ patient_supervisor subgraph, incoming final_output=%r", state.get("final_output"))

    # Create the proper config with thread_id to maintain state
    subgraph_config = {"configurable": {"thread_id": thread_id}} if thread_id else config

    # Invoke the subgraph with the thread_id config
    new_state = await patient_supervisor_graph.ainvoke(state, config=subgraph_config)

    # Log the outgoing state for debugging
    logger.info("↩ patient_supervisor subgraph, outgoing final_output=%r, intent=%r, request_scheduling=%r",
                new_state.get("final_output"), new_state.get("intent"), new_state.get("request_scheduling"))

    if MEMORY_DEBUG and thread_id:
        # Log how many messages the state has after processing
        msg_count = len(new_state.get("messages", []))
        logger.info(f"[MEMORY] Supervisor subgraph returned state with {msg_count} messages")

    return new_state

async def run_conversation_subgraph(state, config):
    """Run the conversation subgraph on the given state with logging."""
    # Ensure we propagate the thread_id to maintain conversation memory
    thread_id = config.get("configurable", {}).get("thread_id")

    if MEMORY_DEBUG and thread_id:
        logger.info(f"[MEMORY] Conversation subgraph using thread_id: {thread_id}")

    # Log the incoming state for debugging
    logger.info("↪ conversation subgraph, incoming final_output=%r", state.get("final_output"))

    # Create the proper config with thread_id to maintain state
    subgraph_config = {"configurable": {"thread_id": thread_id}} if thread_id else config

    # Invoke the subgraph with the thread_id config
    new_state = await conversation_graph.ainvoke(state, config=subgraph_config)

    # Log the outgoing state for debugging
    logger.info("↩ conversation subgraph, outgoing final_output=%r", new_state.get("final_output"))

    if MEMORY_DEBUG and thread_id:
        # Log how many messages the state has after processing
        msg_count = len(new_state.get("messages", []))
        logger.info(f"[MEMORY] Conversation subgraph returned state with {msg_count} messages")

    return new_state

async def run_medical_qa_subgraph(state, config):
    """Run the medical Q&A subgraph on the given state with logging."""
    # Ensure we propagate the thread_id to maintain conversation memory
    thread_id = config.get("configurable", {}).get("thread_id")

    if MEMORY_DEBUG and thread_id:
        logger.info(f"[MEMORY] Medical QA subgraph using thread_id: {thread_id}")

    # Log the incoming state for debugging
    logger.info("↪ medical_qa subgraph, incoming final_output=%r", state.get("final_output"))

    # Create the proper config with thread_id to maintain state
    subgraph_config = {"configurable": {"thread_id": thread_id}} if thread_id else config

    # Invoke the subgraph with the thread_id config
    new_state = await medical_qa_graph.ainvoke(state, config=subgraph_config)

    # Log the outgoing state for debugging
    logger.info("↩ medical_qa subgraph, outgoing final_output=%r", new_state.get("final_output"))

    if MEMORY_DEBUG and thread_id:
        # Log how many messages the state has after processing
        msg_count = len(new_state.get("messages", []))
        logger.info(f"[MEMORY] Medical QA subgraph returned state with {msg_count} messages")

    return new_state

async def run_scheduler_subgraph(state, config):
    """Run the scheduler subgraph on the given state with logging."""
    # Ensure we propagate the thread_id to maintain conversation memory
    thread_id = config.get("configurable", {}).get("thread_id")

    if MEMORY_DEBUG and thread_id:
        logger.info(f"[MEMORY] Scheduler subgraph using thread_id: {thread_id}")

    # Log the incoming state for debugging
    logger.info("↪ scheduler subgraph, incoming final_output=%r", state.get("final_output"))

    # Create the proper config with thread_id to maintain state
    subgraph_config = {"configurable": {"thread_id": thread_id}} if thread_id else config

    # Invoke the subgraph with the thread_id config
    new_state = await scheduler_graph.ainvoke(state, config=subgraph_config)

    # Log the outgoing state for debugging
    logger.info("↩ scheduler subgraph, outgoing final_output=%r", new_state.get("final_output"))

    if MEMORY_DEBUG and thread_id:
        # Log how many messages the state has after processing
        msg_count = len(new_state.get("messages", []))
        logger.info(f"[MEMORY] Scheduler subgraph returned state with {msg_count} messages")

    return new_state

# Doctor-only graphs will be pre-compiled here when they're implemented
