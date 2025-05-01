from app.graphs.sub.conversation import build_conversation_graph
from app.graphs.sub.rag_web import build_medical_qa_graph
from app.graphs.sub.scheduler import build_scheduler_graph
from app.graphs.sub.patient_supervisor import build_patient_supervisor_graph
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Pre-compile sub-graphs at module load time for better performance
patient_supervisor_graph = build_patient_supervisor_graph().compile(checkpointer=None)
conversation_graph = build_conversation_graph().compile(checkpointer=None)
medical_qa_graph = build_medical_qa_graph().compile(checkpointer=None)
scheduler_graph = build_scheduler_graph().compile(checkpointer=None)

# Add proper logging to track data flow through the wrappers
async def run_patient_supervisor_subgraph(state, config):
    """Run the patient supervisor subgraph on the given state with logging."""
    logger.info("↪ patient_supervisor subgraph, incoming final_output=%r", state.get("final_output"))
    new_state = await patient_supervisor_graph.ainvoke(state, config=config)
    logger.info("↩ patient_supervisor subgraph, outgoing final_output=%r, intent=%r, request_scheduling=%r",
                new_state.get("final_output"), new_state.get("intent"), new_state.get("request_scheduling"))
    return new_state

async def run_conversation_subgraph(state, config):
    """Run the conversation subgraph on the given state with logging."""
    logger.info("↪ conversation subgraph, incoming final_output=%r", state.get("final_output"))
    new_state = await conversation_graph.ainvoke(state, config=config)
    logger.info("↩ conversation subgraph, outgoing final_output=%r", new_state.get("final_output"))
    return new_state

async def run_medical_qa_subgraph(state, config):
    """Run the medical Q&A subgraph on the given state with logging."""
    logger.info("↪ medical_qa subgraph, incoming final_output=%r", state.get("final_output"))
    new_state = await medical_qa_graph.ainvoke(state, config=config)
    logger.info("↩ medical_qa subgraph, outgoing final_output=%r", new_state.get("final_output"))
    return new_state

async def run_scheduler_subgraph(state, config):
    """Run the scheduler subgraph on the given state with logging."""
    logger.info("↪ scheduler subgraph, incoming final_output=%r", state.get("final_output"))
    new_state = await scheduler_graph.ainvoke(state, config=config)
    logger.info("↩ scheduler subgraph, outgoing final_output=%r", new_state.get("final_output"))
    return new_state

# Doctor-only graphs will be pre-compiled here when they're implemented
