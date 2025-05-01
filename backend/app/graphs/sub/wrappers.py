from app.graphs.sub.conversation import build_conversation_graph
from app.graphs.sub.rag_web import build_medical_qa_graph
from app.graphs.sub.scheduler import build_scheduler_graph

# Pre-compile sub-graphs at module load time for better performance
conversation_graph = build_conversation_graph().compile(checkpointer=None)
medical_qa_graph = build_medical_qa_graph().compile(checkpointer=None)
scheduler_graph = build_scheduler_graph().compile(checkpointer=None)

# Doctor-only graphs will be pre-compiled here when they're implemented
