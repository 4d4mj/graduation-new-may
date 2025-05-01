from enum import Enum

class AgentName(str, Enum):
    CONVERSATION = "CONVERSATION_AGENT"
    RAG = "RAG_AGENT"
    WEB_SEARCH = "WEB_SEARCH_AGENT"
    INPUT_GUARDRAILS = "INPUT_GUARDRAILS"
    SCHEDULER = "SCHEDULER_AGENT"

class Task(str, Enum):
    CONVERSATION = "conversation"
    MEDICAL_QA = "medical_qa"
    SCHEDULING = "scheduling"
    # doctor-only
    SUMMARY = "summary"
    DB_QUERY = "db_query"
    IMAGE_ANALYSIS = "image_analysis"
