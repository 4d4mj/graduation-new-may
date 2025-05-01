from enum import Enum
from typing import Dict

class Task(Enum):
    CONVERSATION = "conversation"
    MEDICAL_QA = "medical_qa"
    SCHEDULING = "scheduling"
    # doctor-only
    SUMMARY = "summary"
    DB_QUERY = "db_query"
    IMAGE_ANALYSIS = "image_analysis"

def classify_patient_intent(state):
    """
    Classify the patient's input into an intent task.

    Args:
        state: The current state object containing the user input

    Returns:
        Updated state with an "intent" field set
    """
    input_text = state.get("current_input", "")
    if isinstance(input_text, dict):
        input_text = input_text.get("text", "")

    # Simple keyword-based classification
    lower_text = input_text.lower()

    # Check for scheduling keywords
    if any(keyword in lower_text for keyword in [
        "appointment", "schedule", "book", "visit", "meet", "doctor",
        "consultation", "when can i see", "available"
    ]):
        state["intent"] = Task.SCHEDULING.value

    # Check for medical Q&A keywords
    elif any(keyword in lower_text for keyword in [
        "symptoms", "treatment", "disease", "condition", "medicine",
        "diagnosis", "medical", "health", "doctor", "prescription",
        "what is", "how do", "why does", "can you explain", "tell me about"
    ]):
        state["intent"] = Task.MEDICAL_QA.value

    # Check for symptom descriptions (route these to patient analysis)
    elif any(keyword in lower_text for keyword in [
        "headache", "pain", "hurt", "ache", "fever", "sick", "sore",
        "cough", "dizzy", "nausea", "tired", "exhausted", "vomit",
        "burn", "burn", "sting", "rash", "itchy", "bleeding", "blood",
        "i have a", "i've been", "i am feeling", "i'm feeling", "i feel",
        "severe", "moderate", "mild", "chronic", "acute", "worsening"
    ]):
        # Explicitly route symptom descriptions to conversation, which will trigger patient analysis
        state["intent"] = Task.CONVERSATION.value

    # Default to conversation
    else:
        state["intent"] = Task.CONVERSATION.value

    return state

def classify_doctor_intent(state):
    """
    Classify the doctor's input into an intent task.

    Args:
        state: The current state object containing the user input

    Returns:
        Updated state with an "intent" field set
    """
    input_text = state.get("current_input", "")
    if isinstance(input_text, dict):
        input_text = input_text.get("text", "")

    # Simple keyword-based classification
    lower_text = input_text.lower()

    # Check for doctor-specific capabilities
    if any(keyword in lower_text for keyword in [
        "summarize", "summary", "overview", "report", "patient history"
    ]):
        state["intent"] = Task.SUMMARY.value

    elif any(keyword in lower_text for keyword in [
        "database", "query", "find patient", "search records", "patient data"
    ]):
        state["intent"] = Task.DB_QUERY.value

    elif any(keyword in lower_text for keyword in [
        "image", "scan", "xray", "x-ray", "mri", "ct scan", "analyze image"
    ]):
        state["intent"] = Task.IMAGE_ANALYSIS.value

    # Check for scheduling keywords
    elif any(keyword in lower_text for keyword in [
        "appointment", "schedule", "book", "visit", "meet", "patient",
        "consultation", "availability", "calendar"
    ]):
        state["intent"] = Task.SCHEDULING.value

    # Check for medical Q&A keywords
    elif any(keyword in lower_text for keyword in [
        "research", "symptoms", "treatment", "disease", "condition", "medicine",
        "diagnosis", "medical", "health", "prescription", "literature"
    ]):
        state["intent"] = Task.MEDICAL_QA.value

    # Default to conversation
    else:
        state["intent"] = Task.CONVERSATION.value

    return state
