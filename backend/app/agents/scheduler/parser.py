# app/agents/scheduler/parser.py
from typing import TypedDict, Dict, Any, Optional
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from app.config.settings import settings
import json
import logging

logger = logging.getLogger(__name__)

class AppointmentRequest(TypedDict):
    """Schema for parsed appointment booking requests"""
    patient_id: int
    doctor_id: int
    day: str  # YYYY-MM-DD format
    time_preference: Optional[str]  # "morning", "afternoon", "evening", or a specific time


def create_appointment_parser(llm=None):
    """Create a parser for appointment scheduling requests"""
    if llm is None:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-04-17",
            api_key=settings.google_api_key,
            temperature=0.1
        )

    appointment_schema = {
        "patient_id": "integer representing patient ID in database",
        "doctor_id": "integer representing doctor ID in database",
        "day": "YYYY-MM-DD format date string",
        "time_preference": "optional string with 'morning', 'afternoon', 'evening', or specific time (e.g., '14:00')"
    }

    parser_template = """
    Extract structured information about an appointment request from the user's message.
    The message might contain references to scheduling an appointment.

    USER MESSAGE: {user_message}

    USER CONTEXT:
    - User ID (patient_id): {user_id}
    - Default doctor ID if none specified: 1

    OUTPUT SCHEMA:
    ```json
    {
        "patient_id": integer representing patient ID in database,
        "doctor_id": integer representing doctor ID in database,
        "day": "YYYY-MM-DD format date string",
        "time_preference": "optional string with 'morning', 'afternoon', 'evening', or specific time (e.g., '14:00')"
    }
    ```

    If the information is missing or unclear, use these defaults:
    - patient_id: Use the provided user_id
    - doctor_id: 1 (default general practitioner)
    - day: If not specified, use tomorrow's date
    - time_preference: If not specified, use "morning"

    Return ONLY the JSON object with no additional text.
    """

    prompt = ChatPromptTemplate.from_template(parser_template)

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    def parse_appointment_request(user_message: str, user_id: int = 1) -> AppointmentRequest:
        """Parse an appointment request from a user message"""
        try:
            result = chain.invoke({"user_message": user_message, "user_id": user_id})
            # Clean up any potential formatting issues
            json_str = result.strip().replace("```json", "").replace("```", "").strip()
            parsed_data = json.loads(json_str)
            logger.info(f"Successfully parsed appointment request: {parsed_data}")
            return parsed_data
        except Exception as e:
            logger.error(f"Error parsing appointment request: {e}")
            # Return default values if parsing fails
            return {
                "patient_id": user_id,
                "doctor_id": 1,  # Default doctor
                "day": "2025-05-03",  # Tomorrow
                "time_preference": "morning"
            }

    return parse_appointment_request
