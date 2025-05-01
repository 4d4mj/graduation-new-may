# tests/_stubs.py
import logging
from datetime import datetime, timedelta
import random
from langchain_core.messages import AIMessage
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)

class DummyScheduler:
    """A more realistic dummy scheduler that simulates appointment scheduling."""

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize the dummy scheduler with fake data."""
        self.available_slots = {
            # Generate some fake appointment slots for the next 7 days
            (datetime.now() + timedelta(days=d)).strftime("%Y-%m-%d"):
            ["09:00", "10:30", "14:00", "16:30"]
            for d in range(1, 8)
        }
        logger.info("DummyScheduler initialized with fake appointment slots")

    def process_schedule(self, query: str, chat_history: str = "", **kwargs: Any) -> Union[AIMessage, Dict[str, Any]]:
        """
        Process a scheduling request and return a response.

        Args:
            query: The user's scheduling query
            chat_history: Recent chat history for context
            **kwargs: Additional keyword arguments

        Returns:
            Either an AIMessage or a dict with response and appointment_details
        """
        logger.info(f"DummyScheduler processing query: {query}")

        # Extract possible date mentions from the query (very simplistic)
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        next_week = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")

        # Determine if this is an appointment request
        is_appointment_request = any(word in query.lower() for word in
                                   ["schedule", "appointment", "book", "visit", "see", "doctor"])

        if is_appointment_request:
            # Select a random date and time
            date = random.choice(list(self.available_slots.keys()))
            time = random.choice(self.available_slots[date])

            # Create appointment details
            appointment_details = {
                "date": date,
                "time": time,
                "doctor": "Dr. Smith",
                "location": "Main Clinic, Room 302",
                "confirmation_code": f"APPT-{random.randint(1000, 9999)}"
            }

            response = (
                f"I've scheduled an appointment for you on {date} at {time} with {appointment_details['doctor']}. "
                f"Please arrive at {appointment_details['location']} 15 minutes before your appointment. "
                f"Your confirmation code is {appointment_details['confirmation_code']}."
            )

            return {
                "response": response,
                "appointment_details": appointment_details
            }
        else:
            # Information about scheduling
            response = (
                "I can help you schedule an appointment with one of our doctors. "
                f"We have availability from {tomorrow} to {next_week}. "
                "Would you like me to book an appointment for you? Please let me know what day and time would work best for you."
            )

            return AIMessage(content=response)
