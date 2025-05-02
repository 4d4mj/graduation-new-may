# tests/_stubs.py
import logging
from datetime import datetime, timedelta
import random
from langchain_core.messages import AIMessage
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)

class DummyScheduler:
    """A more realistic dummy scheduler that simulates appointment scheduling."""

    def __init__(self):
        """Initialize the dummy scheduler with fake data."""
        self.available_slots = {
            # Generate some fake appointment slots for the next 7 days
            (datetime.now() + timedelta(days=d)).strftime("%Y-%m-%d"):
            ["09:00", "10:30", "14:00", "16:30"]
            for d in range(1, 8)
        }
        self.previous_offer = None  # Track the last appointment offered
        logger.info("DummyScheduler initialized with fake appointment slots")

    def process_schedule(self, query) -> Union[AIMessage, Dict[str, Any]]:
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

        # Check for confirmation of previous offer
        is_confirmation = False
        confirmation_words = ["yes", "yeah", "sure", "okay", "ok", "great", "sounds good", "perfect", "please"]

        if self.previous_offer and any(word in query.lower() for word in confirmation_words):
            is_confirmation = True
            logger.info("Detected confirmation of previous appointment offer")
            appointment_details = self.previous_offer

            # Create confirmation response
            response = (
                f"Great! I've confirmed your appointment for {appointment_details['date']} at {appointment_details['time']} "
                f"with {appointment_details['doctor']}. Please arrive at {appointment_details['location']} "
                f"15 minutes before your appointment. Your confirmation code is {appointment_details['confirmation_code']}."
            )

            return {
                "response": response,
                "appointment_details": appointment_details,
                "status": "confirmed"
            }

        # Determine if this is an explicit appointment request
        is_explicit_request = any(word in query.lower() for word in
                                ["schedule", "appointment", "book", "visit", "see", "doctor"])

        # Check if this is an implicit request (symptom description or severity indicators)
        symptom_words = ["pain", "ache", "hurt", "headache", "migraine", "severe", "worried", "worrying", "bad"]
        severity_indicators = ["7", "8", "9", "10", "severe", "intense", "terrible", "unbearable", "worry"]

        has_symptoms = any(word in query.lower() for word in symptom_words)
        has_severity = any(indicator in query.lower() for indicator in severity_indicators)
        is_implicit_request = has_symptoms and has_severity

        # Respond with appointment if it's either an explicit or implicit request
        if is_explicit_request or is_implicit_request:
            # Select a date and time based on severity
            available_dates = list(self.available_slots.keys())
            # If severe, offer appointment sooner
            if has_severity:
                date = available_dates[0]  # Tomorrow
            else:
                date = random.choice(available_dates[1:3])  # Within next few days

            time = random.choice(self.available_slots[date])

            # Create appointment details
            appointment_details = {
                "date": date,
                "time": time,
                "doctor": "Dr. Smith",
                "location": "Main Clinic, Room 302",
                "confirmation_code": f"APPT-{random.randint(1000, 9999)}"
            }

            # Store this offer for potential confirmation in the next turn
            self.previous_offer = appointment_details

            if is_implicit_request:
                # More empathetic response for symptom-based requests
                response = (
                    f"Based on what you've described, I think it would be good to see a doctor soon. "
                    f"I can offer you an appointment on {date} at {time} with {appointment_details['doctor']}. "
                    f"The appointment will be at {appointment_details['location']}. "
                    f"Would this time work for you, or would you prefer a different day or time?"
                )
            else:
                # Standard booking confirmation
                response = (
                    f"I've scheduled an appointment for you on {date} at {time} with {appointment_details['doctor']}. "
                    f"Please arrive at {appointment_details['location']} 15 minutes before your appointment. "
                    f"Your confirmation code is {appointment_details['confirmation_code']}."
                )

            return {
                "response": response,
                "appointment_details": appointment_details,
                "status": "offered"
            }
        else:
            # Information about scheduling
            response = (
                "I can help you schedule an appointment with one of our doctors. "
                f"We have availability from {tomorrow} to {next_week}. "
                "Would you like me to book an appointment for you? Please let me know what day and time would work best for you."
            )

            # Store a generic appointment offer that will be filled in when confirmed
            self.previous_offer = {
                "date": tomorrow,
                "time": "10:30",
                "doctor": "Dr. Smith",
                "location": "Main Clinic, Room 302",
                "confirmation_code": f"APPT-{random.randint(1000, 9999)}"
            }

            return AIMessage(content=response)
