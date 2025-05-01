# tests/_stubs.py
import logging
import random
from langchain_core.messages import AIMessage
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

class DummyRAG:
    """
    A more realistic dummy RAG system that simulates retrieving information
    from a knowledge base and generating contextual responses.
    """

    def __init__(self):
        """Initialize the dummy RAG system with fake knowledge base."""
        # Simulate a knowledge base with medical information
        self.knowledge_base = {
            "headache": [
                "Headaches can be classified as primary (not caused by another condition) or secondary (caused by another condition).",
                "Common primary headaches include migraines, tension headaches, and cluster headaches.",
                "Treatment options include over-the-counter pain relievers, prescription medications, and lifestyle changes."
            ],
            "diabetes": [
                "Diabetes is a chronic condition that affects how your body turns food into energy.",
                "Type 1 diabetes is an autoimmune reaction where the body stops producing insulin.",
                "Type 2 diabetes occurs when cells become resistant to insulin.",
                "Management includes monitoring blood sugar, medication, healthy eating, and regular physical activity."
            ],
            "hypertension": [
                "Hypertension (high blood pressure) is a common condition where blood flows through blood vessels at higher than normal pressures.",
                "Risk factors include age, family history, obesity, sedentary lifestyle, and high sodium diet.",
                "Treatment may include lifestyle modifications, medication, or a combination of both."
            ],
            "vaccination": [
                "Vaccines work by training the immune system to recognize and combat pathogens.",
                "Recommended adult vaccines include influenza, Tdap, shingles, and pneumococcal vaccines.",
                "Childhood vaccination schedules are designed to protect against serious diseases."
            ],
            "eye": [
                "Common eye conditions include refractive errors, cataracts, glaucoma, and macular degeneration.",
                "Eye pain can be caused by dry eyes, infections, injuries, or underlying conditions.",
                "Regular eye examinations are important for early detection of vision problems."
            ]
        }
        logger.info("DummyRAG initialized with simulated medical knowledge base")

    def process_query(self, query) -> Dict[str, Any]:
        """
        Process a query using the simulated knowledge base.

        Args:
            query: The user's query
            *args, **kwargs: Additional arguments and keyword arguments

        Returns:
            A dictionary containing the response and confidence score
        """
        logger.info(f"DummyRAG processing query: {query}")

        # Simulate retrieval by finding matching topics in the knowledge base
        query_lower = query.lower()
        relevant_topics: List[str] = []

        for topic, facts in self.knowledge_base.items():
            if topic in query_lower:
                relevant_topics.append(topic)

        # Generate response based on retrieved information
        if relevant_topics:
            # Pick a random topic from relevant ones
            chosen_topic = random.choice(relevant_topics)
            facts = self.knowledge_base[chosen_topic]

            # Combine some facts
            num_facts = min(2, len(facts))
            selected_facts = random.sample(facts, num_facts)

            response = " ".join(selected_facts)
            confidence = random.uniform(0.75, 0.98)  # Realistic confidence score

            # Add a disclaimer
            response += " Please consult with a healthcare professional for personalized advice."
        else:
            # No relevant information found
            response = (
                "I don't have specific information about that in my knowledge base. "
                "I'd recommend discussing this with your healthcare provider for accurate information."
            )
            confidence = random.uniform(0.3, 0.5)  # Lower confidence when no relevant info

        return {
            "response": AIMessage(content=response),
            "confidence": confidence,
            "sources": [f"Simulated Medical Database: {topic.capitalize()}" for topic in relevant_topics] if relevant_topics else ["No relevant sources found"]
        }
