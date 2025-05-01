# agents/web_search/core.py
import logging
import random
from datetime import datetime
from langchain_core.messages import AIMessage
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

class DummyWebSearch:
    """
    A more realistic dummy web search agent that simulates retrieving and
    synthesizing information from web sources.
    """

    def __init__(self):
        """Initialize the dummy web search with simulated web sources."""
        # Simulate web search results for different medical topics
        self.simulated_web_results = {
            "headache": [
                {
                    "title": "Mayo Clinic: Headache - Symptoms and causes",
                    "url": "https://www.mayoclinic.org/diseases-conditions/headache/symptoms-causes/syc-20370200",
                    "snippet": "Most headaches aren't caused by a serious illness. Common headache triggers include stress, certain foods, and changes in sleep patterns."
                },
                {
                    "title": "Cleveland Clinic: Headaches",
                    "url": "https://my.clevelandclinic.org/health/diseases/9639-headaches-in-adults",
                    "snippet": "Headaches can be primary, with their stand-alone cause, or secondary, when caused by another medical condition. Tension headaches are the most common type."
                }
            ],
            "diabetes": [
                {
                    "title": "CDC: Diabetes Basics",
                    "url": "https://www.cdc.gov/diabetes/basics/index.html",
                    "snippet": "Diabetes is a chronic health condition that affects how your body turns food into energy. With diabetes, your body doesn't make enough insulin or can't use it as well as it should."
                },
                {
                    "title": "American Diabetes Association: Overview",
                    "url": "https://www.diabetes.org/diabetes",
                    "snippet": "Diabetes causes more deaths per year than breast cancer and AIDS combined. Having diabetes nearly doubles your chance of having a heart attack."
                }
            ],
            "eye pain": [
                {
                    "title": "WebMD: What's Causing My Eye Pain?",
                    "url": "https://www.webmd.com/eye-health/eye-pain-causes-symptoms-diagnosis-treatment",
                    "snippet": "Eye pain has many causes including injury, infection, inflammation, and glaucoma. Some causes of eye pain can be serious and need immediate medical attention."
                },
                {
                    "title": "American Academy of Ophthalmology: Eye Pain",
                    "url": "https://www.aao.org/eye-health/symptoms/eye-pain",
                    "snippet": "Pain in or around the eye can be a symptom of numerous eye conditions. It's important to see an ophthalmologist to determine the cause and receive proper treatment."
                }
            ],
            "vaccination": [
                {
                    "title": "WHO: Vaccines and Immunization",
                    "url": "https://www.who.int/health-topics/vaccines-and-immunization",
                    "snippet": "Vaccines train your immune system to create antibodies, just as it does when it's exposed to a disease. Vaccines contain only killed or weakened forms of germs."
                },
                {
                    "title": "CDC: Vaccines & Immunizations",
                    "url": "https://www.cdc.gov/vaccines/index.html",
                    "snippet": "Vaccines help protect you and your family against serious diseases. Check the CDC's vaccine schedules to make sure you're up to date."
                }
            ]
        }

        # Default search results for general or unknown queries
        self.default_results = [
            {
                "title": "MedlinePlus: Health Information",
                "url": "https://medlineplus.gov/",
                "snippet": "MedlinePlus is an online health information resource for patients and their families and friends. It offers reliable, up-to-date health information."
            },
            {
                "title": "CDC: Health Information for Consumers",
                "url": "https://www.cdc.gov/",
                "snippet": "CDC provides credible health information on diseases, conditions, and other health topics for the general public."
            }
        ]

        logger.info("DummyWebSearch initialized with simulated web search results")

    def process_web_search_results(self, query: str) -> Union[AIMessage, Dict[str, Any]]:
        """
        Process a web search query and return synthesized information.

        Args:
            query: The search query
            **kwargs: Additional keyword arguments

        Returns:
            An AIMessage containing synthesized information from "web sources"
        """
        logger.info(f"DummyWebSearch processing query: {query}")

        # Identify relevant topic based on query keywords
        query_lower = query.lower()

        # Find matching topics in our simulated database
        search_results = []
        for topic, results in self.simulated_web_results.items():
            if topic in query_lower:
                search_results.extend(results)

        # Use default results if no specific matches found
        if not search_results:
            search_results = self.default_results

        # Select a subset of results
        if len(search_results) > 2:
            selected_results = random.sample(search_results, 2)
        else:
            selected_results = search_results

        # Synthesize information from the "web search"
        current_date = datetime.now().strftime("%Y-%m-%d")

        intro = f"Based on the latest web search results (as of {current_date}), I found the following information: "

        content_parts = []
        for result in selected_results:
            content_parts.append(f"According to {result['title']} ({result['url']}): {result['snippet']}")

        conclusion = "Please note that this information is for educational purposes only and shouldn't replace professional medical advice."

        full_response = intro + " " + " ".join(content_parts) + " " + conclusion

        return AIMessage(content=full_response)
