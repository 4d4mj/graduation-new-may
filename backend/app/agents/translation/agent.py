from app.agents.agent import Agent
from app.services.translation_service import TranslationService
from lingua import Language, LanguageDetectorBuilder
import time
import requests

class TranslationAgent(Agent):
    # Mapping of (source_language, target_language) to model names.
    MODEL_MAP = {
        ("en", "ar"): "Helsinki-NLP/opus-mt-en-ar",
        ("ar", "en"): "Helsinki-NLP/opus-mt-ar-en",
        ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
        ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en",
        # You can add more pairs as needed.
    }

    # Map lingua language enum to ISO code
    LINGUA_TO_ISO = {
        Language.ENGLISH: "en",
        Language.FRENCH: "fr",
        Language.ARABIC: "ar",
        # Add more as needed
    }

    # Map ISO code to lingua language enum
    ISO_TO_LINGUA = {v: k for k, v in LINGUA_TO_ISO.items()}

    def __init__(self):
        # Cache TranslationService instances for each language pair.
        self.services = {}
        for lang_pair, model_name in self.MODEL_MAP.items():
            self.services[lang_pair] = TranslationService(model_name)

        # Initialize lingua detector with languages we support
        languages = [lang for lang in self.ISO_TO_LINGUA.values()]
        self.detector = LanguageDetectorBuilder.from_languages(*languages).build()

    def detect_language(self, text: str) -> str:
        try:
            # Use lingua for more accurate detection
            detected_language = self.detector.detect_language_of(text)
            if detected_language:
                return self.LINGUA_TO_ISO.get(detected_language, "unknown")
            return "unknown"
        except Exception as e:
            print(f"Language detection error: {e}")
            return "unknown"

    def handle(self, text: str, target_language: str = "en") -> str:
        source_language = self.detect_language(text)

        print(f"Detected language: {source_language} for text: {text[:30]}...")

        # If the text is already in the target language, return it as-is.
        if source_language == target_language:
            return text

        lang_pair = (source_language, target_language)
        # If no model is available for this pair, you might either return an error or a fallback.
        if lang_pair not in self.services:
            return f"[Translation unavailable for {source_language} to {target_language}]"

        translation_service = self.services[lang_pair]
        return translation_service.translate(text, target_language)

# Convenience function for ease of use.
def translate(text: str, target_language: str = "en") -> str:
    agent = TranslationAgent()
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return agent.handle(text, target_language=target_language)
        except (requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError) as e:
            if attempt < max_retries - 1:
                print(f"Retry {attempt+1}/{max_retries} after error: {e}")
                time.sleep(1)
            else:
                return f"[Translation error: {str(e)}]"
