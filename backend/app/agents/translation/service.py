from transformers import MarianMTModel, MarianTokenizer
from lingua import Language, LanguageDetectorBuilder

class TranslationService:
    # Map lingua language enum to ISO code
    LINGUA_TO_ISO = {
        Language.ENGLISH: "en",
        Language.FRENCH: "fr",
        Language.ARABIC: "ar",
        # Add more as needed
    }

    def __init__(self, model_name: str):
        # Initialize the tokenizer and model for the given model name.
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)

        # Initialize lingua detector with common languages
        languages = [Language.ENGLISH, Language.FRENCH, Language.ARABIC]
        self.detector = LanguageDetectorBuilder.from_languages(*languages).build()

    def translate(self, text: str, target_language: str) -> str:
        """
        Translates the given text.
        (target_language is already implied by the model in this design.)
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        translated_tokens = self.model.generate(**inputs)
        translated_text = self.tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        return translated_text

    def detect_language(self, text: str) -> str:
        try:
            detected_language = self.detector.detect_language_of(text)
            if detected_language:
                return self.LINGUA_TO_ISO.get(detected_language, "unknown")
            return "unknown"
        except Exception as e:
            print(f"Language detection error: {e}")
            return "unknown"
