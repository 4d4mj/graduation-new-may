# app/config/provider/ollama.py
import os
from langchain_ollama import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings

class OllamaProvider:
    def __init__(
        self,
        model_name: str,
        embed_model_name: str,
        base_url: str,
        temperature: float = 0.3,
        num_predict: int | None = None,
    ) -> None:
        """
        Keep just the model names and defaults here;
        actual HTTP host/port come from OLLAMA_BASE_URL env var.
        """
        self.model_name = model_name
        self.embed_model_name = embed_model_name
        self.default_temp = temperature
        self.default_predict = num_predict
        self.base_url = base_url.rstrip("/")

    @classmethod
    def from_env(cls) -> "OllamaProvider":
        return cls(
            model_name=os.getenv("OLLAMA_CHAT_MODEL", "llama3"),
            embed_model_name=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:v1"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=float(os.getenv("OLLAMA_TEMPERATURE", 0.3)),
            num_predict=None,
        )

    def chat(self, *, temperature: float | None = None, num_predict: int | None = None):
        """
        Each call returns a fresh ChatOllama instance,
        configured with the desired sampling params.
        """
        return ChatOllama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=temperature if temperature is not None else self.default_temp,
            num_predict=num_predict if num_predict is not None else self.default_predict,
        )

    def embedder(self):
        """
        Return a new OllamaEmbeddings instance.
        """
        return OllamaEmbeddings(
            model=self.embed_model_name,
            base_url=self.base_url,
        )

# module-level singleton
PROVIDER = OllamaProvider.from_env()
