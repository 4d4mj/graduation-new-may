import logging
from typing import Optional, Dict
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_cohere import CohereRerank
from langchain_core.embeddings import Embeddings
from app.config.settings import settings
from app.config.agent import settings as agent_settings

logger = logging.getLogger(__name__)

_model_cache: Dict[str, object] = {}


def get_llm(model_key: str = "default") -> Optional[ChatGoogleGenerativeAI]:
    """Gets a specific ChatGoogleGenerativeAI Instance"""
    global _model_cache
    cache_key = f"llm_{model_key}"
    if cache_key not in _model_cache:
        logger.info(f"Initializing LLM for key: '{model_key}'")
        try:
            model_name_map = {
                "default": "gemini-2.5-flash-preview-04-17",
                "router": "gemini-2.5-flash-preview-04-17",
                "patient_analyzer": "gemini-2.5-flash-preview-04-17",
                "rag_generator": "gemini-2.5-flash-preview-04-17",
            }
            model_name = model_name_map.get(model_key, model_name_map["default"])

            instance = ChatGoogleGenerativeAI(
                model=model_name, api_key=settings.google_api_key
            )
            _model_cache[cache_key] = instance
            logger.info(f"Initialized LLM '{model_name}' for key '{model_key}'")
        except Exception as e:
            logger.error(
                f"Failed to initialize LLM for key '{model_key}': {e}", exc_info=True
            )
            _model_cache = None
    return _model_cache.get(cache_key)


def get_embedding_model() -> Optional[Embeddings]:
    """Gets the GoogleGenerativeAIEmbeddings instance"""
    global _model_cache
    cache_key = "embeddings"
    if cache_key not in _model_cache:
        logger.info("Initializing Embedding model...")
        try:
            model_name = (
                agent_settings.rag.embedding_model_name or "models/text-embedding-004"
            )
            instance = GoogleGenerativeAIEmbeddings(
                model=model_name, api_key=settings.google_api_key
            )
            _model_cache[cache_key] = instance
            logger.info(f"initializing embedding model '{model_name}'")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}", exc_info=True)
            _model_cache[cache_key] = None
    return _model_cache.get(cache_key)


def get_reranker() -> Optional[CohereRerank]:
    """Get the CohereRereank instance"""
    global _model_cache
    cache_key = "reranker"
    if cache_key not in _model_cache:
        logger.info("Initializing Reranker...")
        if settings.cohere_api_key:
            try:
                instance = CohereRerank(
                    model=agent_settings.rag.reranker or "rerank-v3.5",
                    top_n=agent_settings.rag.reranker_top_k or 3,
                )
                _model_cache[cache_key] = instance
                logger.info(f"initialized reranker model: '{instance.model}'")
            except Exception as e:
                logger.error(f"Failed to initialize Reranker: {e}", exc_info=True)
                _model_cache[cache_key] = None
        else:
            logger.warning("COHERE_API_KEY is not set. Reranker not initialized")
            _model_cache[cache_key] = None
    return _model_cache.get(cache_key)


def clear_model_cache():
    """Clears the model cache"""
    global _model_cache
    _model_cache = {}
