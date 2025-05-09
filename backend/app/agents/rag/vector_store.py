# backend/app/agents/rag/vector_store.py
import logging
from typing import List, Dict, Any, Optional, Iterable

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from sqlalchemy.ext.asyncio import AsyncEngine

# Use the specific PGVector integration
from langchain_postgres.vectorstores import PGVector, DistanceStrategy

# Use the centralized getter for embeddings
from app.core.models import get_embedding_model

# Use the specific RAG settings from agent config
from app.config.agent import settings as agent_settings

# Use the main app settings for the database URL
from app.config.settings import settings as app_settings

logger = logging.getLogger(__name__)

# --- Apply Monkey Patch at Module Level ---
try:

    async def _do_nothing_vector_extension_patch(self, *args, **kwargs):
        # logger.debug("Patched acreate_vector_extension called - doing nothing.")
        pass  # Explicitly do nothing

    # Overwrite the method on the PGVector class itself
    PGVector.acreate_vector_extension = _do_nothing_vector_extension_patch
    logger.info("Applied monkey patch globally to PGVector.acreate_vector_extension")
except AttributeError:
    logger.warning(
        "Could not apply monkey patch: PGVector.acreate_vector_extension not found."
    )
# ------------------------------------------

# --- PGVector Initialization ---

# Instead of a global variable, we'll make the store accessible via a function
# This allows for potential future flexibility if parameters change.
# We still cache it internally for performance.
_vector_store_cache: Dict[str, PGVector] = {}


async def initialize_vector_store(engine: AsyncEngine):
    """
    Initializes the PGVector store if not already done.
    Relies on entrypoint.sh having created the extension.
    Should be called once during application startup or lazily when first needed.
    """
    global _vector_store_cache
    collection_name = agent_settings.rag.vector_collection_name
    cache_key = f"pgvector_{collection_name}"

    if (
        cache_key in _vector_store_cache
    ):  # If already initialized (e.g. by another call)
        logger.info(
            f"PGVector store for '{collection_name}' already initialized and cached."
        )
        return _vector_store_cache[cache_key]  # Return existing instance

    logger.info(
        f"Attempting to initialize PGVector store for collection: {collection_name}"
    )
    embedding_function = get_embedding_model()
    if not embedding_function:
        logger.error("Cannot initialize PGVector: Embedding model not available.")
        raise RuntimeError("PGVector init failed: Embedding model unavailable.")
    if not app_settings.database_url:
        logger.error("Cannot initialize PGVector: DATABASE_URL not set.")
        raise RuntimeError("PGVector init failed: DATABASE_URL missing.")

        # connection_string = str(app_settings.database_url)

    try:
        # --- Corrected Call for Class Method ---
        # Pass 'embedding' positionally, others by keyword
        store = PGVector(
            connection=engine,
            embeddings=embedding_function,
            collection_name=collection_name,
            distance_strategy=DistanceStrategy.COSINE,
            use_jsonb=True,
        )

        # --- TRY A SIMPLE OPERATION TO CONFIRM IT'S WORKING ---
        # This is a bit of a hack, but can confirm table access.
        # Ideally, langchain-postgres would have a more direct "ensure_initialized_and_connected"
        try:
            # Try a dummy search or add; this might create tables if they don't exist
            # and will fail if connection or extension is bad.
            await store.asimilarity_search("test query for init", k=1)
            logger.info(
                "PGVector store: Dummy search successful during initialization."
            )
        except Exception as e:
            logger.error(
                f"PGVector store: Dummy search FAILED during initialization: {e}",
                exc_info=True,
            )
            raise RuntimeError(f"PGVector store failed post-connection check: {e}")
        # --- END SIMPLE OPERATION ---

        _vector_store_cache[cache_key] = store
        logger.info(
            f"PGVector store for collection '{collection_name}' initialized, tested and cached."
        )

        return store

    except Exception as e:
        logger.error(
            f"Failed to initialize PGVector store '{collection_name}': {e}",
            exc_info=True,
        )
        raise RuntimeError(f"PGVector initialization failed: {e}")


def get_vector_store() -> Optional[PGVector]:
    """
    Returns the initialized PGVector store instance from cache.
    Assumes initialize_vector_store() has been called previously (e.g., at startup).
    """
    collection_name = agent_settings.rag.vector_collection_name
    cache_key = f"pgvector_{collection_name}"
    store = _vector_store_cache.get(cache_key)
    if not store:
        logger.warning(
            f"get_vector_store: PGVector store '{collection_name}' cache miss. Store not initialized or init failed."
        )
    # Optionally, attempt lazy initialization here, but better to do it at startup
    # raise RuntimeError("PGVector store not initialized.")
    return store


# --- Vector Store Operations ---


async def add_documents_to_vector_store(
    documents: Iterable[Document], store_instance: Optional[PGVector] = None
):
    """
    Adds documents asynchronously to the PGVector store.

    Args:
        documents: An iterable of LangChain Document objects.
        store_instance: Optional pre-fetched store instance.
    """
    store = store_instance or get_vector_store()
    if not store:
        logger.error("Vector store not available. Cannot add documents.")
        raise ConnectionError("Vector store is not initialized or unavailable.")
    if not documents:
        logger.warning("No documents provided to add_documents_to_vector_store.")
        return []

    doc_list = list(documents)  # Convert iterable for logging length
    logger.info(
        f"Adding {len(doc_list)} documents to collection '{store.collection_name}'..."
    )
    try:
        # Use aadd_documents for async operation
        ids = await store.aadd_documents(doc_list, ids=None)  # Let PGVector handle IDs
        if ids and len(ids) == len(doc_list):
            logger.info(
                f"Successfully received {len(ids)} IDs after adding documents to vector store."
            )
        elif ids:
            logger.warning(
                f"Received {len(ids)} IDs, but expected {len(doc_list)}. Potential partial add?"
            )
        else:
            logger.error(
                "aadd_documents returned None or empty list. Addition likely failed silently."
            )
            raise ValueError("aadd_documents did not return expected IDs.")
        return ids
    except Exception as e:
        logger.error(f"Error adding documents to PGVector: {e}", exc_info=True)
        raise  # Re-raise the exception for the caller to handle


async def search_vector_store(
    query: str,
    k: int = 4,  # Default value or get from agent_settings.rag.retrieval_k
    filter: Optional[Dict[str, Any]] = None,
    store_instance: Optional[PGVector] = None,
) -> List[tuple[Document, float]]:
    """
    Performs similarity search asynchronously and returns documents with scores.

    Args:
        query: The search query string.
        k: The number of documents to return.
        filter: Optional metadata filter dictionary.
        store_instance: Optional pre-fetched store instance.

    Returns:
        A list of tuples, each containing (Document, similarity_score). Returns empty list on error.
    """
    store = store_instance or get_vector_store()
    if not store:
        logger.error("Vector store not available. Cannot perform search.")
        raise ConnectionError("Vector store is not initialized or unavailable.")

    effective_k = k  # Can override with agent_settings later if needed
    logger.debug(
        f"Performing similarity search: k={effective_k}, filter={filter}, query='{query[:50]}...'"
    )
    try:
        # Use asimilarity_search_with_score for async
        results = await store.asimilarity_search_with_score(
            query=query, k=effective_k, filter=filter
        )
        logger.debug(f"Found {len(results)} raw results from vector store.")
        return results
    except Exception as e:
        logger.error(f"Error during similarity search: {e}", exc_info=True)
        return []  # Return empty list on error
