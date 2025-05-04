import os
import argparse
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterable

# Ensure app path is discoverable
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from sqlalchemy.ext.asyncio import AsyncEngine

# --- Core Components ---
from app.config.settings import settings as app_settings
from app.db.base import get_engine as create_async_engine_from_url

# Import PGVector specifically for patching
from langchain_postgres.vectorstores import PGVector
from app.agents.rag.vector_store import (
    initialize_vector_store,
    add_documents_to_vector_store,
    get_vector_store,
)
from app.agents.rag.document_processor import MedicalDocumentProcessor
from app.core.models import get_embedding_model

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ingestion_script")


# --- Apply Monkey Patch ---
# Define the replacement function first
async def _do_nothing_vector_extension(self, *args, **kwargs):
    """A function that does nothing, to replace PGVector.acreate_vector_extension."""
    logger.debug(
        "Skipping internal acreate_vector_extension check (extension created via entrypoint)."
    )
    pass  # Explicitly do nothing


# Overwrite the method on the PGVector class *before* any instances are created
# This ensures that any internal call to this method by langchain-postgres will be bypassed.
try:
    PGVector.acreate_vector_extension = _do_nothing_vector_extension
    logger.info("Applied monkey patch to PGVector.acreate_vector_extension")
except AttributeError:
    logger.warning(
        "Could not apply monkey patch: PGVector.acreate_vector_extension not found (library structure might have changed)."
    )
# --- End Monkey Patch ---


# --- Document Loading Function ---
def load_documents(file_path: Path) -> List[Document]:
    """Loads a single file using appropriate LangChain loader."""
    ext = file_path.suffix.lower()
    loader = None
    docs = []
    try:
        if ext == ".pdf":
            loader = PyPDFLoader(str(file_path), extract_images=False)
        elif ext == ".txt":
            loader = TextLoader(str(file_path), encoding="utf-8")
        # TODO: Add other loaders if needed

        if loader:
            logger.info(f"Loading document: {file_path.name}")
            docs = loader.load()
            for doc in docs:
                doc.metadata.setdefault("source", file_path.name)
            logger.info(
                f"Loaded {len(docs)} document pages/parts from {file_path.name}"
            )
        else:
            logger.warning(
                f"No loader configured for file type: {ext} ({file_path.name})"
            )

    except FileNotFoundError:
        logger.error(f"File not found during loading: {file_path}")
    except ImportError as ie:
        logger.error(f"Missing dependency for {ext} files. Error: {ie}")
    except Exception as e:
        logger.error(f"Error loading {file_path.name}: {e}", exc_info=True)

    return docs


# --- Main Asynchronous Ingestion Logic ---
async def run_ingestion(args):
    """Main asynchronous function to initialize, find files, process, and ingest."""
    logger.info("Starting ingestion process...")
    engine = None

    try:
        # 1. Create DB Engine
        logger.info("Creating DB engine for ingestion...")
        if not app_settings.database_url:
            logger.critical("DATABASE_URL not found in settings. Exiting.")
            return
        engine = await create_async_engine_from_url(str(app_settings.database_url))
        logger.info("DB engine created.")

        # 2. Ensure Embedding Model is Ready
        logger.info("Ensuring embedding model is ready...")
        if not get_embedding_model():
            logger.critical("Failed to initialize embedding model. Exiting.")
            return
        logger.info("Embedding model ready.")

        # 3. Initialize Vector Store (Should now succeed due to patch)
        logger.info("Initializing vector store connection...")
        await initialize_vector_store(engine=engine)
        vector_store_instance = get_vector_store()
        if not vector_store_instance:
            logger.critical(
                "Vector store instance not available after initialization attempt. Exiting."
            )
            return
        logger.info("Vector store connection ready.")

        # 4. Initialize Document Processor
        processor = MedicalDocumentProcessor()
        logger.info("Document processor ready.")

        # 5. Find Files
        base_data_dir = Path("/app/data")
        files_to_process: List[Path] = []
        target_desc = ""
        # ... (file finding logic remains the same) ...
        if args.file:
            target_path = base_data_dir / args.file
            target_desc = f"file '{args.file}'"
            if target_path.is_file():
                files_to_process.append(target_path)
            else:
                logger.error(
                    f"Specified file not found inside container: {target_path}"
                )
                return
        elif args.dir:
            target_path = base_data_dir / args.dir
            target_desc = f"directory '{args.dir}'"
            if target_path.is_dir():
                supported_extensions = ["*.pdf", "*.txt"]
                logger.info(
                    f"Scanning directory {target_path} for {supported_extensions}..."
                )
                for ext in supported_extensions:
                    found = list(target_path.rglob(ext))
                    logger.info(f"Found {len(found)} files with extension {ext}")
                    files_to_process.extend(found)
            else:
                logger.error(
                    f"Specified directory not found inside container: {target_path}"
                )
                return
        else:
            logger.error("Internal error: No file or directory specified.")
            return

        if not files_to_process:
            logger.warning(f"No processable files found in {target_desc}.")
            return
        logger.info(f"Found {len(files_to_process)} files to process in {target_desc}.")

        # 6. Process and Ingest
        total_chunks_added = 0
        successful_files = 0
        failed_files = 0
        # ... (loop through files, load, process, add chunks - this part remains the same) ...
        for file_path in files_to_process:
            logger.info(f"--- Processing file: {file_path.name} ---")
            try:
                loaded_docs = load_documents(file_path)
                if not loaded_docs:
                    logger.warning(
                        f"No content could be loaded from {file_path.name}. Skipping."
                    )
                    failed_files += 1
                    continue

                all_chunks_for_file: List[Document] = []
                for doc_part in loaded_docs:
                    chunks = processor.process_document(
                        content=doc_part.page_content, metadata=doc_part.metadata
                    )
                    if chunks:
                        all_chunks_for_file.extend(chunks)
                    else:
                        logger.warning(
                            f"No chunks generated for a part of {file_path.name}"
                        )

                if all_chunks_for_file:
                    logger.info(
                        f"Adding {len(all_chunks_for_file)} chunks from {file_path.name} to vector store..."
                    )
                    await add_documents_to_vector_store(
                        all_chunks_for_file, store_instance=vector_store_instance
                    )
                    total_chunks_added += len(all_chunks_for_file)
                    successful_files += 1
                else:
                    logger.warning(
                        f"No processable chunks generated for {file_path.name}. Not adding to store."
                    )
                    if loaded_docs:
                        failed_files += 1

            except Exception as file_proc_err:
                logger.error(
                    f"Unexpected error processing file {file_path.name}: {file_proc_err}",
                    exc_info=True,
                )
                failed_files += 1

        # 7. Log Summary
        logger.info("--- Ingestion Summary ---")
        logger.info(f"Successfully processed files: {successful_files}")
        logger.info(f"Failed files: {failed_files}")
        # Safely access collection name for logging
        collection_name_log = (
            vector_store_instance.collection_name
            if vector_store_instance
            else "UNKNOWN"
        )
        logger.info(
            f"Total chunks added to collection '{collection_name_log}': {total_chunks_added}"
        )

    except Exception as e:
        logger.critical(
            f"A critical error occurred during the ingestion process: {e}",
            exc_info=True,
        )
    finally:
        # 8. Cleanup
        if engine:
            logger.info("Disposing of database engine...")
            await engine.dispose()
            logger.info("Database engine disposed.")
        logger.info("Ingestion script finished.")


# --- Script Entry Point ---
if __name__ == "__main__":
    # --- Make sure this block is exactly like this ---
    parser = argparse.ArgumentParser(  # Use argparse.ArgumentParser
        description="Ingest documents into RAG vector store. Paths are relative to the data directory mounted inside the container (/app/data)."
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to a single file relative to the data directory (e.g., 'my_doc.pdf').",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=".",  # Default to processing the root of the data directory
        help="Path to a directory relative to the data directory (e.g., '.' for all, or 'subdir_name'). Default: '.'",
    )
    # Add other arguments like --clear later if needed
    # parser.add_argument('--clear', action='store_true', help='Clear the existing collection before ingesting.')

    parsed_args = parser.parse_args()
    # ----------------------------------------------------

    # Basic validation
    if parsed_args.file and parsed_args.dir != ".":
        parser.error(
            "Cannot specify both --file and a specific --dir. Use --dir . or just --file."
        )
    # This check might be redundant if default is always '.', but safe
    # if not parsed_args.file and not parsed_args.dir:
    #     parser.error("Internal error: No file or directory target.")

    # Run the main async function
    asyncio.run(run_ingestion(parsed_args))
