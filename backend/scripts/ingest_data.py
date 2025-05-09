import os
import argparse
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterable

# Ensure app path is discoverable
import sys

import re

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.sql import text

APP_DIR = Path(__file__).resolve().parent.parent  # Get the /app directory path
sys.path.insert(0, str(APP_DIR))

# --- Core Components ---
from app.config.settings import settings as app_settings
from app.db.base import get_engine as create_async_engine_from_url
from app.config.agent import settings as agent_settings

# Import PGVector specifically for patching
from langchain_postgres.vectorstores import PGVector
from app.tools.rag.vector_store import (
    initialize_vector_store,
    add_documents_to_vector_store,
    get_vector_store,
)
from app.tools.rag.document_processor import MedicalDocumentProcessor
from app.core.models import get_embedding_model

from typing import Optional

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
async def run_ingestion(args, existing_engine: Optional[AsyncEngine] = None):
    """Main asynchronous function to initialize, find files, process, and ingest."""
    logger.info("Starting ingestion process...")
    engine = existing_engine
    engine_created_here = False

    try:
        # 1. Create DB Engine
        if not engine:
            logger.info("Creating DB engine for ingestion...")
            if not app_settings.database_url:
                logger.critical("DATABASE_URL not found in settings. Exiting.")
                return
            engine = await create_async_engine_from_url(str(app_settings.database_url))
            engine_created_here = True
            logger.info("DB engine created.")
        else:
            logger.info("Using preexisting engine from ingestion")

        if not engine:
            logger.critical("DB Engine is not available. Exiting.")
            return

        # 2. Ensure Embedding Model is Ready
        logger.info("Ensuring embedding model is ready...")
        if not get_embedding_model():
            logger.critical("Failed to initialize embedding model. Exiting.")
            if engine_created_here:
                await engine.dispose()
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
            if engine_created_here:  # Dispose if created here
                await engine.dispose()
            return
        logger.info("Vector store connection ready.")

        # 4. Initialize Document Processor
        processor = MedicalDocumentProcessor()
        logger.info("Document processor ready.")

        # 5. Find Files
        base_data_dir = Path("/app/data")
        files_to_process: List[Path] = []
        target_desc = ""

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
            if engine_created_here:
                await engine.dispose()  # Cleanup
            return

        if not files_to_process:
            logger.warning(f"No processable files found in {target_desc}.")
            if engine_created_here:
                await engine.dispose()  # Cleanup
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
                # --- METADATA EXTRACTION
                base_metdata = {}
                filename_lower = file_path.stem.lower()

                # determine document type based on keywords
                if "drug" in filename_lower or "bnf" in filename_lower:
                    base_metdata["document_type"] = "drug_reference"
                elif "guideline" in filename_lower or "nice" in filename_lower:
                    base_metdata["document_type"] = "clinical_guideline"
                elif "anatomy" in filename_lower:
                    base_metdata["document_type"] = "anatomy_textbook"
                elif "physiology" in filename_lower:
                    base_metdata["document_type"] = "physiology_textbook"
                elif "medicine" in filename_lower:
                    base_metdata["document_type"] = "internal_medicine_textbook"
                elif "terminology" in filename_lower:
                    base_metdata["document_type"] = "medical_terminology"
                else:
                    base_metdata["document_type"] = "medical_text"

                cleaned_title = file_path.stem.replace("_", " ")
                cleaned_title = re.sub(r"\s+", " ", cleaned_title).strip()
                base_metdata["title"] = cleaned_title

                base_metdata["source_file"] = file_path.name

                logger.info(f"extracted metadata for {file_path.name}: {base_metdata}")
                # ---END METADATA EXTRACTION

                loaded_docs = load_documents(file_path)
                if not loaded_docs:
                    logger.warning(
                        f"No content could be loaded from {file_path.name}. Skipping."
                    )
                    failed_files += 1
                    continue

                all_chunks_for_file: List[Document] = []
                for doc_part in loaded_docs:
                    combined_metadata = {**base_metdata, **doc_part.metadata}

                    chunks = processor.process_document(
                        content=doc_part.page_content, metadata=combined_metadata
                    )

                    if chunks:
                        if (
                            all_chunks_for_file == []
                        ):  # Log only for the first chunk of the file
                            logger.debug(
                                f"First chunk metadata sample: {chunks[0].metadata}"
                            )
                        all_chunks_for_file.extend(chunks)
                    else:
                        logger.warning(
                            f"No chunks generated for a part of {file_path.name} metadata: {combined_metadata.get('page_number', 'N/A')}"
                        )
                # Add chunks to vector store
                if all_chunks_for_file:
                    logger.info(
                        f"Adding {len(all_chunks_for_file)} chunks from {file_path.name} to vector store..."
                    )
                    if vector_store_instance:
                        await add_documents_to_vector_store(
                            all_chunks_for_file, store_instance=vector_store_instance
                        )
                        total_chunks_added += len(all_chunks_for_file)
                        successful_files += 1
                    else:
                        logger.error(
                            f"Skipping addtition for {file_path.name} as vector store instance is invalid"
                        )
                        failed_files += 1
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
        if engine_created_here and engine:
            logger.info("Disposing of database engine created by run_ingestion...")
            await engine.dispose()
            logger.info("Database engine disposed.")
        elif existing_engine:
            logger.info(
                "Skipping engine disposal in run_ingestion as it was passed in."
            )

    async def clear_collection_if_needed(engine: AsyncEngine, collection_name: str):
        if not parsed_args.clear:
            return
        logger.warning(f"Clearing collection '{collection_name}' as requested...")
        async with engine.connect() as connection:
            async with connection.begin():
                try:
                    await connection.execute(
                        text(
                            f"DELETE FROM langchain_pg_embedding WHERE collection_id = (SELECT uuid FROM langchain_pg_collection WHERE name = :coll_name)"
                        ),
                        {"coll_name": collection_name},
                    )
                    await connection.execute(
                        text(
                            "DELETE FROM langchain_pg_collection WHERE name = :coll_name"
                        ),
                        {"coll_name": collection_name},
                    )
                    logger.info(
                        f"Successfully cleared data for collection '{collection_name}'."
                    )
                except Exception as e:
                    logger.error(
                        f"Error clearing collection '{collection_name}': {e}. Manual cleanup might be needed.",
                        exc_info=True,
                    )
                    raise

    async def run_ingestion_with_clear(args):
        """Wrapper to handle engine creation, optional clearing, and final disposal."""
        engine = None  # Initialize engine variable for the finally block
        try:
            # 1. Create engine *once*
            logger.info("Creating DB engine (wrapper)...")
            if not app_settings.database_url:
                logger.critical("DATABASE_URL not found. Exiting.")
                return
            engine = await create_async_engine_from_url(str(app_settings.database_url))
            logger.info("DB engine created (wrapper).")

            # 2. Call clear logic (uses the created engine)
            collection_name_to_clear = (
                agent_settings.rag.vector_collection_name
            )  # Get collection name
            await clear_collection_if_needed(engine, collection_name_to_clear)

            # 3. Call the main ingestion logic, passing the engine
            await run_ingestion(args, engine)  # <--- Pass the engine here

        except Exception as e:
            # Catch errors from clearing or ingestion
            logger.critical(
                f"Critical error during wrapped ingestion: {e}", exc_info=True
            )
        finally:
            # 4. Dispose the engine created by *this* wrapper function
            if engine:
                logger.info("Disposing DB engine (wrapper)...")
                await engine.dispose()
                logger.info("DB engine disposed (wrapper).")


# --- Script Entry Point ---
if __name__ == "__main__":
    # --- Make sure this block is exactly like this ---
    parser = argparse.ArgumentParser(  # Use argparse.ArgumentParser
        description="Ingest documents into RAG vector store. Paths are relative to the data directory mounted inside the container (/app/data)."
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="medical_documents",
        help="Path to the directory relative to /app/data containing documents to ingest. Default: 'medical_documents'",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="clear the existing collection before ingesting",
    )

    parsed_args = parser.parse_args()

    # Run the main async function
    asyncio.run(run_ingestion(parsed_args))
