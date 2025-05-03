import logging
from typing import List, Dict, Any
from langchain_core.documents import Document

# Use standard text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Use config for settings
from app.config.agent import settings as agent_settings

logger = logging.getLogger(__name__)


class MedicalDocumentProcessor:
    """
    Processes text content into structured document chunks with metadata
    """

    def __init__(self):
        """initializes the processor with settings"""
        self.chunk_size = agent_settings.rag.chunk_size
        self.chunk_overlap = agent_settings.rag.chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            add_start_index=True,  # to know where the chun came fromm
            separators=["\n\n", "\n", ". ", " ", ""],  # common separators
            is_separator_regex=False,
        )
        logger.info(
            f"Initialized MedicalDocumentProcessor with chunk_size={self.chunk_size}, overlap={self.chunk_overlap}"
        )

    def process_document(self, content: str, metadata: dict) -> List[Document]:
        """
        Splits raw text content into Langchain Document object

        Args:
            content (str): the full text content of the document
            metadata (dict): Base metadata (must include 'source') Should contain
                    information like filename, original source URL, etc.

        Returns:
            List[Document]:  A list of LangChain Document objects, each representing a chunk
            with inherited metadata and added 'start_index'.
        """
        if not content or not isinstance(content, str):
            logger.warning("process_document received empty or invalid content")
            return []
        if not metadata or "source" not in metadata:
            logger.error("process_document requires 'source' in metadata")
            return []
        source_name = metadata.get("source", "unknown")
        logger.debug(f"Processing documents: {source_name} length: {len(content)}")

        try:
            # 1. Create a single Document object representing the entire input text
            #    and its associated metadata
            initial_doc = Document(page_content=content, metadata=metadata)

            # 2. Use split_documents to split the initial document.
            #    This automatically handles metadata propagation and adds 'start_index'.
            processed_docs = self.text_splitter.split_documents([initial_doc])

            logger.info(
                f"Split document {source_name} into {len(processed_docs)} chunks"
            )

            if processed_docs:
                logger.debug(f"Metadata of first chunk: {processed_docs[0].metadata}")

            return processed_docs

        except Exception as e:
            logger.error(f"Failed to split document {source_name}: {e}", exc_info=True)
            return []
