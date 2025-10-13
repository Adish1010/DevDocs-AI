"""
Document Processor Module
-------------------------
This module handles the ingestion and preprocessing of documents for the RAG system.
It combines LangChain's robust document loaders with custom metadata enrichment
and code-aware enhancements for technical documentation.
"""

import os
import logging
from datetime import datetime
from typing import List, Dict, Any
from config import CHUNK_SIZE, CHUNK_OVERLAP, SUPPORTED_EXTENSIONS

# LangChain imports
from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Custom Exception
# ---------------------------------------------------------------------------
class DocumentProcessingError(Exception):
    """Raised when document processing fails."""
    pass


# ---------------------------------------------------------------------------
# Smart Document Processor
# ---------------------------------------------------------------------------
class SmartDocumentProcessor:
    """
    Combines LangChain document loaders with custom metadata enrichment and
    code-aware enhancements for processing technical documentation.
    """

    def __init__(self) -> None:
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        )
        self.supported_extensions = SUPPORTED_EXTENSIONS

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    def process_file(self, file_path: str, document_name: str | None = None) -> List[Dict[str, Any]]:
        """
        Processes a single document:
            1. Loads it using the appropriate LangChain loader
            2. Adds custom metadata
            3. Splits text into manageable chunks
            4. Enhances for technical content
            5. Returns standardized chunks ready for vector storage
        """
        document_name = document_name or os.path.basename(file_path)
        logger.info(f"Processing document: {document_name}")

        try:
            raw_documents = self._load_documents(file_path)
            enriched_docs = self._add_custom_metadata(raw_documents, document_name)
            chunks = self.text_splitter.split_documents(enriched_docs)
            enhanced_chunks = self._enhance_for_technical_docs(chunks)
            standardized_chunks = self._standardize_chunks(enhanced_chunks)

            logger.info(f"Successfully processed {document_name}: {len(standardized_chunks)} chunks")
            return standardized_chunks

        except Exception as e:
            logger.exception(f"Failed to process {document_name}: {e}")
            raise DocumentProcessingError(str(e)) from e

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------
    def _load_documents(self, file_path: str) -> List[Document]:
        """Loads document using the appropriate LangChain loader."""
        file_extension = os.path.splitext(file_path)[1].lower()

        try:
            if file_extension == ".pdf":
                loader = PyPDFLoader(file_path)
            elif file_extension == ".txt":
                loader = TextLoader(file_path, encoding="utf-8")
            else:
                logger.warning(f"Unknown extension {file_extension}; using UnstructuredFileLoader.")
                loader = UnstructuredFileLoader(file_path)

            return loader.load()

        except Exception as e:
            logger.error(f"Document loading failed for {file_path}: {e}")
            raise

    def _add_custom_metadata(self, documents: List[Document], document_name: str) -> List[Document]:
        """Adds rich metadata for improved retrieval and traceability."""
        for doc in documents:
            # Copy existing metadata to avoid mutation issues
            doc.metadata = {**doc.metadata}

            doc.metadata.update(
                {
                    "document_name": document_name,
                    "processed_time": datetime.now().isoformat(),
                    "processor_version": "1.0.0",
                    "chunk_strategy": "recursive_character",
                }
            )

            # Detect content type
            preview = doc.page_content[:200].lower()
            if any(k in preview for k in ["api", "endpoint", "route"]):
                doc.metadata["content_type"] = "api_documentation"
            elif any(k in preview for k in ["def ", "class ", "function ", "import "]):
                doc.metadata["content_type"] = "code_example"
            elif any(k in preview for k in ["install", "setup", "configuration"]):
                doc.metadata["content_type"] = "setup_guide"
            else:
                doc.metadata["content_type"] = "general_documentation"

        return documents

    def _enhance_for_technical_docs(self, chunks: List[Document]) -> List[Document]:
        """Adds programming-specific context and importance scoring."""
        enhanced_chunks = []

        for idx, chunk in enumerate(chunks):
            content = chunk.page_content
            chunk.metadata = {**chunk.metadata}  # ensure metadata isolation

            # Detect code snippets
            has_code = self._contains_code_snippets(content)
            chunk.metadata["has_code"] = has_code
            if has_code:
                chunk.metadata["code_language"] = self._detect_code_language(content)

            # Importance scoring
            if self._is_important_section(content):
                chunk.metadata["importance"] = "high"
                chunk.metadata["importance_score"] = 1.0
            else:
                chunk.metadata["importance"] = "medium"
                chunk.metadata["importance_score"] = 0.5

            chunk.metadata["chunk_sequence"] = idx
            enhanced_chunks.append(chunk)

        return enhanced_chunks

    def _standardize_chunks(self, chunks: List[Document]) -> List[Dict[str, Any]]:
        """Converts processed chunks into a consistent dictionary format."""
        standardized: List[Dict[str, Any]] = []
        for chunk in chunks:
            standardized.append(
                {
                    "content": chunk.page_content,
                    "metadata": chunk.metadata,
                }
            )
        return standardized

    # -----------------------------------------------------------------------
    # Utility methods
    # -----------------------------------------------------------------------
    def _contains_code_snippets(self, text: str) -> bool:
        """Detects if text contains programming code."""
        code_indicators = [
            "def ",
            "class ",
            "function ",
            "import ",
            "export ",
            "const ",
            "let ",
            "var ",
            "```",
        ]
        return any(ind in text for ind in code_indicators)

    def _detect_code_language(self, text: str) -> str:
        """Infers likely programming language."""
        text_lower = text.lower()
        if "def " in text_lower or "import " in text_lower:
            return "python"
        if "function " in text_lower or "const " in text_lower:
            return "javascript"
        if "public class " in text_lower or "void " in text_lower:
            return "java"
        return "unknown"

    def _is_important_section(self, text: str) -> bool:
        """Detects important sections in documentation."""
        important_keywords = ["warning", "important", "note:", "caution", "deprecated", "security"]
        return any(k in text.lower() for k in important_keywords)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------
def process_document(file_path: str, document_name: str | None = None) -> List[Dict[str, Any]]:
    """Convenience wrapper for quick document processing."""
    processor = SmartDocumentProcessor()
    return processor.process_file(file_path, document_name)
