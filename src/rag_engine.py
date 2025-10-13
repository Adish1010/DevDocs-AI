"""
RAG Engine Module
-----------------
Core retrieval and generation engine that powers the Intelligent Programming 
Documentation Search Engine. Combines vector search with LLM synthesis for
accurate, context-aware answers from technical documentation.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# Core dependencies
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# LLM - Using Groq for speed and cost efficiency
from groq import Groq

from config import (
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    VECTOR_DB_DIR,
    TOP_K_RESULTS,
    RELEVANCE_THRESHOLD,
    GROQ_API_KEY,
    GROQ_MODEL,
    MAX_TOKENS,
    TEMPERATURE,
    SYSTEM_PROMPT
)


# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------
class RAGEngineError(Exception):
    """Raised when RAG engine operations fail."""
    pass

class RetrievalError(RAGEngineError):
    """Raised when document retrieval fails."""
    pass

class GenerationError(RAGEngineError):
    """Raised when answer generation fails."""
    pass


# ---------------------------------------------------------------------------
# RAG Engine Core
# ---------------------------------------------------------------------------
class RAGEngine:
    """
    Intelligent RAG engine optimized for programming documentation.
    Handles vector storage, semantic search, and LLM-powered answer generation.
    """
    
    def __init__(self, collection_name: str = "programming_docs"):
        self.collection_name = collection_name
        self.embedding_model = None
        self.vector_db = None
        self.llm_client = None
        self.collection = None
        
        self._initialize_components()
        logger.info(f"RAG Engine initialized with collection: {collection_name}")

    def _initialize_components(self) -> None:
        """Initialize embedding model, vector DB, and LLM client."""
        try:
            # 1. Initialize embedding model
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            
            # 2. Initialize vector database
            self._initialize_vector_db()
            
            # 3. Initialize LLM client
            if GROQ_API_KEY:
                self.llm_client = Groq(api_key=GROQ_API_KEY)
            else:
                logger.warning("GROQ_API_KEY not found - generation will be disabled")
                
        except Exception as e:
            logger.exception(f"Failed to initialize RAG engine: {e}")
            raise RAGEngineError(f"Initialization failed: {e}") from e

    def _initialize_vector_db(self) -> None:
        """Initialize ChromaDB with persistent storage."""
        try:
            self.vector_db = chromadb.PersistentClient(
                path=VECTOR_DB_DIR,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            self.collection = self.vector_db.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Programming documentation chunks"}
            )
            
            logger.info(f"Vector DB initialized at: {VECTOR_DB_DIR}")
            
        except Exception as e:
            logger.exception(f"Vector DB initialization failed: {e}")
            raise RAGEngineError(f"Vector DB setup failed: {e}") from e

    # -----------------------------------------------------------------------
    # Document Management
    # -----------------------------------------------------------------------
    def add_documents(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Add processed document chunks to the vector database.
        
        Args:
            chunks: List of chunks from document processor
            
        Returns:
            Number of chunks successfully added
        """
        if not chunks:
            logger.warning("No chunks provided to add_documents")
            return 0
            
        try:
            # Prepare data for vector DB
            documents = []
            metadatas = []
            ids = []
            
            for i, chunk in enumerate(chunks):
                documents.append(chunk["content"])
                metadatas.append(chunk["metadata"])
                ids.append(f"chunk_{i}_{str(abs(hash(chunk['content'])))[:8]}")

            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(documents)} chunks...")
            embeddings = self.embedding_model.encode(documents).tolist()
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Successfully added {len(documents)} chunks to vector DB")
            return len(documents)
            
        except Exception as e:
            logger.exception(f"Failed to add documents: {e}")
            raise RAGEngineError(f"Document addition failed: {e}") from e

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the current collection."""
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "total_chunks": count,
                "vector_db_path": VECTOR_DB_DIR
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}

    # -----------------------------------------------------------------------
    # Core RAG Operations
    # -----------------------------------------------------------------------
    def search(self, query: str, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Perform semantic search with optional metadata filtering.
        
        Args:
            query: User's search query
            filters: Optional metadata filters for targeted search
            
        Returns:
            List of relevant chunks with scores and metadata
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()[0]
            
            # Prepare search parameters
            search_kwargs = {
                "query_embeddings": [query_embedding],
                "n_results": TOP_K_RESULTS,
            }
            
            # Add filters if provided
            if filters:
                search_kwargs["where"] = filters
            
            # Perform search
            results = self.collection.query(**search_kwargs)
            
            # Process and filter results
            relevant_chunks = self._process_search_results(results, query)
            
            logger.info(f"Search for '{query}' returned {len(relevant_chunks)} relevant chunks")
            return relevant_chunks
            
        except Exception as e:
            logger.exception(f"Search failed for query '{query}': {e}")
            raise RetrievalError(f"Search failed: {e}") from e

    def _process_search_results(self, results: Dict, query: str) -> List[Dict[str, Any]]:
        """Process and filter raw search results."""
        relevant_chunks = []
        
        if not results["documents"] or not results["documents"][0]:
            return relevant_chunks
        
        for i, (doc, metadata, distance) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )):
            # Convert distance to similarity score (Chroma uses distance, not similarity)
            similarity_score = 1.0 / (1.0 + distance)
            
            # Apply relevance threshold
            if similarity_score >= RELEVANCE_THRESHOLD:
                chunk_data = {
                    "content": doc,
                    "metadata": metadata,
                    "similarity_score": round(similarity_score, 4),
                    "distance": round(distance, 4),
                    "rank": i + 1
                }
                relevant_chunks.append(chunk_data)
        
        return relevant_chunks

    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate answer using LLM with retrieved context.
        
        Args:
            query: User's question
            context_chunks: Retrieved relevant chunks
            
        Returns:
            Generated answer with sources and confidence
        """
        if not self.llm_client:
            raise GenerationError("LLM client not initialized - check API key")
            
        if not context_chunks:
            return self._handle_no_context(query)
        
        try:
            # Prepare context from chunks
            context_text = self._prepare_context(context_chunks)
            
            # Generate answer
            response = self.llm_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Question: {query}\n\nContext: {context_text}"}
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                stream=False
            )
            
            answer = response.choices[0].message.content
            
            # Calculate confidence based on chunk similarities
            confidence = self._calculate_confidence(context_chunks)
            
            # Extract sources
            sources = self._extract_sources(context_chunks)
            
            return {
                "answer": answer,
                "sources": sources,
                "confidence": confidence,
                "chunks_used": len(context_chunks),
                "model": GROQ_MODEL
            }
            
        except Exception as e:
            logger.exception(f"Answer generation failed for query '{query}': {e}")
            raise GenerationError(f"Answer generation failed: {e}") from e

    def query(self, question: str, filters: Optional[Dict] = None) -> Dict[str, Any]:
        """
        End-to-end RAG pipeline: search + generate.
        
        Args:
            question: User's question
            filters: Optional metadata filters
            
        Returns:
            Complete RAG response with answer and metadata
        """
        logger.info(f"Processing query: {question}")
        
        try:
            # 1. Retrieve relevant chunks
            context_chunks = self.search(question, filters)
            
            # 2. Generate answer
            if context_chunks:
                result = self.generate_answer(question, context_chunks)
            else:
                result = self._handle_no_context(question)
            
            # Add retrieval metadata
            result.update({
                "question": question,
                "chunks_retrieved": len(context_chunks),
                "filters_applied": bool(filters)
            })
            
            logger.info(f"Query processed successfully: {len(context_chunks)} chunks used")
            return result
            
        except Exception as e:
            logger.exception(f"RAG query failed for '{question}': {e}")
            raise RAGEngineError(f"Query failed: {e}") from e

    # -----------------------------------------------------------------------
    # Response Processing Utilities
    # -----------------------------------------------------------------------
    def _prepare_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Prepare context text from retrieved chunks."""
        context_parts = []
        
        for i, chunk in enumerate(chunks):
            source_info = chunk["metadata"].get("document_name", "Unknown")
            content = chunk["content"]
            score = chunk.get("similarity_score", 0)
            
            context_parts.append(
                f"[Source: {source_info} | Relevance: {score}]\n{content}\n"
            )
        
        return "\n---\n".join(context_parts)

    def _calculate_confidence(self, chunks: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence based on chunk similarities."""
        if not chunks:
            return 0.0
        
        scores = [chunk.get("similarity_score", 0) for chunk in chunks]
        avg_score = sum(scores) / len(scores)
        
        # Boost confidence if we have high-importance chunks
        important_chunks = [c for c in chunks if c["metadata"].get("importance") == "high"]
        if important_chunks:
            avg_score = min(1.0, avg_score * 1.2)
        
        return round(avg_score, 3)

    def _extract_sources(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Extract unique source documents from chunks."""
        sources = set()
        for chunk in chunks:
            doc_name = chunk["metadata"].get("document_name")
            if doc_name:
                sources.add(doc_name)
        return list(sources)

    def _handle_no_context(self, query: str) -> Dict[str, Any]:
        """Handle case where no relevant context is found."""
        return {
            "answer": "I couldn't find relevant information in the provided documentation to answer your question. Please try rephrasing or ensure the relevant documents are uploaded.",
            "sources": [],
            "confidence": 0.0,
            "chunks_used": 0,
            "model": "none"
        }


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------
def create_rag_engine(collection_name: str = "programming_docs") -> RAGEngine:
    """Create and initialize a RAG engine instance."""
    return RAGEngine(collection_name=collection_name)