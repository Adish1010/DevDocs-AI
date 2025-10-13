"""
FastAPI Layer for Intelligent Programming Documentation Search Engine
-----------------------------------------------------------------------
Provides RESTful API endpoints for document management and querying.
Designed for production use with proper error handling and documentation.
"""

import os
import logging
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Import our core components
from src.document_processor import process_document
from src.rag_engine import create_rag_engine
from config import (
    PROJECT_NAME,
    VERSION,
    DESCRIPTION,
    EXAMPLE_QUERIES,
    SUPPORTED_EXTENSIONS
)


# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# FastAPI App Configuration
# ---------------------------------------------------------------------------
app = FastAPI(
    title=PROJECT_NAME,
    version=VERSION,
    description=DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware for frontend compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic Models for Request/Response
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str
    filters: Optional[dict] = None

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]
    confidence: float
    chunks_used: int
    processing_time: float

class UploadResponse(BaseModel):
    message: str
    document_name: str
    chunks_processed: int
    document_id: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    version: str
    collection_stats: dict

class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None


# ---------------------------------------------------------------------------
# Global RAG Engine Instance
# ---------------------------------------------------------------------------
rag_engine = None

@app.on_event("startup")
async def startup_event():
    """Initialize RAG engine on application startup."""
    global rag_engine
    try:
        rag_engine = create_rag_engine()
        logger.info("✅ RAG Engine initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG Engine: {e}")
        raise


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Intelligent Programming Documentation Search Engine",
        "version": VERSION,
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "upload": "/upload",
            "query": "/query",
            "examples": "/examples",
            "stats": "/stats"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with system status."""
    try:
        stats = rag_engine.get_collection_stats() if rag_engine else {}
        return HealthResponse(
            status="healthy",
            version=VERSION,
            collection_stats=stats
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="degraded",
            version=VERSION,
            collection_stats={"error": str(e)}
        )

@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    document_name: Optional[str] = None
):
    """
    Upload and process a document for the knowledge base.
    
    Supports:
    - PDF files (.pdf)
    - Text files (.txt)
    """
    try:
        # Validate file type
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type. Supported: {SUPPORTED_EXTENSIONS}"
            )
        
        # Generate document name if not provided
        doc_name = document_name or file.filename
        
        # Create temporary file path
        temp_path = f"data/raw_documents/{doc_name}"
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        
        # Save uploaded file
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"Processing uploaded file: {doc_name}")
        
        # Process document
        chunks = process_document(temp_path, doc_name)
        
        # Add to vector database
        chunks_added = rag_engine.add_documents(chunks)
        
        # Clean up temporary file
        os.remove(temp_path)
        
        logger.info(f"Successfully processed {doc_name}: {chunks_added} chunks")
        
        return UploadResponse(
            message="Document processed successfully",
            document_name=doc_name,
            chunks_processed=chunks_added
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Upload failed for {file.filename}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document processing failed: {str(e)}"
        )

@app.post("/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """
    Query the knowledge base with a natural language question.
    
    Returns:
    - Answer synthesized from relevant documentation
    - Source documents used
    - Confidence score
    - Processing metadata
    """
    import time
    
    try:
        start_time = time.time()
        
        if not rag_engine:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RAG engine not initialized"
            )
        
        # Validate question
        if not request.question or not request.question.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Question cannot be empty"
            )
        
        logger.info(f"Processing query: {request.question}")
        
        # Execute RAG query
        response = rag_engine.query(request.question, request.filters)
        
        processing_time = round(time.time() - start_time, 3)
        
        # Add processing time to response
        response["processing_time"] = processing_time
        
        logger.info(f"Query completed in {processing_time}s: {len(response.get('sources', []))} sources used")
        
        return QueryResponse(**response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Query failed for '{request.question}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}"
        )

@app.get("/examples", response_model=dict)
async def get_example_queries():
    """Get example queries for testing the system."""
    return {
        "example_queries": EXAMPLE_QUERIES,
        "description": "Try these example questions to test the system"
    }

@app.get("/stats", response_model=dict)
async def get_system_stats():
    """Get system statistics and collection information."""
    try:
        stats = rag_engine.get_collection_stats() if rag_engine else {}
        return {
            "system": {
                "name": PROJECT_NAME,
                "version": VERSION,
                "status": "operational"
            },
            "knowledge_base": stats,
            "supported_file_types": SUPPORTED_EXTENSIONS
        }
    except Exception as e:
        logger.error(f"Failed to get system stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve statistics: {str(e)}"
        )


# ---------------------------------------------------------------------------
# Error Handlers
# ---------------------------------------------------------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Global HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            details=str(exc) if exc.status_code == 500 else None
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            details=str(exc)
        ).dict()
    )


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )