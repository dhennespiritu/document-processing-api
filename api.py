"""
FastAPI Application for Document Retrieval from Vector Database
"""
import json
import asyncio
import logging
import traceback
import base64
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

# Import your existing modules
from doc_retrieval import (
    DocumentRetriever, 
    RetrievalConfig, 
    SearchType,
    simple_search,
    enhanced_search
)
from dbase_store import DatabaseConfig
from doc_process import process_document, process_document_with_azure

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global retriever instance
retriever: Optional[DocumentRetriever] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    global retriever
    
    # Startup
    logger.info("Starting up Document Retrieval API...")
    try:
        # Initialize retriever with default config
        config = RetrievalConfig(
            max_results=20,
            min_similarity_threshold=0.3,
            search_type=SearchType.SEMANTIC,
            enable_query_enhancement=True,
            llm_model="azure_openai/gpt-4",
            context_window_size=2
        )
        
        retriever = DocumentRetriever(config)
        await retriever.initialize()
        logger.info("Document retriever initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize retriever: {e}")
        logger.error(traceback.format_exc())
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Document Retrieval API...")
    if retriever:
        try:
            await retriever.close()
            logger.info("Document retriever closed successfully")
        except Exception as e:
            logger.error(f"Error closing retriever: {e}")

# Initialize FastAPI app
app = FastAPI(
    title="Document Extractor API",
    description="API for retrieving documents from vector database using semantic search",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class SearchRequest(BaseModel):
    """Request model for document search."""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    max_results: Optional[int] = Field(10, ge=1, le=100, description="Maximum number of results to return")
    min_similarity_threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Minimum similarity threshold")
    enable_query_enhancement: Optional[bool] = Field(True, description="Enable LLM-based query enhancement")
    enable_context: Optional[bool] = Field(True, description="Include context chunks around matches")
    full_content: Optional[bool] = Field(False, description="Return full content instead of preview")
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty or whitespace only')
        return v.strip()

class SimpleSearchRequest(BaseModel):
    """Simplified request model for basic search."""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    max_results: Optional[int] = Field(5, ge=1, le=50, description="Maximum number of results to return")
    min_similarity_threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Minimum similarity threshold")

class DocumentProcessRequest(BaseModel):
    """Request model for document processing."""
    file_path: str = Field(..., description="Path to the document file to process")

class BinaryFileUploadRequest(BaseModel):
    """Request model for binary file upload."""
    filename: str = Field(..., description="Original filename with extension")
    file_data: str = Field(..., description="Base64 encoded binary file data")
    content_type: Optional[str] = Field(None, description="MIME type of the file")
    
    @validator('filename')
    def validate_filename(cls, v):
        if not v.strip():
            raise ValueError('Filename cannot be empty')
        # Ensure filename has an extension
        if '.' not in v:
            raise ValueError('Filename must include file extension')
        return v.strip()
    
    @validator('file_data')
    def validate_file_data(cls, v):
        if not v.strip():
            raise ValueError('File data cannot be empty')
        try:
            # Validate base64 encoding
            base64.b64decode(v)
        except Exception:
            raise ValueError('File data must be valid base64 encoded')
        return v.strip()

class SearchResponse(BaseModel):
    """Response model for search results."""
    success: bool
    query: str
    enhanced_query: Optional[str]
    total_results: int
    search_time: float
    search_type: str
    results: List[Dict[str, Any]]
    message: Optional[str] = None

class DocumentProcessResponse(BaseModel):
    """Response model for document processing."""
    success: bool
    message: str
    details: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    timestamp: str
    database_status: Optional[Dict[str, Any]] = None

# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API and database health."""
    try:
        health_data = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat()
        }
        
        if retriever and retriever.vector_db:
            db_health = await retriever.vector_db.health_check()
            health_data["database_status"] = db_health
        
        return HealthResponse(**health_data)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

# Simple search endpoint
@app.post("/search/simple", response_model=SearchResponse, tags=["Search"])
async def simple_search_endpoint(request: SimpleSearchRequest):
    """
    Simple document search with basic parameters.
    Fast search without query enhancement or context.
    """
    try:
        logger.info(f"Simple search request: {request.query}")
        
        # Perform simple search
        results = await simple_search(
            query=request.query,
            max_results=request.max_results,
            min_similarity=request.min_similarity_threshold
        )
        
        response_data = results.to_dict(full_content=False, preview_chars=300)
        response_data["success"] = True
        response_data["message"] = f"Found {results.total_results} relevant documents"
        
        logger.info(f"Simple search completed: {results.total_results} results in {results.search_time:.3f}s")
        return SearchResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Simple search failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# Enhanced search endpoint
@app.post("/search/enhanced", response_model=SearchResponse, tags=["Search"])
async def enhanced_search_endpoint(request: SearchRequest):
    """
    Enhanced document search with full features including query enhancement and context.
    """
    try:
        logger.info(f"Enhanced search request: {request.query}")
        
        # Perform enhanced search
        results = await enhanced_search(
            query=request.query,
            max_results=request.max_results,
            min_similarity=request.min_similarity_threshold,
            enable_context=request.enable_context,
            enable_enhancement=request.enable_query_enhancement
        )
        
        response_data = results.to_dict(
            full_content=request.full_content, 
            preview_chars=500 if not request.full_content else None
        )
        response_data["success"] = True
        response_data["message"] = f"Found {results.total_results} relevant documents"
        
        logger.info(f"Enhanced search completed: {results.total_results} results in {results.search_time:.3f}s")
        return SearchResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Enhanced search failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# Custom search endpoint with query parameters
@app.get("/search", response_model=SearchResponse, tags=["Search"])
async def search_with_params(
    query: str = Query(..., description="Search query"),
    max_results: int = Query(10, ge=1, le=100, description="Maximum number of results"),
    min_similarity: float = Query(0.5, ge=0.0, le=1.0, description="Minimum similarity threshold"),
    enable_enhancement: bool = Query(True, description="Enable query enhancement"),
    enable_context: bool = Query(True, description="Include context chunks"),
    full_content: bool = Query(False, description="Return full content")
):
    """
    Search documents using query parameters.
    Convenient for GET requests and testing.
    """
    try:
        logger.info(f"GET search request: {query}")
        
        # Convert to SearchRequest model for validation
        search_request = SearchRequest(
            query=query,
            max_results=max_results,
            min_similarity_threshold=min_similarity,
            enable_query_enhancement=enable_enhancement,
            enable_context=enable_context,
            full_content=full_content
        )
        
        # Use the enhanced search endpoint
        return await enhanced_search_endpoint(search_request)
        
    except Exception as e:
        logger.error(f"GET search failed: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid search parameters: {str(e)}")

# Upload and process document endpoint - Modified to accept binary data
@app.post("/documents/upload", response_model=DocumentProcessResponse, tags=["Documents - Standard Extraction"])
async def upload_and_process_document(
    request: BinaryFileUploadRequest,
    background_tasks: BackgroundTasks = None
):
    """
    Accept binary file data (base64 encoded), create a file in root directory and process it.
    """
    try:
        # Decode the base64 file data
        try:
            file_content = base64.b64decode(request.file_data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 file data: {str(e)}")
        
        # Generate unique filename to avoid conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        file_extension = Path(request.filename).suffix
        safe_filename = f"{timestamp}_{unique_id}_{Path(request.filename).stem}{file_extension}"
        
        # Create file in root directory (current working directory)
        file_path = Path.cwd() / safe_filename
        
        # Write binary content to file
        try:
            with open(file_path, "wb") as f:
                f.write(file_content)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to write file: {str(e)}")
        
        logger.info(f"Binary file created: {file_path} (original: {request.filename})")
        logger.info(f"File size: {len(file_content)} bytes")
        
        # Add processing task to background
        if background_tasks:
            background_tasks.add_task(
                process_document_async, 
                str(file_path.absolute()),
                request.filename,  # Pass original filename for reference
                str(file_path)     # Pass created file path for cleanup
            )
            
            return DocumentProcessResponse(
                success=True,
                message=f"File created and processing started: {request.filename}",
                details={
                    "original_filename": request.filename,
                    "created_file_path": str(file_path.absolute()),
                    "file_size_bytes": len(file_content),
                    "status": "processing"
                }
            )
        else:
            # Process immediately (blocking)
            try:
                result = await process_document(str(file_path.absolute()))
                
                return DocumentProcessResponse(
                    success=True,
                    message=f"File created and processed successfully: {request.filename}",
                    details={
                        "original_filename": request.filename,
                        "created_file_path": str(file_path.absolute()),
                        "file_size_bytes": len(file_content),
                        "processing_result": result,
                        "status": "completed"
                    }
                )
            except Exception as e:
                # Clean up file if processing fails
                try:
                    file_path.unlink()
                except:
                    pass
                raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Binary file upload failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")
    

# Legacy multipart file upload endpoint (keeping for backward compatibility)
@app.post("/documents/upload-multipart", response_model=DocumentProcessResponse, tags=["Documents - Standard Extraction"])
async def upload_multipart_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Legacy multipart file upload endpoint for backward compatibility.
    """
    try:
        # Create uploads directory if it doesn't exist
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        # Save uploaded file
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"Uploaded file saved: {file_path}")
        
        # Add processing task to background
        if background_tasks:
            background_tasks.add_task(process_document_async, str(file_path.absolute()))
            
            return DocumentProcessResponse(
                success=True,
                message=f"File uploaded and processing started: {file.filename}",
                details={"file_path": str(file_path.absolute()), "status": "processing"}
            )
        else:
            # Process immediately (blocking)
            result = await process_document(str(file_path.absolute()))
            
            return DocumentProcessResponse(
                success=True,
                message=f"File uploaded and processed successfully: {file.filename}",
                details=result
            )
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")


# Upload and process document endpoint - Modified to accept binary data
@app.post("/documents/upload/azuredocai", response_model=DocumentProcessResponse, tags=["Documents - Azure Doc AI Extraction"])
async def azure_upload_and_process_document(
    request: BinaryFileUploadRequest,
    background_tasks: BackgroundTasks = None
):
    """
    Accept binary file data (base64 encoded), create a file in root directory and process it.
    """
    try:
        # Decode the base64 file data
        try:
            file_content = base64.b64decode(request.file_data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 file data: {str(e)}")
        
        # Generate unique filename to avoid conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        file_extension = Path(request.filename).suffix
        safe_filename = f"{timestamp}_{unique_id}_{Path(request.filename).stem}{file_extension}"
        
        # Create file in root directory (current working directory)
        file_path = Path.cwd() / safe_filename
        
        # Write binary content to file
        try:
            with open(file_path, "wb") as f:
                f.write(file_content)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to write file: {str(e)}")
        
        logger.info(f"Binary file created: {file_path} (original: {request.filename})")
        logger.info(f"File size: {len(file_content)} bytes")
        
        # Add processing task to background
        if background_tasks:
            background_tasks.add_task(
                process_document_async, 
                str(file_path.absolute()),
                request.filename,  # Pass original filename for reference
                str(file_path)     # Pass created file path for cleanup
            )
            
            return DocumentProcessResponse(
                success=True,
                message=f"File created and processing started: {request.filename}",
                details={
                    "original_filename": request.filename,
                    "created_file_path": str(file_path.absolute()),
                    "file_size_bytes": len(file_content),
                    "status": "processing"
                }
            )
        else:
            # Process immediately (blocking)
            try:
                result = await process_document_with_azure(str(file_path.absolute()))
                
                return DocumentProcessResponse(
                    success=True,
                    message=f"File created and processed successfully: {request.filename}",
                    details={
                        "original_filename": request.filename,
                        "created_file_path": str(file_path.absolute()),
                        "file_size_bytes": len(file_content),
                        "processing_result": result,
                        "status": "completed"
                    }
                )
            except Exception as e:
                # Clean up file if processing fails
                try:
                    file_path.unlink()
                except:
                    pass
                raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Binary file upload failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")
    

# Legacy multipart file upload endpoint (keeping for backward compatibility)
@app.post("/documents/upload-multipart/azuredocai", response_model=DocumentProcessResponse, tags=["Documents - Azure Doc AI Extraction"])
async def azure_upload_multipart_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Legacy multipart file upload endpoint for backward compatibility.
    """
    try:
        # Create uploads directory if it doesn't exist
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        # Save uploaded file
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"Uploaded file saved: {file_path}")
        
        # Add processing task to background
        if background_tasks:
            background_tasks.add_task(process_document_async, str(file_path.absolute()))
            
            return DocumentProcessResponse(
                success=True,
                message=f"File uploaded and processing started: {file.filename}",
                details={"file_path": str(file_path.absolute()), "status": "processing"}
            )
        else:
            # Process immediately (blocking)
            result = await process_document_with_azure(str(file_path.absolute()))
            
            return DocumentProcessResponse(
                success=True,
                message=f"File uploaded and processed successfully: {file.filename}",
                details=result
            )
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")


# Database statistics endpoint
@app.get("/database/stats", tags=["Database"])
async def get_database_stats():
    """Get vector database statistics."""
    try:
        if not retriever or not retriever.vector_db:
            raise HTTPException(status_code=503, detail="Database not available")
        
        stats = await retriever.vector_db.get_stats()
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

# Background task for document processing - Modified to handle cleanup
async def process_document_async(file_path: str, original_filename: str = None, cleanup_path: str = None):
    """Process document in background with optional cleanup."""
    try:
        logger.info(f"Background processing started for: {file_path}")
        if original_filename:
            logger.info(f"Original filename: {original_filename}")

        try:
            result = await process_document(file_path)
        except:
            result = await process_document_with_azure(file_path)
        
        logger.info(f"Background processing completed for: {file_path}")
        logger.info(f"Processing result: {result}")
        
        #Clean up the created file after processing
        if cleanup_path and Path(cleanup_path).exists():
            try:
                Path(cleanup_path).unlink()
                logger.info(f"Cleaned up temporary file: {cleanup_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup file {cleanup_path}: {cleanup_error}")
        
    except Exception as e:
        logger.error(f"Background processing failed for {file_path}: {e}")
        logger.error(traceback.format_exc())
        
        # Clean up file on error
        if cleanup_path and Path(cleanup_path).exists():
            try:
                Path(cleanup_path).unlink()
                logger.info(f"Cleaned up file after processing error: {cleanup_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup file after error {cleanup_path}: {cleanup_error}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

# Root endpoint - Enhanced with parameterized options
@app.get("/", tags=["Root"])
async def root(
    show_endpoints: bool = Query(True, description="Include endpoint information"),
    show_config: bool = Query(False, description="Include configuration details"),
    format_type: str = Query("detailed", enum=["minimal", "detailed", "debug"], description="Response detail level")
):
    """API root endpoint with configurable response details."""
    
    base_response = {
        "message": "Document Extractor & Retrieval API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }
    
    # Minimal response
    if format_type == "minimal":
        return base_response
    
    # Add endpoints information if requested
    if show_endpoints:
        base_response["endpoints"] = {
            "health": "/health",
            "document_enhanced_search": "/search/enhanced", 
            "document_search_parameterized": "/search",
            "upload_binary_document": "/documents/upload",
            "upload_multipart_document": "/documents/upload-multipart",
            "upload_binary_document_using_azure": "/documents/upload/azuredocai",
            "upload_multipart_document_using_azure": "/documents/upload-multipart/azuredocai",
            "database_stats": "/database/stats",
            "docs": "/docs"
        }
    
    # Add configuration details if requested
    if show_config:
        try:
            config_info = {
                "retriever_initialized": retriever is not None,
                "database_available": retriever and retriever.vector_db is not None,
                "cors_enabled": True,
                "background_tasks_enabled": True
            }
            
            if retriever and hasattr(retriever, 'config'):
                config_info["retriever_config"] = {
                    "max_results": getattr(retriever.config, 'max_results', 'unknown'),
                    "min_similarity_threshold": getattr(retriever.config, 'min_similarity_threshold', 'unknown'),
                    "search_type": str(getattr(retriever.config, 'search_type', 'unknown')),
                    "enable_query_enhancement": getattr(retriever.config, 'enable_query_enhancement', 'unknown')
                }
            
            base_response["configuration"] = config_info
            
        except Exception as e:
            base_response["configuration"] = {"error": f"Could not retrieve config: {str(e)}"}
    
    # Add debug information if requested
    if format_type == "debug":
        import sys
        base_response["debug_info"] = {
            "python_version": sys.version,
            "fastapi_version": "unknown",  # Would need to import fastapi to get version
            "working_directory": str(Path.cwd()),
            "environment": "development" if __debug__ else "production"
        }
    
    return base_response

# Configuration for running
if __name__ == "__main__":
    uvicorn.run(
        "api:app",  # Adjust this to match your file name
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

#http://localhost:8000/docs# -> to access