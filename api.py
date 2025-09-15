"""
FastAPI Application for Document Retrieval and Search System

This module provides REST API endpoints for:
- Document upload and processing (binary and multipart)
- Multiple search types: Semantic, Keyword, Hybrid, Enhanced
- Document management and statistics
"""

import json
import asyncio
import logging
import traceback
import base64
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List, Optional, Any, Literal
from pathlib import Path
import os
from enum import Enum

# Document processing imports
from doc_process import process_document, process_document_with_azure

# FastAPI imports
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, File, UploadFile, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

# Document retrieval imports
from doc_retrieval_enhanced import (
    DocumentRetriever, 
    RetrievalConfig, 
    SearchType,
    DatabaseConfig,
    simple_search,
    enhanced_search,
    keyword_search,
    hybrid_search
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# Global retriever instance
retriever: Optional[DocumentRetriever] = None

# Application settings
class Settings:
    """Application settings from environment variables."""
    
    def __init__(self):
        self.cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
        self.max_file_size_mb = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
        self.upload_dir = Path(os.getenv("UPLOAD_DIR", "uploads"))
        self.log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        self.host = os.getenv("HOST", "0.0.0.0")
        self.port = int(os.getenv("PORT", "8000"))
        self.workers = int(os.getenv("WORKERS", "1"))
        
        # Create upload directory
        self.upload_dir.mkdir(exist_ok=True)

settings = Settings()

# Enums
class SearchTypeEnum(str, Enum):
    """Search types available in the API."""
    semantic = "semantic"
    keyword = "keyword"
    hybrid = "hybrid"
    enhanced = "enhanced"

# Statistics tracking
class SearchStats:
    """Track search statistics and performance metrics."""
    
    def __init__(self):
        self.total_searches = 0
        self.searches_by_type = {
            "semantic": 0,
            "keyword": 0,
            "hybrid": 0,
            "enhanced": 0
        }
        self.search_times = []
        self.last_search_timestamp = None
    
    def record_search(self, search_type: str, search_time: float):
        """Record a search operation for statistics."""
        self.total_searches += 1
        self.searches_by_type[search_type] = self.searches_by_type.get(search_type, 0) + 1
        self.search_times.append(search_time)
        if len(self.search_times) > 100:  # Keep only last 100 search times
            self.search_times.pop(0)
        self.last_search_timestamp = datetime.now()
    
    @property
    def average_search_time(self) -> float:
        """Calculate average search time."""
        return sum(self.search_times) / len(self.search_times) if self.search_times else 0.0

stats = SearchStats()

# Pydantic Models

class BaseSearchRequest(BaseModel):
    """Base model for search requests."""
    query: str = Field(..., min_length=1, max_length=500, description="Search query text")
    max_results: int = Field(10, ge=1, le=100, description="Maximum number of results to return")
    metadata_filters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Metadata filters")
    enable_context: bool = Field(True, description="Include context chunks around results")
    full_content: bool = Field(True, description="Return full content in response")
    preview_chars: int = Field(500, ge=100, le=2000, description="Number of preview characters if full_content is False")

    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty or whitespace only')
        return v.strip()

class SemanticSearchRequest(BaseSearchRequest):
    """Request model for semantic search."""
    min_similarity: float = Field(0.6, ge=0.0, le=1.0, description="Minimum similarity threshold")
    enable_query_enhancement: bool = Field(False, description="Enable LLM query enhancement")

class KeywordSearchRequest(BaseSearchRequest):
    """Request model for keyword search."""
    min_keyword_matches: int = Field(1, ge=1, description="Minimum number of keyword matches")
    use_stemming: bool = Field(True, description="Enable word stemming")

class HybridSearchRequest(BaseSearchRequest):
    """Request model for hybrid search."""
    semantic_weight: float = Field(0.7, ge=0.0, le=1.0, description="Weight for semantic scores")
    keyword_weight: float = Field(0.3, ge=0.0, le=1.0, description="Weight for keyword scores")
    min_similarity: float = Field(0.5, ge=0.0, le=1.0, description="Minimum similarity threshold")
    
    @validator('keyword_weight')
    def validate_weights(cls, v, values):
        if 'semantic_weight' in values:
            if abs((values['semantic_weight'] + v) - 1.0) > 0.01:
                raise ValueError('Semantic and keyword weights must sum to 1.0')
        return v

class EnhancedSearchRequest(BaseSearchRequest):
    """Request model for enhanced search."""
    enable_query_enhancement: bool = Field(True, description="Enable LLM query enhancement")
    min_similarity: float = Field(0.6, ge=0.0, le=1.0, description="Minimum similarity threshold")

class UnifiedSearchRequest(BaseModel):
    """Unified search request supporting all search types."""
    query: str = Field(..., min_length=1, max_length=500, description="Search query text")
    search_type: SearchTypeEnum = Field(SearchTypeEnum.enhanced, description="Type of search to perform")
    max_results: int = Field(10, ge=1, le=100, description="Maximum number of results")
    
    # Common parameters
    metadata_filters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Metadata filters")
    enable_context: bool = Field(True, description="Include context chunks")
    full_content: bool = Field(True, description="Return full content")
    preview_chars: int = Field(500, description="Preview character count")
    
    # Semantic/Enhanced parameters
    min_similarity: float = Field(0.6, ge=0.0, le=1.0, description="Minimum similarity threshold")
    enable_query_enhancement: bool = Field(False, description="Enable query enhancement")
    
    # Keyword parameters
    min_keyword_matches: int = Field(1, ge=1, description="Minimum keyword matches")
    use_stemming: bool = Field(True, description="Enable stemming")
    
    # Hybrid parameters
    semantic_weight: float = Field(0.7, ge=0.0, le=1.0, description="Semantic weight")
    keyword_weight: float = Field(0.3, ge=0.0, le=1.0, description="Keyword weight")

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

    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty or whitespace only')
        return v.strip()

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
        if '.' not in v:
            raise ValueError('Filename must include file extension')
        return v.strip()
    
    @validator('file_data')
    def validate_file_data(cls, v):
        if not v.strip():
            raise ValueError('File data cannot be empty')
        try:
            base64.b64decode(v)
        except Exception:
            raise ValueError('File data must be valid base64 encoded')
        return v.strip()

# Response models
class SearchResponse(BaseModel):
    """Response model for search results."""
    success: bool = Field(True, description="Whether the search was successful")
    query: str = Field(..., description="Original search query")
    enhanced_query: Optional[str] = Field(None, description="Enhanced query if enhancement was enabled")
    search_type: str = Field(..., description="Type of search performed")
    total_results: int = Field(..., description="Total number of results found")
    search_time: float = Field(..., description="Search execution time in seconds")
    results: List[Dict[str, Any]] = Field(..., description="Search results")
    error: Optional[str] = Field(None, description="Error message if search failed")
    message: Optional[str] = Field(None, description="Additional information")

class DocumentProcessResponse(BaseModel):
    """Response model for document processing."""
    success: bool
    message: str
    details: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field("healthy", description="Service health status")
    timestamp: datetime = Field(..., description="Current timestamp")
    retriever_initialized: bool = Field(..., description="Whether retriever is initialized")
    database_status: Optional[Dict[str, Any]] = None

class StatsResponse(BaseModel):
    """Statistics response."""
    total_searches: int = Field(..., description="Total searches performed")
    searches_by_type: Dict[str, int] = Field(..., description="Breakdown by search type")
    average_search_time: float = Field(..., description="Average search time in seconds")
    last_search_timestamp: Optional[datetime] = Field(None, description="Timestamp of last search")

# Application lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    global retriever
    
    # Startup
    logger.info("Starting up Document Retrieval & Search API...")
    try:
        config = RetrievalConfig(
            max_results=20,
            min_similarity_threshold=0.3,
            search_type=SearchType.SEMANTIC,
            enable_query_enhancement=True,
            llm_model="azure_openai/gpt-4o",
            context_window_size=2
        )
        
        db_config = DatabaseConfig()
        retriever = DocumentRetriever(config, db_config)
        await retriever.initialize()
        logger.info("Document retriever initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize retriever: {e}")
        logger.error(traceback.format_exc())
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Document Retrieval & Search API...")
    if retriever:
        try:
            await retriever.close()
            logger.info("Document retriever closed successfully")
        except Exception as e:
            logger.error(f"Error closing retriever: {e}")

# Initialize FastAPI app
app = FastAPI(
    title="Document Processing & Search API",
    description="Comprehensive API for document upload, processing, and multi-type search capabilities",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health and Status Endpoints

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API and database health."""
    try:
        health_data = {
            "status": "healthy" if retriever else "degraded",
            "timestamp": datetime.now(),
            "retriever_initialized": retriever is not None
        }
        
        if retriever and retriever.vector_db:
            db_health = await retriever.vector_db.health_check()
            health_data["database_status"] = db_health
        
        return HealthResponse(**health_data)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.get("/stats", response_model=StatsResponse, tags=["System"])
async def get_statistics():
    """Get search statistics."""
    return StatsResponse(
        total_searches=stats.total_searches,
        searches_by_type=stats.searches_by_type,
        average_search_time=stats.average_search_time,
        last_search_timestamp=stats.last_search_timestamp
    )

@app.get("/database/stats", tags=["Database"])
async def get_database_stats():
    """Get vector database statistics."""
    try:
        if not retriever or not retriever.vector_db:
            raise HTTPException(status_code=503, detail="Database not available")
        
        db_stats = await retriever.vector_db.get_stats()
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "stats": db_stats
        }
        
    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

# Search Endpoints

@app.post("/search/simple", response_model=SearchResponse, tags=["Search"])
async def simple_search_endpoint(
    request: SimpleSearchRequest,
    background_tasks: BackgroundTasks
):
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
        
        # Record statistics
        background_tasks.add_task(stats.record_search, "simple", results.search_time)
        
        response_data = results.to_dict(full_content=False, preview_chars=300)
        
        logger.info(f"Simple search completed: {results.total_results} results in {results.search_time:.3f}s")
        return SearchResponse(
            success=True,
            query=results.query,
            enhanced_query=results.enhanced_query,
            search_type="simple",
            total_results=results.total_results,
            search_time=results.search_time,
            results=response_data["results"],
            message=f"Found {results.total_results} relevant documents"
        )
        
    except Exception as e:
        logger.error(f"Simple search failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/search/semantic", response_model=SearchResponse, tags=["Search"])
async def semantic_search_endpoint(
    request: SemanticSearchRequest,
    background_tasks: BackgroundTasks
):
    """
    Perform semantic search using vector embeddings.
    
    This search type uses AI embeddings to find conceptually similar documents,
    even if they don't contain the exact search terms.
    """
    if not retriever:
        raise HTTPException(status_code=503, detail="Search service is not available")
    
    try:
        logger.info(f"Semantic search request: {request.query}")
        
        config = RetrievalConfig(
            search_type=SearchType.SEMANTIC,
            max_results=request.max_results,
            min_similarity_threshold=request.min_similarity,
            #source_filter=request.source_filter,
            metadata_filters=request.metadata_filters,
            context_window_size=3 if request.enable_context else 0,
            enable_query_enhancement=request.enable_query_enhancement
        )
        
        retriever.config = config
        results = await retriever.search(request.query)
        
        # Record statistics
        background_tasks.add_task(stats.record_search, "semantic", results.search_time)
        
        response_data = results.to_dict(
            full_content=request.full_content, 
            preview_chars=request.preview_chars if not request.full_content else None
        )
        
        logger.info(f"Semantic search completed: {results.total_results} results in {results.search_time:.3f}s")
        
        return SearchResponse(
            success=True,
            query=results.query,
            enhanced_query=results.enhanced_query,
            search_type="semantic",
            total_results=results.total_results,
            search_time=results.search_time,
            results=response_data["results"],
            message=f"Found {results.total_results} semantically relevant documents"
        )
        
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Semantic search failed: {str(e)}")

@app.post("/search/keyword", response_model=SearchResponse, tags=["Search"])
async def keyword_search_endpoint(
    request: KeywordSearchRequest,
    background_tasks: BackgroundTasks
):
    """
    Perform keyword-based search.
    
    This search type looks for exact keyword matches in documents,
    useful for finding specific terms or phrases.
    """
    if not retriever:
        raise HTTPException(status_code=503, detail="Search service is not available")
    
    try:
        logger.info(f"Keyword search request: {request.query}")
        
        config = RetrievalConfig(
            search_type=SearchType.KEYWORD,
            max_results=request.max_results,
            #source_filter=request.source_filter,
            metadata_filters=request.metadata_filters,
            context_window_size=3 if request.enable_context else 0,
            min_keyword_matches=request.min_keyword_matches,
            use_stemming=request.use_stemming,
            enable_query_enhancement=False
        )
        
        retriever.config = config
        results = await retriever.search(request.query)
        
        # Record statistics
        background_tasks.add_task(stats.record_search, "keyword", results.search_time)
        
        response_data = results.to_dict(
            full_content=request.full_content, 
            preview_chars=request.preview_chars if not request.full_content else None
        )
        
        logger.info(f"Keyword search completed: {results.total_results} results in {results.search_time:.3f}s")
        
        return SearchResponse(
            success=True,
            query=results.query,
            enhanced_query=None,
            search_type="keyword",
            total_results=results.total_results,
            search_time=results.search_time,
            results=response_data["results"],
            message=f"Found {results.total_results} documents with keyword matches"
        )
        
    except Exception as e:
        logger.error(f"Keyword search failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Keyword search failed: {str(e)}")

@app.post("/search/hybrid", response_model=SearchResponse, tags=["Search"])
async def hybrid_search_endpoint(
    request: HybridSearchRequest,
    background_tasks: BackgroundTasks
):
    """
    Perform hybrid search combining semantic and keyword approaches.
    
    This search type combines the benefits of both semantic similarity
    and keyword matching for comprehensive results.
    """
    if not retriever:
        raise HTTPException(status_code=503, detail="Search service is not available")
    
    try:
        logger.info(f"Hybrid search request: {request.query}")
        
        config = RetrievalConfig(
            search_type=SearchType.HYBRID,
            max_results=request.max_results,
            min_similarity_threshold=request.min_similarity,
            #source_filter=request.source_filter,
            metadata_filters=request.metadata_filters,
            context_window_size=3 if request.enable_context else 0,
            semantic_weight=request.semantic_weight,
            keyword_weight=request.keyword_weight,
            enable_query_enhancement=False
        )
        
        retriever.config = config
        results = await retriever.search(request.query)
        
        # Record statistics
        background_tasks.add_task(stats.record_search, "hybrid", results.search_time)
        
        response_data = results.to_dict(
            full_content=request.full_content, 
            preview_chars=request.preview_chars if not request.full_content else None
        )
        
        logger.info(f"Hybrid search completed: {results.total_results} results in {results.search_time:.3f}s")
        
        return SearchResponse(
            success=True,
            query=results.query,
            enhanced_query=None,
            search_type="hybrid",
            total_results=results.total_results,
            search_time=results.search_time,
            results=response_data["results"],
            message=f"Found {results.total_results} documents using hybrid search"
        )
        
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Hybrid search failed: {str(e)}")

@app.post("/search/enhanced", response_model=SearchResponse, tags=["Search"])
async def enhanced_search_endpoint(
    request: EnhancedSearchRequest,
    background_tasks: BackgroundTasks
):
    """
    Perform enhanced search with automatic optimization.
    
    This search type intelligently analyzes the query and applies
    the most appropriate search strategy with query enhancement.
    """
    if not retriever:
        raise HTTPException(status_code=503, detail="Search service is not available")
    
    try:
        logger.info(f"Enhanced search request: {request.query}")
        
        results = await enhanced_search(
            query=request.query,
            max_results=request.max_results,
            min_similarity=request.min_similarity,
            enable_context=request.enable_context,
            enable_enhancement=request.enable_query_enhancement
        )
        
        # Record statistics
        background_tasks.add_task(stats.record_search, "enhanced", results.search_time)
        
        response_data = results.to_dict(
            full_content=request.full_content, 
            preview_chars=request.preview_chars if not request.full_content else None
        )
        
        logger.info(f"Enhanced search completed: {results.total_results} results in {results.search_time:.3f}s")
        
        return SearchResponse(
            success=True,
            query=results.query,
            enhanced_query=results.enhanced_query,
            search_type="enhanced",
            total_results=results.total_results,
            search_time=results.search_time,
            results=response_data["results"],
            message=f"Found {results.total_results} documents using enhanced search"
        )
        
    except Exception as e:
        logger.error(f"Enhanced search failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Enhanced search failed: {str(e)}")

@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def unified_search_endpoint(
    request: UnifiedSearchRequest,
    background_tasks: BackgroundTasks
):
    """
    Unified search endpoint supporting all search types.
    
    Use the search_type parameter to specify which search method to use:
    - semantic: Conceptual similarity search
    - keyword: Exact term matching
    - hybrid: Combined semantic and keyword
    - enhanced: Intelligent automatic optimization
    """
    if not retriever:
        raise HTTPException(status_code=503, detail="Search service is not available")
    
    try:
        logger.info(f"Unified search request ({request.search_type}): {request.query}")
        
        # Map search type string to enum
        search_type_map = {
            "semantic": SearchType.SEMANTIC,
            "keyword": SearchType.KEYWORD,
            "hybrid": SearchType.HYBRID,
            "enhanced": SearchType.ENHANCED
        }
        
        config = RetrievalConfig(
            search_type=search_type_map[request.search_type],
            max_results=request.max_results,
            min_similarity_threshold=request.min_similarity,
            #source_filter=request.source_filter,
            metadata_filters=request.metadata_filters,
            context_window_size=3 if request.enable_context else 0,
            enable_query_enhancement=request.enable_query_enhancement if request.search_type in ["semantic", "enhanced"] else False,
            semantic_weight=request.semantic_weight,
            keyword_weight=request.keyword_weight,
            min_keyword_matches=request.min_keyword_matches,
            use_stemming=request.use_stemming
        )
        
        retriever.config = config
        results = await retriever.search(request.query)
        
        # Record statistics
        background_tasks.add_task(stats.record_search, request.search_type, results.search_time)
        
        response_data = results.to_dict(
            full_content=request.full_content, 
            preview_chars=request.preview_chars if not request.full_content else None
        )
        
        logger.info(f"Unified search completed: {results.total_results} results in {results.search_time:.3f}s")
        
        return SearchResponse(
            success=True,
            query=results.query,
            enhanced_query=results.enhanced_query,
            search_type=request.search_type,
            total_results=results.total_results,
            search_time=results.search_time,
            results=response_data["results"],
            message=f"Found {results.total_results} documents using {request.search_type} search"
        )
        
    except Exception as e:
        logger.error(f"Unified search failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Unified search failed: {str(e)}")

@app.get("/search", response_model=SearchResponse, tags=["Search"])
async def search_with_params(
    query: str = Query(..., description="Search query"),
    max_results: int = Query(10, ge=1, le=100, description="Maximum number of results"),
    min_similarity: float = Query(0.5, ge=0.0, le=1.0, description="Minimum similarity threshold"),
    enable_enhancement: bool = Query(True, description="Enable query enhancement"),
    enable_context: bool = Query(True, description="Include context chunks"),
    full_content: bool = Query(False, description="Return full content"),
    search_type: SearchTypeEnum = Query(SearchTypeEnum.enhanced, description="Search type")
):
    """
    Search documents using query parameters.
    Convenient for GET requests and testing.
    """
    try:
        search_request = UnifiedSearchRequest(
            query=query,
            search_type=search_type,
            max_results=max_results,
            min_similarity=min_similarity,
            enable_query_enhancement=enable_enhancement,
            enable_context=enable_context,
            full_content=full_content
        )
        
        # Use BackgroundTasks for stats recording
        background_tasks = BackgroundTasks()
        return await unified_search_endpoint(search_request, background_tasks)
        
    except Exception as e:
        logger.error(f"GET search failed: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid search parameters: {str(e)}")

@app.post("/search/batch", response_model=List[SearchResponse], tags=["Search"])
async def batch_search_endpoint(
    background_tasks: BackgroundTasks,
    requests: List[UnifiedSearchRequest] = Body(..., description="List of search requests")
):
    """
    Perform multiple searches in a single request.
    
    Useful for running multiple queries efficiently.
    Maximum 10 searches per batch.
    """
    if not retriever:
        raise HTTPException(status_code=503, detail="Search service is not available")
    
    if len(requests) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 searches per batch")
    
    results = []
    
    for request in requests:
        try:
            search_type_map = {
                "semantic": SearchType.SEMANTIC,
                "keyword": SearchType.KEYWORD,
                "hybrid": SearchType.HYBRID,
                "enhanced": SearchType.ENHANCED
            }
            
            config = RetrievalConfig(
                search_type=search_type_map[request.search_type],
                max_results=request.max_results,
                min_similarity_threshold=request.min_similarity,
                #source_filter=request.source_filter,
                metadata_filters=request.metadata_filters,
                context_window_size=3 if request.enable_context else 0,
                enable_query_enhancement=request.enable_query_enhancement,
                semantic_weight=request.semantic_weight,
                keyword_weight=request.keyword_weight,
                min_keyword_matches=request.min_keyword_matches,
                use_stemming=request.use_stemming
            )
            
            retriever.config = config
            search_results = await retriever.search(request.query)
            
            # Record statistics
            background_tasks.add_task(stats.record_search, request.search_type, search_results.search_time)
            
            response_data = search_results.to_dict(
                full_content=request.full_content,
                preview_chars=request.preview_chars if not request.full_content else None
            )
            
            results.append(SearchResponse(
                success=True,
                query=search_results.query,
                enhanced_query=search_results.enhanced_query,
                search_type=request.search_type,
                total_results=search_results.total_results,
                search_time=search_results.search_time,
                results=response_data["results"]
            ))
            
        except Exception as e:
            logger.error(f"Batch search failed for query '{request.query}': {e}")
            results.append(SearchResponse(
                success=False,
                query=request.query,
                enhanced_query=None,
                search_type=request.search_type,
                total_results=0,
                search_time=0.0,
                results=[],
                error=str(e)
            ))
    
    return results

# Document Processing Endpoints

@app.post("/documents/upload", response_model=DocumentProcessResponse, tags=["Document Processing"])
async def upload_and_process_document(
    request: BinaryFileUploadRequest
):
    """
    Accept binary file data (base64 encoded), create a file and process it.
    Wait for complete processing including vector database storage before returning.
    """
    file_path = None
    try:
        # Decode the base64 file data
        try:
            file_content = base64.b64decode(request.file_data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 file data: {str(e)}")
        
        # Generate unique filename
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
        
        # Process document and wait for completion
        try:
            logger.info(f"Starting document processing for: {request.filename}")
            result = await process_document(str(file_path.absolute()))
            logger.info(f"Document processing completed for: {request.filename}")
            
            return DocumentProcessResponse(
                success=True,
                message=f"File uploaded, processed, and stored in vector database successfully: {request.filename}",
                details={
                    "original_filename": request.filename,
                    "created_file_path": str(file_path.absolute()),
                    "file_size_bytes": len(file_content),
                    "processing_result": result,
                    "status": "completed_and_stored",
                    "processing_method": "standard"
                }
            )
            
        except Exception as e:
            logger.error(f"Standard document processing failed for {request.filename}: {str(e)}")
            logger.info(f"Attempting Azure document processing for: {request.filename}")
            
            # Fallback to Azure processing
            try:
                result = await process_document_with_azure(str(file_path.absolute()))
                logger.info(f"Azure document processing completed for: {request.filename}")
                
                return DocumentProcessResponse(
                    success=True,
                    message=f"File uploaded, processed with Azure Doc AI, and stored in vector database successfully: {request.filename}",
                    details={
                        "original_filename": request.filename,
                        "created_file_path": str(file_path.absolute()),
                        "file_size_bytes": len(file_content),
                        "processing_result": result,
                        "status": "completed_and_stored",
                        "processing_method": "azure_fallback",
                        "fallback_reason": str(e)
                    }
                )
                
            except Exception as azure_error:
                logger.error(f"Both standard and Azure document processing failed for {request.filename}")
                logger.error(f"Standard error: {str(e)}")
                logger.error(f"Azure error: {str(azure_error)}")
                raise HTTPException(
                    status_code=500, 
                    detail=f"Both standard and Azure processing failed. Standard: {str(e)}. Azure: {str(azure_error)}"
                )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Binary file upload failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")
    finally:
        # Clean up the temporary file after processing
        if file_path and file_path.exists():
            try:
                file_path.unlink()
                logger.info(f"Cleaned up temporary file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file {file_path}: {str(e)}")

@app.post("/documents/upload-multipart", response_model=DocumentProcessResponse, tags=["Document Processing"])
async def upload_multipart_document(
    file: UploadFile = File(...)
):
    """
    Legacy multipart file upload endpoint for backward compatibility.
    Wait for complete processing including vector database storage before returning.
    """
    file_path = None
    try:
        # Generate unique filename to avoid conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        file_extension = Path(file.filename).suffix
        safe_filename = f"{timestamp}_{unique_id}_{Path(file.filename).stem}{file_extension}"
        
        # Save uploaded file with unique name
        file_path = settings.upload_dir / safe_filename
        content = await file.read()
        
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        
        logger.info(f"Multipart file saved: {file_path} (original: {file.filename})")
        logger.info(f"File size: {len(content)} bytes")
        
        # Process document and wait for completion
        try:
            logger.info(f"Starting document processing for: {file.filename}")
            result = await process_document(str(file_path.absolute()))
            logger.info(f"Document processing completed for: {file.filename}")
            
            return DocumentProcessResponse(
                success=True,
                message=f"File uploaded, processed, and stored in vector database successfully: {file.filename}",
                details={
                    "original_filename": file.filename,
                    "saved_file_path": str(file_path.absolute()),
                    "file_size_bytes": len(content),
                    "processing_result": result,
                    "status": "completed_and_stored",
                    "processing_method": "standard"
                }
            )
            
        except Exception as e:
            logger.error(f"Standard document processing failed for {file.filename}: {str(e)}")
            logger.info(f"Attempting Azure document processing for: {file.filename}")
            
            # Fallback to Azure processing
            try:
                result = await process_document_with_azure(str(file_path.absolute()))
                logger.info(f"Azure document processing completed for: {file.filename}")
                
                return DocumentProcessResponse(
                    success=True,
                    message=f"File uploaded, processed with Azure, and stored in vector database successfully: {file.filename}",
                    details={
                        "original_filename": file.filename,
                        "saved_file_path": str(file_path.absolute()),
                        "file_size_bytes": len(content),
                        "processing_result": result,
                        "status": "completed_and_stored",
                        "processing_method": "azure_fallback",
                        "fallback_reason": str(e)
                    }
                )
                
            except Exception as azure_error:
                logger.error(f"Both standard and Azure document processing failed for {file.filename}")
                logger.error(f"Standard error: {str(e)}")
                logger.error(f"Azure error: {str(azure_error)}")
                raise HTTPException(
                    status_code=500, 
                    detail=f"Both standard and Azure processing failed. Standard: {str(e)}. Azure: {str(azure_error)}"
                )
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Multipart file upload failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")
    finally:
        # Clean up the uploaded file after processing
        if file_path and file_path.exists():
            try:
                file_path.unlink()
                logger.info(f"Cleaned up uploaded file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up uploaded file {file_path}: {str(e)}")

# Document Management Endpoints

@app.get("/documents/{document_id}/similar", response_model=SearchResponse, tags=["Documents"])
async def get_similar_documents(
    document_id: str,
    limit: int = Query(5, ge=1, le=20, description="Maximum number of similar documents"),
    full_content: bool = Query(True, description="Return full content")
):
    """
    Find documents similar to a given document ID.
    
    Uses semantic similarity to find related documents.
    """
    if not retriever:
        raise HTTPException(status_code=503, detail="Search service is not available")
    
    try:
        similar_docs = await retriever.get_similar_documents(document_id, limit)
        
        return SearchResponse(
            success=True,
            query=f"Similar to document: {document_id}",
            enhanced_query=None,
            search_type="similarity",
            total_results=len(similar_docs),
            search_time=0.0,
            results=[doc.to_dict(full_content=full_content) for doc in similar_docs],
            message=f"Found {len(similar_docs)} similar documents"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Similar documents search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sources", response_model=List[str], tags=["Documents"])
async def list_sources():
    """
    Get a list of all available document sources.
    """
    if not retriever:
        raise HTTPException(status_code=503, detail="Search service is not available")
    
    try:
        async with retriever.vector_db.get_connection() as conn:
            rows = await conn.fetch("SELECT DISTINCT source FROM documents ORDER BY source")
            return [row['source'] for row in rows]
    except Exception as e:
        logger.error(f"Failed to list sources: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents", tags=["Documents"])
async def list_documents(
    source_filter: Optional[str] = Query(None, description="Filter by document source"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of documents to return"),
    offset: int = Query(0, ge=0, description="Number of documents to skip")
):
    """
    List documents in the database with optional filtering and pagination.
    """
    if not retriever:
        raise HTTPException(status_code=503, detail="Search service is not available")
    
    try:
        async with retriever.vector_db.get_connection() as conn:
            if source_filter:
                query = """
                    SELECT id, title, source, created_at, metadata 
                    FROM documents 
                    WHERE source = $1 
                    ORDER BY created_at DESC 
                    LIMIT $2 OFFSET $3
                """
                rows = await conn.fetch(query, source_filter, limit, offset)
                
                count_query = "SELECT COUNT(*) FROM documents WHERE source = $1"
                total_count = await conn.fetchval(count_query, source_filter)
            else:
                query = """
                    SELECT id, title, source, created_at, metadata 
                    FROM documents 
                    ORDER BY created_at DESC 
                    LIMIT $1 OFFSET $2
                """
                rows = await conn.fetch(query, limit, offset)
                
                count_query = "SELECT COUNT(*) FROM documents"
                total_count = await conn.fetchval(count_query)
            
            documents = []
            for row in rows:
                doc = {
                    "id": row['id'],
                    "title": row['title'],
                    "source": row['source'],
                    "created_at": row['created_at'].isoformat(),
                    "metadata": row['metadata'] or {}
                }
                documents.append(doc)
            
            return {
                "success": True,
                "documents": documents,
                "total_count": total_count,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total_count
            }
            
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{document_id}", tags=["Documents"])
async def delete_document(document_id: str):
    """
    Delete a document and all its associated chunks from the database.
    """
    if not retriever:
        raise HTTPException(status_code=503, detail="Search service is not available")
    
    try:
        async with retriever.vector_db.get_connection() as conn:
            # Check if document exists
            doc_exists = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM documents WHERE id = $1)", 
                document_id
            )
            
            if not doc_exists:
                raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
            
            # Delete document chunks first (foreign key constraint)
            chunks_deleted = await conn.fetchval(
                "DELETE FROM document_chunks WHERE document_id = $1", 
                document_id
            )
            
            # Delete the document
            await conn.execute(
                "DELETE FROM documents WHERE id = $1", 
                document_id
            )
            
            logger.info(f"Deleted document {document_id} and {chunks_deleted} associated chunks")
            
            return {
                "success": True,
                "message": f"Document {document_id} deleted successfully",
                "chunks_deleted": chunks_deleted
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint
@app.get("/", tags=["Root"])
async def root(
    show_endpoints: bool = Query(True, description="Include endpoint information"),
    show_config: bool = Query(False, description="Include configuration details"),
    format_type: str = Query("detailed", enum=["minimal", "detailed", "debug"], description="Response detail level")
):
    """API root endpoint with configurable response details."""
    
    base_response = {
        "message": "Document Processing & Search API",
        "version": "2.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "description": "Comprehensive API for document upload, processing, and multi-type search capabilities"
    }
    
    # Minimal response
    if format_type == "minimal":
        return base_response
    
    # Add endpoints information if requested
    if show_endpoints:
        base_response["endpoints"] = {
            # System endpoints
            "health_check": "/health",
            "search_statistics": "/stats", 
            "database_statistics": "/database/stats",
            
            # Search endpoints
            "simple_search": "/search/simple",
            "semantic_search": "/search/semantic",
            "keyword_search": "/search/keyword", 
            "hybrid_search": "/search/hybrid",
            "enhanced_search": "/search/enhanced",
            "unified_search": "/search",
            "batch_search": "/search/batch",
            
            # Document endpoints
            "upload_binary": "/documents/upload",
            "upload_multipart": "/documents/upload-multipart",
            "list_documents": "/documents",
            "delete_document": "/documents/{document_id}",
            "similar_documents": "/documents/{document_id}/similar",
            "list_sources": "/sources",
            
            # Documentation
            "api_docs": "/docs",
            "openapi_spec": "/openapi.json"
        }
    
    # Add configuration details if requested
    if show_config:
        try:
            config_info = {
                "retriever_initialized": retriever is not None,
                "database_available": retriever and retriever.vector_db is not None,
                "cors_enabled": True,
                "upload_directory": str(settings.upload_dir),
                "max_file_size_mb": settings.max_file_size_mb,
                "cors_origins": settings.cors_origins
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
            "working_directory": str(Path.cwd()),
            "environment": "development" if __debug__ else "production",
            "search_stats": {
                "total_searches": stats.total_searches,
                "searches_by_type": stats.searches_by_type,
                "average_search_time": stats.average_search_time
            }
        }
    
    return base_response

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

# Configuration for running
if __name__ == "__main__":
    uvicorn.run(
        "api:app",  # Adjust this to match your file name
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level=settings.log_level.lower(),
        workers=settings.workers
    )

# Access documentation at: http://localhost:8000/docs