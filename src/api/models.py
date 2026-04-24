"""Pydantic models for API request/response validation"""

from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator


class QueryMode(str, Enum):
    """Query mode enumeration"""
    VECTOR = "VECTOR"
    GRAPH = "GRAPH"
    HYBRID = "HYBRID"


class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    query_text: str = Field(..., min_length=1, description="User query text")
    mode: Optional[QueryMode] = Field(None, description="Query mode (VECTOR/GRAPH/HYBRID)")
    top_k: Optional[int] = Field(10, ge=1, le=100, description="Number of chunks to retrieve")
    max_depth: Optional[int] = Field(None, ge=1, le=5, description="Maximum graph traversal depth")
    
    @validator('query_text')
    def query_text_not_empty(cls, v):
        """Validate query text is not empty or whitespace"""
        if not v or not v.strip():
            raise ValueError('query_text cannot be empty or whitespace')
        return v.strip()


class Citation(BaseModel):
    """Citation information"""
    doc_id: str
    section: str
    chunk_id: str
    breadcrumbs: str


class QueryMetrics(BaseModel):
    """Query execution metrics"""
    total_time: float
    routing_time: float
    retrieval_time: float
    fusion_time: float
    reranking_time: float
    assembly_time: float
    generation_time: float
    validation_time: float
    query_mode: str
    chunks_retrieved: int
    chunks_reranked: int
    context_tokens: int


class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    answer: str = Field(..., description="Generated answer with citations")
    citations: Dict[str, Citation] = Field(default_factory=dict, description="Citation map")
    faithfulness_score: float = Field(..., ge=0.0, le=1.0, description="Faithfulness score (0-1)")
    retrieval_mode: str = Field(..., description="Query mode used")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")
    metrics: Optional[QueryMetrics] = Field(None, description="Query execution metrics")


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")


class IngestionRequest(BaseModel):
    """Request model for ingestion endpoint (multipart form data)"""
    # File will be handled separately via UploadFile
    # This model is for additional form fields if needed
    doc_id: Optional[str] = Field(None, description="Optional document ID")


class IngestionResponse(BaseModel):
    """Response model for ingestion endpoint"""
    job_id: str = Field(..., description="Ingestion job ID for tracking")
    status: str = Field(..., description="Initial status (pending)")
    message: str = Field(..., description="Status message")


class IngestionStatus(str, Enum):
    """Ingestion status enumeration"""
    PENDING = "pending"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    EXTRACTING = "extracting"
    RESOLVING = "resolving"
    DETECTING_CONFLICTS = "detecting_conflicts"
    POPULATING_GRAPH = "populating_graph"
    COMPLETED = "completed"
    FAILED = "failed"


class IngestionStatusResponse(BaseModel):
    """Response model for ingestion status endpoint"""
    job_id: str = Field(..., description="Ingestion job ID")
    status: IngestionStatus = Field(..., description="Current ingestion status")
    message: str = Field(..., description="Status message")
    doc_id: Optional[str] = Field(None, description="Document ID if available")
    num_chunks: int = Field(0, description="Number of chunks created")
    num_entities: int = Field(0, description="Number of entities extracted")
    num_relationships: int = Field(0, description="Number of relationships created")
    error: Optional[str] = Field(None, description="Error message if failed")
    started_at: Optional[str] = Field(None, description="Start timestamp")
    completed_at: Optional[str] = Field(None, description="Completion timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
