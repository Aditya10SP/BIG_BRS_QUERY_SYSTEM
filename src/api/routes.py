"""API routes for query and ingestion endpoints"""

import uuid
import logging
from typing import Optional
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request, BackgroundTasks
from fastapi.responses import JSONResponse

from src.api.models import (
    QueryRequest,
    QueryResponse,
    ErrorResponse,
    IngestionResponse,
    IngestionStatusResponse,
    Citation,
    QueryMetrics as QueryMetricsModel
)
from src.pipeline.query_pipeline import QueryPipeline, QueryResponse as PipelineQueryResponse
from src.pipeline.ingestion_pipeline import IngestionPipeline, IngestionStatus


logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


@router.post(
    "/query",
    response_model=QueryResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"}
    },
    summary="Query documents",
    description="Submit a query to retrieve and generate answers from banking documents"
)
async def query_documents(
    request: Request,
    query_request: QueryRequest
) -> QueryResponse:
    """
    Query endpoint for document retrieval and answer generation.
    
    This endpoint:
    1. Validates the query request
    2. Routes the query to appropriate retrieval mode
    3. Retrieves relevant information from vector store and/or knowledge graph
    4. Generates a grounded response with citations
    5. Validates faithfulness of the response
    
    Args:
        request: FastAPI request object (for accessing app state)
        query_request: QueryRequest with query_text and optional parameters
    
    Returns:
        QueryResponse with answer, citations, and metadata
    
    Raises:
        HTTPException 400: If request is malformed or invalid
        HTTPException 500: If query processing fails
    """
    request_id = str(uuid.uuid4())
    
    logger.info(
        f"[{request_id}] Query request received",
        extra={
            "request_id": request_id,
            "query_text": query_request.query_text[:100],  # Log first 100 chars
            "mode": query_request.mode,
            "top_k": query_request.top_k
        }
    )
    
    try:
        # Get query pipeline from app state
        if not hasattr(request.app.state, 'query_pipeline'):
            logger.error(f"[{request_id}] Query pipeline not initialized")
            raise HTTPException(
                status_code=500,
                detail="Query pipeline not initialized. Please check server configuration."
            )
        
        query_pipeline: QueryPipeline = request.app.state.query_pipeline
        
        # Execute query pipeline
        logger.info(f"[{request_id}] Executing query pipeline")
        
        pipeline_response: PipelineQueryResponse = query_pipeline.query(
            query_text=query_request.query_text,
            top_k=query_request.top_k or 10,
            rerank_top_k=request.app.state.config.rerank_top_k,
            max_depth=query_request.max_depth
        )
        
        # Check for pipeline errors
        if pipeline_response.error:
            logger.error(
                f"[{request_id}] Query pipeline failed: {pipeline_response.error}"
            )
            raise HTTPException(
                status_code=500,
                detail=f"Query processing failed: {pipeline_response.error}"
            )
        
        # Convert pipeline response to API response
        citations_dict = {}
        for cid, citation_data in pipeline_response.citations.items():
            citations_dict[cid] = Citation(
                doc_id=citation_data["doc_id"],
                section=citation_data["section"],
                chunk_id=citation_data["chunk_id"],
                breadcrumbs=citation_data["breadcrumbs"]
            )
        
        # Convert metrics if available
        metrics_model = None
        if pipeline_response.metrics:
            metrics_model = QueryMetricsModel(
                total_time=pipeline_response.metrics.total_time,
                routing_time=pipeline_response.metrics.routing_time,
                retrieval_time=pipeline_response.metrics.retrieval_time,
                fusion_time=pipeline_response.metrics.fusion_time,
                reranking_time=pipeline_response.metrics.reranking_time,
                assembly_time=pipeline_response.metrics.assembly_time,
                generation_time=pipeline_response.metrics.generation_time,
                validation_time=pipeline_response.metrics.validation_time,
                query_mode=pipeline_response.metrics.query_mode,
                chunks_retrieved=pipeline_response.metrics.chunks_retrieved,
                chunks_reranked=pipeline_response.metrics.chunks_reranked,
                context_tokens=pipeline_response.metrics.context_tokens
            )
        
        response = QueryResponse(
            answer=pipeline_response.answer,
            citations=citations_dict,
            faithfulness_score=pipeline_response.faithfulness_score,
            retrieval_mode=pipeline_response.retrieval_mode,
            warnings=pipeline_response.warnings,
            metrics=metrics_model
        )
        
        logger.info(
            f"[{request_id}] Query completed successfully",
            extra={
                "request_id": request_id,
                "retrieval_mode": response.retrieval_mode,
                "faithfulness_score": response.faithfulness_score,
                "num_citations": len(response.citations),
                "num_warnings": len(response.warnings)
            }
        )
        
        return response
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        # Log unexpected errors
        logger.error(
            f"[{request_id}] Unexpected error in query endpoint",
            exc_info=True,
            extra={
                "request_id": request_id,
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
        )
        
        # Return 500 error
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.post(
    "/ingest",
    response_model=IngestionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"}
    },
    summary="Ingest document",
    description="Upload and ingest a document (.docx or .pdf) into the system"
)
async def ingest_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Document file (.docx or .pdf)"),
    doc_id: Optional[str] = Form(None, description="Optional document ID")
) -> IngestionResponse:
    """
    Ingestion endpoint for document upload and processing.
    
    This endpoint:
    1. Validates the uploaded file format
    2. Saves the file temporarily
    3. Triggers the ingestion pipeline asynchronously
    4. Returns a job ID for status tracking
    
    Args:
        request: FastAPI request object (for accessing app state)
        file: Uploaded document file
        doc_id: Optional document ID (defaults to filename)
    
    Returns:
        IngestionResponse with job_id and initial status
    
    Raises:
        HTTPException 400: If file format is invalid
        HTTPException 500: If ingestion fails to start
    """
    request_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())
    
    logger.info(
        f"[{request_id}] Ingestion request received",
        extra={
            "request_id": request_id,
            "job_id": job_id,
            "filename": file.filename,
            "content_type": file.content_type,
            "doc_id": doc_id
        }
    )
    
    try:
        # Validate file format
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="Filename is required"
            )
        
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ['.docx', '.pdf']:
            logger.warning(
                f"[{request_id}] Invalid file format: {file_ext}"
            )
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file format: {file_ext}. Only .docx and .pdf files are supported."
            )
        
        file_type = file_ext[1:]  # Remove the dot
        
        # Get ingestion pipeline from app state
        if not hasattr(request.app.state, 'ingestion_pipeline'):
            logger.error(f"[{request_id}] Ingestion pipeline not initialized")
            raise HTTPException(
                status_code=500,
                detail="Ingestion pipeline not initialized. Please check server configuration."
            )
        
        ingestion_pipeline: IngestionPipeline = request.app.state.ingestion_pipeline
        
        # Create temporary directory for uploads if it doesn't exist
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        # Save uploaded file
        file_path = upload_dir / f"{job_id}_{file.filename}"
        
        logger.info(f"[{request_id}] Saving uploaded file to {file_path}")
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(
            f"[{request_id}] File saved successfully: {len(content)} bytes"
        )
        
        # Generate doc_id if not provided
        if not doc_id:
            doc_id = Path(file.filename).stem
        
        # Store job metadata
        if not hasattr(request.app.state, 'ingestion_jobs'):
            request.app.state.ingestion_jobs = {}
        
        request.app.state.ingestion_jobs[job_id] = {
            "job_id": job_id,
            "doc_id": doc_id,
            "file_path": str(file_path),
            "file_type": file_type,
            "filename": file.filename,
            "status": "pending",
            "message": "Ingestion queued"
        }
        
        # Start ingestion in background using FastAPI BackgroundTasks
        def run_ingestion():
            """Background ingestion task"""
            try:
                logger.info(f"[{job_id}] Starting ingestion for {doc_id}")
                
                result = ingestion_pipeline.ingest(
                    file_path=str(file_path),
                    file_type=file_type,
                    doc_id=doc_id
                )
                
                # Update job status
                request.app.state.ingestion_jobs[job_id] = {
                    "job_id": job_id,
                    "doc_id": result.doc_id,
                    "file_path": str(file_path),
                    "file_type": file_type,
                    "filename": file.filename,
                    "status": result.status.value,
                    "message": result.message,
                    "num_chunks": result.num_chunks,
                    "num_entities": result.num_entities,
                    "num_relationships": result.num_relationships,
                    "error": result.error,
                    "started_at": result.started_at.isoformat() if result.started_at else None,
                    "completed_at": result.completed_at.isoformat() if result.completed_at else None,
                    "metadata": result.metadata
                }
                
                logger.info(
                    f"[{job_id}] Ingestion completed: {result.status.value}",
                    extra={
                        "job_id": job_id,
                        "doc_id": result.doc_id,
                        "status": result.status.value,
                        "num_chunks": result.num_chunks
                    }
                )
                
            except Exception as e:
                logger.error(
                    f"[{job_id}] Ingestion failed",
                    exc_info=True,
                    extra={
                        "job_id": job_id,
                        "error": str(e)
                    }
                )
                
                # Update job status with error
                request.app.state.ingestion_jobs[job_id]["status"] = "failed"
                request.app.state.ingestion_jobs[job_id]["error"] = str(e)
                request.app.state.ingestion_jobs[job_id]["message"] = f"Ingestion failed: {str(e)}"
        
        # Add background task
        background_tasks.add_task(run_ingestion)
        
        # Return job ID immediately
        return IngestionResponse(
            job_id=job_id,
            status="pending",
            message="Document upload successful. Ingestion started."
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        # Log unexpected errors
        logger.error(
            f"[{request_id}] Unexpected error in ingestion endpoint",
            exc_info=True,
            extra={
                "request_id": request_id,
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
        )
        
        # Return 500 error
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get(
    "/ingest/{job_id}",
    response_model=IngestionStatusResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Job Not Found"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"}
    },
    summary="Get ingestion status",
    description="Check the status of a document ingestion job"
)
async def get_ingestion_status(
    request: Request,
    job_id: str
) -> IngestionStatusResponse:
    """
    Get ingestion status endpoint.
    
    This endpoint returns the current status of an ingestion job,
    including progress, completion status, and any errors.
    
    Args:
        request: FastAPI request object (for accessing app state)
        job_id: Ingestion job ID
    
    Returns:
        IngestionStatusResponse with current status and metadata
    
    Raises:
        HTTPException 404: If job ID not found
        HTTPException 500: If status retrieval fails
    """
    logger.info(
        f"Status check for job: {job_id}"
    )
    
    try:
        # Check if ingestion jobs exist
        if not hasattr(request.app.state, 'ingestion_jobs'):
            raise HTTPException(
                status_code=404,
                detail=f"Job ID not found: {job_id}"
            )
        
        # Get job status
        job_data = request.app.state.ingestion_jobs.get(job_id)
        
        if not job_data:
            logger.warning(f"Job not found: {job_id}")
            raise HTTPException(
                status_code=404,
                detail=f"Job ID not found: {job_id}"
            )
        
        # Build response
        response = IngestionStatusResponse(
            job_id=job_data["job_id"],
            status=job_data["status"],
            message=job_data["message"],
            doc_id=job_data.get("doc_id"),
            num_chunks=job_data.get("num_chunks", 0),
            num_entities=job_data.get("num_entities", 0),
            num_relationships=job_data.get("num_relationships", 0),
            error=job_data.get("error"),
            started_at=job_data.get("started_at"),
            completed_at=job_data.get("completed_at"),
            metadata=job_data.get("metadata", {})
        )
        
        logger.info(
            f"Status retrieved for job {job_id}: {response.status}",
            extra={
                "job_id": job_id,
                "status": response.status,
                "doc_id": response.doc_id
            }
        )
        
        return response
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        # Log unexpected errors
        logger.error(
            f"Unexpected error retrieving status for job {job_id}",
            exc_info=True,
            extra={
                "job_id": job_id,
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
        )
        
        # Return 500 error
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
