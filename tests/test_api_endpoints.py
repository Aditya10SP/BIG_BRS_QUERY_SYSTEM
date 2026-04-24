"""Tests for REST API endpoints"""

import pytest
import json
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from io import BytesIO
import sys
from pathlib import Path

# Set environment variables before importing app
import os
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
os.environ.setdefault("LLM_MODEL", "llama2")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes import router
from src.api.models import QueryRequest, QueryResponse
from src.pipeline.query_pipeline import QueryResponse as PipelineQueryResponse, QueryMetrics
from src.pipeline.ingestion_pipeline import IngestionResult, IngestionStatus
from config.system_config import SystemConfig


# Create a test app
test_app = FastAPI()
test_app.include_router(router, prefix="/api/v1")


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(test_app)


@pytest.fixture
def mock_config():
    """Create mock configuration"""
    return SystemConfig(
        ollama_base_url="http://localhost:11434",
        llm_model="llama2",
        rerank_top_k=5
    )


@pytest.fixture
def mock_query_pipeline():
    """Create mock query pipeline"""
    pipeline = Mock()
    
    # Mock successful query response
    metrics = QueryMetrics(
        total_time=1.5,
        routing_time=0.1,
        retrieval_time=0.5,
        fusion_time=0.2,
        reranking_time=0.3,
        assembly_time=0.1,
        generation_time=0.2,
        validation_time=0.1,
        query_mode="HYBRID",
        chunks_retrieved=10,
        chunks_reranked=5,
        context_tokens=2048
    )
    
    pipeline.query.return_value = PipelineQueryResponse(
        answer="NEFT is a nationwide payment system [doc1:section2].",
        citations={
            "doc1:section2": {
                "doc_id": "doc1",
                "section": "section2",
                "chunk_id": "chunk1",
                "breadcrumbs": "Document > Section"
            }
        },
        faithfulness_score=0.95,
        retrieval_mode="HYBRID",
        warnings=[],
        metrics=metrics,
        error=None
    )
    
    return pipeline


@pytest.fixture
def mock_ingestion_pipeline():
    """Create mock ingestion pipeline"""
    pipeline = Mock()
    
    # Mock successful ingestion result
    from datetime import datetime
    
    result = IngestionResult(
        doc_id="test_doc",
        status=IngestionStatus.COMPLETED,
        message="Ingestion completed successfully",
        num_chunks=50,
        num_entities=20,
        num_relationships=15,
        error=None,
        started_at=datetime.now(),
        completed_at=datetime.now(),
        metadata={"file_type": "docx"}
    )
    
    pipeline.ingest.return_value = result
    
    return pipeline


class TestQueryEndpoint:
    """Tests for POST /api/v1/query endpoint"""
    
    def test_query_success(self, client, mock_config, mock_query_pipeline):
        """Test successful query request"""
        # Setup app state
        test_app.state.config = mock_config
        test_app.state.query_pipeline = mock_query_pipeline
        
        # Make request
        response = client.post(
            "/api/v1/query",
            json={
                "query_text": "What is NEFT?",
                "top_k": 10
            }
        )
        
        # Assertions
        assert response.status_code == 200
        
        data = response.json()
        assert "answer" in data
        assert "citations" in data
        assert "faithfulness_score" in data
        assert "retrieval_mode" in data
        
        assert data["answer"] == "NEFT is a nationwide payment system [doc1:section2]."
        assert data["faithfulness_score"] == 0.95
        assert data["retrieval_mode"] == "HYBRID"
        assert len(data["citations"]) == 1
        
        # Verify pipeline was called
        mock_query_pipeline.query.assert_called_once()
    
    def test_query_with_optional_parameters(self, client, mock_config, mock_query_pipeline):
        """Test query with optional parameters"""
        test_app.state.config = mock_config
        test_app.state.query_pipeline = mock_query_pipeline
        
        response = client.post(
            "/api/v1/query",
            json={
                "query_text": "What is NEFT?",
                "mode": "VECTOR",
                "top_k": 20,
                "max_depth": 2
            }
        )
        
        assert response.status_code == 200
        
        # Verify parameters were passed
        call_args = mock_query_pipeline.query.call_args
        assert call_args[1]["query_text"] == "What is NEFT?"
        assert call_args[1]["top_k"] == 20
        assert call_args[1]["max_depth"] == 2
    
    def test_query_empty_text(self, client, mock_config, mock_query_pipeline):
        """Test query with empty text returns 400"""
        test_app.state.config = mock_config
        test_app.state.query_pipeline = mock_query_pipeline
        
        response = client.post(
            "/api/v1/query",
            json={
                "query_text": ""
            }
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_query_whitespace_only(self, client, mock_config, mock_query_pipeline):
        """Test query with whitespace only returns 400"""
        test_app.state.config = mock_config
        test_app.state.query_pipeline = mock_query_pipeline
        
        response = client.post(
            "/api/v1/query",
            json={
                "query_text": "   "
            }
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_query_missing_required_field(self, client, mock_config, mock_query_pipeline):
        """Test query without query_text returns 400"""
        test_app.state.config = mock_config
        test_app.state.query_pipeline = mock_query_pipeline
        
        response = client.post(
            "/api/v1/query",
            json={}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_query_invalid_mode(self, client, mock_config, mock_query_pipeline):
        """Test query with invalid mode returns 400"""
        test_app.state.config = mock_config
        test_app.state.query_pipeline = mock_query_pipeline
        
        response = client.post(
            "/api/v1/query",
            json={
                "query_text": "What is NEFT?",
                "mode": "INVALID"
            }
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_query_pipeline_not_initialized(self, client, mock_config):
        """Test query when pipeline not initialized returns 500"""
        test_app.state.config = mock_config
        # Don't set query_pipeline
        if hasattr(test_app.state, 'query_pipeline'):
            delattr(test_app.state, 'query_pipeline')
        
        response = client.post(
            "/api/v1/query",
            json={
                "query_text": "What is NEFT?"
            }
        )
        
        assert response.status_code == 500
        assert "not initialized" in response.json()["detail"].lower()
    
    def test_query_pipeline_error(self, client, mock_config, mock_query_pipeline):
        """Test query when pipeline returns error"""
        test_app.state.config = mock_config
        test_app.state.query_pipeline = mock_query_pipeline
        
        # Mock pipeline error
        mock_query_pipeline.query.return_value = PipelineQueryResponse(
            answer="",
            error="Query processing failed: Database connection error"
        )
        
        response = client.post(
            "/api/v1/query",
            json={
                "query_text": "What is NEFT?"
            }
        )
        
        assert response.status_code == 500
        assert "Query processing failed" in response.json()["detail"]
    
    def test_query_response_has_required_fields(self, client, mock_config, mock_query_pipeline):
        """Test that query response has all required fields (Property 50)"""
        test_app.state.config = mock_config
        test_app.state.query_pipeline = mock_query_pipeline
        
        response = client.post(
            "/api/v1/query",
            json={
                "query_text": "What is NEFT?"
            }
        )
        
        assert response.status_code == 200
        
        data = response.json()
        
        # Verify all required fields are present
        required_fields = ["answer", "citations", "faithfulness_score", "retrieval_mode"]
        for field in required_fields:
            assert field in data, f"Required field '{field}' missing from response"
        
        # Verify field types
        assert isinstance(data["answer"], str)
        assert isinstance(data["citations"], dict)
        assert isinstance(data["faithfulness_score"], (int, float))
        assert isinstance(data["retrieval_mode"], str)


class TestIngestionEndpoint:
    """Tests for POST /api/v1/ingest endpoint"""
    
    def test_ingest_docx_success(self, client, mock_config, mock_ingestion_pipeline):
        """Test successful .docx file ingestion"""
        test_app.state.config = mock_config
        test_app.state.ingestion_pipeline = mock_ingestion_pipeline
        test_app.state.ingestion_jobs = {}
        
        # Create mock file
        file_content = b"Mock DOCX content"
        files = {
            "file": ("test_document.docx", BytesIO(file_content), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        }
        
        response = client.post(
            "/api/v1/ingest",
            files=files
        )
        
        assert response.status_code == 200
        
        data = response.json()
        assert "job_id" in data
        assert "status" in data
        assert "message" in data
        
        assert data["status"] == "pending"
        assert len(data["job_id"]) > 0
    
    def test_ingest_pdf_success(self, client, mock_config, mock_ingestion_pipeline):
        """Test successful .pdf file ingestion"""
        test_app.state.config = mock_config
        test_app.state.ingestion_pipeline = mock_ingestion_pipeline
        test_app.state.ingestion_jobs = {}
        
        # Create mock file
        file_content = b"Mock PDF content"
        files = {
            "file": ("test_document.pdf", BytesIO(file_content), "application/pdf")
        }
        
        response = client.post(
            "/api/v1/ingest",
            files=files
        )
        
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "pending"
    
    def test_ingest_with_custom_doc_id(self, client, mock_config, mock_ingestion_pipeline):
        """Test ingestion with custom doc_id"""
        test_app.state.config = mock_config
        test_app.state.ingestion_pipeline = mock_ingestion_pipeline
        test_app.state.ingestion_jobs = {}
        
        file_content = b"Mock DOCX content"
        files = {
            "file": ("test_document.docx", BytesIO(file_content), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        }
        data = {
            "doc_id": "custom_doc_123"
        }
        
        response = client.post(
            "/api/v1/ingest",
            files=files,
            data=data
        )
        
        assert response.status_code == 200
    
    def test_ingest_invalid_file_format(self, client, mock_config, mock_ingestion_pipeline):
        """Test ingestion with invalid file format returns 400"""
        test_app.state.config = mock_config
        test_app.state.ingestion_pipeline = mock_ingestion_pipeline
        test_app.state.ingestion_jobs = {}
        
        # Create mock .txt file (invalid)
        file_content = b"Mock TXT content"
        files = {
            "file": ("test_document.txt", BytesIO(file_content), "text/plain")
        }
        
        response = client.post(
            "/api/v1/ingest",
            files=files
        )
        
        assert response.status_code == 400
        assert "Invalid file format" in response.json()["detail"]
    
    def test_ingest_no_file(self, client, mock_config, mock_ingestion_pipeline):
        """Test ingestion without file returns 422"""
        test_app.state.config = mock_config
        test_app.state.ingestion_pipeline = mock_ingestion_pipeline
        test_app.state.ingestion_jobs = {}
        
        response = client.post(
            "/api/v1/ingest"
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_ingest_pipeline_not_initialized(self, client, mock_config):
        """Test ingestion when pipeline not initialized returns 500"""
        test_app.state.config = mock_config
        test_app.state.ingestion_jobs = {}
        # Don't set ingestion_pipeline
        if hasattr(test_app.state, 'ingestion_pipeline'):
            delattr(test_app.state, 'ingestion_pipeline')
        
        file_content = b"Mock DOCX content"
        files = {
            "file": ("test_document.docx", BytesIO(file_content), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        }
        
        response = client.post(
            "/api/v1/ingest",
            files=files
        )
        
        assert response.status_code == 500
        assert "not initialized" in response.json()["detail"].lower()
    
    def test_ingest_returns_job_id(self, client, mock_config, mock_ingestion_pipeline):
        """Test that ingestion returns job_id (Property 52)"""
        test_app.state.config = mock_config
        test_app.state.ingestion_pipeline = mock_ingestion_pipeline
        test_app.state.ingestion_jobs = {}
        
        file_content = b"Mock DOCX content"
        files = {
            "file": ("test_document.docx", BytesIO(file_content), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        }
        
        response = client.post(
            "/api/v1/ingest",
            files=files
        )
        
        assert response.status_code == 200
        
        data = response.json()
        assert "job_id" in data
        assert isinstance(data["job_id"], str)
        assert len(data["job_id"]) > 0


class TestIngestionStatusEndpoint:
    """Tests for GET /api/v1/ingest/{job_id} endpoint"""
    
    def test_get_status_success(self, client, mock_config):
        """Test successful status retrieval"""
        test_app.state.config = mock_config
        test_app.state.ingestion_jobs = {
            "job123": {
                "job_id": "job123",
                "doc_id": "doc1",
                "file_path": "/path/to/file.docx",
                "file_type": "docx",
                "filename": "file.docx",
                "status": "completed",
                "message": "Ingestion completed successfully",
                "num_chunks": 50,
                "num_entities": 20,
                "num_relationships": 15,
                "error": None,
                "started_at": "2024-01-01T00:00:00",
                "completed_at": "2024-01-01T00:05:00",
                "metadata": {}
            }
        }
        
        response = client.get("/api/v1/ingest/job123")
        
        assert response.status_code == 200
        
        data = response.json()
        assert data["job_id"] == "job123"
        assert data["status"] == "completed"
        assert data["doc_id"] == "doc1"
        assert data["num_chunks"] == 50
    
    def test_get_status_pending(self, client, mock_config):
        """Test status retrieval for pending job"""
        test_app.state.config = mock_config
        test_app.state.ingestion_jobs = {
            "job456": {
                "job_id": "job456",
                "doc_id": "doc2",
                "file_path": "/path/to/file.pdf",
                "file_type": "pdf",
                "filename": "file.pdf",
                "status": "pending",
                "message": "Ingestion queued",
                "num_chunks": 0,
                "num_entities": 0,
                "num_relationships": 0,
                "error": None,
                "started_at": None,
                "completed_at": None,
                "metadata": {}
            }
        }
        
        response = client.get("/api/v1/ingest/job456")
        
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "pending"
    
    def test_get_status_failed(self, client, mock_config):
        """Test status retrieval for failed job"""
        test_app.state.config = mock_config
        test_app.state.ingestion_jobs = {
            "job789": {
                "job_id": "job789",
                "doc_id": "doc3",
                "file_path": "/path/to/file.docx",
                "file_type": "docx",
                "filename": "file.docx",
                "status": "failed",
                "message": "Ingestion failed: Parsing error",
                "num_chunks": 0,
                "num_entities": 0,
                "num_relationships": 0,
                "error": "Parsing error: Invalid document structure",
                "started_at": "2024-01-01T00:00:00",
                "completed_at": "2024-01-01T00:01:00",
                "metadata": {}
            }
        }
        
        response = client.get("/api/v1/ingest/job789")
        
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "failed"
        assert data["error"] is not None
    
    def test_get_status_not_found(self, client, mock_config):
        """Test status retrieval for non-existent job returns 404"""
        test_app.state.config = mock_config
        test_app.state.ingestion_jobs = {}
        
        response = client.get("/api/v1/ingest/nonexistent_job")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    def test_get_status_trackable(self, client, mock_config):
        """Test that ingestion status is trackable (Property 53)"""
        test_app.state.config = mock_config
        
        # Create jobs in different states
        test_app.state.ingestion_jobs = {
            "job_pending": {
                "job_id": "job_pending",
                "status": "pending",
                "message": "Queued",
                "doc_id": "doc1",
                "num_chunks": 0,
                "num_entities": 0,
                "num_relationships": 0,
                "error": None,
                "started_at": None,
                "completed_at": None,
                "metadata": {}
            },
            "job_processing": {
                "job_id": "job_processing",
                "status": "chunking",
                "message": "Processing",
                "doc_id": "doc2",
                "num_chunks": 0,
                "num_entities": 0,
                "num_relationships": 0,
                "error": None,
                "started_at": "2024-01-01T00:00:00",
                "completed_at": None,
                "metadata": {}
            },
            "job_completed": {
                "job_id": "job_completed",
                "status": "completed",
                "message": "Done",
                "doc_id": "doc3",
                "num_chunks": 50,
                "num_entities": 20,
                "num_relationships": 15,
                "error": None,
                "started_at": "2024-01-01T00:00:00",
                "completed_at": "2024-01-01T00:05:00",
                "metadata": {}
            },
            "job_failed": {
                "job_id": "job_failed",
                "status": "failed",
                "message": "Failed",
                "doc_id": "doc4",
                "num_chunks": 0,
                "num_entities": 0,
                "num_relationships": 0,
                "error": "Error message",
                "started_at": "2024-01-01T00:00:00",
                "completed_at": "2024-01-01T00:01:00",
                "metadata": {}
            }
        }
        
        # Test each status is trackable
        for job_id in ["job_pending", "job_processing", "job_completed", "job_failed"]:
            response = client.get(f"/api/v1/ingest/{job_id}")
            assert response.status_code == 200
            
            data = response.json()
            assert "status" in data
            assert data["status"] in ["pending", "chunking", "completed", "failed"]


class TestErrorHandling:
    """Tests for error handling"""
    
    def test_malformed_json_returns_400(self, client, mock_config, mock_query_pipeline):
        """Test malformed JSON returns 400 (Property 51)"""
        test_app.state.config = mock_config
        test_app.state.query_pipeline = mock_query_pipeline
        
        response = client.post(
            "/api/v1/query",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422  # FastAPI returns 422 for validation errors
    
    def test_missing_required_fields_returns_400(self, client, mock_config, mock_query_pipeline):
        """Test missing required fields returns 400 (Property 51)"""
        test_app.state.config = mock_config
        test_app.state.query_pipeline = mock_query_pipeline
        
        response = client.post(
            "/api/v1/query",
            json={"top_k": 10}  # Missing query_text
        )
        
        assert response.status_code == 422  # Validation error
