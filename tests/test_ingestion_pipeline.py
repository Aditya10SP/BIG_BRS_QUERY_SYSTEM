"""Tests for IngestionPipeline class."""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

from src.pipeline.ingestion_pipeline import (
    IngestionPipeline,
    IngestionStatus,
    IngestionResult
)
from src.parsing.document_parser import ParsedDocument, Section
from src.chunking.hierarchical_chunker import Chunk
from src.extraction.entity_extractor import Entity
from src.extraction.entity_resolver import Relationship


@pytest.fixture
def mock_components():
    """Create mock components for the pipeline."""
    return {
        "parser": Mock(),
        "chunker": Mock(),
        "embedding_generator": Mock(),
        "database_manager": Mock(),
        "vector_store": Mock(),
        "bm25_indexer": Mock(),
        "entity_extractor": Mock(),
        "entity_resolver": Mock(),
        "conflict_detector": Mock(),
        "graph_populator": Mock()
    }


@pytest.fixture
def pipeline(mock_components):
    """Create IngestionPipeline with mocked components."""
    return IngestionPipeline(**mock_components)


@pytest.fixture
def sample_parsed_doc():
    """Create a sample parsed document."""
    return ParsedDocument(
        doc_id="test_doc",
        title="Test Document",
        sections=[
            Section(
                section_id="s1",
                heading="Introduction",
                level=1,
                text="This is a test document about NEFT payments.",
                page_numbers=[1]
            )
        ],
        metadata={"file_name": "test.docx", "file_type": "docx"}
    )


@pytest.fixture
def sample_chunks():
    """Create sample chunks."""
    return [
        Chunk(
            chunk_id="test_doc_s1_parent",
            doc_id="test_doc",
            text="This is a test document about NEFT payments.",
            chunk_type="parent",
            parent_chunk_id=None,
            breadcrumbs="Test Document > Introduction",
            section="Introduction",
            token_count=10,
            metadata={}
        ),
        Chunk(
            chunk_id="test_doc_s1_child_0",
            doc_id="test_doc",
            text="This is a test document about NEFT payments.",
            chunk_type="child",
            parent_chunk_id="test_doc_s1_parent",
            breadcrumbs="Test Document > Introduction",
            section="Introduction",
            token_count=10,
            metadata={}
        )
    ]


@pytest.fixture
def sample_entities():
    """Create sample entities."""
    return [
        Entity(
            entity_id="ent_1",
            entity_type="PaymentMode",
            name="NEFT",
            canonical_name="NEFT",
            source_chunk_id="test_doc_s1_child_0",
            context="This is a test document about NEFT payments.",
            properties={}
        )
    ]


class TestIngestionPipeline:
    """Tests for IngestionPipeline class."""
    
    def test_initialization(self, pipeline):
        """Test pipeline initialization."""
        assert pipeline is not None
        assert pipeline.parser is not None
        assert pipeline.chunker is not None
        assert pipeline.embedding_generator is not None
        assert pipeline.database_manager is not None
        assert pipeline.vector_store is not None
        assert pipeline.bm25_indexer is not None
        assert pipeline.entity_extractor is not None
        assert pipeline.entity_resolver is not None
        assert pipeline.conflict_detector is not None
        assert pipeline.graph_populator is not None
        assert isinstance(pipeline.ingestion_status, dict)
    
    def test_ingest_success(
        self, pipeline, mock_components, sample_parsed_doc, sample_chunks, sample_entities
    ):
        """Test successful document ingestion."""
        # Setup mocks
        mock_components["parser"].parse.return_value = sample_parsed_doc
        mock_components["chunker"].chunk.return_value = sample_chunks
        # Return numpy array for embeddings
        mock_embeddings = np.random.rand(len(sample_chunks), 384)
        mock_components["embedding_generator"].batch_generate.return_value = mock_embeddings
        mock_components["entity_extractor"].extract.return_value = sample_entities
        mock_components["entity_resolver"].resolve.return_value = (sample_entities, [])
        mock_components["conflict_detector"].detect.return_value = []
        mock_components["database_manager"].get_chunks_by_doc_id.return_value = [
            {"chunk_id": chunk.chunk_id} for chunk in sample_chunks
        ]
        mock_components["vector_store"].get_by_chunk_id.return_value = (Mock(), {})
        mock_components["bm25_indexer"].get_index_size.return_value = len(sample_chunks)
        
        # Execute ingestion
        result = pipeline.ingest("test.docx", "docx", "test_doc")
        
        # Verify result
        assert result.doc_id == "test_doc"
        assert result.status == IngestionStatus.COMPLETED
        assert result.num_chunks == len(sample_chunks)
        # num_entities is the count before resolution (extract is called per chunk)
        assert result.num_entities >= len(sample_entities)
        assert result.error is None
        assert result.started_at is not None
        assert result.completed_at is not None
        
        # Verify all steps were called
        mock_components["parser"].parse.assert_called_once()
        mock_components["chunker"].chunk.assert_called_once()
        mock_components["embedding_generator"].batch_generate.assert_called_once()
        mock_components["bm25_indexer"].index.assert_called_once()
        mock_components["entity_extractor"].extract.assert_called()
        mock_components["entity_resolver"].resolve.assert_called_once()
        mock_components["conflict_detector"].detect.assert_called_once()
        mock_components["graph_populator"].populate.assert_called_once()
    
    def test_ingest_parsing_failure(self, pipeline, mock_components):
        """Test ingestion failure during parsing."""
        from src.parsing.document_parser import ParsingError
        
        # Setup mock to raise error
        mock_components["parser"].parse.side_effect = ParsingError("Invalid file")
        
        # Execute ingestion
        result = pipeline.ingest("invalid.docx", "docx", "test_doc")
        
        # Verify result
        assert result.doc_id == "test_doc"
        assert result.status == IngestionStatus.FAILED
        assert result.error is not None
        assert "Invalid file" in result.error
        
        # Verify subsequent steps were not called
        mock_components["chunker"].chunk.assert_not_called()
    
    def test_ingest_chunking_failure(
        self, pipeline, mock_components, sample_parsed_doc
    ):
        """Test ingestion failure during chunking."""
        # Setup mocks
        mock_components["parser"].parse.return_value = sample_parsed_doc
        mock_components["chunker"].chunk.side_effect = ValueError("Chunking failed")
        
        # Execute ingestion
        result = pipeline.ingest("test.docx", "docx", "test_doc")
        
        # Verify result
        assert result.status == IngestionStatus.FAILED
        assert result.error is not None
        
        # Verify subsequent steps were not called
        mock_components["embedding_generator"].batch_generate.assert_not_called()
    
    def test_ingest_batch(self, pipeline, mock_components, sample_parsed_doc, sample_chunks):
        """Test batch ingestion of multiple documents."""
        # Setup mocks
        mock_components["parser"].parse.return_value = sample_parsed_doc
        mock_components["chunker"].chunk.return_value = sample_chunks
        mock_embeddings = np.random.rand(len(sample_chunks), 384)
        mock_components["embedding_generator"].batch_generate.return_value = mock_embeddings
        mock_components["entity_extractor"].extract.return_value = []
        mock_components["entity_resolver"].resolve.return_value = ([], [])
        mock_components["conflict_detector"].detect.return_value = []
        mock_components["database_manager"].get_chunks_by_doc_id.return_value = [
            {"chunk_id": chunk.chunk_id} for chunk in sample_chunks
        ]
        mock_components["vector_store"].get_by_chunk_id.return_value = (Mock(), {})
        mock_components["bm25_indexer"].get_index_size.return_value = len(sample_chunks)
        
        # Prepare batch
        documents = [
            {"file_path": "doc1.docx", "file_type": "docx", "doc_id": "doc1"},
            {"file_path": "doc2.docx", "file_type": "docx", "doc_id": "doc2"}
        ]
        
        # Execute batch ingestion
        results = pipeline.ingest_batch(documents)
        
        # Verify results
        assert len(results) == 2
        assert all(r.status == IngestionStatus.COMPLETED for r in results)
        assert results[0].doc_id == "doc1"
        assert results[1].doc_id == "doc2"
    
    def test_get_status(self, pipeline):
        """Test getting ingestion status."""
        # Initially no status
        assert pipeline.get_status("test_doc") is None
        
        # Add a status
        result = IngestionResult(
            doc_id="test_doc",
            status=IngestionStatus.PARSING,
            message="Parsing in progress"
        )
        pipeline.ingestion_status["test_doc"] = result
        
        # Retrieve status
        retrieved = pipeline.get_status("test_doc")
        assert retrieved is not None
        assert retrieved.doc_id == "test_doc"
        assert retrieved.status == IngestionStatus.PARSING
    
    def test_consistency_verification_failure(
        self, pipeline, mock_components, sample_parsed_doc, sample_chunks
    ):
        """Test consistency verification failure."""
        # Setup mocks
        mock_components["parser"].parse.return_value = sample_parsed_doc
        mock_components["chunker"].chunk.return_value = sample_chunks
        mock_embeddings = np.random.rand(len(sample_chunks), 384)
        mock_components["embedding_generator"].batch_generate.return_value = mock_embeddings
        mock_components["entity_extractor"].extract.return_value = []
        mock_components["entity_resolver"].resolve.return_value = ([], [])
        mock_components["conflict_detector"].detect.return_value = []
        
        # Simulate missing chunk in PostgreSQL
        mock_components["database_manager"].get_chunks_by_doc_id.return_value = []
        
        # Execute ingestion
        result = pipeline.ingest("test.docx", "docx", "test_doc")
        
        # Verify result
        assert result.status == IngestionStatus.FAILED
        assert result.error is not None
        assert "Consistency check failed" in result.error
    
    def test_step_ordering(
        self, pipeline, mock_components, sample_parsed_doc, sample_chunks, sample_entities
    ):
        """Test that pipeline steps execute in correct order."""
        call_order = []
        
        # Track call order
        mock_components["parser"].parse.side_effect = lambda *args: (
            call_order.append("parse"), sample_parsed_doc
        )[1]
        mock_components["chunker"].chunk.side_effect = lambda *args: (
            call_order.append("chunk"), sample_chunks
        )[1]
        mock_embeddings = np.random.rand(len(sample_chunks), 384)
        mock_components["embedding_generator"].batch_generate.side_effect = lambda *args: (
            call_order.append("embed"), mock_embeddings
        )[1]
        mock_components["bm25_indexer"].index.side_effect = lambda *args: (
            call_order.append("index"), None
        )[1]
        mock_components["entity_extractor"].extract.side_effect = lambda *args: (
            call_order.append("extract"), sample_entities
        )[1]
        mock_components["entity_resolver"].resolve.side_effect = lambda *args: (
            call_order.append("resolve"), (sample_entities, [])
        )[1]
        mock_components["conflict_detector"].detect.side_effect = lambda *args: (
            call_order.append("detect"), []
        )[1]
        
        def populate_side_effect(*args, **kwargs):
            call_order.append("populate")
            return None
        
        mock_components["graph_populator"].populate.side_effect = populate_side_effect
        
        # Setup remaining mocks
        mock_components["database_manager"].get_chunks_by_doc_id.return_value = [
            {"chunk_id": chunk.chunk_id} for chunk in sample_chunks
        ]
        mock_components["vector_store"].get_by_chunk_id.return_value = (Mock(), {})
        mock_components["bm25_indexer"].get_index_size.return_value = len(sample_chunks)
        
        # Execute ingestion
        result = pipeline.ingest("test.docx", "docx", "test_doc")
        
        # Verify order (extract is called per chunk, so it appears multiple times)
        # We just verify the key steps are in order
        assert "parse" in call_order
        assert "chunk" in call_order
        assert "embed" in call_order
        assert "index" in call_order
        assert "extract" in call_order
        assert "resolve" in call_order
        assert "detect" in call_order
        assert "populate" in call_order
        
        # Verify parse comes before chunk, chunk before embed, etc.
        assert call_order.index("parse") < call_order.index("chunk")
        assert call_order.index("chunk") < call_order.index("embed")
        assert call_order.index("embed") < call_order.index("index")
        assert call_order.index("index") < call_order.index("extract")
        assert call_order.index("extract") < call_order.index("resolve")
        assert call_order.index("resolve") < call_order.index("detect")
        assert call_order.index("detect") < call_order.index("populate")
        
        assert result.status == IngestionStatus.COMPLETED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
