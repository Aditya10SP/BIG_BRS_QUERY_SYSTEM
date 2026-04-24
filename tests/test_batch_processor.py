"""Tests for batch processing utilities."""

import pytest
import time
from src.utils.batch_processor import (
    BatchProcessor,
    EmbeddingBatchProcessor,
    Neo4jBatchProcessor,
    QdrantBatchProcessor
)


class TestBatchProcessor:
    """Test generic batch processor functionality."""
    
    def test_process_batches_sequential(self):
        """Test sequential batch processing."""
        processor = BatchProcessor(batch_size=3)
        
        items = list(range(10))
        
        def process_batch(batch):
            return sum(batch)
        
        results = processor.process_batches(
            items,
            process_batch,
            parallel=False,
            description="numbers"
        )
        
        # Should have 4 batches: [0,1,2], [3,4,5], [6,7,8], [9]
        assert len(results) == 4
        assert results[0] == 3  # 0+1+2
        assert results[1] == 12  # 3+4+5
        assert results[2] == 21  # 6+7+8
        assert results[3] == 9  # 9
    
    def test_process_batches_parallel(self):
        """Test parallel batch processing."""
        processor = BatchProcessor(batch_size=3, max_workers=2)
        
        items = list(range(10))
        
        def process_batch(batch):
            return sum(batch)
        
        results = processor.process_batches(
            items,
            process_batch,
            parallel=True,
            description="numbers"
        )
        
        # Should have same results as sequential
        assert len(results) == 4
        assert results[0] == 3
        assert results[1] == 12
        assert results[2] == 21
        assert results[3] == 9
    
    def test_empty_items(self):
        """Test processing empty list."""
        processor = BatchProcessor(batch_size=10)
        
        results = processor.process_batches(
            [],
            lambda x: x,
            description="empty"
        )
        
        assert results == []
    
    def test_custom_batch_size(self):
        """Test using custom batch size."""
        processor = BatchProcessor(batch_size=10)
        
        items = list(range(20))
        
        def process_batch(batch):
            return len(batch)
        
        # Use custom batch size of 5
        results = processor.process_batches(
            items,
            process_batch,
            batch_size=5,
            description="numbers"
        )
        
        # Should have 4 batches of 5 items each
        assert len(results) == 4
        assert all(r == 5 for r in results)


class TestEmbeddingBatchProcessor:
    """Test embedding batch processor functionality."""
    
    def test_small_text_batch_size(self):
        """Test batch size for small texts."""
        texts = ["short"] * 10
        
        batch_size = EmbeddingBatchProcessor.get_optimal_batch_size(texts, avg_token_count=50)
        
        assert batch_size == EmbeddingBatchProcessor.SMALL_TEXT_BATCH_SIZE
    
    def test_medium_text_batch_size(self):
        """Test batch size for medium texts."""
        texts = ["medium length text"] * 10
        
        batch_size = EmbeddingBatchProcessor.get_optimal_batch_size(texts, avg_token_count=150)
        
        assert batch_size == EmbeddingBatchProcessor.MEDIUM_TEXT_BATCH_SIZE
    
    def test_large_text_batch_size(self):
        """Test batch size for large texts."""
        texts = ["very long text " * 50] * 10
        
        batch_size = EmbeddingBatchProcessor.get_optimal_batch_size(texts, avg_token_count=400)
        
        assert batch_size == EmbeddingBatchProcessor.LARGE_TEXT_BATCH_SIZE
    
    def test_auto_estimate_token_count(self):
        """Test automatic token count estimation."""
        # Short texts (< 100 tokens estimated)
        short_texts = ["short"] * 10
        batch_size = EmbeddingBatchProcessor.get_optimal_batch_size(short_texts)
        assert batch_size == EmbeddingBatchProcessor.SMALL_TEXT_BATCH_SIZE
        
        # Medium texts (100-300 tokens estimated)
        medium_texts = ["a" * 500] * 10  # ~125 tokens
        batch_size = EmbeddingBatchProcessor.get_optimal_batch_size(medium_texts)
        assert batch_size == EmbeddingBatchProcessor.MEDIUM_TEXT_BATCH_SIZE


class TestNeo4jBatchProcessor:
    """Test Neo4j batch processor functionality."""
    
    def test_node_batch_size(self):
        """Test batch size for node operations."""
        batch_size = Neo4jBatchProcessor.get_optimal_batch_size('node')
        assert batch_size == Neo4jBatchProcessor.NODE_BATCH_SIZE
    
    def test_relationship_batch_size(self):
        """Test batch size for relationship operations."""
        batch_size = Neo4jBatchProcessor.get_optimal_batch_size('relationship')
        assert batch_size == Neo4jBatchProcessor.RELATIONSHIP_BATCH_SIZE
    
    def test_property_batch_size(self):
        """Test batch size for property operations."""
        batch_size = Neo4jBatchProcessor.get_optimal_batch_size('property')
        assert batch_size == Neo4jBatchProcessor.PROPERTY_UPDATE_BATCH_SIZE
    
    def test_unknown_operation_type(self):
        """Test default batch size for unknown operation."""
        batch_size = Neo4jBatchProcessor.get_optimal_batch_size('unknown')
        assert batch_size == 100  # Default


class TestQdrantBatchProcessor:
    """Test Qdrant batch processor functionality."""
    
    def test_upsert_batch_size(self):
        """Test batch size for upsert operations."""
        batch_size = QdrantBatchProcessor.get_optimal_batch_size('upsert')
        assert batch_size == QdrantBatchProcessor.UPSERT_BATCH_SIZE
    
    def test_search_batch_size(self):
        """Test batch size for search operations."""
        batch_size = QdrantBatchProcessor.get_optimal_batch_size('search')
        assert batch_size == QdrantBatchProcessor.SEARCH_BATCH_SIZE
    
    def test_delete_batch_size(self):
        """Test batch size for delete operations."""
        batch_size = QdrantBatchProcessor.get_optimal_batch_size('delete')
        assert batch_size == QdrantBatchProcessor.DELETE_BATCH_SIZE
    
    def test_unknown_operation_type(self):
        """Test default batch size for unknown operation."""
        batch_size = QdrantBatchProcessor.get_optimal_batch_size('unknown')
        assert batch_size == 100  # Default
