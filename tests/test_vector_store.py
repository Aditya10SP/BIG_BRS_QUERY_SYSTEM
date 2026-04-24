"""Tests for VectorStore wrapper for Qdrant"""

import pytest
import numpy as np
from src.storage.vector_store import VectorStore


class TestVectorStore:
    """Unit tests for VectorStore class"""
    
    @pytest.fixture
    def vector_store(self):
        """Create a VectorStore instance for testing"""
        # Use a test collection name to avoid conflicts
        store = VectorStore(
            url="http://localhost:6333",
            collection_name="test_banking_docs",
            vector_size=384,
        )
        yield store
        # Cleanup: delete the test collection after tests
        try:
            store.client.delete_collection("test_banking_docs")
        except:
            pass
        store.close()
    
    @pytest.fixture
    def sample_embeddings(self):
        """Generate sample embeddings for testing"""
        # Create 5 sample embeddings with 384 dimensions
        embeddings = np.random.rand(5, 384).astype(np.float32)
        # Normalize for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings
    
    @pytest.fixture
    def sample_metadata(self):
        """Generate sample metadata for testing"""
        return [
            {
                "doc_id": "doc1",
                "text": "NEFT is a payment system",
                "breadcrumbs": "Banking > Payments > NEFT",
                "section": "Payments",
                "chunk_type": "child",
            },
            {
                "doc_id": "doc1",
                "text": "RTGS is for large transactions",
                "breadcrumbs": "Banking > Payments > RTGS",
                "section": "Payments",
                "chunk_type": "child",
            },
            {
                "doc_id": "doc2",
                "text": "Core Banking System overview",
                "breadcrumbs": "Systems > Core Banking",
                "section": "Systems",
                "chunk_type": "parent",
            },
            {
                "doc_id": "doc2",
                "text": "Account management features",
                "breadcrumbs": "Systems > Core Banking > Accounts",
                "section": "Systems",
                "chunk_type": "child",
            },
            {
                "doc_id": "doc3",
                "text": "Risk rules for transactions",
                "breadcrumbs": "Rules > Risk Management",
                "section": "Rules",
                "chunk_type": "child",
            },
        ]
    
    def test_initialization(self, vector_store):
        """Test VectorStore initialization and collection creation"""
        assert vector_store.collection_name == "test_banking_docs"
        assert vector_store.vector_size == 384
        assert vector_store.client is not None
        
        # Verify collection was created
        collections = vector_store.client.get_collections().collections
        collection_names = [col.name for col in collections]
        assert "test_banking_docs" in collection_names
    
    def test_store_embeddings(self, vector_store, sample_embeddings, sample_metadata):
        """Test storing embeddings with batch upsert (Requirement 3.2)"""
        chunk_ids = ["chunk1", "chunk2", "chunk3", "chunk4", "chunk5"]
        
        # Store embeddings
        vector_store.store_embeddings(chunk_ids, sample_embeddings, sample_metadata)
        
        # Verify count
        count = vector_store.count()
        assert count == 5
    
    def test_store_embeddings_validation(self, vector_store, sample_embeddings, sample_metadata):
        """Test input validation for store_embeddings"""
        chunk_ids = ["chunk1", "chunk2"]
        
        # Test length mismatch between chunk_ids and metadata
        with pytest.raises(ValueError, match="Length mismatch.*chunk_ids.*metadata"):
            vector_store.store_embeddings(chunk_ids, sample_embeddings, sample_metadata)
        
        # Test length mismatch between embeddings and chunk_ids
        with pytest.raises(ValueError, match="Length mismatch.*embeddings.*chunk_ids"):
            vector_store.store_embeddings(chunk_ids, sample_embeddings[:3], sample_metadata[:2])
        
        # Test wrong embedding dimension
        wrong_dim_embeddings = np.random.rand(2, 256).astype(np.float32)
        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            vector_store.store_embeddings(chunk_ids, wrong_dim_embeddings, sample_metadata[:2])
    
    def test_search_cosine_similarity(self, vector_store, sample_embeddings, sample_metadata):
        """Test search with cosine similarity (Requirements 3.3, 3.5)"""
        chunk_ids = ["chunk1", "chunk2", "chunk3", "chunk4", "chunk5"]
        
        # Store embeddings
        vector_store.store_embeddings(chunk_ids, sample_embeddings, sample_metadata)
        
        # Search using the first embedding as query
        query_embedding = sample_embeddings[0]
        results = vector_store.search(query_embedding, top_k=3)
        
        # Verify results
        assert len(results) <= 3
        assert len(results) > 0
        
        # First result should be the exact match (chunk1)
        chunk_id, score, metadata = results[0]
        assert chunk_id == "chunk1"
        assert score > 0.99  # Should be very close to 1.0 for exact match
        
        # Verify results are sorted by score (descending)
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)
    
    def test_search_with_threshold(self, vector_store, sample_embeddings, sample_metadata):
        """Test search with similarity threshold (Requirement 3.5)"""
        chunk_ids = ["chunk1", "chunk2", "chunk3", "chunk4", "chunk5"]
        
        # Store embeddings
        vector_store.store_embeddings(chunk_ids, sample_embeddings, sample_metadata)
        
        # Search with high threshold
        query_embedding = sample_embeddings[0]
        results = vector_store.search(query_embedding, top_k=10, score_threshold=0.95)
        
        # All results should have score >= threshold
        for chunk_id, score, metadata in results:
            assert score >= 0.95
    
    def test_search_with_filters(self, vector_store, sample_embeddings, sample_metadata):
        """Test search with filter conditions"""
        chunk_ids = ["chunk1", "chunk2", "chunk3", "chunk4", "chunk5"]
        
        # Store embeddings
        vector_store.store_embeddings(chunk_ids, sample_embeddings, sample_metadata)
        
        # Search with doc_id filter
        query_embedding = sample_embeddings[0]
        results = vector_store.search(
            query_embedding,
            top_k=10,
            filter_conditions={"doc_id": "doc1"}
        )
        
        # All results should be from doc1
        for chunk_id, score, metadata in results:
            assert metadata["doc_id"] == "doc1"
        
        # Should return at most 2 results (only 2 chunks from doc1)
        assert len(results) <= 2
    
    def test_search_validation(self, vector_store):
        """Test query embedding validation"""
        # Test wrong dimension
        wrong_dim_query = np.random.rand(256).astype(np.float32)
        with pytest.raises(ValueError, match="Query embedding dimension mismatch"):
            vector_store.search(wrong_dim_query)
    
    def test_get_by_chunk_id(self, vector_store, sample_embeddings, sample_metadata):
        """Test retrieving embedding by chunk_id (Requirement 3.4)"""
        chunk_ids = ["chunk1", "chunk2", "chunk3", "chunk4", "chunk5"]
        
        # Store embeddings
        vector_store.store_embeddings(chunk_ids, sample_embeddings, sample_metadata)
        
        # Retrieve by chunk_id
        result = vector_store.get_by_chunk_id("chunk1")
        assert result is not None
        
        embedding, metadata = result
        assert embedding.shape == (384,)
        assert metadata["chunk_id"] == "chunk1"
        assert metadata["doc_id"] == "doc1"
        assert metadata["text"] == "NEFT is a payment system"
        
        # Test non-existent chunk
        result = vector_store.get_by_chunk_id("nonexistent")
        assert result is None
    
    def test_delete_by_chunk_ids(self, vector_store, sample_embeddings, sample_metadata):
        """Test deleting embeddings by chunk IDs"""
        chunk_ids = ["chunk1", "chunk2", "chunk3", "chunk4", "chunk5"]
        
        # Store embeddings
        vector_store.store_embeddings(chunk_ids, sample_embeddings, sample_metadata)
        assert vector_store.count() == 5
        
        # Delete some chunks
        vector_store.delete_by_chunk_ids(["chunk1", "chunk2"])
        assert vector_store.count() == 3
        
        # Verify deleted chunks are gone
        result = vector_store.get_by_chunk_id("chunk1")
        assert result is None
        
        # Verify remaining chunks still exist
        result = vector_store.get_by_chunk_id("chunk3")
        assert result is not None
    
    def test_delete_by_doc_id(self, vector_store, sample_embeddings, sample_metadata):
        """Test deleting all embeddings for a document"""
        chunk_ids = ["chunk1", "chunk2", "chunk3", "chunk4", "chunk5"]
        
        # Store embeddings
        vector_store.store_embeddings(chunk_ids, sample_embeddings, sample_metadata)
        assert vector_store.count() == 5
        
        # Delete all chunks from doc1
        vector_store.delete_by_doc_id("doc1")
        
        # Verify doc1 chunks are gone
        result = vector_store.get_by_chunk_id("chunk1")
        assert result is None
        result = vector_store.get_by_chunk_id("chunk2")
        assert result is None
        
        # Verify other docs still exist
        result = vector_store.get_by_chunk_id("chunk3")
        assert result is not None
    
    def test_batch_upsert(self, vector_store, sample_embeddings, sample_metadata):
        """Test batch upsert updates existing embeddings"""
        chunk_ids = ["chunk1", "chunk2", "chunk3"]
        
        # Store initial embeddings
        vector_store.store_embeddings(
            chunk_ids,
            sample_embeddings[:3],
            sample_metadata[:3]
        )
        assert vector_store.count() == 3
        
        # Update with new embeddings (upsert)
        new_embeddings = np.random.rand(3, 384).astype(np.float32)
        new_embeddings = new_embeddings / np.linalg.norm(new_embeddings, axis=1, keepdims=True)
        new_metadata = [
            {**meta, "text": "Updated text"} for meta in sample_metadata[:3]
        ]
        
        vector_store.store_embeddings(chunk_ids, new_embeddings, new_metadata)
        
        # Count should still be 3 (upsert, not insert)
        assert vector_store.count() == 3
        
        # Verify metadata was updated
        result = vector_store.get_by_chunk_id("chunk1")
        assert result is not None
        _, metadata = result
        assert metadata["text"] == "Updated text"
    
    def test_metadata_preservation(self, vector_store, sample_embeddings, sample_metadata):
        """Test that all metadata fields are preserved (Requirement 3.4)"""
        chunk_ids = ["chunk1"]
        
        # Add extra metadata fields
        metadata_with_extra = [{
            **sample_metadata[0],
            "custom_field": "custom_value",
            "page_number": 42,
        }]
        
        # Store with extra metadata
        vector_store.store_embeddings(
            chunk_ids,
            sample_embeddings[:1],
            metadata_with_extra
        )
        
        # Retrieve and verify all fields are preserved
        result = vector_store.get_by_chunk_id("chunk1")
        assert result is not None
        _, metadata = result
        
        assert metadata["chunk_id"] == "chunk1"
        assert metadata["doc_id"] == "doc1"
        assert metadata["text"] == "NEFT is a payment system"
        assert metadata["breadcrumbs"] == "Banking > Payments > NEFT"
        assert metadata["section"] == "Payments"
        assert metadata["chunk_type"] == "child"
        assert metadata["custom_field"] == "custom_value"
        assert metadata["page_number"] == 42


@pytest.mark.integration
class TestVectorStoreIntegration:
    """Integration tests requiring Qdrant to be running"""
    
    def test_connection_failure(self):
        """Test handling of connection failure"""
        with pytest.raises(Exception):
            VectorStore(url="http://localhost:9999")  # Invalid port
    
    def test_large_batch_storage(self):
        """Test storing a large batch of embeddings"""
        vector_store = VectorStore(
            url="http://localhost:6333",
            collection_name="test_large_batch",
            vector_size=384,
        )
        
        try:
            # Create 100 embeddings
            num_embeddings = 100
            embeddings = np.random.rand(num_embeddings, 384).astype(np.float32)
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            chunk_ids = [f"chunk{i}" for i in range(num_embeddings)]
            metadata = [
                {
                    "doc_id": f"doc{i % 10}",
                    "text": f"Sample text {i}",
                    "breadcrumbs": f"Section {i % 5}",
                    "section": f"Section {i % 5}",
                    "chunk_type": "child",
                }
                for i in range(num_embeddings)
            ]
            
            # Store all embeddings
            vector_store.store_embeddings(chunk_ids, embeddings, metadata)
            
            # Verify count
            assert vector_store.count() == num_embeddings
            
            # Test search returns results
            query_embedding = embeddings[0]
            results = vector_store.search(query_embedding, top_k=10)
            assert len(results) == 10
            
        finally:
            # Cleanup
            vector_store.client.delete_collection("test_large_batch")
            vector_store.close()
