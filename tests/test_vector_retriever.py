"""Unit tests for VectorRetriever class."""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from src.retrieval.vector_retriever import VectorRetriever, RetrievedChunk


class TestVectorRetriever:
    """Test suite for VectorRetriever class."""
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create mock VectorStore."""
        mock = Mock()
        mock.search = Mock(return_value=[])
        return mock
    
    @pytest.fixture
    def mock_bm25_index(self):
        """Create mock BM25Indexer."""
        mock = Mock()
        mock.search = Mock(return_value=[])
        return mock
    
    @pytest.fixture
    def mock_embedding_generator(self):
        """Create mock EmbeddingGenerator."""
        mock = Mock()
        mock.generate = Mock(return_value=np.random.rand(384))
        return mock
    
    @pytest.fixture
    def mock_doc_store(self):
        """Create mock DatabaseManager."""
        mock = Mock()
        mock.get_chunk_by_id = Mock(return_value=None)
        return mock
    
    @pytest.fixture
    def retriever(self, mock_vector_store, mock_bm25_index, mock_embedding_generator, mock_doc_store):
        """Create VectorRetriever instance with mocks."""
        return VectorRetriever(
            vector_store=mock_vector_store,
            bm25_index=mock_bm25_index,
            embedding_generator=mock_embedding_generator,
            doc_store=mock_doc_store,
            similarity_threshold=0.7,
            rrf_k=60
        )
    
    def test_initialization(self, retriever):
        """Test VectorRetriever initialization."""
        assert retriever.similarity_threshold == 0.7
        assert retriever.rrf_k == 60
        assert retriever.vector_store is not None
        assert retriever.bm25_index is not None
        assert retriever.embedding_generator is not None
        assert retriever.doc_store is not None
    
    def test_retrieve_empty_query_raises_error(self, retriever):
        """Test that empty query raises ValueError."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            retriever.retrieve("")
        
        with pytest.raises(ValueError, match="Query cannot be empty"):
            retriever.retrieve("   ")
    
    def test_retrieve_with_no_results(self, retriever, mock_vector_store, mock_bm25_index):
        """Test retrieve when no results are found."""
        # Setup mocks to return empty results
        mock_vector_store.search.return_value = []
        mock_bm25_index.search.return_value = []
        
        results = retriever.retrieve("test query")
        
        assert results == []
        assert mock_vector_store.search.called
        assert mock_bm25_index.search.called
    
    def test_retrieve_with_vector_results_only(
        self, retriever, mock_vector_store, mock_bm25_index, mock_doc_store
    ):
        """Test retrieve with only vector search results."""
        # Setup vector results above threshold
        mock_vector_store.search.return_value = [
            ("chunk1", 0.9, {"text": "test"}),
            ("chunk2", 0.8, {"text": "test"}),
        ]
        mock_bm25_index.search.return_value = []
        
        # Setup doc store to return chunk data
        mock_doc_store.get_chunk_by_id.side_effect = lambda cid: {
            "chunk_id": cid,
            "text": f"Text for {cid}",
            "breadcrumbs": "Doc > Section",
            "doc_id": "doc1",
            "section": "Section 1"
        }
        
        results = retriever.retrieve("test query", top_k=10)
        
        assert len(results) == 2
        assert all(isinstance(chunk, RetrievedChunk) for chunk in results)
        assert results[0].chunk_id == "chunk1"
        assert results[0].retrieval_source == "vector"
        assert results[0].score > 0
    
    def test_retrieve_with_bm25_results_only(
        self, retriever, mock_vector_store, mock_bm25_index, mock_doc_store
    ):
        """Test retrieve with only BM25 search results."""
        # Setup BM25 results
        mock_vector_store.search.return_value = []
        mock_bm25_index.search.return_value = [
            ("chunk1", 5.2),
            ("chunk2", 4.8),
        ]
        
        # Setup doc store to return chunk data
        mock_doc_store.get_chunk_by_id.side_effect = lambda cid: {
            "chunk_id": cid,
            "text": f"Text for {cid}",
            "breadcrumbs": "Doc > Section",
            "doc_id": "doc1",
            "section": "Section 1"
        }
        
        results = retriever.retrieve("test query", top_k=10)
        
        assert len(results) == 2
        assert results[0].chunk_id == "chunk1"
        assert results[0].retrieval_source == "bm25"
        assert results[0].score > 0
    
    def test_retrieve_with_hybrid_results(
        self, retriever, mock_vector_store, mock_bm25_index, mock_doc_store
    ):
        """Test retrieve with both vector and BM25 results."""
        # Setup both vector and BM25 results with overlap
        mock_vector_store.search.return_value = [
            ("chunk1", 0.9, {"text": "test"}),
            ("chunk2", 0.8, {"text": "test"}),
            ("chunk3", 0.75, {"text": "test"}),
        ]
        mock_bm25_index.search.return_value = [
            ("chunk2", 5.2),  # Overlaps with vector
            ("chunk4", 4.8),
        ]
        
        # Setup doc store to return chunk data
        mock_doc_store.get_chunk_by_id.side_effect = lambda cid: {
            "chunk_id": cid,
            "text": f"Text for {cid}",
            "breadcrumbs": "Doc > Section",
            "doc_id": "doc1",
            "section": "Section 1"
        }
        
        results = retriever.retrieve("test query", top_k=10)
        
        assert len(results) == 4  # chunk1, chunk2, chunk3, chunk4
        
        # chunk2 should have highest score (appears in both)
        chunk2 = next(c for c in results if c.chunk_id == "chunk2")
        assert chunk2.retrieval_source == "vector+bm25"
        assert chunk2.vector_score > 0
        assert chunk2.bm25_score > 0
    
    def test_similarity_threshold_filtering(
        self, retriever, mock_vector_store, mock_bm25_index, mock_doc_store
    ):
        """Test that similarity threshold filters vector results correctly."""
        # Setup vector results with some below threshold
        mock_vector_store.search.return_value = [
            ("chunk1", 0.9, {"text": "test"}),   # Above threshold
            ("chunk2", 0.75, {"text": "test"}),  # Above threshold
            ("chunk3", 0.65, {"text": "test"}),  # Below threshold (0.7)
            ("chunk4", 0.5, {"text": "test"}),   # Below threshold
        ]
        mock_bm25_index.search.return_value = []
        
        # Setup doc store
        mock_doc_store.get_chunk_by_id.side_effect = lambda cid: {
            "chunk_id": cid,
            "text": f"Text for {cid}",
            "breadcrumbs": "Doc > Section",
            "doc_id": "doc1",
            "section": "Section 1"
        }
        
        results = retriever.retrieve("test query", top_k=10)
        
        # Only chunk1 and chunk2 should be returned (above 0.7 threshold)
        assert len(results) == 2
        chunk_ids = [c.chunk_id for c in results]
        assert "chunk1" in chunk_ids
        assert "chunk2" in chunk_ids
        assert "chunk3" not in chunk_ids
        assert "chunk4" not in chunk_ids
    
    def test_top_k_limiting(
        self, retriever, mock_vector_store, mock_bm25_index, mock_doc_store
    ):
        """Test that top_k limits the number of results."""
        # Setup many results
        vector_results = [(f"chunk{i}", 0.9 - i*0.01, {"text": "test"}) for i in range(20)]
        mock_vector_store.search.return_value = vector_results
        mock_bm25_index.search.return_value = []
        
        # Setup doc store
        mock_doc_store.get_chunk_by_id.side_effect = lambda cid: {
            "chunk_id": cid,
            "text": f"Text for {cid}",
            "breadcrumbs": "Doc > Section",
            "doc_id": "doc1",
            "section": "Section 1"
        }
        
        # Request only top 5
        results = retriever.retrieve("test query", top_k=5)
        
        assert len(results) == 5
    
    def test_reciprocal_rank_fusion_calculation(self, retriever):
        """Test RRF score calculation."""
        vector_results = [
            ("chunk1", 0.9),
            ("chunk2", 0.8),
            ("chunk3", 0.7),
        ]
        bm25_results = [
            ("chunk2", 5.0),  # Overlaps with vector
            ("chunk4", 4.0),
        ]
        
        fused = retriever._reciprocal_rank_fusion(vector_results, bm25_results)
        
        # Check that we have all unique chunks
        chunk_ids = [chunk_id for chunk_id, _, _, _, _ in fused]
        assert set(chunk_ids) == {"chunk1", "chunk2", "chunk3", "chunk4"}
        
        # chunk2 should have highest score (appears in both at top positions)
        chunk2_score = next(score for cid, score, _, _, _ in fused if cid == "chunk2")
        chunk1_score = next(score for cid, score, _, _, _ in fused if cid == "chunk1")
        assert chunk2_score > chunk1_score
        
        # Check sources
        chunk2_source = next(src for cid, _, src, _, _ in fused if cid == "chunk2")
        chunk1_source = next(src for cid, _, src, _, _ in fused if cid == "chunk1")
        assert chunk2_source == "vector+bm25"
        assert chunk1_source == "vector"
    
    def test_parallel_search_execution(
        self, retriever, mock_vector_store, mock_bm25_index, mock_embedding_generator
    ):
        """Test that vector and BM25 searches are executed."""
        mock_vector_store.search.return_value = [("chunk1", 0.9, {})]
        mock_bm25_index.search.return_value = [("chunk2", 5.0)]
        
        vector_results, bm25_results = retriever._parallel_search("test query")
        
        # Both searches should be called
        assert mock_embedding_generator.generate.called
        assert mock_vector_store.search.called
        assert mock_bm25_index.search.called
        
        # Results should be returned
        assert len(vector_results) > 0
        assert len(bm25_results) > 0
    
    def test_fetch_chunk_data_with_missing_chunks(self, retriever, mock_doc_store):
        """Test that missing chunks are handled gracefully."""
        fused_results = [
            ("chunk1", 0.5, "vector", 0.9, 0.0),
            ("chunk2", 0.4, "bm25", 0.0, 5.0),
            ("chunk3", 0.3, "vector", 0.8, 0.0),
        ]
        
        # Setup doc store to return None for chunk2 (missing)
        def get_chunk(cid):
            if cid == "chunk2":
                return None
            return {
                "chunk_id": cid,
                "text": f"Text for {cid}",
                "breadcrumbs": "Doc > Section",
                "doc_id": "doc1",
                "section": "Section 1"
            }
        
        mock_doc_store.get_chunk_by_id.side_effect = get_chunk
        
        chunks = retriever._fetch_chunk_data(fused_results)
        
        # Only chunk1 and chunk3 should be returned
        assert len(chunks) == 2
        chunk_ids = [c.chunk_id for c in chunks]
        assert "chunk1" in chunk_ids
        assert "chunk3" in chunk_ids
        assert "chunk2" not in chunk_ids
    
    def test_retrieved_chunk_dataclass(self):
        """Test RetrievedChunk dataclass."""
        chunk = RetrievedChunk(
            chunk_id="chunk1",
            text="Test text",
            breadcrumbs="Doc > Section",
            doc_id="doc1",
            section="Section 1",
            score=0.85,
            retrieval_source="vector+bm25",
            vector_score=0.9,
            bm25_score=5.2
        )
        
        assert chunk.chunk_id == "chunk1"
        assert chunk.text == "Test text"
        assert chunk.score == 0.85
        assert chunk.retrieval_source == "vector+bm25"
        assert chunk.vector_score == 0.9
        assert chunk.bm25_score == 5.2
    
    def test_vector_search_failure_handling(
        self, retriever, mock_vector_store, mock_bm25_index, mock_doc_store
    ):
        """Test that vector search failures are handled gracefully."""
        # Make vector search fail
        mock_vector_store.search.side_effect = Exception("Vector search failed")
        mock_bm25_index.search.return_value = [("chunk1", 5.0)]
        
        # Setup doc store
        mock_doc_store.get_chunk_by_id.return_value = {
            "chunk_id": "chunk1",
            "text": "Text",
            "breadcrumbs": "Doc > Section",
            "doc_id": "doc1",
            "section": "Section 1"
        }
        
        # Should still return BM25 results
        results = retriever.retrieve("test query")
        
        assert len(results) == 1
        assert results[0].retrieval_source == "bm25"
    
    def test_bm25_search_failure_handling(
        self, retriever, mock_vector_store, mock_bm25_index, mock_doc_store
    ):
        """Test that BM25 search failures are handled gracefully."""
        # Make BM25 search fail
        mock_vector_store.search.return_value = [("chunk1", 0.9, {})]
        mock_bm25_index.search.side_effect = Exception("BM25 search failed")
        
        # Setup doc store
        mock_doc_store.get_chunk_by_id.return_value = {
            "chunk_id": "chunk1",
            "text": "Text",
            "breadcrumbs": "Doc > Section",
            "doc_id": "doc1",
            "section": "Section 1"
        }
        
        # Should still return vector results
        results = retriever.retrieve("test query")
        
        assert len(results) == 1
        assert results[0].retrieval_source == "vector"
    
    def test_rrf_with_empty_vector_results(self, retriever):
        """Test RRF when vector results are empty."""
        vector_results = []
        bm25_results = [("chunk1", 5.0), ("chunk2", 4.0)]
        
        fused = retriever._reciprocal_rank_fusion(vector_results, bm25_results)
        
        assert len(fused) == 2
        assert all(src == "bm25" for _, _, src, _, _ in fused)
    
    def test_rrf_with_empty_bm25_results(self, retriever):
        """Test RRF when BM25 results are empty."""
        vector_results = [("chunk1", 0.9), ("chunk2", 0.8)]
        bm25_results = []
        
        fused = retriever._reciprocal_rank_fusion(vector_results, bm25_results)
        
        assert len(fused) == 2
        assert all(src == "vector" for _, _, src, _, _ in fused)
    
    def test_results_sorted_by_score(
        self, retriever, mock_vector_store, mock_bm25_index, mock_doc_store
    ):
        """Test that results are sorted by score in descending order."""
        # Setup results with different scores
        mock_vector_store.search.return_value = [
            ("chunk1", 0.75, {}),
            ("chunk2", 0.9, {}),
            ("chunk3", 0.8, {}),
        ]
        mock_bm25_index.search.return_value = []
        
        # Setup doc store
        mock_doc_store.get_chunk_by_id.side_effect = lambda cid: {
            "chunk_id": cid,
            "text": f"Text for {cid}",
            "breadcrumbs": "Doc > Section",
            "doc_id": "doc1",
            "section": "Section 1"
        }
        
        results = retriever.retrieve("test query")
        
        # Results should be sorted by score descending
        scores = [chunk.score for chunk in results]
        assert scores == sorted(scores, reverse=True)
