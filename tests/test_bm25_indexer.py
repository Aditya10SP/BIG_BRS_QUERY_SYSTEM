"""Unit tests for BM25Indexer class."""

import pytest
from src.indexing.bm25_indexer import BM25Indexer
from src.chunking.hierarchical_chunker import Chunk


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing."""
    return [
        Chunk(
            chunk_id="chunk_1",
            doc_id="doc_1",
            text="NEFT (National Electronic Funds Transfer) is a payment system.",
            chunk_type="child",
            parent_chunk_id="parent_1",
            breadcrumbs="Banking > Payment Systems > NEFT",
            section="NEFT",
            token_count=15,
            metadata={}
        ),
        Chunk(
            chunk_id="chunk_2",
            doc_id="doc_1",
            text="RTGS (Real Time Gross Settlement) processes high-value transactions.",
            chunk_type="child",
            parent_chunk_id="parent_2",
            breadcrumbs="Banking > Payment Systems > RTGS",
            section="RTGS",
            token_count=12,
            metadata={}
        ),
        Chunk(
            chunk_id="chunk_3",
            doc_id="doc_2",
            text="The transaction limit for NEFT is 2 lakhs per transaction.",
            chunk_type="child",
            parent_chunk_id="parent_3",
            breadcrumbs="Banking > Limits > NEFT Limits",
            section="Transaction Limits",
            token_count=11,
            metadata={}
        ),
    ]


class TestBM25Indexer:
    """Test suite for BM25Indexer class."""
    
    def test_initialization(self):
        """Test BM25Indexer initialization with default parameters."""
        indexer = BM25Indexer()
        assert indexer.k1 == 1.5
        assert indexer.b == 0.75
        assert indexer.bm25 is None
        assert indexer.chunk_ids == []
        assert indexer.corpus_tokens == []
    
    def test_initialization_custom_parameters(self):
        """Test BM25Indexer initialization with custom parameters."""
        indexer = BM25Indexer(k1=2.0, b=0.5)
        assert indexer.k1 == 2.0
        assert indexer.b == 0.5
    
    def test_index_builds_successfully(self, sample_chunks):
        """Test that index() method builds BM25 index successfully."""
        indexer = BM25Indexer()
        indexer.index(sample_chunks)
        
        assert indexer.bm25 is not None
        assert len(indexer.chunk_ids) == 3
        assert len(indexer.corpus_tokens) == 3
        assert indexer.get_index_size() == 3
    
    def test_index_empty_chunks_raises_error(self):
        """Test that indexing empty chunks list raises ValueError."""
        indexer = BM25Indexer()
        with pytest.raises(ValueError, match="Cannot index empty chunks list"):
            indexer.index([])
    
    def test_search_returns_chunk_ids_and_scores(self, sample_chunks):
        """Test that search() returns chunk IDs with BM25 scores."""
        indexer = BM25Indexer()
        indexer.index(sample_chunks)
        
        results = indexer.search("NEFT", top_k=5)
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        # Check result structure
        for chunk_id, score in results:
            assert isinstance(chunk_id, str)
            assert isinstance(score, float)
            assert chunk_id in [c.chunk_id for c in sample_chunks]
    
    def test_search_returns_ranked_results(self, sample_chunks):
        """Test that search results are ranked by BM25 score (descending)."""
        indexer = BM25Indexer()
        indexer.index(sample_chunks)
        
        results = indexer.search("NEFT transaction", top_k=5)
        
        # Verify results are sorted by score descending
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)
    
    def test_search_respects_top_k_limit(self, sample_chunks):
        """Test that search() respects top_k parameter."""
        indexer = BM25Indexer()
        indexer.index(sample_chunks)
        
        results = indexer.search("payment", top_k=2)
        
        assert len(results) <= 2
    
    def test_search_with_score_threshold(self, sample_chunks):
        """Test that search() filters results by score threshold."""
        indexer = BM25Indexer()
        indexer.index(sample_chunks)
        
        results = indexer.search("NEFT", top_k=10, score_threshold=0.5)
        
        # All results should have score >= threshold
        for _, score in results:
            assert score >= 0.5
    
    def test_search_before_index_raises_error(self):
        """Test that searching before indexing raises ValueError."""
        indexer = BM25Indexer()
        with pytest.raises(ValueError, match="Index not built"):
            indexer.search("test")
    
    def test_search_empty_query_raises_error(self, sample_chunks):
        """Test that searching with empty query raises ValueError."""
        indexer = BM25Indexer()
        indexer.index(sample_chunks)
        
        with pytest.raises(ValueError, match="Query cannot be empty"):
            indexer.search("")
        
        with pytest.raises(ValueError, match="Query cannot be empty"):
            indexer.search("   ")
    
    def test_acronym_preservation(self):
        """Test that tokenization preserves acronyms."""
        indexer = BM25Indexer()
        
        # Test various acronym formats
        tokens = indexer._tokenize("NEFT and RTGS are payment systems")
        assert "NEFT" in tokens
        assert "RTGS" in tokens
        assert "and" in tokens  # lowercase
        
        # Test acronym with numbers
        tokens = indexer._tokenize("ISO20022 standard")
        assert "ISO20022" in tokens
        
        # Test acronym with punctuation
        tokens = indexer._tokenize("NEFT, RTGS, and IMPS")
        assert "NEFT" in tokens
        assert "RTGS" in tokens
        assert "IMPS" in tokens
    
    def test_tokenization_lowercase_non_acronyms(self):
        """Test that non-acronyms are converted to lowercase."""
        indexer = BM25Indexer()
        
        tokens = indexer._tokenize("Payment System")
        assert "payment" in tokens
        assert "system" in tokens
        assert "Payment" not in tokens
        assert "System" not in tokens
    
    def test_tokenization_removes_punctuation(self):
        """Test that punctuation is removed from non-acronyms."""
        indexer = BM25Indexer()
        
        tokens = indexer._tokenize("transaction, limit, and processing.")
        assert "transaction" in tokens
        assert "limit" in tokens
        assert "and" in tokens
        assert "processing" in tokens
        # No punctuation tokens
        assert "," not in tokens
        assert "." not in tokens
    
    def test_is_acronym_detection(self):
        """Test acronym detection logic."""
        indexer = BM25Indexer()
        
        # Valid acronyms
        assert indexer._is_acronym("NEFT") is True
        assert indexer._is_acronym("RTGS") is True
        assert indexer._is_acronym("ISO20022") is True
        assert indexer._is_acronym("UPI") is True
        
        # Not acronyms
        assert indexer._is_acronym("Payment") is False
        assert indexer._is_acronym("system") is False
        assert indexer._is_acronym("NEFT,") is True  # Punctuation removed internally
        assert indexer._is_acronym("A") is False  # Too short
        assert indexer._is_acronym("123") is False  # No letters
    
    def test_index_includes_breadcrumbs(self, sample_chunks):
        """Test that indexing includes breadcrumbs for context matching."""
        indexer = BM25Indexer()
        indexer.index(sample_chunks)
        
        # Search for term in breadcrumbs
        results = indexer.search("Limits", top_k=5)
        
        # chunk_3 has "Limits" in breadcrumbs
        chunk_ids = [cid for cid, _ in results]
        assert "chunk_3" in chunk_ids
    
    def test_clear_resets_index(self, sample_chunks):
        """Test that clear() resets the index state."""
        indexer = BM25Indexer()
        indexer.index(sample_chunks)
        
        assert indexer.bm25 is not None
        assert len(indexer.chunk_ids) > 0
        
        indexer.clear()
        
        assert indexer.bm25 is None
        assert indexer.chunk_ids == []
        assert indexer.corpus_tokens == []
    
    def test_get_index_size(self, sample_chunks):
        """Test get_index_size() returns correct count."""
        indexer = BM25Indexer()
        
        assert indexer.get_index_size() == 0
        
        indexer.index(sample_chunks)
        assert indexer.get_index_size() == len(sample_chunks)
    
    def test_reindexing_replaces_old_index(self, sample_chunks):
        """Test that calling index() again replaces the old index."""
        indexer = BM25Indexer()
        indexer.index(sample_chunks)
        
        original_size = indexer.get_index_size()
        
        # Index with fewer chunks
        indexer.index(sample_chunks[:2])
        
        assert indexer.get_index_size() == 2
        assert indexer.get_index_size() != original_size
    
    def test_search_with_multiple_keywords(self, sample_chunks):
        """Test search with multiple keywords."""
        indexer = BM25Indexer()
        indexer.index(sample_chunks)
        
        results = indexer.search("NEFT RTGS payment", top_k=5)
        
        # Should return results for all chunks
        assert len(results) > 0
        
        # Chunks with more matching terms should rank higher
        chunk_ids = [cid for cid, _ in results]
        assert len(chunk_ids) > 0
    
    def test_empty_tokenization(self):
        """Test tokenization of empty or whitespace-only text."""
        indexer = BM25Indexer()
        
        assert indexer._tokenize("") == []
        assert indexer._tokenize("   ") == []
        assert indexer._tokenize("\n\t") == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
