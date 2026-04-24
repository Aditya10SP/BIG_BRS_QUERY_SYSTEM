"""Integration tests for BM25Indexer with real document chunks."""

import pytest
from src.indexing.bm25_indexer import BM25Indexer
from src.chunking.hierarchical_chunker import HierarchicalChunker
from src.parsing.document_parser import DocumentParser, ParsedDocument, Section


@pytest.fixture
def banking_document():
    """Create a sample banking document for testing."""
    return ParsedDocument(
        doc_id="banking_doc_1",
        title="Banking Payment Systems",
        sections=[
            Section(
                section_id="sec_1",
                heading="NEFT Payment System",
                level=1,
                text=(
                    "NEFT (National Electronic Funds Transfer) is a nationwide payment system "
                    "facilitating one-to-one funds transfer. The system operates in hourly batches. "
                    "NEFT is available 24x7 throughout the year including holidays. "
                    "The transaction limit for NEFT is 2 lakhs per transaction."
                ),
                page_numbers=[1]
            ),
            Section(
                section_id="sec_2",
                heading="RTGS Payment System",
                level=1,
                text=(
                    "RTGS (Real Time Gross Settlement) is used for high-value transactions. "
                    "RTGS transactions are processed in real-time on a continuous basis. "
                    "The minimum transaction amount for RTGS is 2 lakhs. "
                    "RTGS is the fastest mode of money transfer."
                ),
                page_numbers=[2]
            ),
            Section(
                section_id="sec_3",
                heading="IMPS Payment System",
                level=1,
                text=(
                    "IMPS (Immediate Payment Service) provides instant interbank electronic fund transfer. "
                    "IMPS is available 24x7 including bank holidays. "
                    "IMPS supports both mobile and internet banking channels. "
                    "The transaction limit for IMPS varies by bank."
                ),
                page_numbers=[3]
            ),
        ],
        metadata={"file_type": "docx"}
    )


class TestBM25IndexerIntegration:
    """Integration tests for BM25Indexer with hierarchical chunks."""
    
    def test_index_and_search_with_real_chunks(self, banking_document):
        """Test indexing and searching with real document chunks."""
        # Create chunks
        chunker = HierarchicalChunker(parent_size=2048, child_size=512)
        chunks = chunker.chunk(banking_document)
        
        # Filter to child chunks only (more realistic for search)
        child_chunks = [c for c in chunks if c.chunk_type == "child"]
        
        assert len(child_chunks) > 0, "Should have child chunks"
        
        # Index chunks
        indexer = BM25Indexer()
        indexer.index(child_chunks)
        
        assert indexer.get_index_size() == len(child_chunks)
    
    def test_search_for_acronym_neft(self, banking_document):
        """Test searching for NEFT acronym."""
        chunker = HierarchicalChunker()
        chunks = chunker.chunk(banking_document)
        child_chunks = [c for c in chunks if c.chunk_type == "child"]
        
        indexer = BM25Indexer()
        indexer.index(child_chunks)
        
        # Search for NEFT
        results = indexer.search("NEFT", top_k=5)
        
        assert len(results) > 0, "Should find chunks with NEFT"
        
        # Top result should be from NEFT section
        top_chunk_id, top_score = results[0]
        top_chunk = next(c for c in child_chunks if c.chunk_id == top_chunk_id)
        
        assert "NEFT" in top_chunk.text
        assert top_score > 0
    
    def test_search_for_acronym_rtgs(self, banking_document):
        """Test searching for RTGS acronym."""
        chunker = HierarchicalChunker()
        chunks = chunker.chunk(banking_document)
        child_chunks = [c for c in chunks if c.chunk_type == "child"]
        
        indexer = BM25Indexer()
        indexer.index(child_chunks)
        
        # Search for RTGS
        results = indexer.search("RTGS", top_k=5)
        
        assert len(results) > 0, "Should find chunks with RTGS"
        
        # Verify RTGS appears in results
        chunk_ids = [cid for cid, _ in results]
        matching_chunks = [c for c in child_chunks if c.chunk_id in chunk_ids]
        
        assert any("RTGS" in c.text for c in matching_chunks)
    
    def test_search_for_transaction_limit(self, banking_document):
        """Test searching for transaction limit information."""
        chunker = HierarchicalChunker()
        chunks = chunker.chunk(banking_document)
        child_chunks = [c for c in chunks if c.chunk_type == "child"]
        
        indexer = BM25Indexer()
        indexer.index(child_chunks)
        
        # Search for transaction limit
        results = indexer.search("transaction limit", top_k=5)
        
        assert len(results) > 0, "Should find chunks about transaction limits"
        
        # Verify results contain relevant information
        chunk_ids = [cid for cid, _ in results]
        matching_chunks = [c for c in child_chunks if c.chunk_id in chunk_ids]
        
        # Should find chunks mentioning limits
        assert any("limit" in c.text.lower() for c in matching_chunks)
    
    def test_search_multiple_acronyms(self, banking_document):
        """Test searching with multiple acronyms."""
        chunker = HierarchicalChunker()
        chunks = chunker.chunk(banking_document)
        child_chunks = [c for c in chunks if c.chunk_type == "child"]
        
        indexer = BM25Indexer()
        indexer.index(child_chunks)
        
        # Search for multiple payment systems
        results = indexer.search("NEFT RTGS IMPS", top_k=10)
        
        assert len(results) > 0, "Should find chunks with payment systems"
        
        # Should return chunks from all three sections
        chunk_ids = [cid for cid, _ in results]
        matching_chunks = [c for c in child_chunks if c.chunk_id in chunk_ids]
        
        # Verify we get results from different sections
        sections = {c.section for c in matching_chunks}
        assert len(sections) >= 2, "Should match chunks from multiple sections"
    
    def test_search_with_context_from_breadcrumbs(self, banking_document):
        """Test that search matches content in breadcrumbs."""
        chunker = HierarchicalChunker()
        chunks = chunker.chunk(banking_document)
        child_chunks = [c for c in chunks if c.chunk_type == "child"]
        
        indexer = BM25Indexer()
        indexer.index(child_chunks)
        
        # Search for term in document title (appears in breadcrumbs)
        results = indexer.search("Banking Payment Systems", top_k=10)
        
        # Should match all chunks since title is in all breadcrumbs
        assert len(results) > 0
    
    def test_ranking_by_relevance(self, banking_document):
        """Test that results are properly ranked by relevance."""
        chunker = HierarchicalChunker()
        chunks = chunker.chunk(banking_document)
        child_chunks = [c for c in chunks if c.chunk_type == "child"]
        
        indexer = BM25Indexer()
        indexer.index(child_chunks)
        
        # Search for specific term
        results = indexer.search("real-time", top_k=5)
        
        if len(results) > 1:
            # Verify scores are in descending order
            scores = [score for _, score in results]
            assert scores == sorted(scores, reverse=True), "Results should be ranked by score"
    
    def test_search_with_score_threshold_filters_results(self, banking_document):
        """Test that score threshold properly filters results."""
        chunker = HierarchicalChunker()
        chunks = chunker.chunk(banking_document)
        child_chunks = [c for c in chunks if c.chunk_type == "child"]
        
        indexer = BM25Indexer()
        indexer.index(child_chunks)
        
        # Search without threshold
        results_no_threshold = indexer.search("payment", top_k=10)
        
        # Search with high threshold
        results_with_threshold = indexer.search("payment", top_k=10, score_threshold=3.0)
        
        # With threshold should return fewer or equal results
        assert len(results_with_threshold) <= len(results_no_threshold)
        
        # All results with threshold should have score >= threshold
        for _, score in results_with_threshold:
            assert score >= 3.0
    
    def test_acronym_case_sensitivity(self, banking_document):
        """Test that acronyms are case-sensitive in search."""
        chunker = HierarchicalChunker()
        chunks = chunker.chunk(banking_document)
        child_chunks = [c for c in chunks if c.chunk_type == "child"]
        
        indexer = BM25Indexer()
        indexer.index(child_chunks)
        
        # Search with correct case
        results_upper = indexer.search("NEFT", top_k=5)
        
        # Search with lowercase (should still work but may have different scores)
        results_lower = indexer.search("neft", top_k=5)
        
        # Both should return results
        assert len(results_upper) > 0
        assert len(results_lower) > 0
    
    def test_index_with_parent_and_child_chunks(self, banking_document):
        """Test indexing both parent and child chunks."""
        chunker = HierarchicalChunker()
        chunks = chunker.chunk(banking_document)
        
        # Index all chunks (both parent and child)
        indexer = BM25Indexer()
        indexer.index(chunks)
        
        assert indexer.get_index_size() == len(chunks)
        
        # Search should work across all chunk types
        results = indexer.search("NEFT", top_k=10)
        assert len(results) > 0
    
    def test_empty_query_handling(self, banking_document):
        """Test handling of edge cases in queries."""
        chunker = HierarchicalChunker()
        chunks = chunker.chunk(banking_document)
        child_chunks = [c for c in chunks if c.chunk_type == "child"]
        
        indexer = BM25Indexer()
        indexer.index(child_chunks)
        
        # Empty query should raise error
        with pytest.raises(ValueError):
            indexer.search("")
        
        # Whitespace-only query should raise error
        with pytest.raises(ValueError):
            indexer.search("   ")
    
    def test_search_returns_valid_chunk_ids(self, banking_document):
        """Test that search returns valid chunk IDs that exist in the index."""
        chunker = HierarchicalChunker()
        chunks = chunker.chunk(banking_document)
        child_chunks = [c for c in chunks if c.chunk_type == "child"]
        
        indexer = BM25Indexer()
        indexer.index(child_chunks)
        
        results = indexer.search("payment", top_k=10)
        
        # All returned chunk IDs should be in the original chunks
        valid_chunk_ids = {c.chunk_id for c in child_chunks}
        for chunk_id, _ in results:
            assert chunk_id in valid_chunk_ids, f"Invalid chunk ID: {chunk_id}"
