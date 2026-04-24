"""Unit tests for ContextAssembler."""

import pytest
from src.retrieval.context_assembler import ContextAssembler, AssembledContext, Citation
from src.retrieval.vector_retriever import RetrievedChunk


class TestContextAssembler:
    """Test cases for ContextAssembler class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.assembler = ContextAssembler(max_tokens=1000)
        
        # Sample chunks for testing
        self.chunks = [
            RetrievedChunk(
                chunk_id="chunk1",
                text="NEFT is a nationwide payment system that enables bank customers to transfer funds from any bank branch to any individual having an account with any other bank branch in the country.",
                breadcrumbs="Banking Systems > Payment Systems > NEFT",
                doc_id="doc1",
                section="Payment Systems",
                score=0.95,
                retrieval_source="vector"
            ),
            RetrievedChunk(
                chunk_id="chunk2", 
                text="RTGS is a real-time gross settlement system that processes high-value transactions immediately. The minimum amount for RTGS is Rs. 2 lakhs.",
                breadcrumbs="Banking Systems > Payment Systems > RTGS",
                doc_id="doc1",
                section="Payment Systems",
                score=0.88,
                retrieval_source="vector"
            ),
            RetrievedChunk(
                chunk_id="chunk3",
                text="Transaction limits for NEFT are set at Rs. 10 lakhs per transaction for retail customers.",
                breadcrumbs="Banking Rules > Transaction Limits > NEFT Limits",
                doc_id="doc2",
                section="Transaction Limits",
                score=0.82,
                retrieval_source="bm25"
            )
        ]
        
        self.graph_facts = [
            "NEFT DEPENDS_ON Core Banking System",
            "RTGS INTEGRATES_WITH NEFT",
            "Transaction Limit Rule APPLIES_TO NEFT"
        ]
    
    def test_assemble_basic_context(self):
        """Test basic context assembly with chunks and graph facts."""
        query = "What is NEFT and what are its limits?"
        
        result = self.assembler.assemble(query, self.chunks, self.graph_facts)
        
        assert isinstance(result, AssembledContext)
        assert result.context_text is not None
        assert len(result.citations) > 0
        assert result.token_count > 0
        
        # Check that query is included
        assert query in result.context_text
        
        # Check that graph facts are included
        assert "Knowledge Graph Facts:" in result.context_text
        assert "NEFT DEPENDS_ON Core Banking System" in result.context_text
        
        # Check that chunks are included
        assert "Relevant Document Excerpts:" in result.context_text
        assert "NEFT is a nationwide payment system" in result.context_text
        
        # Check citations format
        assert "[doc1:Payment Systems]" in result.context_text
        assert "[doc2:Transaction Limits]" in result.context_text
    
    def test_assemble_with_empty_inputs(self):
        """Test context assembly with empty chunks and facts."""
        query = "What is NEFT?"
        
        result = self.assembler.assemble(query, [], [])
        
        assert isinstance(result, AssembledContext)
        assert query in result.context_text
        assert len(result.citations) == 0
        assert "Knowledge Graph Facts:" not in result.context_text
        assert "Relevant Document Excerpts:" not in result.context_text
    
    def test_assemble_only_chunks(self):
        """Test context assembly with only chunks (no graph facts)."""
        query = "What is NEFT?"
        
        result = self.assembler.assemble(query, self.chunks, [])
        
        assert isinstance(result, AssembledContext)
        assert query in result.context_text
        assert len(result.citations) > 0
        assert "Knowledge Graph Facts:" not in result.context_text
        assert "Relevant Document Excerpts:" in result.context_text
    
    def test_assemble_only_graph_facts(self):
        """Test context assembly with only graph facts (no chunks)."""
        query = "What systems does NEFT depend on?"
        
        result = self.assembler.assemble(query, [], self.graph_facts)
        
        assert isinstance(result, AssembledContext)
        assert query in result.context_text
        assert "Knowledge Graph Facts:" in result.context_text
        assert "NEFT DEPENDS_ON Core Banking System" in result.context_text
        assert "Relevant Document Excerpts:" not in result.context_text
    
    def test_citations_creation(self):
        """Test that citations are created correctly."""
        query = "What is NEFT?"
        
        result = self.assembler.assemble(query, self.chunks[:2], [])
        
        # Should have citations for both chunks (but they share same doc_id:section)
        assert len(result.citations) == 1  # Both chunks have same doc_id:section
        
        citation = result.citations["doc1:Payment Systems"]
        assert citation.doc_id == "doc1"
        assert citation.section == "Payment Systems"
        assert citation.citation_id == "doc1:Payment Systems"
        # The breadcrumbs will be from the last chunk processed (chunk2 - RTGS)
        assert citation.breadcrumbs == "Banking Systems > Payment Systems > RTGS"
    
    def test_empty_query_raises_error(self):
        """Test that empty query raises ValueError."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            self.assembler.assemble("", self.chunks, self.graph_facts)
        
        with pytest.raises(ValueError, match="Query cannot be empty"):
            self.assembler.assemble("   ", self.chunks, self.graph_facts)
    
    def test_token_counting(self):
        """Test that token counting works."""
        query = "What is NEFT?"
        
        result = self.assembler.assemble(query, self.chunks, self.graph_facts)
        
        assert result.token_count > 0
        # Token count should be reasonable for the content
        assert result.token_count < 1000  # Should be well under our limit
    
    def test_context_structure(self):
        """Test that context has the expected structure."""
        query = "What is NEFT and its limits?"
        
        result = self.assembler.assemble(query, self.chunks, self.graph_facts)
        
        lines = result.context_text.split('\n')
        
        # Should start with query
        assert lines[0].startswith("Query:")
        
        # Should have graph facts section
        graph_facts_line = None
        excerpts_line = None
        for i, line in enumerate(lines):
            if "Knowledge Graph Facts:" in line:
                graph_facts_line = i
            elif "Relevant Document Excerpts:" in line:
                excerpts_line = i
        
        assert graph_facts_line is not None
        assert excerpts_line is not None
        assert graph_facts_line < excerpts_line  # Graph facts should come first


class TestContextAssemblerTruncation:
    """Test cases for context truncation functionality."""
    
    def test_truncation_with_small_limit(self):
        """Test that context is truncated when it exceeds token limit."""
        # Use very small token limit to force truncation
        assembler = ContextAssembler(max_tokens=100)
        
        # Create many chunks to exceed limit
        chunks = []
        for i in range(10):
            chunks.append(RetrievedChunk(
                chunk_id=f"chunk{i}",
                text=f"This is a long chunk of text number {i} that contains detailed information about banking systems and payment processing mechanisms.",
                breadcrumbs=f"Doc{i} > Section{i}",
                doc_id=f"doc{i}",
                section=f"Section{i}",
                score=0.9 - i * 0.05,
                retrieval_source="vector"
            ))
        
        query = "What are the banking systems?"
        result = assembler.assemble(query, chunks, [])
        
        # Should be truncated to fit limit
        assert result.token_count <= 100
        
        # Should still have query
        assert query in result.context_text
        
        # Should have fewer chunks than input
        assert len(result.citations) < len(chunks)
    
    def test_truncation_preserves_citations(self):
        """Test that truncation preserves citations for included chunks."""
        assembler = ContextAssembler(max_tokens=200)
        
        chunks = [
            RetrievedChunk(
                chunk_id="chunk1",
                text="Short chunk one.",
                breadcrumbs="Doc1 > Section1",
                doc_id="doc1",
                section="Section1",
                score=0.95,
                retrieval_source="vector"
            ),
            RetrievedChunk(
                chunk_id="chunk2",
                text="This is a much longer chunk that contains a lot of detailed information about banking systems and might cause the context to exceed the token limit when combined with other chunks.",
                breadcrumbs="Doc2 > Section2",
                doc_id="doc2", 
                section="Section2",
                score=0.85,
                retrieval_source="vector"
            )
        ]
        
        query = "What are banking systems?"
        result = assembler.assemble(query, chunks, [])
        
        # All included chunks should have citations
        for citation_id in result.citations:
            assert f"[{citation_id}]" in result.context_text


class TestCitation:
    """Test cases for Citation dataclass."""
    
    def test_citation_creation(self):
        """Test Citation object creation."""
        citation = Citation(
            citation_id="doc1:section1",
            doc_id="doc1",
            section="section1",
            chunk_id="chunk1",
            breadcrumbs="Doc1 > Section1"
        )
        
        assert citation.citation_id == "doc1:section1"
        assert citation.doc_id == "doc1"
        assert citation.section == "section1"
        assert citation.chunk_id == "chunk1"
        assert citation.breadcrumbs == "Doc1 > Section1"