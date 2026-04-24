"""Tests for HierarchicalChunker class."""

import pytest
from src.chunking.hierarchical_chunker import HierarchicalChunker, Chunk
from src.parsing.document_parser import ParsedDocument, Section


class TestHierarchicalChunker:
    """Test suite for HierarchicalChunker."""
    
    @pytest.fixture
    def chunker(self):
        """Create a chunker instance with default settings."""
        return HierarchicalChunker(parent_size=2048, child_size=512, overlap=50)
    
    @pytest.fixture
    def sample_document(self):
        """Create a sample parsed document for testing."""
        section1 = Section(
            section_id="s1",
            heading="Introduction",
            level=1,
            text="This is the introduction. It contains multiple sentences. "
                 "Each sentence provides context. The introduction sets the stage.",
            page_numbers=[1]
        )
        
        section2 = Section(
            section_id="s2",
            heading="Main Content",
            level=1,
            text="This is the main content section. " * 100,  # Long text
            page_numbers=[2, 3]
        )
        
        return ParsedDocument(
            doc_id="test_doc",
            title="Test Document",
            sections=[section1, section2],
            metadata={"test": True}
        )
    
    def test_chunk_creates_parent_and_child_chunks(self, chunker, sample_document):
        """Test that chunking creates both parent and child chunks."""
        chunks = chunker.chunk(sample_document)
        
        # Should have chunks
        assert len(chunks) > 0
        
        # Should have both parent and child chunks
        parent_chunks = [c for c in chunks if c.chunk_type == "parent"]
        child_chunks = [c for c in chunks if c.chunk_type == "child"]
        
        assert len(parent_chunks) > 0
        assert len(child_chunks) > 0
    
    def test_child_chunks_have_parent_reference(self, chunker, sample_document):
        """Test that child chunks reference their parent."""
        chunks = chunker.chunk(sample_document)
        
        parent_chunks = {c.chunk_id: c for c in chunks if c.chunk_type == "parent"}
        child_chunks = [c for c in chunks if c.chunk_type == "child"]
        
        for child in child_chunks:
            assert child.parent_chunk_id is not None
            assert child.parent_chunk_id in parent_chunks
    
    def test_child_chunks_respect_token_limit(self, chunker, sample_document):
        """Test that child chunks do not exceed the token limit."""
        chunks = chunker.chunk(sample_document)
        
        child_chunks = [c for c in chunks if c.chunk_type == "child"]
        
        for child in child_chunks:
            assert child.token_count <= chunker.child_size, \
                f"Child chunk {child.chunk_id} exceeds token limit: {child.token_count} > {chunker.child_size}"
    
    def test_chunks_have_breadcrumbs(self, chunker, sample_document):
        """Test that all chunks have breadcrumb metadata."""
        chunks = chunker.chunk(sample_document)
        
        for chunk in chunks:
            assert chunk.breadcrumbs is not None
            assert len(chunk.breadcrumbs) > 0
            # Breadcrumbs should contain document title
            assert sample_document.title in chunk.breadcrumbs
    
    def test_breadcrumbs_format(self, chunker, sample_document):
        """Test that breadcrumbs follow the expected format."""
        chunks = chunker.chunk(sample_document)
        
        for chunk in chunks:
            # Breadcrumbs should use " > " separator
            assert " > " in chunk.breadcrumbs
            # Should start with document title
            assert chunk.breadcrumbs.startswith(sample_document.title)
    
    def test_chunks_preserve_section_info(self, chunker, sample_document):
        """Test that chunks preserve section information."""
        chunks = chunker.chunk(sample_document)
        
        for chunk in chunks:
            assert chunk.section is not None
            assert chunk.doc_id == sample_document.doc_id
            assert "section_id" in chunk.metadata
    
    def test_empty_section_handling(self, chunker):
        """Test handling of empty sections."""
        doc = ParsedDocument(
            doc_id="empty_doc",
            title="Empty Document",
            sections=[
                Section(
                    section_id="s1",
                    heading="Empty Section",
                    level=1,
                    text="",
                    page_numbers=[]
                )
            ],
            metadata={}
        )
        
        chunks = chunker.chunk(doc)
        
        # Should handle empty sections gracefully (may produce no chunks or empty chunks)
        # The implementation should not crash
        assert isinstance(chunks, list)
    
    def test_single_sentence_section(self, chunker):
        """Test chunking of a section with a single sentence."""
        doc = ParsedDocument(
            doc_id="single_doc",
            title="Single Sentence Document",
            sections=[
                Section(
                    section_id="s1",
                    heading="Single Sentence",
                    level=1,
                    text="This is a single sentence.",
                    page_numbers=[1]
                )
            ],
            metadata={}
        )
        
        chunks = chunker.chunk(doc)
        
        # Should create at least a parent chunk
        parent_chunks = [c for c in chunks if c.chunk_type == "parent"]
        assert len(parent_chunks) > 0
        
        # Parent should contain the sentence
        assert "This is a single sentence." in parent_chunks[0].text
    
    def test_chunk_ids_are_unique(self, chunker, sample_document):
        """Test that all chunk IDs are unique."""
        chunks = chunker.chunk(sample_document)
        
        chunk_ids = [c.chunk_id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids)), "Chunk IDs are not unique"
    
    def test_token_counting(self, chunker):
        """Test that token counting works."""
        text = "This is a test sentence."
        token_count = chunker._count_tokens(text)
        
        # Should return a positive integer
        assert isinstance(token_count, int)
        assert token_count > 0
    
    def test_sentence_splitting(self, chunker):
        """Test sentence splitting functionality."""
        text = "First sentence. Second sentence! Third sentence? Fourth sentence."
        sentences = chunker._split_into_sentences(text)
        
        # Should split into multiple sentences
        assert len(sentences) >= 3
        
        # Each sentence should be non-empty
        for sentence in sentences:
            assert len(sentence.strip()) > 0


class TestHierarchicalChunkerEdgeCases:
    """Test edge cases for HierarchicalChunker."""
    
    def test_very_long_sentence(self):
        """Test handling of a sentence that exceeds child_size."""
        chunker = HierarchicalChunker(parent_size=2048, child_size=50, overlap=10)
        
        # Create a very long sentence
        long_sentence = "word " * 100  # Very long sentence
        
        doc = ParsedDocument(
            doc_id="long_doc",
            title="Long Sentence Document",
            sections=[
                Section(
                    section_id="s1",
                    heading="Long Sentence",
                    level=1,
                    text=long_sentence,
                    page_numbers=[1]
                )
            ],
            metadata={}
        )
        
        chunks = chunker.chunk(doc)
        
        # Should handle long sentences by splitting them
        child_chunks = [c for c in chunks if c.chunk_type == "child"]
        
        # All child chunks should respect token limit
        for child in child_chunks:
            assert child.token_count <= chunker.child_size
    
    def test_multiple_sections(self):
        """Test chunking document with multiple sections."""
        chunker = HierarchicalChunker()
        
        sections = [
            Section(
                section_id=f"s{i}",
                heading=f"Section {i}",
                level=1,
                text=f"This is section {i}. " * 50,
                page_numbers=[i]
            )
            for i in range(1, 4)
        ]
        
        doc = ParsedDocument(
            doc_id="multi_doc",
            title="Multi-Section Document",
            sections=sections,
            metadata={}
        )
        
        chunks = chunker.chunk(doc)
        
        # Should create chunks for each section
        parent_chunks = [c for c in chunks if c.chunk_type == "parent"]
        assert len(parent_chunks) == 3
        
        # Each parent should have corresponding child chunks
        for parent in parent_chunks:
            children = [c for c in chunks if c.parent_chunk_id == parent.chunk_id]
            assert len(children) > 0
