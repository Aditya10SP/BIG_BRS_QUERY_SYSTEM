"""Integration tests for HierarchicalChunker with DocumentParser."""

import pytest
from pathlib import Path
from src.chunking.hierarchical_chunker import HierarchicalChunker
from src.parsing.document_parser import DocumentParser


class TestHierarchicalChunkerIntegration:
    """Integration tests combining DocumentParser and HierarchicalChunker."""
    
    @pytest.fixture
    def parser(self):
        """Create a DocumentParser instance."""
        return DocumentParser()
    
    @pytest.fixture
    def chunker(self):
        """Create a HierarchicalChunker instance."""
        return HierarchicalChunker(parent_size=2048, child_size=512, overlap=50)
    
    def test_end_to_end_document_processing(self, parser, chunker, tmp_path):
        """Test complete flow from document parsing to chunking."""
        # Create a temporary test document
        test_file = tmp_path / "test_doc.txt"
        test_content = """
        Banking System Documentation
        
        Introduction
        This document describes the NEFT payment system. NEFT stands for National Electronic Funds Transfer.
        It is a nationwide payment system facilitating one-to-one funds transfer.
        
        System Architecture
        The NEFT system consists of multiple components. The core banking system interfaces with the NEFT gateway.
        Transaction processing happens in batches. Each batch is processed hourly during business hours.
        
        Transaction Limits
        Individual transactions are limited to 2 lakhs. Daily limits apply per account.
        Special accounts may have higher limits. All limits are configurable by the bank.
        """
        
        # Write content to file (simulating a simple text document)
        test_file.write_text(test_content)
        
        # For this test, we'll create a ParsedDocument manually since we don't have
        # a text parser (only docx and pdf)
        from src.parsing.document_parser import ParsedDocument, Section
        
        sections = [
            Section(
                section_id="s1",
                heading="Introduction",
                level=1,
                text="This document describes the NEFT payment system. NEFT stands for National Electronic Funds Transfer. It is a nationwide payment system facilitating one-to-one funds transfer.",
                page_numbers=[1]
            ),
            Section(
                section_id="s2",
                heading="System Architecture",
                level=1,
                text="The NEFT system consists of multiple components. The core banking system interfaces with the NEFT gateway. Transaction processing happens in batches. Each batch is processed hourly during business hours.",
                page_numbers=[1]
            ),
            Section(
                section_id="s3",
                heading="Transaction Limits",
                level=1,
                text="Individual transactions are limited to 2 lakhs. Daily limits apply per account. Special accounts may have higher limits. All limits are configurable by the bank.",
                page_numbers=[2]
            )
        ]
        
        parsed_doc = ParsedDocument(
            doc_id="banking_doc",
            title="Banking System Documentation",
            sections=sections,
            metadata={"source": "test"}
        )
        
        # Chunk the document
        chunks = chunker.chunk(parsed_doc)
        
        # Verify results
        assert len(chunks) > 0
        
        # Check that we have both parent and child chunks
        parent_chunks = [c for c in chunks if c.chunk_type == "parent"]
        child_chunks = [c for c in chunks if c.chunk_type == "child"]
        
        assert len(parent_chunks) == 3  # One per section
        assert len(child_chunks) > 0
        
        # Verify all chunks have proper metadata
        for chunk in chunks:
            assert chunk.doc_id == "banking_doc"
            assert chunk.breadcrumbs is not None
            assert "Banking System Documentation" in chunk.breadcrumbs
            assert chunk.token_count > 0
            
            if chunk.chunk_type == "child":
                assert chunk.token_count <= 512
                assert chunk.parent_chunk_id is not None
        
        # Verify breadcrumbs contain section information
        intro_chunks = [c for c in chunks if "Introduction" in c.breadcrumbs]
        assert len(intro_chunks) > 0
        
        # Verify content is preserved
        all_text = " ".join(c.text for c in chunks)
        assert "NEFT" in all_text
        assert "payment system" in all_text
        assert "Transaction Limits" in all_text or "Transaction" in all_text
    
    def test_chunking_preserves_document_structure(self, chunker):
        """Test that chunking preserves the hierarchical structure of the document."""
        from src.parsing.document_parser import ParsedDocument, Section
        
        # Create a document with nested structure
        sections = [
            Section(
                section_id="s1",
                heading="Chapter 1: Overview",
                level=1,
                text="This is the overview chapter. " * 20,
                page_numbers=[1]
            ),
            Section(
                section_id="s2",
                heading="Section 1.1: Details",
                level=2,
                text="This section provides details. " * 30,
                page_numbers=[2]
            ),
            Section(
                section_id="s3",
                heading="Chapter 2: Implementation",
                level=1,
                text="This chapter covers implementation. " * 25,
                page_numbers=[3]
            )
        ]
        
        doc = ParsedDocument(
            doc_id="structured_doc",
            title="Technical Manual",
            sections=sections,
            metadata={}
        )
        
        chunks = chunker.chunk(doc)
        
        # Verify each section has corresponding chunks
        for section in sections:
            section_chunks = [c for c in chunks if section.heading in c.breadcrumbs]
            assert len(section_chunks) > 0, f"No chunks found for section: {section.heading}"
            
            # Verify parent chunk exists for each section
            parent = [c for c in section_chunks if c.chunk_type == "parent"]
            assert len(parent) == 1, f"Expected 1 parent chunk for {section.heading}"
    
    def test_token_limit_enforcement(self, chunker):
        """Test that token limits are strictly enforced."""
        from src.parsing.document_parser import ParsedDocument, Section
        
        # Create a document with very long content
        long_text = "This is a sentence that will be repeated many times. " * 200
        
        section = Section(
            section_id="s1",
            heading="Long Section",
            level=1,
            text=long_text,
            page_numbers=[1]
        )
        
        doc = ParsedDocument(
            doc_id="long_doc",
            title="Long Document",
            sections=[section],
            metadata={}
        )
        
        chunks = chunker.chunk(doc)
        
        # Verify all child chunks respect the token limit
        child_chunks = [c for c in chunks if c.chunk_type == "child"]
        
        for child in child_chunks:
            assert child.token_count <= chunker.child_size, \
                f"Child chunk exceeds limit: {child.token_count} > {chunker.child_size}"
        
        # Verify parent chunks respect their limit
        parent_chunks = [c for c in chunks if c.chunk_type == "parent"]
        
        for parent in parent_chunks:
            assert parent.token_count <= chunker.parent_size, \
                f"Parent chunk exceeds limit: {parent.token_count} > {chunker.parent_size}"
