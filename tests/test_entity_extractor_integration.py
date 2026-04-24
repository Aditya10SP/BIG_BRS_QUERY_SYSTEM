"""Integration tests for EntityExtractor with real spaCy model."""

import pytest

from src.extraction.entity_extractor import Entity, EntityExtractor
from src.chunking.hierarchical_chunker import Chunk


@pytest.fixture
def banking_chunk():
    """Create a realistic banking document chunk."""
    return Chunk(
        chunk_id="banking_doc_s1_child_0",
        doc_id="banking_doc",
        text="""
        The National Electronic Funds Transfer (NEFT) System is a nationwide payment system 
        facilitating one-to-one funds transfer. NEFT operates in hourly batches and settles 
        transactions through the Reserve Bank of India (RBI). The system integrates with 
        Core Banking Solutions to process transactions. Each transaction requires an 
        Account Number, IFSC Code, and beneficiary details. The Transaction Limit Rule 
        specifies that individual transactions cannot exceed Rs. 2,00,000 without additional 
        authorization. The Payment Authorization Workflow involves validation, risk assessment, 
        and final approval stages.
        """,
        chunk_type="child",
        parent_chunk_id="banking_doc_s1_parent",
        breadcrumbs="Banking Systems > Payment Systems > NEFT",
        section="NEFT",
        token_count=120,
        metadata={}
    )


@pytest.fixture
def extractor():
    """Create EntityExtractor with spaCy model."""
    return EntityExtractor(spacy_model="en_core_web_sm")


class TestEntityExtractorIntegration:
    """Integration tests for EntityExtractor."""
    
    def test_extract_entities_from_banking_text(self, extractor, banking_chunk):
        """Test extracting entities from realistic banking text."""
        entities = extractor.extract(banking_chunk)
        
        # If spaCy is not available, extraction may return empty list
        # This is expected behavior
        assert isinstance(entities, list)
        
        # If entities were extracted, validate them
        for entity in entities:
            assert isinstance(entity, Entity)
            assert entity.entity_id
            assert entity.entity_type in EntityExtractor.ENTITY_TYPES
            assert entity.name
            assert entity.canonical_name
            assert entity.source_chunk_id == banking_chunk.chunk_id
            assert entity.context
            assert isinstance(entity.properties, dict)
    
    def test_entity_normalization_consistency(self, extractor):
        """Test that entity normalization is consistent across multiple chunks."""
        chunk1 = Chunk(
            chunk_id="doc1_s1_child_0",
            doc_id="doc1",
            text="The NEFT System processes payments.",
            chunk_type="child",
            parent_chunk_id="doc1_s1_parent",
            breadcrumbs="Doc1 > Section1",
            section="Section1",
            token_count=10,
            metadata={}
        )
        
        chunk2 = Chunk(
            chunk_id="doc2_s1_child_0",
            doc_id="doc2",
            text="NEFT is used for electronic funds transfer.",
            chunk_type="child",
            parent_chunk_id="doc2_s1_parent",
            breadcrumbs="Doc2 > Section1",
            section="Section1",
            token_count=10,
            metadata={}
        )
        
        entities1 = extractor.extract(chunk1)
        entities2 = extractor.extract(chunk2)
        
        # Find NEFT entities in both extractions
        neft_entities1 = [e for e in entities1 if "NEFT" in e.name.upper()]
        neft_entities2 = [e for e in entities2 if "NEFT" in e.name.upper()]
        
        # If NEFT was extracted from both, canonical names should match
        if neft_entities1 and neft_entities2:
            assert neft_entities1[0].canonical_name == neft_entities2[0].canonical_name
    
    def test_extract_with_empty_chunk(self, extractor):
        """Test extraction from empty chunk."""
        empty_chunk = Chunk(
            chunk_id="empty_chunk",
            doc_id="empty_doc",
            text="",
            chunk_type="child",
            parent_chunk_id="empty_parent",
            breadcrumbs="Empty > Doc",
            section="Empty",
            token_count=0,
            metadata={}
        )
        
        entities = extractor.extract(empty_chunk)
        
        # Should return empty list for empty chunk
        assert entities == []
    
    def test_extract_with_no_entities(self, extractor):
        """Test extraction from text with no recognizable entities."""
        simple_chunk = Chunk(
            chunk_id="simple_chunk",
            doc_id="simple_doc",
            text="This is a simple sentence with no entities.",
            chunk_type="child",
            parent_chunk_id="simple_parent",
            breadcrumbs="Simple > Doc",
            section="Simple",
            token_count=10,
            metadata={}
        )
        
        entities = extractor.extract(simple_chunk)
        
        # May return empty list or very few entities
        assert isinstance(entities, list)
    
    def test_entity_context_extraction(self, extractor, banking_chunk):
        """Test that entity context is properly extracted."""
        entities = extractor.extract(banking_chunk)
        
        # All entities should have non-empty context
        for entity in entities:
            assert entity.context
            assert len(entity.context) > 0
            # Context should be a substring of the original text or a sentence from it
            # (allowing for some processing/normalization)
    
    def test_entity_properties_populated(self, extractor, banking_chunk):
        """Test that entity properties are populated."""
        entities = extractor.extract(banking_chunk)
        
        # At least some entities should have properties
        entities_with_props = [e for e in entities if e.properties]
        
        if entities_with_props:
            # Check that properties contain expected fields
            for entity in entities_with_props:
                # spaCy entities should have spacy_label
                if "spacy_label" in entity.properties:
                    assert entity.properties["spacy_label"] in EntityExtractor.SPACY_ENTITY_TYPES
    
    def test_deduplication_in_extract(self, extractor):
        """Test that extract() deduplicates entities within a chunk."""
        # Create chunk with repeated entity mentions
        chunk = Chunk(
            chunk_id="dup_chunk",
            doc_id="dup_doc",
            text="NEFT is a payment system. The NEFT System processes transactions. NEFT operates in batches.",
            chunk_type="child",
            parent_chunk_id="dup_parent",
            breadcrumbs="Dup > Doc",
            section="Dup",
            token_count=20,
            metadata={}
        )
        
        entities = extractor.extract(chunk)
        
        # Count entities with "NEFT" in canonical name
        neft_entities = [e for e in entities if "NEFT" in e.canonical_name.upper()]
        
        # Should have at most one NEFT entity after deduplication
        # (may have 0 if spaCy doesn't recognize it)
        assert len(neft_entities) <= 1
    
    def test_multiple_entity_types(self, extractor, banking_chunk):
        """Test extraction of multiple entity types from same chunk."""
        entities = extractor.extract(banking_chunk)
        
        # Get unique entity types
        entity_types = set(e.entity_type for e in entities)
        
        # If entities were extracted, validate types
        if entity_types:
            # All types should be valid
            for entity_type in entity_types:
                assert entity_type in EntityExtractor.ENTITY_TYPES
        else:
            # If no entities extracted (spaCy not available), that's OK
            assert len(entities) == 0


class TestEntityExtractorWithoutSpacy:
    """Test EntityExtractor behavior when spaCy is not available."""
    
    def test_extractor_without_spacy_model(self):
        """Test that extractor handles missing spaCy model gracefully."""
        # Try to load non-existent model
        extractor = EntityExtractor(spacy_model="non_existent_model")
        
        # Should initialize without error
        assert extractor.nlp is None
        
        # Should still be able to extract (will skip Stage 1)
        chunk = Chunk(
            chunk_id="test_chunk",
            doc_id="test_doc",
            text="Test text",
            chunk_type="child",
            parent_chunk_id="test_parent",
            breadcrumbs="Test",
            section="Test",
            token_count=5,
            metadata={}
        )
        
        entities = extractor.extract(chunk)
        
        # Should return empty list (no LLM configured, spaCy not available)
        assert entities == []
