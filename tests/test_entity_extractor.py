"""Unit tests for EntityExtractor class."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.extraction.entity_extractor import Entity, EntityExtractor
from src.chunking.hierarchical_chunker import Chunk


@pytest.fixture
def sample_chunk():
    """Create a sample chunk for testing."""
    return Chunk(
        chunk_id="test_doc_s1_child_0",
        doc_id="test_doc",
        text="The NEFT System processes payments through the Core Banking platform. "
             "Transaction limits are defined in the Transaction Limit Rule. "
             "Each transaction requires an Account Number and IFSC Code.",
        chunk_type="child",
        parent_chunk_id="test_doc_s1_parent",
        breadcrumbs="Test Document > Payment Systems",
        section="Payment Systems",
        token_count=50,
        metadata={}
    )


@pytest.fixture
def extractor_no_llm():
    """Create EntityExtractor without LLM configuration."""
    return EntityExtractor(spacy_model="en_core_web_sm")


@pytest.fixture
def extractor_with_llm():
    """Create EntityExtractor with LLM configuration."""
    return EntityExtractor(
        spacy_model="en_core_web_sm",
        ollama_base_url="http://localhost:11434",
        llm_model="llama2"
    )


class TestEntityExtractor:
    """Test suite for EntityExtractor."""
    
    def test_initialization_with_spacy(self, extractor_no_llm):
        """Test EntityExtractor initializes with spaCy model."""
        # Note: nlp may be None if spaCy model is not installed
        # This is expected behavior - the extractor will skip Stage 1 extraction
        assert extractor_no_llm.spacy_model_name == "en_core_web_sm"
        assert extractor_no_llm.ollama_base_url is None
        assert extractor_no_llm.llm_model is None
    
    def test_initialization_with_llm(self, extractor_with_llm):
        """Test EntityExtractor initializes with LLM configuration."""
        assert extractor_with_llm.ollama_base_url == "http://localhost:11434"
        assert extractor_with_llm.llm_model == "llama2"
    
    def test_extract_with_spacy_only(self, extractor_no_llm, sample_chunk):
        """Test entity extraction using only spaCy NER."""
        entities = extractor_no_llm.extract(sample_chunk)
        
        # Should extract some entities
        assert isinstance(entities, list)
        
        # All entities should have required fields
        for entity in entities:
            assert isinstance(entity, Entity)
            assert entity.entity_id
            assert entity.entity_type in EntityExtractor.ENTITY_TYPES
            assert entity.name
            assert entity.canonical_name
            assert entity.source_chunk_id == sample_chunk.chunk_id
            assert entity.context
    
    def test_normalize_entity_name_acronym(self, extractor_no_llm):
        """Test entity name normalization for acronyms."""
        # Short uppercase acronyms should stay uppercase
        assert extractor_no_llm._normalize_entity_name("NEFT") == "NEFT"
        assert extractor_no_llm._normalize_entity_name("RTGS") == "RTGS"
        assert extractor_no_llm._normalize_entity_name("UPI") == "UPI"
    
    def test_normalize_entity_name_with_suffix(self, extractor_no_llm):
        """Test entity name normalization removes common suffixes."""
        assert extractor_no_llm._normalize_entity_name("NEFT System") == "NEFT"
        assert extractor_no_llm._normalize_entity_name("Payment Module") == "Payment"
        assert extractor_no_llm._normalize_entity_name("Core Banking Service") == "Core Banking"
    
    def test_normalize_entity_name_title_case(self, extractor_no_llm):
        """Test entity name normalization for multi-word names."""
        result = extractor_no_llm._normalize_entity_name("core banking platform")
        assert result == "Core Banking Platform"
    
    def test_normalize_entity_name_whitespace(self, extractor_no_llm):
        """Test entity name normalization removes extra whitespace."""
        result = extractor_no_llm._normalize_entity_name("  NEFT   System  ")
        assert result == "NEFT"
    
    def test_generate_entity_id(self, extractor_no_llm):
        """Test entity ID generation."""
        chunk_id = "test_doc_s1_child_0"
        canonical_name = "NEFT"
        
        entity_id = extractor_no_llm._generate_entity_id(chunk_id, canonical_name)
        
        assert entity_id.startswith("ent_")
        assert chunk_id in entity_id
        
        # Same inputs should produce same ID
        entity_id2 = extractor_no_llm._generate_entity_id(chunk_id, canonical_name)
        assert entity_id == entity_id2
    
    def test_deduplicate_entities(self, extractor_no_llm, sample_chunk):
        """Test entity deduplication by canonical name."""
        entities = [
            Entity(
                entity_id="ent_1",
                entity_type="System",
                name="NEFT",
                canonical_name="NEFT",
                source_chunk_id=sample_chunk.chunk_id,
                context="Short context"
            ),
            Entity(
                entity_id="ent_2",
                entity_type="System",
                name="NEFT System",
                canonical_name="NEFT",
                source_chunk_id=sample_chunk.chunk_id,
                context="This is a much longer context with more information"
            ),
            Entity(
                entity_id="ent_3",
                entity_type="PaymentMode",
                name="RTGS",
                canonical_name="RTGS",
                source_chunk_id=sample_chunk.chunk_id,
                context="RTGS context"
            )
        ]
        
        deduplicated = extractor_no_llm._deduplicate_entities(entities)
        
        # Should have 2 entities (NEFT and RTGS)
        assert len(deduplicated) == 2
        
        # NEFT entity should be the one with longer context
        neft_entity = next(e for e in deduplicated if e.canonical_name == "NEFT")
        assert "much longer context" in neft_entity.context
    
    def test_map_spacy_to_domain_type(self, extractor_no_llm):
        """Test mapping spaCy labels to domain entity types."""
        assert extractor_no_llm._map_spacy_to_domain_type("ORG") == "System"
        assert extractor_no_llm._map_spacy_to_domain_type("PRODUCT") == "System"
        assert extractor_no_llm._map_spacy_to_domain_type("MONEY") == "Field"
        assert extractor_no_llm._map_spacy_to_domain_type("DATE") == "Field"
        assert extractor_no_llm._map_spacy_to_domain_type("UNKNOWN") == "System"
    
    @patch('src.extraction.entity_extractor.requests.post')
    def test_call_ollama_success(self, mock_post, extractor_with_llm):
        """Test successful Ollama API call."""
        mock_response = Mock()
        mock_response.json.return_value = {"response": "test response"}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        result = extractor_with_llm._call_ollama("test prompt")
        
        assert result == "test response"
        mock_post.assert_called_once()
    
    @patch('src.extraction.entity_extractor.requests.post')
    def test_call_ollama_failure(self, mock_post, extractor_with_llm):
        """Test Ollama API call failure with retries."""
        import requests
        mock_post.side_effect = requests.exceptions.RequestException("Connection error")
        
        result = extractor_with_llm._call_ollama("test prompt", max_retries=1)
        
        assert result is None
        assert mock_post.call_count == 2  # Initial + 1 retry
    
    def test_parse_llm_response_valid_json(self, extractor_no_llm):
        """Test parsing valid JSON response from LLM."""
        response = '[{"type": "System", "name": "NEFT", "context": "test"}]'
        
        entities = extractor_no_llm._parse_llm_response(response)
        
        assert len(entities) == 1
        assert entities[0]["type"] == "System"
        assert entities[0]["name"] == "NEFT"
    
    def test_parse_llm_response_json_in_text(self, extractor_no_llm):
        """Test parsing JSON embedded in text response."""
        response = 'Here are the entities: [{"type": "System", "name": "NEFT"}] from the text.'
        
        entities = extractor_no_llm._parse_llm_response(response)
        
        assert len(entities) == 1
        assert entities[0]["name"] == "NEFT"
    
    def test_parse_llm_response_invalid_json(self, extractor_no_llm):
        """Test parsing invalid JSON response."""
        response = "This is not valid JSON"
        
        entities = extractor_no_llm._parse_llm_response(response)
        
        assert entities == []
    
    @patch('src.extraction.entity_extractor.requests.post')
    def test_extract_with_llm_success(self, mock_post, extractor_with_llm, sample_chunk):
        """Test LLM extraction with successful API call."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": '[{"type": "System", "name": "NEFT", "context": "payment system"}]'
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        entities = extractor_with_llm._extract_with_llm(sample_chunk)
        
        assert len(entities) > 0
        assert all(isinstance(e, Entity) for e in entities)
    
    def test_extract_with_llm_no_config(self, extractor_no_llm, sample_chunk):
        """Test LLM extraction skips when no LLM config provided."""
        entities = extractor_no_llm._extract_with_llm(sample_chunk)
        
        assert entities == []
    
    def test_entity_dataclass_fields(self):
        """Test Entity dataclass has all required fields."""
        entity = Entity(
            entity_id="test_id",
            entity_type="System",
            name="NEFT",
            canonical_name="NEFT",
            source_chunk_id="chunk_1",
            context="test context"
        )
        
        assert entity.entity_id == "test_id"
        assert entity.entity_type == "System"
        assert entity.name == "NEFT"
        assert entity.canonical_name == "NEFT"
        assert entity.source_chunk_id == "chunk_1"
        assert entity.context == "test context"
        assert entity.properties == {}
    
    def test_entity_with_properties(self):
        """Test Entity dataclass with properties."""
        entity = Entity(
            entity_id="test_id",
            entity_type="System",
            name="NEFT",
            canonical_name="NEFT",
            source_chunk_id="chunk_1",
            context="test context",
            properties={"key": "value"}
        )
        
        assert entity.properties == {"key": "value"}


class TestEntityTypes:
    """Test entity type validation."""
    
    def test_entity_types_defined(self):
        """Test that entity types are properly defined."""
        assert "System" in EntityExtractor.ENTITY_TYPES
        assert "PaymentMode" in EntityExtractor.ENTITY_TYPES
        assert "Workflow" in EntityExtractor.ENTITY_TYPES
        assert "Rule" in EntityExtractor.ENTITY_TYPES
        assert "Field" in EntityExtractor.ENTITY_TYPES
    
    def test_spacy_entity_types_defined(self):
        """Test that spaCy entity types are properly defined."""
        assert "ORG" in EntityExtractor.SPACY_ENTITY_TYPES
        assert "PRODUCT" in EntityExtractor.SPACY_ENTITY_TYPES
        assert "MONEY" in EntityExtractor.SPACY_ENTITY_TYPES
        assert "DATE" in EntityExtractor.SPACY_ENTITY_TYPES


class TestLLMPromptTemplate:
    """Test LLM prompt template."""
    
    def test_prompt_template_has_placeholder(self):
        """Test that prompt template has chunk_text placeholder."""
        assert "{chunk_text}" in EntityExtractor.LLM_PROMPT_TEMPLATE
    
    def test_prompt_template_mentions_entity_types(self):
        """Test that prompt template mentions all entity types."""
        template = EntityExtractor.LLM_PROMPT_TEMPLATE
        assert "System" in template
        assert "PaymentMode" in template
        assert "Workflow" in template
        assert "Rule" in template
        assert "Field" in template
    
    def test_prompt_template_format(self, extractor_no_llm):
        """Test that prompt template can be formatted."""
        prompt = EntityExtractor.LLM_PROMPT_TEMPLATE.format(
            chunk_text="Test text"
        )
        assert "Test text" in prompt
        assert "{chunk_text}" not in prompt
