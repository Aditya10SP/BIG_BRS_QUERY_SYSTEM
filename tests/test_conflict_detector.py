"""Tests for ConflictDetector class."""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock

from src.chunking.hierarchical_chunker import Chunk
from src.extraction.entity_extractor import Entity
from src.extraction.conflict_detector import ConflictDetector
from src.extraction.entity_resolver import Relationship


@pytest.fixture
def conflict_detector():
    """Create ConflictDetector instance with mock LLM configuration."""
    return ConflictDetector(
        ollama_base_url="http://localhost:11434",
        llm_model="llama2"
    )


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing."""
    return [
        Chunk(
            chunk_id="chunk1",
            doc_id="doc1",
            text="NEFT transaction limit is 2 lakhs per transaction.",
            chunk_type="child",
            parent_chunk_id="parent1",
            breadcrumbs="Doc1 > Section1",
            section="Section1",
            token_count=10,
            metadata={}
        ),
        Chunk(
            chunk_id="chunk2",
            doc_id="doc2",
            text="NEFT transaction limit is 5 lakhs per transaction.",
            chunk_type="child",
            parent_chunk_id="parent2",
            breadcrumbs="Doc2 > Section1",
            section="Section1",
            token_count=10,
            metadata={}
        ),
        Chunk(
            chunk_id="chunk3",
            doc_id="doc3",
            text="RTGS is used for high-value transactions.",
            chunk_type="child",
            parent_chunk_id="parent3",
            breadcrumbs="Doc3 > Section1",
            section="Section1",
            token_count=8,
            metadata={}
        )
    ]


@pytest.fixture
def sample_entities():
    """Create sample entities for testing."""
    return [
        Entity(
            entity_id="ent1",
            entity_type="PaymentMode",
            name="NEFT",
            canonical_name="NEFT",
            source_chunk_id="chunk1",
            context="NEFT transaction limit is 2 lakhs per transaction.",
            properties={"limit": "2 lakhs"}
        ),
        Entity(
            entity_id="ent2",
            entity_type="PaymentMode",
            name="NEFT",
            canonical_name="NEFT",
            source_chunk_id="chunk2",
            context="NEFT transaction limit is 5 lakhs per transaction.",
            properties={"limit": "5 lakhs"}
        ),
        Entity(
            entity_id="ent3",
            entity_type="PaymentMode",
            name="RTGS",
            canonical_name="RTGS",
            source_chunk_id="chunk3",
            context="RTGS is used for high-value transactions.",
            properties={}
        )
    ]


class TestConflictDetectorInitialization:
    """Test ConflictDetector initialization."""
    
    def test_initialization_with_llm_config(self):
        """Test initialization with LLM configuration."""
        detector = ConflictDetector(
            ollama_base_url="http://localhost:11434",
            llm_model="llama2"
        )
        
        assert detector.ollama_base_url == "http://localhost:11434"
        assert detector.llm_model == "llama2"
    
    def test_initialization_without_llm_config(self):
        """Test initialization without LLM configuration."""
        detector = ConflictDetector()
        
        assert detector.ollama_base_url is None
        assert detector.llm_model is None


class TestConflictDetection:
    """Test conflict detection functionality."""
    
    def test_detect_with_empty_entities(self, conflict_detector):
        """Test detect with empty entity list."""
        conflicts = conflict_detector.detect([], [])
        
        assert conflicts == []
    
    def test_detect_without_llm_config(self, sample_entities, sample_chunks):
        """Test detect without LLM configuration returns empty list."""
        detector = ConflictDetector()  # No LLM config
        
        conflicts = detector.detect(sample_entities, sample_chunks)
        
        assert conflicts == []
    
    def test_detect_with_single_entity(self, conflict_detector, sample_chunks):
        """Test detect with single entity (no conflicts possible)."""
        entities = [
            Entity(
                entity_id="ent1",
                entity_type="PaymentMode",
                name="NEFT",
                canonical_name="NEFT",
                source_chunk_id="chunk1",
                context="NEFT is a payment mode.",
                properties={}
            )
        ]
        
        conflicts = conflict_detector.detect(entities, sample_chunks)
        
        assert conflicts == []
    
    def test_detect_property_conflict(self, conflict_detector, sample_entities, sample_chunks):
        """Test detection of property conflicts."""
        # Use only entities with property conflicts
        entities = sample_entities[:2]  # Both NEFT entities with different limits
        
        conflicts = conflict_detector.detect(entities, sample_chunks)
        
        # Should detect property conflict without calling LLM
        assert len(conflicts) == 2  # Bidirectional relationships
        
        # Check first relationship
        assert conflicts[0].rel_type == "CONFLICTS_WITH"
        assert conflicts[0].source_entity_id == "ent1"
        assert conflicts[0].target_entity_id == "ent2"
        assert conflicts[0].properties["conflict_type"] == "property"
        assert "limit" in conflicts[0].properties["explanation"]
        assert conflicts[0].properties["source_chunk_ids"] == ["chunk1", "chunk2"]
        
        # Check second relationship (bidirectional)
        assert conflicts[1].rel_type == "CONFLICTS_WITH"
        assert conflicts[1].source_entity_id == "ent2"
        assert conflicts[1].target_entity_id == "ent1"
        assert conflicts[1].properties["conflict_type"] == "property"
    
    @patch('requests.post')
    def test_detect_semantic_conflict(self, mock_post, conflict_detector, sample_chunks):
        """Test detection of semantic conflicts using LLM."""
        # Create entities without property conflicts but with semantic conflicts
        entities = [
            Entity(
                entity_id="ent1",
                entity_type="Rule",
                name="Transaction Rule",
                canonical_name="Transaction Rule",
                source_chunk_id="chunk1",
                context="Approve transactions below 1 lakh.",
                properties={}
            ),
            Entity(
                entity_id="ent2",
                entity_type="Rule",
                name="Transaction Rule",
                canonical_name="Transaction Rule",
                source_chunk_id="chunk2",
                context="Reject transactions below 1 lakh.",
                properties={}
            )
        ]
        
        # Mock LLM response indicating conflict
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": json.dumps({
                "conflicts": True,
                "conflict_type": "rule",
                "explanation": "Contradictory approval rules"
            })
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        conflicts = conflict_detector.detect(entities, sample_chunks)
        
        # Should detect semantic conflict via LLM
        assert len(conflicts) == 2  # Bidirectional
        assert conflicts[0].rel_type == "CONFLICTS_WITH"
        assert conflicts[0].properties["conflict_type"] == "rule"
        assert "Contradictory approval rules" in conflicts[0].properties["explanation"]
    
    @patch('requests.post')
    def test_detect_no_conflict(self, mock_post, conflict_detector, sample_chunks):
        """Test when LLM determines no conflict exists."""
        entities = [
            Entity(
                entity_id="ent1",
                entity_type="System",
                name="NEFT",
                canonical_name="NEFT",
                source_chunk_id="chunk1",
                context="NEFT is a payment system.",
                properties={}
            ),
            Entity(
                entity_id="ent2",
                entity_type="System",
                name="NEFT",
                canonical_name="NEFT",
                source_chunk_id="chunk2",
                context="NEFT enables electronic fund transfers.",
                properties={}
            )
        ]
        
        # Mock LLM response indicating no conflict
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": json.dumps({
                "conflicts": False,
                "conflict_type": None,
                "explanation": "Both statements are complementary"
            })
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        conflicts = conflict_detector.detect(entities, sample_chunks)
        
        # Should not create conflict relationships
        assert len(conflicts) == 0


class TestPropertyConflictDetection:
    """Test property conflict detection."""
    
    def test_check_property_conflict_with_different_values(self, conflict_detector):
        """Test property conflict detection with different values."""
        entity1 = Entity(
            entity_id="ent1",
            entity_type="PaymentMode",
            name="NEFT",
            canonical_name="NEFT",
            source_chunk_id="chunk1",
            context="",
            properties={"limit": "2 lakhs", "fee": "5 rupees"}
        )
        
        entity2 = Entity(
            entity_id="ent2",
            entity_type="PaymentMode",
            name="NEFT",
            canonical_name="NEFT",
            source_chunk_id="chunk2",
            context="",
            properties={"limit": "5 lakhs", "fee": "5 rupees"}
        )
        
        conflict = conflict_detector._check_property_conflict(entity1, entity2)
        
        assert conflict is not None
        assert conflict["conflicts"] is True
        assert conflict["conflict_type"] == "property"
        assert "limit" in conflict["explanation"]
        assert len(conflict["conflicting_properties"]) == 1
        assert conflict["conflicting_properties"][0]["property"] == "limit"
    
    def test_check_property_conflict_with_same_values(self, conflict_detector):
        """Test no conflict when properties have same values."""
        entity1 = Entity(
            entity_id="ent1",
            entity_type="PaymentMode",
            name="NEFT",
            canonical_name="NEFT",
            source_chunk_id="chunk1",
            context="",
            properties={"limit": "2 lakhs"}
        )
        
        entity2 = Entity(
            entity_id="ent2",
            entity_type="PaymentMode",
            name="NEFT",
            canonical_name="NEFT",
            source_chunk_id="chunk2",
            context="",
            properties={"limit": "2 lakhs"}
        )
        
        conflict = conflict_detector._check_property_conflict(entity1, entity2)
        
        assert conflict is None
    
    def test_check_property_conflict_ignores_metadata(self, conflict_detector):
        """Test that metadata properties are ignored."""
        entity1 = Entity(
            entity_id="ent1",
            entity_type="PaymentMode",
            name="NEFT",
            canonical_name="NEFT",
            source_chunk_id="chunk1",
            context="",
            properties={"extraction_method": "llm", "spacy_label": "ORG"}
        )
        
        entity2 = Entity(
            entity_id="ent2",
            entity_type="PaymentMode",
            name="NEFT",
            canonical_name="NEFT",
            source_chunk_id="chunk2",
            context="",
            properties={"extraction_method": "spacy", "spacy_label": "PRODUCT"}
        )
        
        conflict = conflict_detector._check_property_conflict(entity1, entity2)
        
        # Should not detect conflict for metadata properties
        assert conflict is None


class TestConflictRelationshipCreation:
    """Test conflict relationship creation."""
    
    def test_create_bidirectional_relationships(self, conflict_detector, sample_chunks):
        """Test creation of bidirectional CONFLICTS_WITH relationships."""
        entity1 = Entity(
            entity_id="ent1",
            entity_type="PaymentMode",
            name="NEFT",
            canonical_name="NEFT",
            source_chunk_id="chunk1",
            context="NEFT limit is 2 lakhs",
            properties={}
        )
        
        entity2 = Entity(
            entity_id="ent2",
            entity_type="PaymentMode",
            name="NEFT",
            canonical_name="NEFT",
            source_chunk_id="chunk2",
            context="NEFT limit is 5 lakhs",
            properties={}
        )
        
        conflict_data = {
            "conflicts": True,
            "conflict_type": "property",
            "explanation": "Different transaction limits"
        }
        
        relationships = conflict_detector._create_conflict_relationships(
            entity1, entity2, sample_chunks[0], sample_chunks[1], conflict_data
        )
        
        # Should create 2 relationships (bidirectional)
        assert len(relationships) == 2
        
        # Check forward relationship
        assert relationships[0].rel_type == "CONFLICTS_WITH"
        assert relationships[0].source_entity_id == "ent1"
        assert relationships[0].target_entity_id == "ent2"
        assert relationships[0].properties["conflict_type"] == "property"
        assert relationships[0].properties["explanation"] == "Different transaction limits"
        assert relationships[0].properties["source_chunk_ids"] == ["chunk1", "chunk2"]
        
        # Check backward relationship
        assert relationships[1].rel_type == "CONFLICTS_WITH"
        assert relationships[1].source_entity_id == "ent2"
        assert relationships[1].target_entity_id == "ent1"
        assert relationships[1].properties["conflict_type"] == "property"
    
    def test_relationship_metadata_completeness(self, conflict_detector, sample_chunks):
        """Test that conflict relationships have complete metadata."""
        entity1 = Entity(
            entity_id="ent1",
            entity_type="Rule",
            name="Rule1",
            canonical_name="Rule1",
            source_chunk_id="chunk1",
            context="Approve if amount < 1L",
            properties={}
        )
        
        entity2 = Entity(
            entity_id="ent2",
            entity_type="Rule",
            name="Rule1",
            canonical_name="Rule1",
            source_chunk_id="chunk2",
            context="Reject if amount < 1L",
            properties={}
        )
        
        conflict_data = {
            "conflicts": True,
            "conflict_type": "rule",
            "explanation": "Contradictory rules"
        }
        
        relationships = conflict_detector._create_conflict_relationships(
            entity1, entity2, sample_chunks[0], sample_chunks[1], conflict_data
        )
        
        # Check metadata completeness
        rel = relationships[0]
        assert "conflict_type" in rel.properties
        assert "explanation" in rel.properties
        assert "source_chunk_ids" in rel.properties
        assert "doc_ids" in rel.properties
        assert "entity1_context" in rel.properties
        assert "entity2_context" in rel.properties
        
        # Validate metadata values
        assert rel.properties["conflict_type"] == "rule"
        assert rel.properties["explanation"] == "Contradictory rules"
        assert len(rel.properties["source_chunk_ids"]) == 2
        assert len(rel.properties["doc_ids"]) == 2


class TestLLMIntegration:
    """Test LLM integration for semantic conflict detection."""
    
    @patch('requests.post')
    def test_call_ollama_success(self, mock_post, conflict_detector):
        """Test successful Ollama API call."""
        mock_response = Mock()
        mock_response.json.return_value = {"response": "test response"}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        response = conflict_detector._call_ollama("test prompt")
        
        assert response == "test response"
        mock_post.assert_called_once()
    
    @patch('requests.post')
    def test_call_ollama_failure_with_retry(self, mock_post, conflict_detector):
        """Test Ollama API call failure with retry."""
        import requests
        mock_post.side_effect = requests.exceptions.RequestException("Connection error")
        
        response = conflict_detector._call_ollama("test prompt", max_retries=2)
        
        assert response is None
        assert mock_post.call_count == 3  # Initial + 2 retries
    
    def test_parse_conflict_response_valid_json(self, conflict_detector):
        """Test parsing valid JSON conflict response."""
        response = '{"conflicts": true, "conflict_type": "property", "explanation": "Different values"}'
        
        result = conflict_detector._parse_conflict_response(response)
        
        assert result is not None
        assert result["conflicts"] is True
        assert result["conflict_type"] == "property"
    
    def test_parse_conflict_response_json_in_text(self, conflict_detector):
        """Test parsing JSON embedded in text."""
        response = 'Here is the analysis: {"conflicts": true, "conflict_type": "rule", "explanation": "Contradictory"} as you can see.'
        
        result = conflict_detector._parse_conflict_response(response)
        
        assert result is not None
        assert result["conflicts"] is True
        assert result["conflict_type"] == "rule"
    
    def test_parse_conflict_response_invalid_json(self, conflict_detector):
        """Test parsing invalid JSON response."""
        response = "This is not valid JSON"
        
        result = conflict_detector._parse_conflict_response(response)
        
        assert result is None


class TestGroupingByCanonicalName:
    """Test entity grouping by canonical name."""
    
    def test_group_by_canonical_name(self, conflict_detector, sample_entities):
        """Test grouping entities by canonical name."""
        groups = conflict_detector._group_by_canonical_name(sample_entities)
        
        assert len(groups) == 2  # NEFT and RTGS
        assert "neft" in groups
        assert "rtgs" in groups
        assert len(groups["neft"]) == 2
        assert len(groups["rtgs"]) == 1
    
    def test_group_by_canonical_name_case_insensitive(self, conflict_detector):
        """Test grouping is case-insensitive."""
        entities = [
            Entity(
                entity_id="ent1",
                entity_type="System",
                name="NEFT",
                canonical_name="NEFT",
                source_chunk_id="chunk1",
                context="",
                properties={}
            ),
            Entity(
                entity_id="ent2",
                entity_type="System",
                name="neft",
                canonical_name="neft",
                source_chunk_id="chunk2",
                context="",
                properties={}
            ),
            Entity(
                entity_id="ent3",
                entity_type="System",
                name="Neft",
                canonical_name="Neft",
                source_chunk_id="chunk3",
                context="",
                properties={}
            )
        ]
        
        groups = conflict_detector._group_by_canonical_name(entities)
        
        assert len(groups) == 1
        assert "neft" in groups
        assert len(groups["neft"]) == 3


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_detect_with_missing_chunks(self, conflict_detector, sample_entities):
        """Test detection when chunks are missing."""
        # Provide empty chunk list
        conflicts = conflict_detector.detect(sample_entities, [])
        
        # Should handle gracefully without crashing
        assert isinstance(conflicts, list)
    
    def test_detect_with_entities_from_same_chunk(self, conflict_detector, sample_chunks):
        """Test detection when entities are from the same chunk."""
        entities = [
            Entity(
                entity_id="ent1",
                entity_type="System",
                name="NEFT",
                canonical_name="NEFT",
                source_chunk_id="chunk1",
                context="NEFT system",
                properties={"version": "1.0"}
            ),
            Entity(
                entity_id="ent2",
                entity_type="System",
                name="NEFT",
                canonical_name="NEFT",
                source_chunk_id="chunk1",  # Same chunk
                context="NEFT system",
                properties={"version": "2.0"}
            )
        ]
        
        conflicts = conflict_detector.detect(entities, sample_chunks)
        
        # Should still detect property conflict
        assert len(conflicts) == 2  # Bidirectional
