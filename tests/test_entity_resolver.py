"""Unit tests for EntityResolver class."""

import pytest
import numpy as np

from src.extraction.entity_resolver import EntityResolver, Relationship
from src.extraction.entity_extractor import Entity


@pytest.fixture
def sample_entities():
    """Create sample entities for testing."""
    return [
        Entity(
            entity_id="ent_1",
            entity_type="System",
            name="NEFT",
            canonical_name="NEFT",
            source_chunk_id="chunk_1",
            context="NEFT is a payment system"
        ),
        Entity(
            entity_id="ent_2",
            entity_type="System",
            name="NEFT System",
            canonical_name="NEFT",
            source_chunk_id="chunk_2",
            context="The NEFT System processes payments"
        ),
        Entity(
            entity_id="ent_3",
            entity_type="System",
            name="National Electronic Funds Transfer",
            canonical_name="National Electronic Funds Transfer",
            source_chunk_id="chunk_3",
            context="National Electronic Funds Transfer (NEFT) is used for transfers"
        ),
        Entity(
            entity_id="ent_4",
            entity_type="PaymentMode",
            name="RTGS",
            canonical_name="RTGS",
            source_chunk_id="chunk_4",
            context="RTGS is for large value transfers"
        ),
        Entity(
            entity_id="ent_5",
            entity_type="PaymentMode",
            name="RTGS System",
            canonical_name="RTGS",
            source_chunk_id="chunk_5",
            context="The RTGS System handles real-time transfers"
        )
    ]


@pytest.fixture
def resolver():
    """Create EntityResolver with default threshold."""
    return EntityResolver(similarity_threshold=0.85)


class TestEntityResolver:
    """Test suite for EntityResolver."""
    
    def test_initialization(self, resolver):
        """Test EntityResolver initializes with correct threshold."""
        assert resolver.similarity_threshold == 0.85
    
    def test_initialization_custom_threshold(self):
        """Test EntityResolver with custom threshold."""
        resolver = EntityResolver(similarity_threshold=0.9)
        assert resolver.similarity_threshold == 0.9
    
    def test_resolve_empty_list(self, resolver):
        """Test resolving empty entity list."""
        canonical, relationships = resolver.resolve([])
        
        assert canonical == []
        assert relationships == []
    
    def test_resolve_single_entity(self, resolver):
        """Test resolving single entity."""
        entity = Entity(
            entity_id="ent_1",
            entity_type="System",
            name="NEFT",
            canonical_name="NEFT",
            source_chunk_id="chunk_1",
            context="test"
        )
        
        canonical, relationships = resolver.resolve([entity])
        
        assert len(canonical) == 1
        assert canonical[0].entity_id == "ent_1"
        assert len(relationships) == 0
    
    def test_resolve_duplicate_entities(self, sample_entities):
        """Test resolving duplicate entities creates SAME_AS relationships."""
        # Use lower threshold to catch similar entities
        resolver = EntityResolver(similarity_threshold=0.6)
        
        # Use only NEFT entities (first 3)
        neft_entities = sample_entities[:3]
        
        canonical, relationships = resolver.resolve(neft_entities)
        
        # Should have 1 canonical entity (with threshold 0.6, acronym matching helps)
        assert len(canonical) == 1
        assert canonical[0].entity_type == "System"
        
        # Should have 2 SAME_AS relationships (pointing to canonical)
        assert len(relationships) == 2
        assert all(rel.rel_type == "SAME_AS" for rel in relationships)
    
    def test_resolve_different_types_separate(self, sample_entities):
        """Test entities of different types are resolved separately."""
        # Use lower threshold to merge similar entities
        resolver = EntityResolver(similarity_threshold=0.6)
        
        canonical, relationships = resolver.resolve(sample_entities)
        
        # Should have 2 canonical entities (1 System, 1 PaymentMode)
        assert len(canonical) == 2
        
        types = {e.entity_type for e in canonical}
        assert "System" in types
        assert "PaymentMode" in types
    
    def test_group_by_type(self, resolver, sample_entities):
        """Test grouping entities by type."""
        groups = resolver._group_by_type(sample_entities)
        
        assert "System" in groups
        assert "PaymentMode" in groups
        assert len(groups["System"]) == 3
        assert len(groups["PaymentMode"]) == 2
    
    def test_string_similarity_identical(self, resolver):
        """Test string similarity for identical strings."""
        sim = resolver._string_similarity("NEFT", "NEFT")
        assert sim == 1.0
    
    def test_string_similarity_different(self, resolver):
        """Test string similarity for different strings."""
        sim = resolver._string_similarity("NEFT", "RTGS")
        assert 0.0 <= sim < 1.0
    
    def test_string_similarity_similar(self, resolver):
        """Test string similarity for similar strings."""
        sim = resolver._string_similarity("NEFT", "NEFT System")
        # "NEFT" vs "NEFT System" - 4 chars match out of 11 total
        # Should have some similarity but not very high
        assert sim > 0.3
    
    def test_string_similarity_empty(self, resolver):
        """Test string similarity with empty strings."""
        assert resolver._string_similarity("", "") == 1.0
        assert resolver._string_similarity("NEFT", "") == 0.0
        assert resolver._string_similarity("", "NEFT") == 0.0
    
    def test_levenshtein_distance_identical(self, resolver):
        """Test Levenshtein distance for identical strings."""
        dist = resolver._levenshtein_distance("NEFT", "NEFT")
        assert dist == 0
    
    def test_levenshtein_distance_one_char(self, resolver):
        """Test Levenshtein distance for one character difference."""
        dist = resolver._levenshtein_distance("NEFT", "NEXT")
        assert dist == 1
    
    def test_levenshtein_distance_insertion(self, resolver):
        """Test Levenshtein distance for insertion."""
        dist = resolver._levenshtein_distance("NEFT", "NEFTS")
        assert dist == 1
    
    def test_levenshtein_distance_deletion(self, resolver):
        """Test Levenshtein distance for deletion."""
        dist = resolver._levenshtein_distance("NEFTS", "NEFT")
        assert dist == 1
    
    def test_compute_entity_similarity(self, resolver):
        """Test computing similarity between two entities."""
        entity1 = Entity(
            entity_id="ent_1",
            entity_type="System",
            name="NEFT",
            canonical_name="NEFT",
            source_chunk_id="chunk_1",
            context="test"
        )
        entity2 = Entity(
            entity_id="ent_2",
            entity_type="System",
            name="NEFT System",
            canonical_name="NEFT",
            source_chunk_id="chunk_2",
            context="test"
        )
        
        sim = resolver._compute_entity_similarity(entity1, entity2)
        
        # Should be high similarity (same canonical name)
        assert sim > 0.8
    
    def test_compute_similarity_matrix(self, resolver):
        """Test computing similarity matrix."""
        entities = [
            Entity(
                entity_id=f"ent_{i}",
                entity_type="System",
                name="NEFT",
                canonical_name="NEFT",
                source_chunk_id=f"chunk_{i}",
                context="test"
            )
            for i in range(3)
        ]
        
        matrix = resolver._compute_similarity_matrix(entities)
        
        assert matrix.shape == (3, 3)
        # Diagonal should be 1.0
        assert np.allclose(np.diag(matrix), 1.0)
        # Matrix should be symmetric
        assert np.allclose(matrix, matrix.T)
    
    def test_select_canonical_single(self, resolver):
        """Test selecting canonical from single entity."""
        entity = Entity(
            entity_id="ent_1",
            entity_type="System",
            name="NEFT",
            canonical_name="NEFT",
            source_chunk_id="chunk_1",
            context="test"
        )
        
        canonical = resolver._select_canonical([entity])
        
        assert canonical.entity_id == "ent_1"
    
    def test_select_canonical_prefers_longer_name(self, resolver):
        """Test canonical selection prefers longer canonical name."""
        entities = [
            Entity(
                entity_id="ent_1",
                entity_type="System",
                name="NEFT",
                canonical_name="NEFT",
                source_chunk_id="chunk_1",
                context="short"
            ),
            Entity(
                entity_id="ent_2",
                entity_type="System",
                name="National Electronic Funds Transfer",
                canonical_name="National Electronic Funds Transfer",
                source_chunk_id="chunk_2",
                context="short"
            )
        ]
        
        canonical = resolver._select_canonical(entities)
        
        # Should prefer longer canonical name
        assert len(canonical.canonical_name) > 4
    
    def test_select_canonical_prefers_longer_context(self, resolver):
        """Test canonical selection prefers longer context."""
        entities = [
            Entity(
                entity_id="ent_1",
                entity_type="System",
                name="NEFT",
                canonical_name="NEFT",
                source_chunk_id="chunk_1",
                context="short"
            ),
            Entity(
                entity_id="ent_2",
                entity_type="System",
                name="NEFT",
                canonical_name="NEFT",
                source_chunk_id="chunk_2",
                context="This is a much longer context with more information"
            )
        ]
        
        canonical = resolver._select_canonical(entities)
        
        # Should prefer longer context
        assert "much longer context" in canonical.context
    
    def test_merge_entity_properties(self, resolver):
        """Test merging properties from cluster members."""
        canonical = Entity(
            entity_id="ent_1",
            entity_type="System",
            name="NEFT",
            canonical_name="NEFT",
            source_chunk_id="chunk_1",
            context="test"
        )
        
        members = [
            canonical,
            Entity(
                entity_id="ent_2",
                entity_type="System",
                name="NEFT System",
                canonical_name="NEFT",
                source_chunk_id="chunk_2",
                context="test"
            ),
            Entity(
                entity_id="ent_3",
                entity_type="System",
                name="National Electronic Funds Transfer",
                canonical_name="NEFT",
                source_chunk_id="chunk_3",
                context="test"
            )
        ]
        
        merged = resolver._merge_entity_properties(canonical, members)
        
        # Should have source_chunk_ids from all members
        assert "source_chunk_ids" in merged.properties
        assert len(merged.properties["source_chunk_ids"]) == 3
        assert "chunk_1" in merged.properties["source_chunk_ids"]
        assert "chunk_2" in merged.properties["source_chunk_ids"]
        assert "chunk_3" in merged.properties["source_chunk_ids"]
        
        # Should have mention count
        assert merged.properties["mention_count"] == 3
        
        # Should have aliases
        assert "aliases" in merged.properties
        assert "NEFT System" in merged.properties["aliases"]
        assert "National Electronic Funds Transfer" in merged.properties["aliases"]
    
    def test_create_same_as_relationship(self, resolver):
        """Test creating SAME_AS relationship."""
        source = Entity(
            entity_id="ent_1",
            entity_type="System",
            name="NEFT System",
            canonical_name="NEFT",
            source_chunk_id="chunk_1",
            context="test"
        )
        
        target = Entity(
            entity_id="ent_2",
            entity_type="System",
            name="NEFT",
            canonical_name="NEFT",
            source_chunk_id="chunk_2",
            context="test"
        )
        
        rel = resolver._create_same_as_relationship(source, target, "System")
        
        assert rel.rel_type == "SAME_AS"
        assert rel.source_entity_id == "ent_1"
        assert rel.target_entity_id == "ent_2"
        assert rel.properties["entity_type"] == "System"
        assert rel.properties["source_name"] == "NEFT System"
        assert rel.properties["target_name"] == "NEFT"
        assert rel.properties["canonical_name"] == "NEFT"
    
    def test_cluster_entities_single_cluster(self, resolver):
        """Test clustering entities into single cluster."""
        # Create entities with high similarity
        entities = [
            Entity(
                entity_id=f"ent_{i}",
                entity_type="System",
                name="NEFT",
                canonical_name="NEFT",
                source_chunk_id=f"chunk_{i}",
                context="test"
            )
            for i in range(3)
        ]
        
        # Create similarity matrix (all similar)
        similarity_matrix = np.ones((3, 3))
        
        clusters = resolver._cluster_entities(similarity_matrix, entities)
        
        # Should have 1 cluster with all entities
        assert len(clusters) == 1
        cluster_sizes = [len(members) for members in clusters.values()]
        assert 3 in cluster_sizes
    
    def test_cluster_entities_multiple_clusters(self, resolver):
        """Test clustering entities into multiple clusters."""
        entities = [
            Entity(
                entity_id="ent_1",
                entity_type="System",
                name="NEFT",
                canonical_name="NEFT",
                source_chunk_id="chunk_1",
                context="test"
            ),
            Entity(
                entity_id="ent_2",
                entity_type="System",
                name="RTGS",
                canonical_name="RTGS",
                source_chunk_id="chunk_2",
                context="test"
            )
        ]
        
        # Create similarity matrix (dissimilar)
        similarity_matrix = np.array([
            [1.0, 0.3],
            [0.3, 1.0]
        ])
        
        clusters = resolver._cluster_entities(similarity_matrix, entities)
        
        # Should have 2 clusters
        assert len(clusters) == 2


class TestRelationship:
    """Test Relationship dataclass."""
    
    def test_relationship_creation(self):
        """Test creating a Relationship."""
        rel = Relationship(
            rel_id="rel_1",
            rel_type="SAME_AS",
            source_entity_id="ent_1",
            target_entity_id="ent_2"
        )
        
        assert rel.rel_id == "rel_1"
        assert rel.rel_type == "SAME_AS"
        assert rel.source_entity_id == "ent_1"
        assert rel.target_entity_id == "ent_2"
        assert rel.properties == {}
    
    def test_relationship_with_properties(self):
        """Test creating a Relationship with properties."""
        rel = Relationship(
            rel_id="rel_1",
            rel_type="SAME_AS",
            source_entity_id="ent_1",
            target_entity_id="ent_2",
            properties={"key": "value"}
        )
        
        assert rel.properties == {"key": "value"}


class TestEntityResolutionIntegration:
    """Integration tests for entity resolution."""
    
    def test_resolve_preserves_source_references(self, resolver, sample_entities):
        """Test that resolution preserves all source chunk references."""
        canonical, relationships = resolver.resolve(sample_entities)
        
        # Check that canonical entities have source_chunk_ids
        for entity in canonical:
            if "source_chunk_ids" in entity.properties:
                assert len(entity.properties["source_chunk_ids"]) > 0
    
    def test_resolve_creates_bidirectional_mapping(self, resolver, sample_entities):
        """Test that SAME_AS relationships create proper mapping."""
        canonical, relationships = resolver.resolve(sample_entities)
        
        # All relationships should point to canonical entities
        canonical_ids = {e.entity_id for e in canonical}
        
        for rel in relationships:
            assert rel.target_entity_id in canonical_ids
    
    def test_resolve_different_thresholds(self, sample_entities):
        """Test resolution with different similarity thresholds."""
        # High threshold (strict matching)
        strict_resolver = EntityResolver(similarity_threshold=0.95)
        strict_canonical, strict_rels = strict_resolver.resolve(sample_entities)
        
        # Low threshold (loose matching)
        loose_resolver = EntityResolver(similarity_threshold=0.7)
        loose_canonical, loose_rels = loose_resolver.resolve(sample_entities)
        
        # Strict should have more canonical entities (less merging)
        # Loose should have fewer canonical entities (more merging)
        assert len(strict_canonical) >= len(loose_canonical)
