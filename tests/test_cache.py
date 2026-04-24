"""Tests for caching utilities."""

import pytest
import numpy as np
from src.utils.cache import (
    EmbeddingCache,
    EntityResolutionCache,
    CypherQueryCache,
    CrossEncoderScoreCache
)


class TestEmbeddingCache:
    """Test embedding cache functionality."""
    
    def test_cache_put_and_get(self):
        """Test storing and retrieving embeddings."""
        cache = EmbeddingCache(max_size=10)
        
        text = "test text"
        embedding = np.array([0.1, 0.2, 0.3])
        
        # Store embedding
        cache.put(text, embedding)
        
        # Retrieve embedding
        cached = cache.get(text)
        
        assert cached is not None
        assert np.array_equal(cached, embedding)
    
    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = EmbeddingCache(max_size=10)
        
        result = cache.get("nonexistent text")
        
        assert result is None
    
    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = EmbeddingCache(max_size=3)
        
        # Fill cache
        cache.put("text1", np.array([1.0]))
        cache.put("text2", np.array([2.0]))
        cache.put("text3", np.array([3.0]))
        
        # Access text1 to make it recently used
        cache.get("text1")
        
        # Add new item, should evict text2 (least recently used)
        cache.put("text4", np.array([4.0]))
        
        assert cache.get("text1") is not None
        assert cache.get("text2") is None  # Evicted
        assert cache.get("text3") is not None
        assert cache.get("text4") is not None
    
    def test_cache_size(self):
        """Test cache size tracking."""
        cache = EmbeddingCache(max_size=10)
        
        assert cache.size() == 0
        
        cache.put("text1", np.array([1.0]))
        assert cache.size() == 1
        
        cache.put("text2", np.array([2.0]))
        assert cache.size() == 2
    
    def test_cache_clear(self):
        """Test clearing cache."""
        cache = EmbeddingCache(max_size=10)
        
        cache.put("text1", np.array([1.0]))
        cache.put("text2", np.array([2.0]))
        
        assert cache.size() == 2
        
        cache.clear()
        
        assert cache.size() == 0
        assert cache.get("text1") is None


class TestEntityResolutionCache:
    """Test entity resolution cache functionality."""
    
    def test_cache_put_and_get(self):
        """Test storing and retrieving entity mappings."""
        cache = EntityResolutionCache(max_size=10)
        
        entity_name = "NEFT"
        entity_type = "System"
        canonical_id = "entity_123"
        
        # Store mapping
        cache.put(entity_name, entity_type, canonical_id)
        
        # Retrieve mapping
        cached_id = cache.get(entity_name, entity_type)
        
        assert cached_id == canonical_id
    
    def test_cache_case_insensitive(self):
        """Test cache key is case-insensitive."""
        cache = EntityResolutionCache(max_size=10)
        
        cache.put("NEFT", "System", "entity_123")
        
        # Should find with different case
        result = cache.get("neft", "System")
        assert result == "entity_123"
    
    def test_cache_type_specific(self):
        """Test cache is specific to entity type."""
        cache = EntityResolutionCache(max_size=10)
        
        cache.put("NEFT", "System", "entity_123")
        
        # Different type should not match
        result = cache.get("NEFT", "PaymentMode")
        assert result is None


class TestCypherQueryCache:
    """Test Cypher query cache functionality."""
    
    def test_cache_put_and_get(self):
        """Test storing and retrieving query results."""
        cache = CypherQueryCache(max_size=10)
        
        query = "MATCH (n:Entity) RETURN n"
        params = {"name": "NEFT"}
        result = {"nodes": [{"id": 1, "name": "NEFT"}]}
        
        # Store result
        cache.put(query, params, result)
        
        # Retrieve result
        cached = cache.get(query, params)
        
        assert cached == result
    
    def test_cache_params_sensitive(self):
        """Test cache is sensitive to parameter changes."""
        cache = CypherQueryCache(max_size=10)
        
        query = "MATCH (n:Entity {name: $name}) RETURN n"
        
        cache.put(query, {"name": "NEFT"}, {"result": "neft"})
        cache.put(query, {"name": "RTGS"}, {"result": "rtgs"})
        
        # Different params should return different results
        assert cache.get(query, {"name": "NEFT"}) == {"result": "neft"}
        assert cache.get(query, {"name": "RTGS"}) == {"result": "rtgs"}


class TestCrossEncoderScoreCache:
    """Test cross-encoder score cache functionality."""
    
    def test_cache_put_and_get(self):
        """Test storing and retrieving scores."""
        cache = CrossEncoderScoreCache(max_size=10)
        
        query = "What is NEFT?"
        text = "NEFT is a payment system"
        score = 0.85
        
        # Store score
        cache.put(query, text, score)
        
        # Retrieve score
        cached_score = cache.get(query, text)
        
        assert cached_score == score
    
    def test_cache_pair_specific(self):
        """Test cache is specific to query-text pairs."""
        cache = CrossEncoderScoreCache(max_size=10)
        
        cache.put("query1", "text1", 0.8)
        cache.put("query1", "text2", 0.6)
        cache.put("query2", "text1", 0.7)
        
        assert cache.get("query1", "text1") == 0.8
        assert cache.get("query1", "text2") == 0.6
        assert cache.get("query2", "text1") == 0.7
        assert cache.get("query2", "text2") is None
