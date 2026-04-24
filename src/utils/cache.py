"""Caching utilities for performance optimization."""

import hashlib
import logging
from functools import lru_cache, wraps
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """
    LRU cache for embedding vectors.
    
    Caches embeddings for frequently queried text chunks to avoid
    recomputation. Uses text hash as key for efficient lookup.
    
    Attributes:
        max_size: Maximum number of embeddings to cache
        cache: Dictionary mapping text hash to embedding vector
        access_order: List tracking access order for LRU eviction
    """
    
    def __init__(self, max_size: int = 10000):
        """
        Initialize embedding cache with maximum size.
        
        Args:
            max_size: Maximum number of embeddings to cache (default: 10000)
        """
        self.max_size = max_size
        self.cache: Dict[str, np.ndarray] = {}
        self.access_order: List[str] = []
        logger.info(f"Initialized EmbeddingCache with max_size={max_size}")
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """
        Get cached embedding for text.
        
        Args:
            text: Input text
        
        Returns:
            Cached embedding vector or None if not found
        """
        key = self._hash_text(text)
        
        if key in self.cache:
            # Update access order (move to end)
            self.access_order.remove(key)
            self.access_order.append(key)
            
            logger.debug(f"Cache hit for text hash: {key[:8]}...")
            return self.cache[key]
        
        logger.debug(f"Cache miss for text hash: {key[:8]}...")
        return None
    
    def put(self, text: str, embedding: np.ndarray) -> None:
        """
        Store embedding in cache.
        
        Args:
            text: Input text
            embedding: Embedding vector to cache
        """
        key = self._hash_text(text)
        
        # Check if cache is full
        if len(self.cache) >= self.max_size and key not in self.cache:
            # Evict least recently used
            lru_key = self.access_order.pop(0)
            del self.cache[lru_key]
            logger.debug(f"Evicted LRU embedding: {lru_key[:8]}...")
        
        # Add to cache
        self.cache[key] = embedding
        
        # Update access order
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
        
        logger.debug(f"Cached embedding for text hash: {key[:8]}...")
    
    def clear(self) -> None:
        """Clear all cached embeddings."""
        self.cache.clear()
        self.access_order.clear()
        logger.info("Cleared embedding cache")
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)
    
    def _hash_text(self, text: str) -> str:
        """
        Generate hash key for text.
        
        Args:
            text: Input text
        
        Returns:
            SHA256 hash of text
        """
        return hashlib.sha256(text.encode('utf-8')).hexdigest()


class EntityResolutionCache:
    """
    LRU cache for entity resolution results.
    
    Caches canonical entity mappings to avoid recomputing
    entity similarity and clustering.
    
    Attributes:
        max_size: Maximum number of entity mappings to cache
        cache: Dictionary mapping entity name to canonical entity ID
        access_order: List tracking access order for LRU eviction
    """
    
    def __init__(self, max_size: int = 5000):
        """
        Initialize entity resolution cache.
        
        Args:
            max_size: Maximum number of mappings to cache (default: 5000)
        """
        self.max_size = max_size
        self.cache: Dict[str, str] = {}
        self.access_order: List[str] = []
        logger.info(f"Initialized EntityResolutionCache with max_size={max_size}")
    
    def get(self, entity_name: str, entity_type: str) -> Optional[str]:
        """
        Get canonical entity ID for entity name.
        
        Args:
            entity_name: Entity name to resolve
            entity_type: Entity type
        
        Returns:
            Canonical entity ID or None if not found
        """
        key = self._make_key(entity_name, entity_type)
        
        if key in self.cache:
            # Update access order
            self.access_order.remove(key)
            self.access_order.append(key)
            
            logger.debug(f"Cache hit for entity: {entity_name} ({entity_type})")
            return self.cache[key]
        
        logger.debug(f"Cache miss for entity: {entity_name} ({entity_type})")
        return None
    
    def put(self, entity_name: str, entity_type: str, canonical_id: str) -> None:
        """
        Store entity resolution mapping.
        
        Args:
            entity_name: Entity name
            entity_type: Entity type
            canonical_id: Canonical entity ID
        """
        key = self._make_key(entity_name, entity_type)
        
        # Check if cache is full
        if len(self.cache) >= self.max_size and key not in self.cache:
            # Evict least recently used
            lru_key = self.access_order.pop(0)
            del self.cache[lru_key]
            logger.debug(f"Evicted LRU entity mapping: {lru_key}")
        
        # Add to cache
        self.cache[key] = canonical_id
        
        # Update access order
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
        
        logger.debug(f"Cached entity mapping: {entity_name} -> {canonical_id}")
    
    def clear(self) -> None:
        """Clear all cached mappings."""
        self.cache.clear()
        self.access_order.clear()
        logger.info("Cleared entity resolution cache")
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)
    
    def _make_key(self, entity_name: str, entity_type: str) -> str:
        """
        Create cache key from entity name and type.
        
        Args:
            entity_name: Entity name
            entity_type: Entity type
        
        Returns:
            Cache key
        """
        return f"{entity_type}:{entity_name.lower()}"


class CypherQueryCache:
    """
    LRU cache for Cypher query results.
    
    Caches graph query results to avoid repeated Neo4j queries
    for common patterns.
    
    Attributes:
        max_size: Maximum number of query results to cache
        cache: Dictionary mapping query hash to result
        access_order: List tracking access order for LRU eviction
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize Cypher query cache.
        
        Args:
            max_size: Maximum number of results to cache (default: 1000)
        """
        self.max_size = max_size
        self.cache: Dict[str, Any] = {}
        self.access_order: List[str] = []
        logger.info(f"Initialized CypherQueryCache with max_size={max_size}")
    
    def get(self, query: str, params: Dict[str, Any]) -> Optional[Any]:
        """
        Get cached query result.
        
        Args:
            query: Cypher query string
            params: Query parameters
        
        Returns:
            Cached result or None if not found
        """
        key = self._hash_query(query, params)
        
        if key in self.cache:
            # Update access order
            self.access_order.remove(key)
            self.access_order.append(key)
            
            logger.debug(f"Cache hit for Cypher query: {key[:8]}...")
            return self.cache[key]
        
        logger.debug(f"Cache miss for Cypher query: {key[:8]}...")
        return None
    
    def put(self, query: str, params: Dict[str, Any], result: Any) -> None:
        """
        Store query result in cache.
        
        Args:
            query: Cypher query string
            params: Query parameters
            result: Query result to cache
        """
        key = self._hash_query(query, params)
        
        # Check if cache is full
        if len(self.cache) >= self.max_size and key not in self.cache:
            # Evict least recently used
            lru_key = self.access_order.pop(0)
            del self.cache[lru_key]
            logger.debug(f"Evicted LRU query result: {lru_key[:8]}...")
        
        # Add to cache
        self.cache[key] = result
        
        # Update access order
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
        
        logger.debug(f"Cached query result: {key[:8]}...")
    
    def clear(self) -> None:
        """Clear all cached results."""
        self.cache.clear()
        self.access_order.clear()
        logger.info("Cleared Cypher query cache")
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)
    
    def _hash_query(self, query: str, params: Dict[str, Any]) -> str:
        """
        Generate hash key for query and parameters.
        
        Args:
            query: Cypher query string
            params: Query parameters
        
        Returns:
            SHA256 hash of query and params
        """
        # Create deterministic string representation
        params_str = str(sorted(params.items()))
        combined = f"{query}|{params_str}"
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()


class CrossEncoderScoreCache:
    """
    LRU cache for cross-encoder relevance scores.
    
    Caches (query, text) pair scores to avoid recomputation
    for frequently seen pairs.
    
    Attributes:
        max_size: Maximum number of scores to cache
        cache: Dictionary mapping pair hash to score
        access_order: List tracking access order for LRU eviction
    """
    
    def __init__(self, max_size: int = 5000):
        """
        Initialize cross-encoder score cache.
        
        Args:
            max_size: Maximum number of scores to cache (default: 5000)
        """
        self.max_size = max_size
        self.cache: Dict[str, float] = {}
        self.access_order: List[str] = []
        logger.info(f"Initialized CrossEncoderScoreCache with max_size={max_size}")
    
    def get(self, query: str, text: str) -> Optional[float]:
        """
        Get cached score for (query, text) pair.
        
        Args:
            query: Query string
            text: Text string
        
        Returns:
            Cached score or None if not found
        """
        key = self._hash_pair(query, text)
        
        if key in self.cache:
            # Update access order
            self.access_order.remove(key)
            self.access_order.append(key)
            
            logger.debug(f"Cache hit for cross-encoder pair: {key[:8]}...")
            return self.cache[key]
        
        logger.debug(f"Cache miss for cross-encoder pair: {key[:8]}...")
        return None
    
    def put(self, query: str, text: str, score: float) -> None:
        """
        Store score in cache.
        
        Args:
            query: Query string
            text: Text string
            score: Relevance score
        """
        key = self._hash_pair(query, text)
        
        # Check if cache is full
        if len(self.cache) >= self.max_size and key not in self.cache:
            # Evict least recently used
            lru_key = self.access_order.pop(0)
            del self.cache[lru_key]
            logger.debug(f"Evicted LRU score: {lru_key[:8]}...")
        
        # Add to cache
        self.cache[key] = score
        
        # Update access order
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
        
        logger.debug(f"Cached cross-encoder score: {key[:8]}...")
    
    def clear(self) -> None:
        """Clear all cached scores."""
        self.cache.clear()
        self.access_order.clear()
        logger.info("Cleared cross-encoder score cache")
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)
    
    def _hash_pair(self, query: str, text: str) -> str:
        """
        Generate hash key for (query, text) pair.
        
        Args:
            query: Query string
            text: Text string
        
        Returns:
            SHA256 hash of pair
        """
        combined = f"{query}|{text}"
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()


def cached_method(cache_attr: str):
    """
    Decorator for caching method results using a cache attribute.
    
    Args:
        cache_attr: Name of the cache attribute on the class instance
    
    Returns:
        Decorated method with caching
    
    Example:
        class MyClass:
            def __init__(self):
                self.embedding_cache = EmbeddingCache()
            
            @cached_method('embedding_cache')
            def generate_embedding(self, text: str) -> np.ndarray:
                # Expensive computation
                return compute_embedding(text)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Get cache from instance
            cache = getattr(self, cache_attr, None)
            
            if cache is None:
                # No cache available, call function directly
                return func(self, *args, **kwargs)
            
            # Try to get from cache
            # Assume first argument is the key
            if args:
                key = args[0]
                cached_result = cache.get(key)
                
                if cached_result is not None:
                    return cached_result
            
            # Cache miss, compute result
            result = func(self, *args, **kwargs)
            
            # Store in cache
            if args:
                key = args[0]
                cache.put(key, result)
            
            return result
        
        return wrapper
    return decorator
