"""Embedding generation using sentence-transformers"""

import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
import logging

from src.utils.cache import EmbeddingCache
from src.utils.batch_processor import EmbeddingBatchProcessor

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generate vector embeddings for semantic search using sentence-transformers.
    
    This class provides methods to generate embeddings for text chunks using
    the all-MiniLM-L6-v2 model (384 dimensions). Embeddings are L2-normalized
    for cosine similarity search.
    
    Attributes:
        model_name: Name of the sentence-transformers model
        model: Loaded SentenceTransformer model
        embedding_dimension: Dimension of the embedding vectors (384)
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", enable_cache: bool = True, cache_size: int = 10000):
        """
        Initialize the embedding generator with a sentence-transformers model.
        
        Args:
            model_name: Name of the sentence-transformers model to use.
                       Default is "sentence-transformers/all-MiniLM-L6-v2"
                       which produces 384-dimensional embeddings.
            enable_cache: Whether to enable embedding caching (default: True)
            cache_size: Maximum number of embeddings to cache (default: 10000)
        
        Raises:
            Exception: If the model cannot be loaded
        """
        self.model_name = model_name
        self.enable_cache = enable_cache
        
        # Initialize cache if enabled
        if enable_cache:
            self.cache = EmbeddingCache(max_size=cache_size)
            logger.info(f"Embedding cache enabled with size={cache_size}")
        else:
            self.cache = None
            logger.info("Embedding cache disabled")
        
        logger.info(f"Initializing EmbeddingGenerator with model: {model_name}")
        
        try:
            self.model = SentenceTransformer(model_name)
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dimension}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise
    
    def generate(self, text: str) -> np.ndarray:
        """
        Generate a normalized embedding vector for a single text.
        
        Args:
            text: Input text to embed
        
        Returns:
            L2-normalized embedding vector as numpy array of shape (embedding_dimension,)
        
        Raises:
            ValueError: If text is empty or None
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty or None")
        
        # Check cache first
        if self.cache:
            cached_embedding = self.cache.get(text)
            if cached_embedding is not None:
                logger.debug(f"Using cached embedding for text of length {len(text)}")
                return cached_embedding
        
        try:
            # Generate embedding
            embedding = self.model.encode(text, convert_to_numpy=True)
            
            # L2 normalize for cosine similarity
            normalized_embedding = self._l2_normalize(embedding)
            
            # Store in cache
            if self.cache:
                self.cache.put(text, normalized_embedding)
            
            logger.debug(f"Generated embedding for text of length {len(text)}")
            return normalized_embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise
    
    def batch_generate(self, texts: List[str]) -> np.ndarray:
        """
        Generate normalized embedding vectors for multiple texts efficiently.
        
        This method processes multiple texts in a single batch for better performance.
        Uses cache for texts that have been embedded before.
        
        Args:
            texts: List of input texts to embed
        
        Returns:
            L2-normalized embedding matrix as numpy array of shape (num_texts, embedding_dimension)
        
        Raises:
            ValueError: If texts list is empty or contains empty strings
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")
        
        # Validate all texts are non-empty
        for i, text in enumerate(texts):
            if not text or not text.strip():
                raise ValueError(f"Text at index {i} is empty or None")
        
        # Check cache for each text
        cached_embeddings = []
        texts_to_compute = []
        text_indices = []
        
        if self.cache:
            for i, text in enumerate(texts):
                cached = self.cache.get(text)
                if cached is not None:
                    cached_embeddings.append((i, cached))
                else:
                    texts_to_compute.append(text)
                    text_indices.append(i)
        else:
            texts_to_compute = texts
            text_indices = list(range(len(texts)))
        
        try:
            # Generate embeddings for uncached texts
            if texts_to_compute:
                # Determine optimal batch size based on text characteristics
                optimal_batch_size = EmbeddingBatchProcessor.get_optimal_batch_size(texts_to_compute)
                
                embeddings = self.model.encode(
                    texts_to_compute,
                    convert_to_numpy=True,
                    batch_size=optimal_batch_size,
                    show_progress_bar=len(texts_to_compute) > 32  # Show progress for large batches
                )
                
                # L2 normalize all embeddings
                normalized_embeddings = self._l2_normalize_batch(embeddings)
                
                # Store in cache
                if self.cache:
                    for text, embedding in zip(texts_to_compute, normalized_embeddings):
                        self.cache.put(text, embedding)
                
                logger.info(
                    f"Generated {len(texts_to_compute)} new embeddings in batch "
                    f"(batch_size={optimal_batch_size})"
                )
            else:
                normalized_embeddings = np.array([])
            
            # Combine cached and newly computed embeddings in correct order
            if cached_embeddings:
                result = np.zeros((len(texts), self.embedding_dimension))
                
                # Place cached embeddings
                for idx, embedding in cached_embeddings:
                    result[idx] = embedding
                
                # Place newly computed embeddings
                for i, idx in enumerate(text_indices):
                    result[idx] = normalized_embeddings[i]
                
                logger.info(f"Used {len(cached_embeddings)} cached embeddings, generated {len(texts_to_compute)} new")
                return result
            else:
                return normalized_embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {str(e)}")
            raise
    
    def _l2_normalize(self, vector: np.ndarray) -> np.ndarray:
        """
        L2 normalize a single vector for cosine similarity.
        
        Args:
            vector: Input vector to normalize
        
        Returns:
            L2-normalized vector
        """
        norm = np.linalg.norm(vector)
        if norm == 0:
            logger.warning("Zero vector encountered during normalization")
            return vector
        return vector / norm
    
    def _l2_normalize_batch(self, vectors: np.ndarray) -> np.ndarray:
        """
        L2 normalize a batch of vectors for cosine similarity.
        
        Args:
            vectors: Input matrix of vectors to normalize (shape: num_vectors x dimension)
        
        Returns:
            L2-normalized matrix
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return vectors / norms
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            Embedding dimension (384 for all-MiniLM-L6-v2)
        """
        return self.embedding_dimension
