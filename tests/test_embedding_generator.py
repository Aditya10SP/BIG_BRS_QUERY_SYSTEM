"""Unit tests for EmbeddingGenerator"""

import pytest
import numpy as np
from src.embedding.embedding_generator import EmbeddingGenerator


class TestEmbeddingGenerator:
    """Test suite for EmbeddingGenerator class"""
    
    @pytest.fixture
    def generator(self):
        """Create an EmbeddingGenerator instance for testing"""
        return EmbeddingGenerator()
    
    def test_initialization(self, generator):
        """Test that the generator initializes correctly"""
        assert generator.model is not None
        assert generator.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert generator.embedding_dimension == 384
    
    def test_custom_model_initialization(self):
        """Test initialization with a custom model name"""
        generator = EmbeddingGenerator(model_name="sentence-transformers/all-MiniLM-L6-v2")
        assert generator.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert generator.embedding_dimension == 384
    
    def test_generate_single_embedding(self, generator):
        """Test generating a single embedding"""
        text = "This is a test document about banking systems."
        embedding = generator.generate(text)
        
        # Check shape
        assert embedding.shape == (384,)
        
        # Check type
        assert isinstance(embedding, np.ndarray)
        
        # Check L2 normalization (norm should be approximately 1)
        norm = np.linalg.norm(embedding)
        assert np.isclose(norm, 1.0, atol=1e-6)
    
    def test_generate_empty_text_raises_error(self, generator):
        """Test that empty text raises ValueError"""
        with pytest.raises(ValueError, match="Text cannot be empty or None"):
            generator.generate("")
        
        with pytest.raises(ValueError, match="Text cannot be empty or None"):
            generator.generate("   ")
    
    def test_generate_none_text_raises_error(self, generator):
        """Test that None text raises ValueError"""
        with pytest.raises(ValueError, match="Text cannot be empty or None"):
            generator.generate(None)
    
    def test_batch_generate(self, generator):
        """Test generating embeddings in batch"""
        texts = [
            "NEFT is a payment system.",
            "RTGS handles large transactions.",
            "Core Banking System manages accounts."
        ]
        
        embeddings = generator.batch_generate(texts)
        
        # Check shape
        assert embeddings.shape == (3, 384)
        
        # Check type
        assert isinstance(embeddings, np.ndarray)
        
        # Check L2 normalization for all vectors
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)
    
    def test_batch_generate_single_text(self, generator):
        """Test batch generation with a single text"""
        texts = ["Single text for batch processing"]
        embeddings = generator.batch_generate(texts)
        
        assert embeddings.shape == (1, 384)
        norm = np.linalg.norm(embeddings[0])
        assert np.isclose(norm, 1.0, atol=1e-6)
    
    def test_batch_generate_empty_list_raises_error(self, generator):
        """Test that empty list raises ValueError"""
        with pytest.raises(ValueError, match="Texts list cannot be empty"):
            generator.batch_generate([])
    
    def test_batch_generate_with_empty_string_raises_error(self, generator):
        """Test that list with empty string raises ValueError"""
        texts = ["Valid text", "", "Another valid text"]
        with pytest.raises(ValueError, match="Text at index 1 is empty or None"):
            generator.batch_generate(texts)
    
    def test_batch_generate_with_none_raises_error(self, generator):
        """Test that list with None raises ValueError"""
        texts = ["Valid text", None, "Another valid text"]
        with pytest.raises(ValueError, match="Text at index 1 is empty or None"):
            generator.batch_generate(texts)
    
    def test_embedding_consistency(self, generator):
        """Test that the same text produces the same embedding"""
        text = "Consistent embedding test"
        
        embedding1 = generator.generate(text)
        embedding2 = generator.generate(text)
        
        # Embeddings should be identical
        assert np.allclose(embedding1, embedding2, atol=1e-6)
    
    def test_different_texts_produce_different_embeddings(self, generator):
        """Test that different texts produce different embeddings"""
        text1 = "NEFT payment system"
        text2 = "Weather forecast today"
        
        embedding1 = generator.generate(text1)
        embedding2 = generator.generate(text2)
        
        # Embeddings should be different
        assert not np.allclose(embedding1, embedding2, atol=0.1)
    
    def test_similar_texts_have_high_similarity(self, generator):
        """Test that similar texts have high cosine similarity"""
        text1 = "NEFT is a payment system"
        text2 = "NEFT is a payment method"
        
        embedding1 = generator.generate(text1)
        embedding2 = generator.generate(text2)
        
        # Compute cosine similarity (dot product of normalized vectors)
        similarity = np.dot(embedding1, embedding2)
        
        # Similar texts should have high similarity (> 0.8)
        assert similarity > 0.8
    
    def test_dissimilar_texts_have_low_similarity(self, generator):
        """Test that dissimilar texts have low cosine similarity"""
        text1 = "Banking payment system"
        text2 = "Quantum physics theory"
        
        embedding1 = generator.generate(text1)
        embedding2 = generator.generate(text2)
        
        # Compute cosine similarity
        similarity = np.dot(embedding1, embedding2)
        
        # Dissimilar texts should have lower similarity (< 0.5)
        assert similarity < 0.5
    
    def test_batch_vs_single_consistency(self, generator):
        """Test that batch generation produces same results as single generation"""
        texts = [
            "First test text",
            "Second test text",
            "Third test text"
        ]
        
        # Generate individually
        single_embeddings = np.array([generator.generate(text) for text in texts])
        
        # Generate in batch
        batch_embeddings = generator.batch_generate(texts)
        
        # Should be identical
        assert np.allclose(single_embeddings, batch_embeddings, atol=1e-6)
    
    def test_get_embedding_dimension(self, generator):
        """Test getting the embedding dimension"""
        dimension = generator.get_embedding_dimension()
        assert dimension == 384
        assert dimension == generator.embedding_dimension
    
    def test_long_text_handling(self, generator):
        """Test handling of long text (model should truncate automatically)"""
        # Create a very long text (more than model's max sequence length)
        long_text = "Banking system. " * 1000
        
        # Should not raise an error
        embedding = generator.generate(long_text)
        
        assert embedding.shape == (384,)
        norm = np.linalg.norm(embedding)
        assert np.isclose(norm, 1.0, atol=1e-6)
    
    def test_special_characters_handling(self, generator):
        """Test handling of special characters"""
        text = "Payment: $1,000.00 (NEFT) - Account #12345"
        
        embedding = generator.generate(text)
        
        assert embedding.shape == (384,)
        norm = np.linalg.norm(embedding)
        assert np.isclose(norm, 1.0, atol=1e-6)
    
    def test_unicode_text_handling(self, generator):
        """Test handling of unicode characters"""
        text = "Banking system with unicode: ₹ € £ ¥"
        
        embedding = generator.generate(text)
        
        assert embedding.shape == (384,)
        norm = np.linalg.norm(embedding)
        assert np.isclose(norm, 1.0, atol=1e-6)
    
    def test_l2_normalize_single_vector(self, generator):
        """Test L2 normalization of a single vector"""
        vector = np.array([3.0, 4.0, 0.0])
        normalized = generator._l2_normalize(vector)
        
        # Check norm is 1
        norm = np.linalg.norm(normalized)
        assert np.isclose(norm, 1.0, atol=1e-6)
        
        # Check values
        expected = np.array([0.6, 0.8, 0.0])
        assert np.allclose(normalized, expected, atol=1e-6)
    
    def test_l2_normalize_zero_vector(self, generator):
        """Test L2 normalization of zero vector"""
        vector = np.array([0.0, 0.0, 0.0])
        normalized = generator._l2_normalize(vector)
        
        # Should return the zero vector unchanged
        assert np.allclose(normalized, vector)
    
    def test_l2_normalize_batch(self, generator):
        """Test L2 normalization of a batch of vectors"""
        vectors = np.array([
            [3.0, 4.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0]
        ])
        
        normalized = generator._l2_normalize_batch(vectors)
        
        # Check all norms are 1
        norms = np.linalg.norm(normalized, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)
    
    def test_l2_normalize_batch_with_zero_vector(self, generator):
        """Test L2 normalization of batch containing zero vector"""
        vectors = np.array([
            [3.0, 4.0, 0.0],
            [0.0, 0.0, 0.0],  # Zero vector
            [1.0, 0.0, 0.0]
        ])
        
        normalized = generator._l2_normalize_batch(vectors)
        
        # Zero vector should remain zero
        assert np.allclose(normalized[1], [0.0, 0.0, 0.0])
        
        # Other vectors should be normalized
        assert np.isclose(np.linalg.norm(normalized[0]), 1.0, atol=1e-6)
        assert np.isclose(np.linalg.norm(normalized[2]), 1.0, atol=1e-6)
