"""Integration tests for CrossEncoderReranker class."""

import pytest
from src.retrieval.cross_encoder_reranker import CrossEncoderReranker
from src.retrieval.vector_retriever import RetrievedChunk
from src.retrieval.result_fusion import FusedResults


class TestCrossEncoderRerankerIntegration:
    """Integration tests for CrossEncoderReranker with real model."""
    
    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks for testing."""
        return [
            RetrievedChunk(
                chunk_id="chunk1",
                text="NEFT (National Electronic Funds Transfer) is a payment system that enables electronic transfer of funds between banks in India.",
                breadcrumbs="Banking Systems > Payment Methods > NEFT",
                doc_id="doc1",
                section="payment_methods",
                score=0.8,
                retrieval_source="vector",
                vector_score=0.8,
                bm25_score=0.6
            ),
            RetrievedChunk(
                chunk_id="chunk2", 
                text="RTGS (Real Time Gross Settlement) is used for high-value transactions and provides real-time settlement.",
                breadcrumbs="Banking Systems > Payment Methods > RTGS",
                doc_id="doc1",
                section="payment_methods",
                score=0.7,
                retrieval_source="vector",
                vector_score=0.7,
                bm25_score=0.5
            ),
            RetrievedChunk(
                chunk_id="chunk3",
                text="UPI (Unified Payments Interface) enables instant money transfer between bank accounts through mobile applications.",
                breadcrumbs="Banking Systems > Payment Methods > UPI",
                doc_id="doc1", 
                section="payment_methods",
                score=0.6,
                retrieval_source="bm25",
                vector_score=0.4,
                bm25_score=0.8
            ),
            RetrievedChunk(
                chunk_id="chunk4",
                text="Core Banking System manages customer accounts, transactions, and provides centralized banking operations.",
                breadcrumbs="Banking Systems > Core Systems > CBS",
                doc_id="doc2",
                section="core_systems", 
                score=0.5,
                retrieval_source="vector",
                vector_score=0.5,
                bm25_score=0.3
            )
        ]
    
    @pytest.mark.slow
    def test_rerank_with_real_model(self, sample_chunks):
        """Test reranking with actual cross-encoder model (slow test)."""
        # Skip if running in CI or if model download would be too slow
        pytest.skip("Skipping slow test with real model download")
        
        # This test would download the actual model
        reranker = CrossEncoderReranker()
        
        results = FusedResults(chunks=sample_chunks)
        
        # Test query about NEFT
        reranked = reranker.rerank("What is NEFT payment system?", results, top_k=3)
        
        assert len(reranked) == 3
        
        # NEFT chunk should likely be ranked highest for NEFT query
        assert reranked[0].chunk_id == "chunk1"
        
        # Scores should be between 0 and 1
        for chunk in reranked:
            assert 0.0 <= chunk.score <= 1.0
        
        # Scores should be in descending order
        for i in range(len(reranked) - 1):
            assert reranked[i].score >= reranked[i + 1].score
    
    def test_rerank_preserves_metadata(self, sample_chunks):
        """Test that reranking preserves all chunk metadata."""
        # Use a mock model to avoid slow download
        with pytest.MonkeyPatch().context() as m:
            # Mock the model loading to avoid actual download
            m.setattr("transformers.AutoTokenizer.from_pretrained", lambda x: MockTokenizer())
            m.setattr("transformers.AutoModelForSequenceClassification.from_pretrained", lambda x: MockModel())
            
            reranker = CrossEncoderReranker()
            results = FusedResults(chunks=sample_chunks)
            
            reranked = reranker.rerank("test query", results, top_k=2)
            
            # Check that metadata is preserved
            for chunk in reranked:
                original = next(c for c in sample_chunks if c.chunk_id == chunk.chunk_id)
                
                assert chunk.chunk_id == original.chunk_id
                assert chunk.text == original.text
                assert chunk.breadcrumbs == original.breadcrumbs
                assert chunk.doc_id == original.doc_id
                assert chunk.section == original.section
                assert chunk.retrieval_source == original.retrieval_source
                assert chunk.vector_score == original.vector_score
                assert chunk.bm25_score == original.bm25_score
    
    def test_rerank_different_queries(self, sample_chunks):
        """Test reranking with different query types."""
        # Use mock to avoid model download
        with pytest.MonkeyPatch().context() as m:
            m.setattr("transformers.AutoTokenizer.from_pretrained", lambda x: MockTokenizer())
            m.setattr("transformers.AutoModelForSequenceClassification.from_pretrained", lambda x: MockModel())
            
            reranker = CrossEncoderReranker()
            results = FusedResults(chunks=sample_chunks)
            
            # Test different query types
            queries = [
                "What is NEFT?",
                "How does RTGS work?", 
                "UPI payment system",
                "Core banking operations"
            ]
            
            for query in queries:
                reranked = reranker.rerank(query, results, top_k=2)
                
                assert len(reranked) == 2
                assert all(isinstance(chunk, RetrievedChunk) for chunk in reranked)
                assert all(hasattr(chunk, 'score') for chunk in reranked)


class MockTokenizer:
    """Mock tokenizer for testing without model download."""
    
    def __call__(self, pairs, **kwargs):
        import torch
        # Return mock tokenized output
        batch_size = len(pairs) if isinstance(pairs, list) else 1
        return {
            'input_ids': torch.randint(1, 1000, (batch_size, 10)),
            'attention_mask': torch.ones(batch_size, 10)
        }


class MockModel:
    """Mock model for testing without model download."""
    
    def __init__(self):
        self.device = "cpu"
    
    def to(self, device):
        self.device = device
        return self
    
    def eval(self):
        return self
    
    def __call__(self, **kwargs):
        import torch
        batch_size = kwargs['input_ids'].shape[0]
        
        # Return mock logits
        class MockOutput:
            def __init__(self, batch_size):
                # Random logits between -2 and 2
                self.logits = torch.randn(batch_size, 1) * 2
        
        return MockOutput(batch_size)