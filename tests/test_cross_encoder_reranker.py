"""Tests for CrossEncoderReranker class."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import torch
import numpy as np

from src.retrieval.cross_encoder_reranker import CrossEncoderReranker
from src.retrieval.vector_retriever import RetrievedChunk
from src.retrieval.result_fusion import FusedResults


class TestCrossEncoderReranker:
    """Test suite for CrossEncoderReranker class."""
    
    @patch('src.retrieval.cross_encoder_reranker.AutoTokenizer')
    @patch('src.retrieval.cross_encoder_reranker.AutoModelForSequenceClassification')
    def test_initialization_success(self, mock_model_class, mock_tokenizer_class):
        """Test successful initialization of CrossEncoderReranker."""
        # Mock tokenizer and model
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model
        
        reranker = CrossEncoderReranker(
            model_name="test-model",
            max_length=256
        )
        
        assert reranker.model_name == "test-model"
        assert reranker.max_length == 256
        assert reranker.tokenizer == mock_tokenizer
        assert reranker.model == mock_model
        
        # Verify model setup calls
        mock_tokenizer_class.from_pretrained.assert_called_once_with("test-model")
        mock_model_class.from_pretrained.assert_called_once_with("test-model")
        mock_model.to.assert_called_once()
        mock_model.eval.assert_called_once()
    
    @patch('src.retrieval.cross_encoder_reranker.AutoTokenizer')
    @patch('src.retrieval.cross_encoder_reranker.AutoModelForSequenceClassification')
    def test_initialization_default_params(self, mock_model_class, mock_tokenizer_class):
        """Test initialization with default parameters."""
        mock_tokenizer_class.from_pretrained.return_value = Mock()
        mock_model_class.from_pretrained.return_value = Mock()
        
        reranker = CrossEncoderReranker()
        
        assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert reranker.max_length == 512
    
    @patch('src.retrieval.cross_encoder_reranker.AutoTokenizer')
    @patch('src.retrieval.cross_encoder_reranker.AutoModelForSequenceClassification')
    def test_initialization_failure(self, mock_model_class, mock_tokenizer_class):
        """Test initialization failure when model loading fails."""
        mock_tokenizer_class.from_pretrained.side_effect = Exception("Model not found")
        
        with pytest.raises(Exception, match="Model not found"):
            CrossEncoderReranker()
    
    def test_rerank_empty_query(self):
        """Test rerank with empty query raises ValueError."""
        with patch('src.retrieval.cross_encoder_reranker.AutoTokenizer'), \
             patch('src.retrieval.cross_encoder_reranker.AutoModelForSequenceClassification'):
            reranker = CrossEncoderReranker()
            
            results = FusedResults(chunks=[])
            
            with pytest.raises(ValueError, match="Query cannot be empty"):
                reranker.rerank("", results)
            
            with pytest.raises(ValueError, match="Query cannot be empty"):
                reranker.rerank("   ", results)
    
    def test_rerank_empty_chunks(self):
        """Test rerank with empty chunks returns empty list."""
        with patch('src.retrieval.cross_encoder_reranker.AutoTokenizer'), \
             patch('src.retrieval.cross_encoder_reranker.AutoModelForSequenceClassification'):
            reranker = CrossEncoderReranker()
            
            results = FusedResults(chunks=[])
            
            reranked = reranker.rerank("test query", results)
            
            assert reranked == []
    
    @patch('src.retrieval.cross_encoder_reranker.AutoTokenizer')
    @patch('src.retrieval.cross_encoder_reranker.AutoModelForSequenceClassification')
    def test_rerank_success(self, mock_model_class, mock_tokenizer_class):
        """Test successful reranking of chunks."""
        # Setup mocks
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Mock tokenizer output
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3], [4, 5, 6]]),
            'attention_mask': torch.tensor([[1, 1, 1], [1, 1, 1]])
        }
        
        # Mock model output (logits for 2 chunks)
        mock_outputs = Mock()
        mock_outputs.logits = torch.tensor([[2.0], [1.0]])  # Higher score for first chunk
        mock_model.return_value = mock_outputs
        
        reranker = CrossEncoderReranker()
        
        # Create test chunks
        chunks = [
            RetrievedChunk(
                chunk_id="chunk1",
                text="First chunk text",
                breadcrumbs="Doc > Section1",
                doc_id="doc1",
                section="section1",
                score=0.5,
                retrieval_source="vector",
                vector_score=0.5,
                bm25_score=0.3
            ),
            RetrievedChunk(
                chunk_id="chunk2",
                text="Second chunk text",
                breadcrumbs="Doc > Section2",
                doc_id="doc1",
                section="section2",
                score=0.8,
                retrieval_source="bm25",
                vector_score=0.2,
                bm25_score=0.8
            )
        ]
        
        results = FusedResults(chunks=chunks)
        
        # Rerank
        reranked = reranker.rerank("test query", results, top_k=2)
        
        # Verify results
        assert len(reranked) == 2
        
        # First chunk should have higher score (sigmoid(2.0) ≈ 0.88)
        assert reranked[0].chunk_id == "chunk1"
        assert reranked[0].score > reranked[1].score
        
        # Second chunk should have lower score (sigmoid(1.0) ≈ 0.73)
        assert reranked[1].chunk_id == "chunk2"
        
        # Verify metadata preservation
        assert reranked[0].text == "First chunk text"
        assert reranked[0].breadcrumbs == "Doc > Section1"
        assert reranked[0].doc_id == "doc1"
        assert reranked[0].section == "section1"
        assert reranked[0].retrieval_source == "vector"
        assert reranked[0].vector_score == 0.5
        assert reranked[0].bm25_score == 0.3
    
    @patch('src.retrieval.cross_encoder_reranker.AutoTokenizer')
    @patch('src.retrieval.cross_encoder_reranker.AutoModelForSequenceClassification')
    def test_rerank_top_k_limit(self, mock_model_class, mock_tokenizer_class):
        """Test that rerank respects top_k limit."""
        # Setup mocks
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model
        
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2], [3, 4], [5, 6]]),
            'attention_mask': torch.tensor([[1, 1], [1, 1], [1, 1]])
        }
        
        # Mock scores in descending order
        mock_outputs = Mock()
        mock_outputs.logits = torch.tensor([[3.0], [2.0], [1.0]])
        mock_model.return_value = mock_outputs
        
        reranker = CrossEncoderReranker()
        
        # Create 3 test chunks
        chunks = [
            RetrievedChunk(
                chunk_id=f"chunk{i}",
                text=f"Chunk {i} text",
                breadcrumbs="Doc > Section",
                doc_id="doc1",
                section="section1",
                score=0.5,
                retrieval_source="vector"
            )
            for i in range(3)
        ]
        
        results = FusedResults(chunks=chunks)
        
        # Rerank with top_k=2
        reranked = reranker.rerank("test query", results, top_k=2)
        
        # Should return only 2 chunks
        assert len(reranked) == 2
        
        # Should be sorted by score (highest first)
        assert reranked[0].score > reranked[1].score
    
    @patch('src.retrieval.cross_encoder_reranker.AutoTokenizer')
    @patch('src.retrieval.cross_encoder_reranker.AutoModelForSequenceClassification')
    def test_rerank_fewer_chunks_than_top_k(self, mock_model_class, mock_tokenizer_class):
        """Test rerank when fewer chunks available than top_k."""
        # Setup mocks
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model
        
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2]]),
            'attention_mask': torch.tensor([[1, 1]])
        }
        
        mock_outputs = Mock()
        mock_outputs.logits = torch.tensor([[1.5]])
        mock_model.return_value = mock_outputs
        
        reranker = CrossEncoderReranker()
        
        # Create only 1 chunk
        chunks = [
            RetrievedChunk(
                chunk_id="chunk1",
                text="Single chunk",
                breadcrumbs="Doc > Section",
                doc_id="doc1",
                section="section1",
                score=0.5,
                retrieval_source="vector"
            )
        ]
        
        results = FusedResults(chunks=chunks)
        
        # Request top_k=5 but only 1 chunk available
        reranked = reranker.rerank("test query", results, top_k=5)
        
        # Should return only 1 chunk
        assert len(reranked) == 1
        assert reranked[0].chunk_id == "chunk1"
    
    @patch('src.retrieval.cross_encoder_reranker.AutoTokenizer')
    @patch('src.retrieval.cross_encoder_reranker.AutoModelForSequenceClassification')
    def test_score_pairs_single_pair(self, mock_model_class, mock_tokenizer_class):
        """Test scoring a single query-text pair."""
        # Setup mocks
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model
        
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        
        # Mock single logit output
        mock_outputs = Mock()
        mock_outputs.logits = torch.tensor([1.5])  # Single value, not 2D
        mock_model.return_value = mock_outputs
        
        reranker = CrossEncoderReranker()
        
        # Test single pair
        pairs = [("query", "text")]
        scores = reranker._score_pairs(pairs)
        
        assert len(scores) == 1
        # sigmoid(1.5) ≈ 0.818
        assert 0.8 < scores[0] < 0.85
    
    @patch('src.retrieval.cross_encoder_reranker.AutoTokenizer')
    @patch('src.retrieval.cross_encoder_reranker.AutoModelForSequenceClassification')
    def test_score_pairs_multiple_pairs(self, mock_model_class, mock_tokenizer_class):
        """Test scoring multiple query-text pairs."""
        # Setup mocks
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model
        
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2], [3, 4]]),
            'attention_mask': torch.tensor([[1, 1], [1, 1]])
        }
        
        # Mock multiple logits
        mock_outputs = Mock()
        mock_outputs.logits = torch.tensor([[2.0], [1.0]])
        mock_model.return_value = mock_outputs
        
        reranker = CrossEncoderReranker()
        
        # Test multiple pairs
        pairs = [("query", "text1"), ("query", "text2")]
        scores = reranker._score_pairs(pairs)
        
        assert len(scores) == 2
        assert scores[0] > scores[1]  # First should have higher score
    
    @patch('src.retrieval.cross_encoder_reranker.AutoTokenizer')
    @patch('src.retrieval.cross_encoder_reranker.AutoModelForSequenceClassification')
    def test_score_pairs_empty_list(self, mock_model_class, mock_tokenizer_class):
        """Test scoring empty pairs list."""
        mock_tokenizer_class.from_pretrained.return_value = Mock()
        mock_model_class.from_pretrained.return_value = Mock()
        
        reranker = CrossEncoderReranker()
        
        scores = reranker._score_pairs([])
        
        assert scores == []
    
    @patch('src.retrieval.cross_encoder_reranker.AutoTokenizer')
    @patch('src.retrieval.cross_encoder_reranker.AutoModelForSequenceClassification')
    def test_score_pairs_model_failure(self, mock_model_class, mock_tokenizer_class):
        """Test scoring when model inference fails."""
        # Setup mocks
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model
        
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2]]),
            'attention_mask': torch.tensor([[1, 1]])
        }
        
        # Make model inference fail
        mock_model.side_effect = Exception("CUDA out of memory")
        
        reranker = CrossEncoderReranker()
        
        # Should return zero scores as fallback
        pairs = [("query", "text")]
        scores = reranker._score_pairs(pairs)
        
        assert scores == [0.0]
    
    @patch('src.retrieval.cross_encoder_reranker.AutoTokenizer')
    @patch('src.retrieval.cross_encoder_reranker.AutoModelForSequenceClassification')
    def test_rerank_model_inference_failure(self, mock_model_class, mock_tokenizer_class):
        """Test rerank when model inference fails gracefully returns zero scores."""
        # Setup mocks
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Mock tokenizer to return proper format
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2]]),
            'attention_mask': torch.tensor([[1, 1]])
        }
        
        # Make model inference fail
        mock_model.side_effect = Exception("Model error")
        
        reranker = CrossEncoderReranker()
        
        chunks = [
            RetrievedChunk(
                chunk_id="chunk1",
                text="Test chunk",
                breadcrumbs="Doc > Section",
                doc_id="doc1",
                section="section1",
                score=0.5,
                retrieval_source="vector"
            )
        ]
        
        results = FusedResults(chunks=chunks)
        
        # Should return chunks with zero scores (graceful degradation)
        reranked = reranker.rerank("test query", results)
        
        assert len(reranked) == 1
        assert reranked[0].chunk_id == "chunk1"
        assert reranked[0].score == 0.0  # Fallback score
    
    @patch('src.retrieval.cross_encoder_reranker.torch.cuda.is_available')
    @patch('src.retrieval.cross_encoder_reranker.AutoTokenizer')
    @patch('src.retrieval.cross_encoder_reranker.AutoModelForSequenceClassification')
    def test_device_selection_cuda_available(self, mock_model_class, mock_tokenizer_class, mock_cuda):
        """Test device selection when CUDA is available."""
        mock_cuda.return_value = True
        mock_tokenizer_class.from_pretrained.return_value = Mock()
        mock_model_class.from_pretrained.return_value = Mock()
        
        reranker = CrossEncoderReranker()
        
        assert str(reranker.device) == "cuda"
    
    @patch('src.retrieval.cross_encoder_reranker.torch.cuda.is_available')
    @patch('src.retrieval.cross_encoder_reranker.AutoTokenizer')
    @patch('src.retrieval.cross_encoder_reranker.AutoModelForSequenceClassification')
    def test_device_selection_cuda_unavailable(self, mock_model_class, mock_tokenizer_class, mock_cuda):
        """Test device selection when CUDA is unavailable."""
        mock_cuda.return_value = False
        mock_tokenizer_class.from_pretrained.return_value = Mock()
        mock_model_class.from_pretrained.return_value = Mock()
        
        reranker = CrossEncoderReranker()
        
        assert str(reranker.device) == "cpu"