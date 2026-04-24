"""Unit tests for QueryRouter"""

import pytest
import json
from unittest.mock import Mock, patch
from src.query.query_router import QueryRouter, QueryMode


class TestQueryRouter:
    """Test suite for QueryRouter class"""
    
    @pytest.fixture
    def router(self):
        """Create QueryRouter instance for testing"""
        return QueryRouter(
            ollama_base_url="http://localhost:11434",
            llm_model="llama2",
            confidence_threshold=0.7
        )
    
    def test_initialization(self, router):
        """Test QueryRouter initialization"""
        assert router.ollama_base_url == "http://localhost:11434"
        assert router.llm_model == "llama2"
        assert router.confidence_threshold == 0.7
    
    def test_empty_query_defaults_to_hybrid(self, router):
        """Test that empty query defaults to HYBRID mode"""
        mode, confidence = router.route("")
        assert mode == QueryMode.HYBRID
        assert confidence == 0.0
        
        mode, confidence = router.route("   ")
        assert mode == QueryMode.HYBRID
        assert confidence == 0.0
    
    @patch('src.query.query_router.requests.post')
    def test_vector_mode_classification(self, mock_post, router):
        """Test classification of factual/definitional queries as VECTOR mode"""
        # Mock LLM response
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": json.dumps({
                "mode": "VECTOR",
                "confidence": 0.9,
                "reasoning": "This is a definitional question about a single concept"
            })
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        mode, confidence = router.route("What is NEFT?")
        
        assert mode == QueryMode.VECTOR
        assert confidence == 0.9
        assert mock_post.called
    
    @patch('src.query.query_router.requests.post')
    def test_graph_mode_classification(self, mock_post, router):
        """Test classification of relational queries as GRAPH mode"""
        # Mock LLM response
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": json.dumps({
                "mode": "GRAPH",
                "confidence": 0.85,
                "reasoning": "This is a relational question about dependencies"
            })
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        mode, confidence = router.route("What systems depend on NEFT?")
        
        assert mode == QueryMode.GRAPH
        assert confidence == 0.85
    
    @patch('src.query.query_router.requests.post')
    def test_hybrid_mode_classification(self, mock_post, router):
        """Test classification of complex queries as HYBRID mode"""
        # Mock LLM response
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": json.dumps({
                "mode": "HYBRID",
                "confidence": 0.95,
                "reasoning": "This requires both relationship context and full text"
            })
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        mode, confidence = router.route(
            "How does NEFT integrate with Core Banking and what are the limits?"
        )
        
        assert mode == QueryMode.HYBRID
        assert confidence == 0.95
    
    @patch('src.query.query_router.requests.post')
    def test_low_confidence_defaults_to_hybrid(self, mock_post, router):
        """Test that low confidence classification defaults to HYBRID mode"""
        # Mock LLM response with low confidence
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": json.dumps({
                "mode": "VECTOR",
                "confidence": 0.5,  # Below threshold of 0.7
                "reasoning": "Uncertain classification"
            })
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        mode, confidence = router.route("Some ambiguous query")
        
        assert mode == QueryMode.HYBRID
        assert confidence == 0.5  # Original confidence preserved
    
    @patch('src.query.query_router.requests.post')
    def test_invalid_mode_defaults_to_hybrid(self, mock_post, router):
        """Test that invalid mode defaults to HYBRID"""
        # Mock LLM response with invalid mode
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": json.dumps({
                "mode": "INVALID_MODE",
                "confidence": 0.9,
                "reasoning": "Invalid mode"
            })
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        mode, confidence = router.route("Test query")
        
        assert mode == QueryMode.HYBRID
        assert confidence == 0.0
    
    @patch('src.query.query_router.requests.post')
    def test_llm_failure_defaults_to_hybrid(self, mock_post, router):
        """Test that LLM API failure defaults to HYBRID mode"""
        # Mock LLM API failure
        mock_post.side_effect = Exception("API connection failed")
        
        mode, confidence = router.route("Test query")
        
        assert mode == QueryMode.HYBRID
        assert confidence == 0.0
    
    @patch('src.query.query_router.requests.post')
    def test_json_parsing_failure_defaults_to_hybrid(self, mock_post, router):
        """Test that JSON parsing failure defaults to HYBRID mode"""
        # Mock LLM response with invalid JSON
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "This is not valid JSON"
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        mode, confidence = router.route("Test query")
        
        assert mode == QueryMode.HYBRID
        assert confidence == 0.0
    
    @patch('src.query.query_router.requests.post')
    def test_markdown_json_extraction(self, mock_post, router):
        """Test extraction of JSON from markdown code blocks"""
        # Mock LLM response with JSON in markdown code block
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": """```json
{
  "mode": "VECTOR",
  "confidence": 0.8,
  "reasoning": "Definitional query"
}
```"""
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        mode, confidence = router.route("What is RTGS?")
        
        assert mode == QueryMode.VECTOR
        assert confidence == 0.8
    
    @patch('src.query.query_router.requests.post')
    def test_missing_required_fields_defaults_to_hybrid(self, mock_post, router):
        """Test that missing required fields defaults to HYBRID mode"""
        # Mock LLM response missing confidence field
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": json.dumps({
                "mode": "VECTOR",
                "reasoning": "Missing confidence field"
            })
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        mode, confidence = router.route("Test query")
        
        assert mode == QueryMode.HYBRID
        assert confidence == 0.0
    
    @patch('src.query.query_router.requests.post')
    def test_retry_on_failure(self, mock_post, router):
        """Test that router retries on API failure"""
        # First call fails, second succeeds
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": json.dumps({
                "mode": "VECTOR",
                "confidence": 0.9,
                "reasoning": "Success on retry"
            })
        }
        mock_response.raise_for_status = Mock()
        
        mock_post.side_effect = [
            Exception("First attempt failed"),
            mock_response
        ]
        
        mode, confidence = router.route("Test query")
        
        assert mode == QueryMode.VECTOR
        assert confidence == 0.9
        assert mock_post.call_count == 2
    
    def test_query_mode_enum_values(self):
        """Test QueryMode enum has correct values"""
        assert QueryMode.VECTOR.value == "vector"
        assert QueryMode.GRAPH.value == "graph"
        assert QueryMode.HYBRID.value == "hybrid"
    
    @patch('src.query.query_router.requests.post')
    def test_custom_confidence_threshold(self, mock_post):
        """Test router with custom confidence threshold"""
        router = QueryRouter(
            ollama_base_url="http://localhost:11434",
            llm_model="llama2",
            confidence_threshold=0.9  # Higher threshold
        )
        
        # Mock LLM response with confidence below custom threshold
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": json.dumps({
                "mode": "VECTOR",
                "confidence": 0.85,  # Below 0.9 threshold
                "reasoning": "Below custom threshold"
            })
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        mode, confidence = router.route("Test query")
        
        assert mode == QueryMode.HYBRID  # Should default due to low confidence
        assert confidence == 0.85


class TestQueryModeExamples:
    """Test specific query examples for each mode"""
    
    @pytest.fixture
    def router(self):
        """Create QueryRouter instance for testing"""
        return QueryRouter(
            ollama_base_url="http://localhost:11434",
            llm_model="llama2"
        )
    
    @patch('src.query.query_router.requests.post')
    def test_vector_mode_examples(self, mock_post, router):
        """Test various factual/definitional queries"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": json.dumps({
                "mode": "VECTOR",
                "confidence": 0.9,
                "reasoning": "Definitional query"
            })
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        vector_queries = [
            "What is NEFT?",
            "Define transaction limit",
            "Explain RTGS process",
            "What are the features of IMPS?"
        ]
        
        for query in vector_queries:
            mode, confidence = router.route(query)
            assert mode == QueryMode.VECTOR
            assert confidence >= 0.7
    
    @patch('src.query.query_router.requests.post')
    def test_graph_mode_examples(self, mock_post, router):
        """Test various relational/structural queries"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": json.dumps({
                "mode": "GRAPH",
                "confidence": 0.85,
                "reasoning": "Relational query"
            })
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        graph_queries = [
            "What systems depend on NEFT?",
            "Compare RTGS and IMPS",
            "Show payment workflow",
            "What are the dependencies of Core Banking?"
        ]
        
        for query in graph_queries:
            mode, confidence = router.route(query)
            assert mode == QueryMode.GRAPH
            assert confidence >= 0.7
    
    @patch('src.query.query_router.requests.post')
    def test_hybrid_mode_examples(self, mock_post, router):
        """Test various complex queries requiring both modes"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": json.dumps({
                "mode": "HYBRID",
                "confidence": 0.95,
                "reasoning": "Complex query requiring both"
            })
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        hybrid_queries = [
            "How does NEFT integrate with Core Banking and what are the limits?",
            "What are the conflicts between payment rules across documents?",
            "Compare the workflows and limits of RTGS and NEFT"
        ]
        
        for query in hybrid_queries:
            mode, confidence = router.route(query)
            assert mode == QueryMode.HYBRID
            assert confidence >= 0.7
