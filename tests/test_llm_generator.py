"""Unit tests for LLMGenerator class."""

import json
import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import requests

from src.query.llm_generator import LLMGenerator, GeneratedResponse
from src.retrieval.context_assembler import AssembledContext, Citation


class TestLLMGenerator:
    """Test cases for LLMGenerator class."""
    
    def test_init_valid_config(self):
        """Test LLMGenerator initialization with valid configuration."""
        generator = LLMGenerator(
            base_url="http://localhost:11434",
            model="llama2"
        )
        
        assert generator.base_url == "http://localhost:11434"
        assert generator.model == "llama2"
    
    def test_init_strips_trailing_slash(self):
        """Test that trailing slash is stripped from base_url."""
        generator = LLMGenerator(
            base_url="http://localhost:11434/",
            model="llama2"
        )
        
        assert generator.base_url == "http://localhost:11434"
    
    def test_init_empty_base_url(self):
        """Test initialization fails with empty base_url."""
        with pytest.raises(ValueError, match="base_url cannot be empty"):
            LLMGenerator(base_url="", model="llama2")
        
        with pytest.raises(ValueError, match="base_url cannot be empty"):
            LLMGenerator(base_url="   ", model="llama2")
    
    def test_init_empty_model(self):
        """Test initialization fails with empty model."""
        with pytest.raises(ValueError, match="model cannot be empty"):
            LLMGenerator(base_url="http://localhost:11434", model="")
        
        with pytest.raises(ValueError, match="model cannot be empty"):
            LLMGenerator(base_url="http://localhost:11434", model="   ")
    
    def test_generate_empty_query(self):
        """Test generate fails with empty query."""
        generator = LLMGenerator("http://localhost:11434", "llama2")
        context = AssembledContext(context_text="Some context", citations={})
        
        with pytest.raises(ValueError, match="Query cannot be empty"):
            generator.generate("", context)
        
        with pytest.raises(ValueError, match="Query cannot be empty"):
            generator.generate("   ", context)
    
    def test_generate_empty_context(self):
        """Test generate fails with empty context."""
        generator = LLMGenerator("http://localhost:11434", "llama2")
        
        with pytest.raises(ValueError, match="Context cannot be empty"):
            generator.generate("test query", None)
        
        empty_context = AssembledContext(context_text="", citations={})
        with pytest.raises(ValueError, match="Context cannot be empty"):
            generator.generate("test query", empty_context)
    
    @patch('requests.post')
    def test_generate_successful_response(self, mock_post):
        """Test successful response generation with citations."""
        # Mock successful Ollama API response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "response": "NEFT is a payment system [doc1:section2] that processes transfers [doc2:section1]."
        }
        mock_post.return_value = mock_response
        
        # Create generator and context
        generator = LLMGenerator("http://localhost:11434", "llama2")
        
        citations = {
            "doc1:section2": Citation("doc1:section2", "doc1", "section2", "chunk1", "breadcrumb1"),
            "doc2:section1": Citation("doc2:section1", "doc2", "section1", "chunk2", "breadcrumb2")
        }
        
        context = AssembledContext(
            context_text="Context about NEFT",
            citations=citations,
            token_count=50
        )
        
        # Generate response
        response = generator.generate("What is NEFT?", context)
        
        # Verify response
        assert isinstance(response, GeneratedResponse)
        assert "NEFT is a payment system" in response.answer
        assert response.model == "llama2"
        assert isinstance(response.timestamp, datetime)
        assert "doc1:section2" in response.citations_used
        assert "doc2:section1" in response.citations_used
        
        # Verify API call
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]['json']['model'] == "llama2"
        assert call_args[1]['json']['stream'] is False
        assert "What is NEFT?" in call_args[1]['json']['prompt']
        assert "Context about NEFT" in call_args[1]['json']['prompt']
    
    @patch('requests.post')
    def test_generate_api_failure(self, mock_post):
        """Test handling of API failure."""
        # Mock API failure
        mock_post.side_effect = requests.exceptions.RequestException("Connection failed")
        
        generator = LLMGenerator("http://localhost:11434", "llama2")
        context = AssembledContext(context_text="Some context", citations={})
        
        with pytest.raises(RuntimeError, match="LLM generation failed after retries"):
            generator.generate("test query", context)
    
    @patch('requests.post')
    def test_generate_invalid_citations(self, mock_post):
        """Test handling of invalid citations in response."""
        # Mock response with invalid citations
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "response": "Answer with valid [doc1:section2] and invalid [invalid_citation] citations."
        }
        mock_post.return_value = mock_response
        
        generator = LLMGenerator("http://localhost:11434", "llama2")
        
        citations = {
            "doc1:section2": Citation("doc1:section2", "doc1", "section2", "chunk1", "breadcrumb1")
        }
        
        context = AssembledContext(
            context_text="Context",
            citations=citations,
            token_count=10
        )
        
        response = generator.generate("test query", context)
        
        # Should only include valid citations
        assert response.citations_used == ["doc1:section2"]
    
    def test_extract_citations_valid_format(self):
        """Test citation extraction with valid format."""
        generator = LLMGenerator("http://localhost:11434", "llama2")
        
        text = "This is from [doc1:section2] and also [doc2:section1]. More info [doc1:section3]."
        citations = generator._extract_citations(text)
        
        assert citations == ["doc1:section2", "doc2:section1", "doc1:section3"]
    
    def test_extract_citations_invalid_format(self):
        """Test citation extraction filters invalid formats."""
        generator = LLMGenerator("http://localhost:11434", "llama2")
        
        text = "Valid [doc1:section2] but invalid [1] and [abc] and [no_colon]."
        citations = generator._extract_citations(text)
        
        assert citations == ["doc1:section2"]
    
    def test_extract_citations_duplicates(self):
        """Test citation extraction removes duplicates."""
        generator = LLMGenerator("http://localhost:11434", "llama2")
        
        text = "First [doc1:section2] and second [doc1:section2] mention."
        citations = generator._extract_citations(text)
        
        assert citations == ["doc1:section2"]
    
    def test_extract_citations_empty_text(self):
        """Test citation extraction with empty text."""
        generator = LLMGenerator("http://localhost:11434", "llama2")
        
        assert generator._extract_citations("") == []
        assert generator._extract_citations(None) == []
    
    @patch('requests.post')
    def test_call_ollama_timeout(self, mock_post):
        """Test Ollama API timeout handling."""
        mock_post.side_effect = requests.exceptions.Timeout("Request timed out")
        
        generator = LLMGenerator("http://localhost:11434", "llama2")
        result = generator._call_ollama("test prompt")
        
        assert result is None
        assert mock_post.call_count == 3  # Initial + 2 retries
    
    @patch('requests.post')
    def test_call_ollama_json_decode_error(self, mock_post):
        """Test handling of invalid JSON response."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_post.return_value = mock_response
        
        generator = LLMGenerator("http://localhost:11434", "llama2")
        result = generator._call_ollama("test prompt")
        
        assert result is None
    
    @patch('requests.post')
    def test_call_ollama_unexpected_response_format(self, mock_post):
        """Test handling of unexpected response format."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"error": "Model not found"}
        mock_post.return_value = mock_response
        
        generator = LLMGenerator("http://localhost:11434", "llama2")
        result = generator._call_ollama("test prompt")
        
        assert result is None
    
    @patch('requests.post')
    def test_call_ollama_success_after_retry(self, mock_post):
        """Test successful call after initial failure."""
        # First call fails, second succeeds
        mock_post.side_effect = [
            requests.exceptions.RequestException("First failure"),
            Mock(
                raise_for_status=Mock(return_value=None),
                json=Mock(return_value={"response": "Success"})
            )
        ]
        
        generator = LLMGenerator("http://localhost:11434", "llama2")
        result = generator._call_ollama("test prompt")
        
        assert result == "Success"
        assert mock_post.call_count == 2
    
    def test_system_prompt_template_format(self):
        """Test that system prompt template contains required elements."""
        template = LLMGenerator.SYSTEM_PROMPT_TEMPLATE
        
        # Check for required placeholders
        assert "{context_text}" in template
        assert "{query}" in template
        
        # Check for required instructions
        assert "banking documentation assistant" in template.lower()
        assert "only use information from the provided context" in template.lower()
        assert "cite sources using [citation_id] format" in template.lower()
        assert "insufficient information" in template.lower()
    
    @patch('requests.post')
    def test_generate_with_system_prompt_formatting(self, mock_post):
        """Test that system prompt is properly formatted with context and query."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"response": "Test response"}
        mock_post.return_value = mock_response
        
        generator = LLMGenerator("http://localhost:11434", "llama2")
        context = AssembledContext(
            context_text="Test context content",
            citations={},
            token_count=10
        )
        
        generator.generate("Test query", context)
        
        # Verify the prompt contains both context and query
        call_args = mock_post.call_args
        prompt = call_args[1]['json']['prompt']
        
        assert "Test context content" in prompt
        assert "Test query" in prompt
        assert "banking documentation assistant" in prompt.lower()