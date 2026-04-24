"""Integration tests for LLMGenerator with real Ollama API."""

import os
import pytest
from unittest.mock import patch, Mock

from src.query.llm_generator import LLMGenerator, GeneratedResponse
from src.retrieval.context_assembler import AssembledContext, Citation


class TestLLMGeneratorIntegration:
    """Integration tests for LLMGenerator with real Ollama API."""
    
    @pytest.fixture
    def ollama_config(self):
        """Get Ollama configuration from environment."""
        return {
            "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            "model": os.getenv("LLM_MODEL", "llama2")
        }
    
    @pytest.fixture
    def sample_context(self):
        """Create sample context for testing."""
        citations = {
            "doc1:section2": Citation(
                citation_id="doc1:section2",
                doc_id="doc1",
                section="section2",
                chunk_id="chunk_123",
                breadcrumbs="NEFT Specification > Transaction Processing"
            ),
            "doc2:section1": Citation(
                citation_id="doc2:section1",
                doc_id="doc2",
                section="section1",
                chunk_id="chunk_456",
                breadcrumbs="Core Banking > Payment Systems"
            )
        }
        
        context_text = """Query: What is NEFT?

Knowledge Graph Facts:
1. NEFT DEPENDS_ON Core Banking System [doc1:section2]
2. NEFT INTEGRATES_WITH RTGS System [doc2:section1]

Relevant Document Excerpts:

[doc1:section2] (NEFT Specification > Transaction Processing)
NEFT (National Electronic Funds Transfer) is a nation-wide payment system facilitating one-to-one funds transfer. It operates in hourly batches and is available 24x7 throughout the year including holidays.

[doc2:section1] (Core Banking > Payment Systems)
The Core Banking System integrates with NEFT to process electronic fund transfers. All NEFT transactions are validated against account balances and regulatory limits before processing.
"""
        
        return AssembledContext(
            context_text=context_text,
            citations=citations,
            token_count=120
        )
    
    @pytest.mark.integration
    def test_generate_with_real_ollama(self, ollama_config, sample_context):
        """
        Test LLM generation with real Ollama API.
        
        Note: This test requires Ollama to be running locally.
        Skip if Ollama is not available.
        """
        generator = LLMGenerator(
            base_url=ollama_config["base_url"],
            model=ollama_config["model"]
        )
        
        query = "What is NEFT and how does it work?"
        
        try:
            response = generator.generate(query, sample_context)
            
            # Verify response structure
            assert isinstance(response, GeneratedResponse)
            assert response.answer
            assert response.model == ollama_config["model"]
            assert isinstance(response.citations_used, list)
            
            # Response should contain information about NEFT
            assert "NEFT" in response.answer or "neft" in response.answer.lower()
            
            # Should have citations if properly formatted
            if response.citations_used:
                for citation in response.citations_used:
                    assert citation in sample_context.citations
            
            print(f"Generated response: {response.answer}")
            print(f"Citations used: {response.citations_used}")
            
        except RuntimeError as e:
            if "LLM generation failed after retries" in str(e):
                pytest.skip("Ollama server not available for integration test")
            else:
                raise
    
    @pytest.mark.integration
    def test_insufficient_context_handling(self, ollama_config):
        """Test handling of insufficient context with real Ollama API."""
        generator = LLMGenerator(
            base_url=ollama_config["base_url"],
            model=ollama_config["model"]
        )
        
        # Create minimal context that doesn't answer the question
        minimal_context = AssembledContext(
            context_text="Query: What is the capital of Mars?\n\nNo relevant information available.",
            citations={},
            token_count=10
        )
        
        query = "What is the capital of Mars?"
        
        try:
            response = generator.generate(query, minimal_context)
            
            # Should indicate insufficient information
            assert "insufficient" in response.answer.lower() or \
                   "not enough" in response.answer.lower() or \
                   "cannot answer" in response.answer.lower() or \
                   "no information" in response.answer.lower()
            
            print(f"Insufficient context response: {response.answer}")
            
        except RuntimeError as e:
            if "LLM generation failed after retries" in str(e):
                pytest.skip("Ollama server not available for integration test")
            else:
                raise
    
    @pytest.mark.integration
    def test_citation_extraction_real_response(self, ollama_config, sample_context):
        """Test citation extraction from real LLM response."""
        generator = LLMGenerator(
            base_url=ollama_config["base_url"],
            model=ollama_config["model"]
        )
        
        query = "What is NEFT?"
        
        try:
            response = generator.generate(query, sample_context)
            
            # Check that citations are properly extracted
            for citation in response.citations_used:
                # Should be in valid format
                assert ":" in citation
                parts = citation.split(":")
                assert len(parts) == 2
                
                # Should exist in context
                assert citation in sample_context.citations
            
            print(f"Extracted citations: {response.citations_used}")
            
        except RuntimeError as e:
            if "LLM generation failed after retries" in str(e):
                pytest.skip("Ollama server not available for integration test")
            else:
                raise
    
    def test_integration_with_mocked_ollama(self, sample_context):
        """Test integration behavior with mocked Ollama responses."""
        
        # Mock a realistic Ollama response with citations
        mock_response_text = """NEFT (National Electronic Funds Transfer) is a nation-wide payment system [doc1:section2] that facilitates one-to-one funds transfer. The system operates in hourly batches and integrates with the Core Banking System [doc2:section1] to process electronic fund transfers."""
        
        with patch('requests.post') as mock_post:
            mock_post.return_value.raise_for_status.return_value = None
            mock_post.return_value.json.return_value = {
                "response": mock_response_text
            }
            
            generator = LLMGenerator("http://localhost:11434", "llama2")
            response = generator.generate("What is NEFT?", sample_context)
            
            # Verify integration behavior
            assert response.answer == mock_response_text
            assert "doc1:section2" in response.citations_used
            assert "doc2:section1" in response.citations_used
            assert len(response.citations_used) == 2
            
            # Verify API call parameters
            call_args = mock_post.call_args
            assert call_args[1]['json']['model'] == "llama2"
            assert call_args[1]['json']['stream'] is False
            assert call_args[1]['json']['options']['temperature'] == 0.1
    
    def test_error_recovery_integration(self):
        """Test error recovery in integration scenarios."""
        
        with patch('requests.post') as mock_post:
            # First two calls fail with RequestException, third succeeds
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"response": "Success after retry"}
            
            import requests
            mock_post.side_effect = [
                requests.exceptions.RequestException("Network error"),
                requests.exceptions.RequestException("Timeout"),
                mock_response
            ]
            
            generator = LLMGenerator("http://localhost:11434", "llama2")
            context = AssembledContext(
                context_text="Test context",
                citations={},
                token_count=5
            )
            
            response = generator.generate("Test query", context)
            
            assert response.answer == "Success after retry"
            assert mock_post.call_count == 3