"""Unit tests for FaithfulnessValidator class."""

import json
import pytest
from unittest.mock import Mock, patch
import requests

from src.query.faithfulness_validator import FaithfulnessValidator, ValidationResult
from src.query.llm_generator import GeneratedResponse
from src.retrieval.context_assembler import AssembledContext, Citation
from datetime import datetime


class TestFaithfulnessValidator:
    """Test cases for FaithfulnessValidator class."""
    
    def test_init_valid_config(self):
        """Test FaithfulnessValidator initialization with valid configuration."""
        validator = FaithfulnessValidator(
            base_url="http://localhost:11434",
            model="llama2",
            faithfulness_threshold=0.8
        )
        
        assert validator.base_url == "http://localhost:11434"
        assert validator.model == "llama2"
        assert validator.faithfulness_threshold == 0.8
    
    def test_init_default_threshold(self):
        """Test initialization with default threshold."""
        validator = FaithfulnessValidator(
            base_url="http://localhost:11434",
            model="llama2"
        )
        
        assert validator.faithfulness_threshold == 0.8
    
    def test_init_strips_trailing_slash(self):
        """Test that trailing slash is stripped from base_url."""
        validator = FaithfulnessValidator(
            base_url="http://localhost:11434/",
            model="llama2"
        )
        
        assert validator.base_url == "http://localhost:11434"
    
    def test_init_empty_base_url(self):
        """Test initialization fails with empty base_url."""
        with pytest.raises(ValueError, match="base_url cannot be empty"):
            FaithfulnessValidator(base_url="", model="llama2")
        
        with pytest.raises(ValueError, match="base_url cannot be empty"):
            FaithfulnessValidator(base_url="   ", model="llama2")
    
    def test_init_empty_model(self):
        """Test initialization fails with empty model."""
        with pytest.raises(ValueError, match="model cannot be empty"):
            FaithfulnessValidator(base_url="http://localhost:11434", model="")
        
        with pytest.raises(ValueError, match="model cannot be empty"):
            FaithfulnessValidator(base_url="http://localhost:11434", model="   ")
    
    def test_init_invalid_threshold(self):
        """Test initialization fails with invalid threshold."""
        with pytest.raises(ValueError, match="faithfulness_threshold must be between 0.0 and 1.0"):
            FaithfulnessValidator(
                base_url="http://localhost:11434",
                model="llama2",
                faithfulness_threshold=-0.1
            )
        
        with pytest.raises(ValueError, match="faithfulness_threshold must be between 0.0 and 1.0"):
            FaithfulnessValidator(
                base_url="http://localhost:11434",
                model="llama2",
                faithfulness_threshold=1.5
            )
    
    def test_validate_empty_response(self):
        """Test validate fails with empty response."""
        validator = FaithfulnessValidator("http://localhost:11434", "llama2")
        context = AssembledContext(context_text="Some context", citations={})
        
        with pytest.raises(ValueError, match="Response cannot be empty"):
            validator.validate(None, context)
        
        empty_response = GeneratedResponse(
            answer="",
            citations_used=[],
            model="llama2",
            timestamp=datetime.now()
        )
        
        with pytest.raises(ValueError, match="Response cannot be empty"):
            validator.validate(empty_response, context)
    
    def test_validate_empty_context(self):
        """Test validate fails with empty context."""
        validator = FaithfulnessValidator("http://localhost:11434", "llama2")
        response = GeneratedResponse(
            answer="Test answer",
            citations_used=[],
            model="llama2",
            timestamp=datetime.now()
        )
        
        with pytest.raises(ValueError, match="Context cannot be empty"):
            validator.validate(response, None)
        
        empty_context = AssembledContext(context_text="", citations={})
        with pytest.raises(ValueError, match="Context cannot be empty"):
            validator.validate(response, empty_context)
    
    @patch('requests.post')
    def test_validate_all_claims_supported(self, mock_post):
        """Test validation with all claims supported."""
        # Mock claim extraction
        mock_post.side_effect = [
            # Claim extraction response
            Mock(
                raise_for_status=Mock(return_value=None),
                json=Mock(return_value={
                    "response": '["NEFT is a payment system", "NEFT processes transfers"]'
                })
            ),
            # Entailment check 1
            Mock(
                raise_for_status=Mock(return_value=None),
                json=Mock(return_value={
                    "response": '{"supported": true, "confidence": 0.95, "explanation": "Supported"}'
                })
            ),
            # Entailment check 2
            Mock(
                raise_for_status=Mock(return_value=None),
                json=Mock(return_value={
                    "response": '{"supported": true, "confidence": 0.90, "explanation": "Supported"}'
                })
            )
        ]
        
        validator = FaithfulnessValidator("http://localhost:11434", "llama2")
        
        response = GeneratedResponse(
            answer="NEFT is a payment system [doc1:section2]. NEFT processes transfers [doc1:section2].",
            citations_used=["doc1:section2"],
            model="llama2",
            timestamp=datetime.now()
        )
        
        context = AssembledContext(
            context_text="NEFT is a payment system that processes electronic fund transfers.",
            citations={"doc1:section2": Citation("doc1:section2", "doc1", "section2", "chunk1", "breadcrumb1")},
            token_count=50
        )
        
        result = validator.validate(response, context)
        
        assert isinstance(result, ValidationResult)
        assert result.faithfulness_score == 1.0
        assert result.total_claims == 2
        assert result.supported_claims == 2
        assert len(result.unsupported_claims) == 0
        assert len(result.warnings) == 0
    
    @patch('requests.post')
    def test_validate_some_claims_unsupported(self, mock_post):
        """Test validation with some unsupported claims."""
        # Mock responses
        mock_post.side_effect = [
            # Claim extraction
            Mock(
                raise_for_status=Mock(return_value=None),
                json=Mock(return_value={
                    "response": '["NEFT is a payment system", "NEFT has no transaction limits"]'
                })
            ),
            # Entailment check 1 - supported
            Mock(
                raise_for_status=Mock(return_value=None),
                json=Mock(return_value={
                    "response": '{"supported": true, "confidence": 0.95, "explanation": "Supported"}'
                })
            ),
            # Entailment check 2 - not supported
            Mock(
                raise_for_status=Mock(return_value=None),
                json=Mock(return_value={
                    "response": '{"supported": false, "confidence": 0.85, "explanation": "Not mentioned"}'
                })
            )
        ]
        
        validator = FaithfulnessValidator("http://localhost:11434", "llama2", faithfulness_threshold=0.8)
        
        response = GeneratedResponse(
            answer="NEFT is a payment system. NEFT has no transaction limits.",
            citations_used=[],
            model="llama2",
            timestamp=datetime.now()
        )
        
        context = AssembledContext(
            context_text="NEFT is a payment system.",
            citations={},
            token_count=10
        )
        
        result = validator.validate(response, context)
        
        assert result.faithfulness_score == 0.5
        assert result.total_claims == 2
        assert result.supported_claims == 1
        assert len(result.unsupported_claims) == 1
        assert "NEFT has no transaction limits" in result.unsupported_claims
        assert len(result.warnings) == 2  # Low score warning + unsupported claims warning
        assert any("Low faithfulness score" in w for w in result.warnings)
        assert any("unsupported claim" in w for w in result.warnings)
    
    @patch('requests.post')
    def test_validate_no_claims_extracted(self, mock_post):
        """Test validation when no claims can be extracted."""
        # Mock empty claim extraction
        mock_post.return_value = Mock(
            raise_for_status=Mock(return_value=None),
            json=Mock(return_value={"response": "[]"})
        )
        
        validator = FaithfulnessValidator("http://localhost:11434", "llama2")
        
        response = GeneratedResponse(
            answer="Insufficient information in documents to answer this question.",
            citations_used=[],
            model="llama2",
            timestamp=datetime.now()
        )
        
        context = AssembledContext(
            context_text="Some context",
            citations={},
            token_count=10
        )
        
        result = validator.validate(response, context)
        
        assert result.faithfulness_score == 1.0  # Vacuously true
        assert result.total_claims == 0
        assert result.supported_claims == 0
        assert len(result.unsupported_claims) == 0
        assert len(result.warnings) == 1
        assert "No claims extracted" in result.warnings[0]
    
    @patch('requests.post')
    def test_validate_low_faithfulness_triggers_warning(self, mock_post):
        """Test that low faithfulness score triggers warning."""
        # Mock responses with low support
        mock_post.side_effect = [
            # Claim extraction
            Mock(
                raise_for_status=Mock(return_value=None),
                json=Mock(return_value={
                    "response": '["Claim 1", "Claim 2", "Claim 3", "Claim 4", "Claim 5"]'
                })
            ),
            # Only 3 out of 5 supported (60% < 80% threshold)
            Mock(raise_for_status=Mock(return_value=None), json=Mock(return_value={"response": '{"supported": true, "confidence": 0.9}'})),
            Mock(raise_for_status=Mock(return_value=None), json=Mock(return_value={"response": '{"supported": true, "confidence": 0.9}'})),
            Mock(raise_for_status=Mock(return_value=None), json=Mock(return_value={"response": '{"supported": true, "confidence": 0.9}'})),
            Mock(raise_for_status=Mock(return_value=None), json=Mock(return_value={"response": '{"supported": false, "confidence": 0.8}'})),
            Mock(raise_for_status=Mock(return_value=None), json=Mock(return_value={"response": '{"supported": false, "confidence": 0.8}'}))
        ]
        
        validator = FaithfulnessValidator("http://localhost:11434", "llama2", faithfulness_threshold=0.8)
        
        response = GeneratedResponse(
            answer="Multiple claims here.",
            citations_used=[],
            model="llama2",
            timestamp=datetime.now()
        )
        
        context = AssembledContext(
            context_text="Context",
            citations={},
            token_count=10
        )
        
        result = validator.validate(response, context)
        
        assert result.faithfulness_score == 0.6
        assert len(result.warnings) > 0
        assert any("Low faithfulness score" in w for w in result.warnings)
    
    def test_fallback_claim_extraction(self):
        """Test fallback claim extraction using sentence splitting."""
        validator = FaithfulnessValidator("http://localhost:11434", "llama2")
        
        answer = "NEFT is a payment system [doc1:section2]. It processes transfers. RTGS is different."
        claims = validator._fallback_claim_extraction(answer)
        
        assert len(claims) == 3
        assert "NEFT is a payment system" in claims[0]
        assert "It processes transfers" in claims[1]
        assert "RTGS is different" in claims[2]
    
    def test_fallback_claim_extraction_filters_short_fragments(self):
        """Test that fallback extraction filters very short fragments."""
        validator = FaithfulnessValidator("http://localhost:11434", "llama2")
        
        answer = "NEFT is a payment system. Yes. No. Maybe. It processes electronic transfers."
        claims = validator._fallback_claim_extraction(answer)
        
        # Should filter out "Yes", "No", "Maybe" (too short)
        assert len(claims) == 2
        assert any("NEFT is a payment system" in c for c in claims)
        assert any("It processes electronic transfers" in c for c in claims)
    
    def test_fallback_claim_extraction_removes_citations(self):
        """Test that fallback extraction removes citation brackets."""
        validator = FaithfulnessValidator("http://localhost:11434", "llama2")
        
        answer = "NEFT is a payment system [doc1:section2]. RTGS is different [doc2:section1]."
        claims = validator._fallback_claim_extraction(answer)
        
        # Citations should be removed
        assert all("[" not in claim for claim in claims)
        assert all("]" not in claim for claim in claims)
    
    @patch('requests.post')
    def test_extract_claims_with_llm(self, mock_post):
        """Test claim extraction using LLM."""
        mock_post.return_value = Mock(
            raise_for_status=Mock(return_value=None),
            json=Mock(return_value={
                "response": '["Claim 1", "Claim 2", "Claim 3"]'
            })
        )
        
        validator = FaithfulnessValidator("http://localhost:11434", "llama2")
        claims = validator._extract_claims("Some answer text")
        
        assert claims == ["Claim 1", "Claim 2", "Claim 3"]
    
    @patch('requests.post')
    def test_extract_claims_fallback_on_llm_failure(self, mock_post):
        """Test fallback to sentence splitting when LLM fails."""
        mock_post.side_effect = requests.exceptions.RequestException("Connection failed")
        
        validator = FaithfulnessValidator("http://localhost:11434", "llama2")
        claims = validator._extract_claims("NEFT is a payment system. RTGS is different.")
        
        # Should use fallback
        assert len(claims) == 2
        assert any("NEFT is a payment system" in c for c in claims)
    
    @patch('requests.post')
    def test_extract_claims_fallback_on_invalid_json(self, mock_post):
        """Test fallback when LLM returns invalid JSON."""
        mock_post.return_value = Mock(
            raise_for_status=Mock(return_value=None),
            json=Mock(return_value={"response": "Not valid JSON array"})
        )
        
        validator = FaithfulnessValidator("http://localhost:11434", "llama2")
        claims = validator._extract_claims("NEFT is a payment system. RTGS is different.")
        
        # Should use fallback
        assert len(claims) >= 1
    
    @patch('requests.post')
    def test_check_entailment_supported(self, mock_post):
        """Test entailment check for supported claim."""
        mock_post.return_value = Mock(
            raise_for_status=Mock(return_value=None),
            json=Mock(return_value={
                "response": '{"supported": true, "confidence": 0.95, "explanation": "Clearly stated"}'
            })
        )
        
        validator = FaithfulnessValidator("http://localhost:11434", "llama2")
        is_supported = validator._check_entailment(
            "NEFT is a payment system",
            "NEFT is a payment system that processes transfers."
        )
        
        assert is_supported is True
    
    @patch('requests.post')
    def test_check_entailment_not_supported(self, mock_post):
        """Test entailment check for unsupported claim."""
        mock_post.return_value = Mock(
            raise_for_status=Mock(return_value=None),
            json=Mock(return_value={
                "response": '{"supported": false, "confidence": 0.90, "explanation": "Not mentioned"}'
            })
        )
        
        validator = FaithfulnessValidator("http://localhost:11434", "llama2")
        is_supported = validator._check_entailment(
            "NEFT has no limits",
            "NEFT is a payment system."
        )
        
        assert is_supported is False
    
    @patch('requests.post')
    def test_check_entailment_truncates_long_context(self, mock_post):
        """Test that entailment check truncates very long context."""
        mock_post.return_value = Mock(
            raise_for_status=Mock(return_value=None),
            json=Mock(return_value={
                "response": '{"supported": true, "confidence": 0.9}'
            })
        )
        
        validator = FaithfulnessValidator("http://localhost:11434", "llama2")
        
        # Create very long context
        long_context = "A" * 5000
        
        validator._check_entailment("Test claim", long_context)
        
        # Verify API was called
        call_args = mock_post.call_args
        prompt = call_args[1]['json']['prompt']
        
        # Context should be truncated to 2000 chars
        assert len(prompt) < len(long_context)
    
    @patch('requests.post')
    def test_check_entailment_returns_false_on_failure(self, mock_post):
        """Test that entailment check returns False on API failure."""
        mock_post.side_effect = requests.exceptions.RequestException("Connection failed")
        
        validator = FaithfulnessValidator("http://localhost:11434", "llama2")
        is_supported = validator._check_entailment("Test claim", "Test context")
        
        # Conservative: assume not supported on failure
        assert is_supported is False
    
    @patch('requests.post')
    def test_check_entailment_returns_false_on_invalid_json(self, mock_post):
        """Test that entailment check returns False on invalid JSON."""
        mock_post.return_value = Mock(
            raise_for_status=Mock(return_value=None),
            json=Mock(return_value={"response": "Invalid JSON"})
        )
        
        validator = FaithfulnessValidator("http://localhost:11434", "llama2")
        is_supported = validator._check_entailment("Test claim", "Test context")
        
        assert is_supported is False
    
    def test_check_entailment_empty_inputs(self):
        """Test entailment check with empty inputs."""
        validator = FaithfulnessValidator("http://localhost:11434", "llama2")
        
        assert validator._check_entailment("", "context") is False
        assert validator._check_entailment("claim", "") is False
        assert validator._check_entailment("", "") is False
    
    @patch('requests.post')
    def test_call_ollama_timeout(self, mock_post):
        """Test Ollama API timeout handling."""
        mock_post.side_effect = requests.exceptions.Timeout("Request timed out")
        
        validator = FaithfulnessValidator("http://localhost:11434", "llama2")
        result = validator._call_ollama("test prompt")
        
        assert result is None
        assert mock_post.call_count == 3  # Initial + 2 retries
    
    @patch('requests.post')
    def test_call_ollama_success_after_retry(self, mock_post):
        """Test successful call after initial failure."""
        mock_post.side_effect = [
            requests.exceptions.RequestException("First failure"),
            Mock(
                raise_for_status=Mock(return_value=None),
                json=Mock(return_value={"response": "Success"})
            )
        ]
        
        validator = FaithfulnessValidator("http://localhost:11434", "llama2")
        result = validator._call_ollama("test prompt")
        
        assert result == "Success"
        assert mock_post.call_count == 2
    
    def test_validation_result_dataclass(self):
        """Test ValidationResult dataclass structure."""
        result = ValidationResult(
            faithfulness_score=0.8,
            total_claims=5,
            supported_claims=4,
            unsupported_claims=["Unsupported claim"],
            warnings=["Warning message"]
        )
        
        assert result.faithfulness_score == 0.8
        assert result.total_claims == 5
        assert result.supported_claims == 4
        assert len(result.unsupported_claims) == 1
        assert len(result.warnings) == 1
    
    def test_claim_extraction_prompt_format(self):
        """Test that claim extraction prompt contains required elements."""
        prompt = FaithfulnessValidator.CLAIM_EXTRACTION_PROMPT
        
        assert "{answer}" in prompt
        assert "JSON array" in prompt or "json" in prompt.lower()
    
    def test_entailment_prompt_format(self):
        """Test that entailment prompt contains required elements."""
        prompt = FaithfulnessValidator.ENTAILMENT_PROMPT
        
        assert "{context_excerpt}" in prompt
        assert "{claim}" in prompt
        assert "supported" in prompt.lower()
        assert "JSON" in prompt or "json" in prompt.lower()
