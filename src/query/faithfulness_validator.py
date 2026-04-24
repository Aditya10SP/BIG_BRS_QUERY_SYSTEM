"""Faithfulness validation for LLM responses to ensure grounding in source context."""

import json
import logging
import re
from dataclasses import dataclass
from typing import List, Optional

import requests

from src.query.llm_generator import GeneratedResponse
from src.retrieval.context_assembler import AssembledContext

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """
    Represents the result of faithfulness validation.
    
    Attributes:
        faithfulness_score: Score between 0 and 1 indicating proportion of supported claims
        total_claims: Total number of claims extracted from the response
        supported_claims: Number of claims supported by the context
        unsupported_claims: List of claims not supported by the context
        warnings: List of warning messages for low scores or issues
    """
    faithfulness_score: float
    total_claims: int
    supported_claims: int
    unsupported_claims: List[str]
    warnings: List[str]


class FaithfulnessValidator:
    """
    Validates LLM responses for faithfulness to source context.
    
    This class:
    1. Extracts individual claims from generated responses
    2. Checks each claim against source context using entailment
    3. Computes faithfulness score (supported_claims / total_claims)
    4. Generates warnings for low scores (< 0.8)
    5. Identifies specific unsupported claims
    
    The validation uses LLM-based entailment checking to determine if
    each claim is supported by the provided context.
    
    Attributes:
        base_url: Base URL for Ollama API
        model: Name of LLM model to use for entailment checking
        faithfulness_threshold: Threshold below which warnings are generated (default: 0.8)
    """
    
    # Prompt template for claim extraction
    CLAIM_EXTRACTION_PROMPT = """Extract individual factual claims from the following answer. Each claim should be a single, verifiable statement.

Answer: {answer}

Return the claims as a JSON array of strings:
["claim 1", "claim 2", "claim 3"]

Claims:"""
    
    # Prompt template for entailment checking
    ENTAILMENT_PROMPT = """Does the following context support this claim?

Context: {context_excerpt}

Claim: {claim}

Respond with JSON:
{{
  "supported": true/false,
  "confidence": 0.0-1.0,
  "explanation": "..."
}}

Response:"""
    
    def __init__(
        self,
        base_url: str,
        model: str,
        faithfulness_threshold: float = 0.8
    ):
        """
        Initialize with LLM client configuration.
        
        Args:
            base_url: Base URL for Ollama API (e.g., "http://localhost:11434")
            model: Name of LLM model to use for entailment checking
            faithfulness_threshold: Threshold below which warnings are generated (default: 0.8)
        
        Raises:
            ValueError: If base_url or model is empty, or threshold is invalid
        """
        if not base_url or not base_url.strip():
            raise ValueError("base_url cannot be empty")
        
        if not model or not model.strip():
            raise ValueError("model cannot be empty")
        
        if not 0.0 <= faithfulness_threshold <= 1.0:
            raise ValueError("faithfulness_threshold must be between 0.0 and 1.0")
        
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.faithfulness_threshold = faithfulness_threshold
        
        logger.info(
            "FaithfulnessValidator initialized",
            extra={
                "base_url": self.base_url,
                "model": self.model,
                "faithfulness_threshold": self.faithfulness_threshold
            }
        )
    
    def validate(
        self,
        response: GeneratedResponse,
        context: AssembledContext
    ) -> ValidationResult:
        """
        Check if response claims are supported by context.
        
        This method:
        1. Extracts individual claims from the response
        2. For each claim, checks if context entails it
        3. Computes faithfulness score
        4. Generates warnings if score is below threshold
        
        Args:
            response: GeneratedResponse to validate
            context: AssembledContext that was used for generation
        
        Returns:
            ValidationResult with score and unsupported claims
        
        Raises:
            ValueError: If response or context is invalid
        """
        if not response or not response.answer:
            raise ValueError("Response cannot be empty")
        
        if not context or not context.context_text:
            raise ValueError("Context cannot be empty")
        
        logger.info(
            f"Validating response faithfulness: {len(response.answer)} chars, "
            f"{len(context.context_text)} context chars"
        )
        
        # Step 1: Extract claims from response
        claims = self._extract_claims(response.answer)
        
        if not claims:
            logger.warning("No claims extracted from response")
            return ValidationResult(
                faithfulness_score=1.0,  # No claims = vacuously true
                total_claims=0,
                supported_claims=0,
                unsupported_claims=[],
                warnings=["No claims extracted from response"]
            )
        
        logger.debug(f"Extracted {len(claims)} claims from response")
        
        # Step 2: Check each claim for entailment
        supported_count = 0
        unsupported_claims = []
        
        for claim in claims:
            is_supported = self._check_entailment(claim, context.context_text)
            
            if is_supported:
                supported_count += 1
            else:
                unsupported_claims.append(claim)
        
        # Step 3: Compute faithfulness score
        faithfulness_score = supported_count / len(claims) if claims else 1.0
        
        # Step 4: Generate warnings
        warnings = []
        if faithfulness_score < self.faithfulness_threshold:
            warnings.append(
                f"Low faithfulness score ({faithfulness_score:.2f} < {self.faithfulness_threshold}). "
                f"Response may contain unsupported claims."
            )
        
        if unsupported_claims:
            warnings.append(
                f"Found {len(unsupported_claims)} unsupported claim(s). "
                "Please verify these statements against source documents."
            )
        
        logger.info(
            f"Validation complete: score={faithfulness_score:.2f}, "
            f"supported={supported_count}/{len(claims)}, "
            f"warnings={len(warnings)}"
        )
        
        return ValidationResult(
            faithfulness_score=faithfulness_score,
            total_claims=len(claims),
            supported_claims=supported_count,
            unsupported_claims=unsupported_claims,
            warnings=warnings
        )
    
    def _extract_claims(self, answer: str) -> List[str]:
        """
        Extract individual claims from answer text.
        
        Uses LLM to parse the answer into discrete factual claims.
        
        Args:
            answer: Generated answer text
        
        Returns:
            List of claim strings
        """
        if not answer or not answer.strip():
            return []
        
        # Format prompt
        prompt = self.CLAIM_EXTRACTION_PROMPT.format(answer=answer)
        
        # Call LLM for claim extraction
        response_text = self._call_ollama(prompt)
        
        if not response_text:
            logger.warning("Failed to extract claims from response")
            # Fallback: split by sentences
            return self._fallback_claim_extraction(answer)
        
        # Parse JSON response
        try:
            claims = json.loads(response_text)
            
            if isinstance(claims, list) and all(isinstance(c, str) for c in claims):
                return [claim.strip() for claim in claims if claim.strip()]
            else:
                logger.warning(f"Invalid claims format: {claims}")
                return self._fallback_claim_extraction(answer)
        
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse claims JSON: {e}")
            return self._fallback_claim_extraction(answer)
    
    def _fallback_claim_extraction(self, answer: str) -> List[str]:
        """
        Fallback claim extraction using simple sentence splitting.
        
        Args:
            answer: Answer text
        
        Returns:
            List of sentences as claims
        """
        # Remove citations
        text_without_citations = re.sub(r'\[[^\]]+\]', '', answer)
        
        # Split by sentence-ending punctuation
        sentences = re.split(r'[.!?]+', text_without_citations)
        
        # Filter and clean
        claims = [
            s.strip() for s in sentences
            if s.strip() and len(s.strip()) > 10  # Ignore very short fragments
        ]
        
        logger.debug(f"Fallback extraction: {len(claims)} claims")
        return claims
    
    def _check_entailment(self, claim: str, context: str) -> bool:
        """
        Check if context entails (supports) the claim.
        
        Uses LLM-based entailment checking to determine if the claim
        is supported by the context.
        
        Args:
            claim: Claim to check
            context: Context text
        
        Returns:
            True if claim is supported, False otherwise
        """
        if not claim or not context:
            return False
        
        # Truncate context if too long (keep first 2000 chars for efficiency)
        context_excerpt = context[:2000] if len(context) > 2000 else context
        
        # Format prompt
        prompt = self.ENTAILMENT_PROMPT.format(
            context_excerpt=context_excerpt,
            claim=claim
        )
        
        # Call LLM for entailment check
        response_text = self._call_ollama(prompt)
        
        if not response_text:
            logger.warning(f"Failed to check entailment for claim: {claim[:50]}...")
            # Conservative: assume not supported if check fails
            return False
        
        # Parse JSON response
        try:
            result = json.loads(response_text)
            
            if isinstance(result, dict) and "supported" in result:
                is_supported = result.get("supported", False)
                confidence = result.get("confidence", 0.0)
                
                logger.debug(
                    f"Entailment check: supported={is_supported}, "
                    f"confidence={confidence:.2f}"
                )
                
                return is_supported
            else:
                logger.warning(f"Invalid entailment response format: {result}")
                return False
        
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse entailment JSON: {e}")
            return False
    
    def _call_ollama(self, prompt: str, max_retries: int = 2) -> Optional[str]:
        """
        Call Ollama API for LLM operations.
        
        Args:
            prompt: Prompt text
            max_retries: Maximum number of retry attempts (default: 2)
        
        Returns:
            LLM response text or None if failed
        """
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for consistent results
                "top_p": 0.9,
                "num_predict": 512  # Reasonable limit for validation tasks
            }
        }
        
        for attempt in range(max_retries + 1):
            try:
                logger.debug(f"Calling Ollama API (attempt {attempt + 1}/{max_retries + 1})")
                
                response = requests.post(
                    url,
                    json=payload,
                    timeout=30,  # 30 second timeout for validation
                    headers={"Content-Type": "application/json"}
                )
                
                response.raise_for_status()
                
                # Parse response
                response_data = response.json()
                
                if "response" in response_data:
                    generated_text = response_data["response"].strip()
                    logger.debug(f"Ollama API call successful: {len(generated_text)} chars")
                    return generated_text
                else:
                    logger.warning(f"Unexpected Ollama response format: {response_data}")
                    return None
            
            except requests.exceptions.Timeout:
                logger.warning(f"Ollama API timeout (attempt {attempt + 1}/{max_retries + 1})")
                if attempt == max_retries:
                    logger.error("Ollama API call failed after timeout retries")
                    return None
            
            except requests.exceptions.RequestException as e:
                logger.warning(f"Ollama API call failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                if attempt == max_retries:
                    logger.error(f"Ollama API call failed after {max_retries + 1} attempts")
                    return None
            
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse Ollama response (attempt {attempt + 1}/{max_retries + 1}): {e}")
                if attempt == max_retries:
                    logger.error("Ollama API response parsing failed after retries")
                    return None
        
        return None
