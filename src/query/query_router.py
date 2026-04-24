"""Query routing component for classifying query intent"""

import json
import logging
import requests
from enum import Enum
from typing import Tuple, Optional

from src.utils.errors import (
    LLMError,
    ErrorContext,
    retry_with_backoff,
    RetryConfig,
    log_error_with_context
)

logger = logging.getLogger(__name__)


class QueryMode(Enum):
    """Query retrieval modes"""
    VECTOR = "vector"      # Factual/definitional queries
    GRAPH = "graph"        # Relational/structural queries
    HYBRID = "hybrid"      # Complex queries needing both


class QueryRouter:
    """
    Routes queries to appropriate retrieval mode using LLM-based classification.
    
    Classifies queries into VECTOR, GRAPH, or HYBRID modes based on intent analysis.
    Defaults to HYBRID mode when classification confidence is below threshold.
    """
    
    CLASSIFICATION_PROMPT_TEMPLATE = """Classify this query into one of three retrieval modes:

VECTOR: Factual or definitional questions about single concepts
- Examples: "What is NEFT?", "Define transaction limit", "Explain RTGS process"

GRAPH: Relational, structural, or comparison questions
- Examples: "What systems depend on NEFT?", "Compare RTGS and IMPS", "Show payment workflow"

HYBRID: Complex questions requiring both relationships and full text
- Examples: "How does NEFT integrate with Core Banking and what are the limits?", 
  "What are the conflicts between payment rules across documents?"

Query: {query}

Respond with JSON:
{{
  "mode": "VECTOR|GRAPH|HYBRID",
  "confidence": 0.0-1.0,
  "reasoning": "..."
}}"""
    
    def __init__(self, ollama_base_url: str, llm_model: str, confidence_threshold: float = 0.7):
        """
        Initialize with LLM client for classification.
        
        Args:
            ollama_base_url: Base URL for Ollama API
            llm_model: Name of LLM model to use
            confidence_threshold: Minimum confidence to accept classification (default: 0.7)
        """
        self.ollama_base_url = ollama_base_url
        self.llm_model = llm_model
        self.confidence_threshold = confidence_threshold
        logger.info(f"Initialized QueryRouter with model: {llm_model}, threshold: {confidence_threshold}")
    
    def route(self, query: str) -> Tuple[QueryMode, float]:
        """
        Classify query and return mode with confidence.
        
        Args:
            query: User query string
            
        Returns:
            Tuple of (QueryMode enum, confidence_score)
            
        Defaults to HYBRID mode if:
        - Classification fails
        - Confidence is below threshold
        - Invalid mode returned
        """
        if not query or not query.strip():
            logger.warning("Empty query provided, defaulting to HYBRID mode")
            return QueryMode.HYBRID, 0.0
        
        # Call LLM for classification
        classification = self._classify_with_llm(query)
        
        if not classification:
            logger.warning("LLM classification failed, defaulting to HYBRID mode")
            return QueryMode.HYBRID, 0.0
        
        mode_str = classification.get("mode", "").upper()
        confidence = classification.get("confidence", 0.0)
        reasoning = classification.get("reasoning", "")
        
        # Validate mode
        try:
            mode = QueryMode[mode_str]
        except KeyError:
            logger.warning(f"Invalid mode '{mode_str}' returned, defaulting to HYBRID")
            return QueryMode.HYBRID, 0.0
        
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            logger.info(
                f"Low confidence ({confidence:.2f} < {self.confidence_threshold}), "
                f"defaulting to HYBRID mode. Original: {mode_str}"
            )
            return QueryMode.HYBRID, confidence
        
        logger.info(
            f"Query classified as {mode_str} with confidence {confidence:.2f}. "
            f"Reasoning: {reasoning}"
        )
        
        return mode, confidence
    
    def _classify_with_llm(self, query: str) -> Optional[dict]:
        """
        Call LLM to classify query intent.
        
        Args:
            query: User query string
            
        Returns:
            Dictionary with mode, confidence, and reasoning, or None if failed
        """
        prompt = self.CLASSIFICATION_PROMPT_TEMPLATE.format(query=query)
        
        # Call Ollama API
        response_text = self._call_ollama(prompt)
        
        if not response_text:
            return None
        
        # Parse JSON response
        try:
            # Extract JSON from response (handle markdown code blocks)
            response_text = response_text.strip()
            if "```json" in response_text:
                # Extract JSON from markdown code block
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            elif "```" in response_text:
                # Extract from generic code block
                start = response_text.find("```") + 3
                end = response_text.find("```", start)
                response_text = response_text[start:end].strip()
            
            classification = json.loads(response_text)
            
            # Validate required fields
            if "mode" not in classification or "confidence" not in classification:
                logger.error(f"Invalid classification response: missing required fields")
                return None
            
            # Ensure confidence is a float
            classification["confidence"] = float(classification["confidence"])
            
            return classification
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse LLM classification response: {e}")
            logger.debug(f"Response text: {response_text}")
            return None
    
    def _call_ollama(self, prompt: str, max_retries: int = 2) -> Optional[str]:
        """
        Call Ollama API for LLM generation with retry logic.
        
        Args:
            prompt: Prompt text to send to LLM
            max_retries: Maximum number of retry attempts
            
        Returns:
            LLM response text or None if failed
        """
        url = f"{self.ollama_base_url}/api/generate"
        
        payload = {
            "model": self.llm_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for consistent classification
            }
        }
        
        # Configure retry with exponential backoff
        retry_config = RetryConfig(
            max_attempts=max_retries + 1,
            initial_delay=1.0,
            max_delay=10.0,
            exponential_base=2.0,
            retryable_exceptions=(
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.RequestException,
                Exception  # Catch all for compatibility with existing tests
            )
        )
        
        @retry_with_backoff(retry_config)
        def _make_request():
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        
        try:
            return _make_request()
        except Exception as e:
            log_error_with_context(
                e,
                component="QueryRouter",
                operation="_call_ollama",
                url=url,
                model=self.llm_model,
                max_retries=max_retries
            )
            return None
