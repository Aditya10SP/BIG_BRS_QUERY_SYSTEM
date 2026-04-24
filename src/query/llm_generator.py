"""LLM generation with grounded responses and citation enforcement."""

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import requests

from src.retrieval.context_assembler import AssembledContext
from src.utils.errors import (
    LLMError,
    ErrorContext,
    retry_with_backoff,
    RetryConfig,
    log_error_with_context
)

logger = logging.getLogger(__name__)


@dataclass
class GeneratedResponse:
    """
    Represents a generated response from the LLM.
    
    Attributes:
        answer: Generated answer text with inline citations
        citations_used: List of citation IDs referenced in the answer
        model: Name of the LLM model used for generation
        timestamp: When the response was generated
    """
    answer: str
    citations_used: List[str]
    model: str
    timestamp: datetime


class LLMGenerator:
    """
    Generates grounded responses using assembled context with citation enforcement.
    
    This class:
    1. Sends context and query to Ollama LLM API
    2. Enforces citation requirements through system prompts
    3. Handles insufficient context scenarios
    4. Extracts citations from generated responses
    5. Provides structured response objects
    
    The system prompt instructs the LLM to:
    - Answer only using provided context
    - Include citations in [citation_id] format
    - Return "Insufficient information" when context is inadequate
    - Be precise and factual
    - Include relevant graph facts in answers
    
    Attributes:
        base_url: Base URL for Ollama API
        model: Name of LLM model to use
    """
    
    # System prompt template for grounded generation with citations
    SYSTEM_PROMPT_TEMPLATE = """You are a banking documentation assistant. Answer the user's question using ONLY the provided context.

Rules:
1. Only use information from the provided context
2. Cite sources using [citation_id] format after each claim
3. If the context doesn't contain enough information, respond: "Insufficient information in documents to answer this question."
4. Be precise and factual
5. Include relevant graph facts (dependencies, integrations, conflicts) in your answer

Context:
{context_text}

Question: {query}

Answer:"""
    
    def __init__(self, base_url: str, model: str):
        """
        Initialize with Ollama configuration.
        
        Args:
            base_url: Base URL for Ollama API (e.g., "http://localhost:11434")
            model: Name of LLM model to use (e.g., "llama2", "mistral")
        
        Raises:
            ValueError: If base_url or model is empty
        """
        if not base_url or not base_url.strip():
            raise ValueError("base_url cannot be empty")
        
        if not model or not model.strip():
            raise ValueError("model cannot be empty")
        
        self.base_url = base_url.rstrip('/')
        self.model = model
        
        logger.info(
            "LLMGenerator initialized",
            extra={
                "base_url": self.base_url,
                "model": self.model
            }
        )
    
    def generate(self, query: str, context: AssembledContext) -> GeneratedResponse:
        """
        Generate response from context with citations.
        
        This method:
        1. Formats the system prompt with context and query
        2. Calls Ollama API for generation
        3. Extracts citations from the response
        4. Returns structured GeneratedResponse object
        
        Args:
            query: User's question
            context: AssembledContext with formatted text and citations
        
        Returns:
            GeneratedResponse with answer and inline citations
        
        Raises:
            ValueError: If query is empty or context is invalid
            RuntimeError: If LLM generation fails after retries
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        if not context or not context.context_text:
            raise ValueError("Context cannot be empty")
        
        logger.info(
            f"Generating response for query: '{query}' "
            f"(context: {context.token_count} tokens, {len(context.citations)} citations)"
        )
        
        # Format system prompt with context and query
        prompt = self.SYSTEM_PROMPT_TEMPLATE.format(
            context_text=context.context_text,
            query=query
        )
        
        # Call Ollama API
        response_text = self._call_ollama(prompt)
        
        if not response_text:
            raise RuntimeError("LLM generation failed after retries")
        
        # Extract citations from response
        citations_used = self._extract_citations(response_text)
        
        # Validate citations exist in context
        valid_citations = [
            citation for citation in citations_used
            if citation in context.citations
        ]
        
        if len(valid_citations) != len(citations_used):
            invalid_citations = set(citations_used) - set(valid_citations)
            logger.warning(
                f"Response contains invalid citations: {invalid_citations}"
            )
        
        logger.info(
            f"Response generated: {len(response_text)} chars, "
            f"{len(valid_citations)} valid citations"
        )
        
        return GeneratedResponse(
            answer=response_text,
            citations_used=valid_citations,
            model=self.model,
            timestamp=datetime.now()
        )
    
    def _call_ollama(self, prompt: str, max_retries: int = 2) -> Optional[str]:
        """
        Call Ollama API for LLM generation with retry logic.
        
        Uses the Ollama generate API endpoint with streaming disabled
        for simpler response handling.
        
        Args:
            prompt: Complete prompt including system instructions
            max_retries: Maximum number of retry attempts (default: 2)
        
        Returns:
            LLM response text or None if failed
        
        Raises:
            LLMError: If all retry attempts fail
        """
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,  # Get complete response at once
            "options": {
                "temperature": 0.1,  # Low temperature for factual responses
                "top_p": 0.9,
                "num_predict": 1024  # Reasonable response length limit
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
                requests.exceptions.RequestException
            )
        )
        
        @retry_with_backoff(retry_config)
        def _make_request():
            try:
                logger.debug(f"Calling Ollama API at {url}")
                
                response = requests.post(
                    url,
                    json=payload,
                    timeout=60,  # 60 second timeout for generation
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
                    context = ErrorContext().add("response_data", response_data).add("url", url)
                    raise LLMError(
                        "Unexpected Ollama response format",
                        context=context.build()
                    )
            
            except json.JSONDecodeError as e:
                log_error_with_context(
                    e,
                    component="LLMGenerator",
                    operation="_call_ollama",
                    url=url,
                    model=self.model
                )
                context = ErrorContext().add("url", url).add("model", self.model)
                raise LLMError(
                    "Failed to parse Ollama response",
                    context=context.build(),
                    cause=e
                )
        
        try:
            return _make_request()
        except Exception as e:
            log_error_with_context(
                e,
                component="LLMGenerator",
                operation="_call_ollama",
                url=url,
                model=self.model,
                max_retries=max_retries
            )
            return None
    
    def _extract_citations(self, response_text: str) -> List[str]:
        """
        Extract citation IDs from response text.
        
        Looks for citations in [citation_id] format and returns unique citation IDs.
        
        Args:
            response_text: Generated response text
        
        Returns:
            List of unique citation IDs found in the response
        """
        if not response_text:
            return []
        
        # Find all citations in [citation_id] format
        citation_pattern = r'\[([^\]]+)\]'
        matches = re.findall(citation_pattern, response_text)
        
        # Filter out non-citation brackets (e.g., [1], [a], etc.)
        # Valid citations should contain a colon (doc_id:section format)
        valid_citations = [
            match for match in matches
            if ':' in match and len(match.split(':')) == 2
        ]
        
        # Return unique citations while preserving order
        seen = set()
        unique_citations = []
        for citation in valid_citations:
            if citation not in seen:
                seen.add(citation)
                unique_citations.append(citation)
        
        logger.debug(f"Extracted {len(unique_citations)} unique citations from response")
        
        return unique_citations