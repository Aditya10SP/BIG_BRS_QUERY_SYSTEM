"""
Comprehensive error handling for the Graph RAG Layer system.

This module provides:
1. Custom error classes for different failure scenarios
2. Retry logic with exponential backoff
3. Graceful degradation utilities
4. Error context tracking
"""

import functools
import logging
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

logger = logging.getLogger(__name__)


# ============================================================================
# Error Classes
# ============================================================================

class GraphRAGError(Exception):
    """Base exception for all Graph RAG Layer errors."""
    
    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """
        Initialize error with message, context, and optional cause.
        
        Args:
            message: Human-readable error message
            context: Additional context information (e.g., file path, query)
            cause: Original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.cause = cause
        self.timestamp = datetime.now()
        self.stack_trace = traceback.format_exc() if cause else None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "error_message": self.message,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "cause": str(self.cause) if self.cause else None,
            "stack_trace": self.stack_trace
        }


class ParsingError(GraphRAGError):
    """
    Exception raised when document parsing fails.
    
    Scenarios:
    - Invalid file format (not .docx or .pdf)
    - Corrupted file content
    - Unsupported document structure
    - Missing required metadata
    """
    pass


class StorageError(GraphRAGError):
    """
    Exception raised when storage operations fail.
    
    Scenarios:
    - Qdrant connection failure
    - Neo4j connection failure
    - PostgreSQL connection failure
    - Write operation timeout
    - Insufficient storage space
    - Index creation failure
    """
    pass


class RetrievalError(GraphRAGError):
    """
    Exception raised when retrieval operations fail.
    
    Scenarios:
    - Empty query string
    - Invalid query parameters
    - No results found above threshold
    - Graph traversal timeout
    - Cypher query syntax error
    - Vector search failure
    """
    pass


class LLMError(GraphRAGError):
    """
    Exception raised when LLM operations fail.
    
    Scenarios:
    - Ollama service unavailable
    - Model not found
    - Context too large for model
    - Generation timeout
    - Rate limiting
    - Invalid response format
    """
    pass


class ValidationError(GraphRAGError):
    """
    Exception raised when validation fails.
    
    Scenarios:
    - Missing required configuration
    - Invalid configuration values
    - Malformed API request
    - Invalid file format
    - Schema validation failure
    """
    pass


# ============================================================================
# Error Severity Levels
# ============================================================================

class ErrorSeverity(Enum):
    """Error severity levels for logging and handling."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# ============================================================================
# Retry Configuration
# ============================================================================

@dataclass
class RetryConfig:
    """Configuration for retry logic with exponential backoff."""
    
    max_attempts: int = 3
    """Maximum number of retry attempts (including initial attempt)."""
    
    initial_delay: float = 1.0
    """Initial delay in seconds before first retry."""
    
    max_delay: float = 60.0
    """Maximum delay in seconds between retries."""
    
    exponential_base: float = 2.0
    """Base for exponential backoff calculation."""
    
    jitter: bool = True
    """Add random jitter to prevent thundering herd."""
    
    retryable_exceptions: tuple = (
        ConnectionError,
        TimeoutError,
        StorageError,
        LLMError
    )
    """Exception types that should trigger retry."""


# ============================================================================
# Retry Decorator with Exponential Backoff
# ============================================================================

T = TypeVar('T')


def retry_with_backoff(
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[Exception, int], None]] = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        config: RetryConfig object with retry parameters
        on_retry: Optional callback function called on each retry
                 Signature: on_retry(exception, attempt_number)
    
    Returns:
        Decorated function with retry logic
    
    Example:
        @retry_with_backoff(RetryConfig(max_attempts=3))
        def call_external_api():
            # API call that might fail
            pass
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                
                except config.retryable_exceptions as e:
                    last_exception = e
                    
                    # Don't retry on last attempt
                    if attempt == config.max_attempts - 1:
                        break
                    
                    # Calculate delay with exponential backoff
                    delay = min(
                        config.initial_delay * (config.exponential_base ** attempt),
                        config.max_delay
                    )
                    
                    # Add jitter if enabled
                    if config.jitter:
                        import random
                        delay *= (0.5 + random.random())
                    
                    logger.warning(
                        f"Attempt {attempt + 1}/{config.max_attempts} failed for {func.__name__}. "
                        f"Retrying in {delay:.2f}s. Error: {e}"
                    )
                    
                    # Call retry callback if provided
                    if on_retry:
                        on_retry(e, attempt + 1)
                    
                    time.sleep(delay)
                
                except Exception as e:
                    # Non-retryable exception, fail immediately
                    logger.error(
                        f"Non-retryable exception in {func.__name__}: {e}",
                        extra={"stack_trace": traceback.format_exc()}
                    )
                    raise
            
            # All retries exhausted
            logger.error(
                f"All {config.max_attempts} attempts failed for {func.__name__}",
                extra={
                    "last_exception": str(last_exception),
                    "stack_trace": traceback.format_exc()
                }
            )
            raise last_exception
        
        return wrapper
    return decorator


# ============================================================================
# Graceful Degradation
# ============================================================================

@dataclass
class DegradationMode:
    """Represents a degraded operational mode."""
    
    mode_name: str
    """Name of the degradation mode."""
    
    description: str
    """Description of what functionality is degraded."""
    
    fallback_behavior: str
    """Description of fallback behavior."""
    
    severity: ErrorSeverity = ErrorSeverity.WARNING
    """Severity level of this degradation."""


class GracefulDegradation:
    """
    Manages graceful degradation when components fail.
    
    Tracks active degradations and provides fallback strategies.
    """
    
    def __init__(self):
        """Initialize degradation tracker."""
        self.active_degradations: List[DegradationMode] = []
        self.degradation_history: List[Dict[str, Any]] = []
    
    def enter_degraded_mode(self, mode: DegradationMode) -> None:
        """
        Enter a degraded operational mode.
        
        Args:
            mode: DegradationMode describing the degradation
        """
        self.active_degradations.append(mode)
        
        log_level = getattr(logger, mode.severity.value)
        log_level(
            f"Entering degraded mode: {mode.mode_name}",
            extra={
                "mode": mode.mode_name,
                "description": mode.description,
                "fallback": mode.fallback_behavior,
                "severity": mode.severity.value
            }
        )
        
        self.degradation_history.append({
            "mode": mode.mode_name,
            "timestamp": datetime.now().isoformat(),
            "action": "enter"
        })
    
    def exit_degraded_mode(self, mode_name: str) -> None:
        """
        Exit a degraded operational mode.
        
        Args:
            mode_name: Name of the mode to exit
        """
        self.active_degradations = [
            m for m in self.active_degradations
            if m.mode_name != mode_name
        ]
        
        logger.info(
            f"Exiting degraded mode: {mode_name}",
            extra={"mode": mode_name}
        )
        
        self.degradation_history.append({
            "mode": mode_name,
            "timestamp": datetime.now().isoformat(),
            "action": "exit"
        })
    
    def is_degraded(self, mode_name: Optional[str] = None) -> bool:
        """
        Check if system is in degraded mode.
        
        Args:
            mode_name: Optional specific mode to check. If None, checks any degradation.
        
        Returns:
            True if in specified degraded mode (or any mode if mode_name is None)
        """
        if mode_name is None:
            return len(self.active_degradations) > 0
        
        return any(m.mode_name == mode_name for m in self.active_degradations)
    
    def get_active_degradations(self) -> List[DegradationMode]:
        """Get list of currently active degradations."""
        return self.active_degradations.copy()
    
    def get_degradation_summary(self) -> Dict[str, Any]:
        """Get summary of degradation status."""
        return {
            "is_degraded": len(self.active_degradations) > 0,
            "active_modes": [
                {
                    "name": m.mode_name,
                    "description": m.description,
                    "fallback": m.fallback_behavior,
                    "severity": m.severity.value
                }
                for m in self.active_degradations
            ],
            "history_count": len(self.degradation_history)
        }


# ============================================================================
# Predefined Degradation Modes
# ============================================================================

# BM25 Index Unavailable
BM25_DEGRADATION = DegradationMode(
    mode_name="bm25_unavailable",
    description="BM25 keyword index is unavailable",
    fallback_behavior="Fall back to vector-only search",
    severity=ErrorSeverity.WARNING
)

# Neo4j Unavailable
NEO4J_DEGRADATION = DegradationMode(
    mode_name="neo4j_unavailable",
    description="Neo4j graph database is unavailable",
    fallback_behavior="Fall back to vector-only mode (no graph retrieval)",
    severity=ErrorSeverity.ERROR
)

# Cross-Encoder Unavailable
CROSS_ENCODER_DEGRADATION = DegradationMode(
    mode_name="cross_encoder_unavailable",
    description="Cross-encoder reranking model is unavailable",
    fallback_behavior="Use original retrieval scores without reranking",
    severity=ErrorSeverity.WARNING
)

# Qdrant Unavailable
QDRANT_DEGRADATION = DegradationMode(
    mode_name="qdrant_unavailable",
    description="Qdrant vector store is unavailable",
    fallback_behavior="System cannot perform vector search (critical failure)",
    severity=ErrorSeverity.CRITICAL
)

# LLM Unavailable
LLM_DEGRADATION = DegradationMode(
    mode_name="llm_unavailable",
    description="LLM service (Ollama) is unavailable",
    fallback_behavior="Cannot generate responses or perform LLM-based operations",
    severity=ErrorSeverity.CRITICAL
)


# ============================================================================
# Error Context Builder
# ============================================================================

class ErrorContext:
    """
    Builder for error context information.
    
    Helps collect relevant context when errors occur.
    """
    
    def __init__(self):
        """Initialize empty context."""
        self._context: Dict[str, Any] = {}
    
    def add(self, key: str, value: Any) -> 'ErrorContext':
        """
        Add context information.
        
        Args:
            key: Context key
            value: Context value
        
        Returns:
            Self for chaining
        """
        self._context[key] = value
        return self
    
    def add_all(self, context: Dict[str, Any]) -> 'ErrorContext':
        """
        Add multiple context items.
        
        Args:
            context: Dictionary of context items
        
        Returns:
            Self for chaining
        """
        self._context.update(context)
        return self
    
    def build(self) -> Dict[str, Any]:
        """Build and return context dictionary."""
        return self._context.copy()


# ============================================================================
# Error Logging Utilities
# ============================================================================

def log_error(
    error: Exception,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    context: Optional[Dict[str, Any]] = None,
    component: Optional[str] = None
) -> None:
    """
    Log error with complete context information.
    
    Args:
        error: Exception to log
        severity: Error severity level
        context: Additional context information
        component: Component name where error occurred
    """
    log_level = getattr(logger, severity.value)
    
    error_info = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "severity": severity.value,
        "timestamp": datetime.now().isoformat(),
        "stack_trace": traceback.format_exc()
    }
    
    if context:
        error_info["context"] = context
    
    if component:
        error_info["component"] = component
    
    # Add GraphRAGError specific information
    if isinstance(error, GraphRAGError):
        error_info.update(error.to_dict())
    
    log_level(
        f"Error in {component or 'unknown component'}: {error}",
        extra=error_info
    )


def log_error_with_context(
    error: Exception,
    component: str,
    operation: str,
    **context_kwargs
) -> None:
    """
    Convenience function to log error with context.
    
    Args:
        error: Exception to log
        component: Component name
        operation: Operation being performed
        **context_kwargs: Additional context as keyword arguments
    """
    context = ErrorContext()
    context.add("operation", operation)
    context.add_all(context_kwargs)
    
    log_error(
        error,
        severity=ErrorSeverity.ERROR,
        context=context.build(),
        component=component
    )


# ============================================================================
# Global Degradation Manager
# ============================================================================

# Global instance for tracking system-wide degradations
_global_degradation_manager = GracefulDegradation()


def get_degradation_manager() -> GracefulDegradation:
    """Get the global degradation manager instance."""
    return _global_degradation_manager
