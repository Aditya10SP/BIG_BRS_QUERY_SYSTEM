"""Tests for error handling utilities."""

import time
import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from src.utils.errors import (
    GraphRAGError,
    ParsingError,
    StorageError,
    RetrievalError,
    LLMError,
    ValidationError,
    ErrorSeverity,
    RetryConfig,
    retry_with_backoff,
    DegradationMode,
    GracefulDegradation,
    ErrorContext,
    log_error,
    log_error_with_context,
    get_degradation_manager,
    BM25_DEGRADATION,
    NEO4J_DEGRADATION,
    CROSS_ENCODER_DEGRADATION,
    QDRANT_DEGRADATION,
    LLM_DEGRADATION
)


class TestErrorClasses:
    """Test custom error classes."""
    
    def test_graph_rag_error_basic(self):
        """Test basic GraphRAGError creation."""
        error = GraphRAGError("Test error")
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.context == {}
        assert error.cause is None
        assert isinstance(error.timestamp, datetime)
    
    def test_graph_rag_error_with_context(self):
        """Test GraphRAGError with context."""
        context = {"file_path": "/test/file.txt", "line": 42}
        error = GraphRAGError("Test error", context=context)
        assert error.context == context
    
    def test_graph_rag_error_with_cause(self):
        """Test GraphRAGError with cause."""
        cause = ValueError("Original error")
        error = GraphRAGError("Test error", cause=cause)
        assert error.cause == cause
    
    def test_graph_rag_error_to_dict(self):
        """Test error serialization to dict."""
        context = {"key": "value"}
        error = GraphRAGError("Test error", context=context)
        error_dict = error.to_dict()
        
        assert error_dict["error_type"] == "GraphRAGError"
        assert error_dict["error_message"] == "Test error"
        assert error_dict["context"] == context
        assert "timestamp" in error_dict
    
    def test_parsing_error(self):
        """Test ParsingError is a GraphRAGError."""
        error = ParsingError("Parse failed")
        assert isinstance(error, GraphRAGError)
        assert str(error) == "Parse failed"
    
    def test_storage_error(self):
        """Test StorageError is a GraphRAGError."""
        error = StorageError("Storage failed")
        assert isinstance(error, GraphRAGError)
    
    def test_retrieval_error(self):
        """Test RetrievalError is a GraphRAGError."""
        error = RetrievalError("Retrieval failed")
        assert isinstance(error, GraphRAGError)
    
    def test_llm_error(self):
        """Test LLMError is a GraphRAGError."""
        error = LLMError("LLM failed")
        assert isinstance(error, GraphRAGError)
    
    def test_validation_error(self):
        """Test ValidationError is a GraphRAGError."""
        error = ValidationError("Validation failed")
        assert isinstance(error, GraphRAGError)


class TestRetryConfig:
    """Test retry configuration."""
    
    def test_default_config(self):
        """Test default retry configuration."""
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
    
    def test_custom_config(self):
        """Test custom retry configuration."""
        config = RetryConfig(
            max_attempts=5,
            initial_delay=0.5,
            max_delay=30.0,
            exponential_base=3.0,
            jitter=False
        )
        assert config.max_attempts == 5
        assert config.initial_delay == 0.5
        assert config.max_delay == 30.0
        assert config.exponential_base == 3.0
        assert config.jitter is False


class TestRetryDecorator:
    """Test retry decorator with exponential backoff."""
    
    def test_success_on_first_attempt(self):
        """Test function succeeds on first attempt."""
        mock_func = Mock(return_value="success")
        
        @retry_with_backoff(RetryConfig(max_attempts=3))
        def test_func():
            return mock_func()
        
        result = test_func()
        assert result == "success"
        assert mock_func.call_count == 1
    
    def test_success_after_retry(self):
        """Test function succeeds after retry."""
        mock_func = Mock(side_effect=[
            ConnectionError("First attempt"),
            "success"
        ])
        
        @retry_with_backoff(RetryConfig(max_attempts=3, initial_delay=0.01, jitter=False))
        def test_func():
            return mock_func()
        
        result = test_func()
        assert result == "success"
        assert mock_func.call_count == 2
    
    def test_all_attempts_fail(self):
        """Test all retry attempts fail."""
        mock_func = Mock(side_effect=ConnectionError("Always fails"))
        
        @retry_with_backoff(RetryConfig(max_attempts=3, initial_delay=0.01, jitter=False))
        def test_func():
            return mock_func()
        
        with pytest.raises(ConnectionError, match="Always fails"):
            test_func()
        
        assert mock_func.call_count == 3
    
    def test_non_retryable_exception(self):
        """Test non-retryable exception fails immediately."""
        mock_func = Mock(side_effect=ValueError("Not retryable"))
        
        @retry_with_backoff(RetryConfig(
            max_attempts=3,
            initial_delay=0.01,
            retryable_exceptions=(ConnectionError,)
        ))
        def test_func():
            return mock_func()
        
        with pytest.raises(ValueError, match="Not retryable"):
            test_func()
        
        assert mock_func.call_count == 1
    
    def test_retry_callback(self):
        """Test retry callback is called."""
        callback_mock = Mock()
        mock_func = Mock(side_effect=[
            ConnectionError("First"),
            ConnectionError("Second"),
            "success"
        ])
        
        @retry_with_backoff(
            RetryConfig(max_attempts=3, initial_delay=0.01, jitter=False),
            on_retry=callback_mock
        )
        def test_func():
            return mock_func()
        
        result = test_func()
        assert result == "success"
        assert callback_mock.call_count == 2


class TestGracefulDegradation:
    """Test graceful degradation manager."""
    
    def test_initialization(self):
        """Test degradation manager initialization."""
        manager = GracefulDegradation()
        assert len(manager.active_degradations) == 0
        assert len(manager.degradation_history) == 0
        assert not manager.is_degraded()
    
    def test_enter_degraded_mode(self):
        """Test entering degraded mode."""
        manager = GracefulDegradation()
        mode = DegradationMode(
            mode_name="test_mode",
            description="Test degradation",
            fallback_behavior="Use fallback",
            severity=ErrorSeverity.WARNING
        )
        
        manager.enter_degraded_mode(mode)
        
        assert manager.is_degraded()
        assert manager.is_degraded("test_mode")
        assert len(manager.active_degradations) == 1
        assert len(manager.degradation_history) == 1
    
    def test_exit_degraded_mode(self):
        """Test exiting degraded mode."""
        manager = GracefulDegradation()
        mode = DegradationMode(
            mode_name="test_mode",
            description="Test degradation",
            fallback_behavior="Use fallback"
        )
        
        manager.enter_degraded_mode(mode)
        assert manager.is_degraded("test_mode")
        
        manager.exit_degraded_mode("test_mode")
        assert not manager.is_degraded("test_mode")
        assert len(manager.active_degradations) == 0
        assert len(manager.degradation_history) == 2
    
    def test_multiple_degradations(self):
        """Test multiple active degradations."""
        manager = GracefulDegradation()
        
        mode1 = DegradationMode("mode1", "Desc1", "Fallback1")
        mode2 = DegradationMode("mode2", "Desc2", "Fallback2")
        
        manager.enter_degraded_mode(mode1)
        manager.enter_degraded_mode(mode2)
        
        assert manager.is_degraded()
        assert manager.is_degraded("mode1")
        assert manager.is_degraded("mode2")
        assert len(manager.active_degradations) == 2
    
    def test_get_active_degradations(self):
        """Test getting active degradations."""
        manager = GracefulDegradation()
        mode = DegradationMode("test", "Test", "Fallback")
        
        manager.enter_degraded_mode(mode)
        active = manager.get_active_degradations()
        
        assert len(active) == 1
        assert active[0].mode_name == "test"
    
    def test_get_degradation_summary(self):
        """Test getting degradation summary."""
        manager = GracefulDegradation()
        mode = DegradationMode("test", "Test", "Fallback", ErrorSeverity.ERROR)
        
        manager.enter_degraded_mode(mode)
        summary = manager.get_degradation_summary()
        
        assert summary["is_degraded"] is True
        assert len(summary["active_modes"]) == 1
        assert summary["active_modes"][0]["name"] == "test"
        assert summary["active_modes"][0]["severity"] == "error"


class TestPredefinedDegradations:
    """Test predefined degradation modes."""
    
    def test_bm25_degradation(self):
        """Test BM25 degradation mode."""
        assert BM25_DEGRADATION.mode_name == "bm25_unavailable"
        assert BM25_DEGRADATION.severity == ErrorSeverity.WARNING
    
    def test_neo4j_degradation(self):
        """Test Neo4j degradation mode."""
        assert NEO4J_DEGRADATION.mode_name == "neo4j_unavailable"
        assert NEO4J_DEGRADATION.severity == ErrorSeverity.ERROR
    
    def test_cross_encoder_degradation(self):
        """Test cross-encoder degradation mode."""
        assert CROSS_ENCODER_DEGRADATION.mode_name == "cross_encoder_unavailable"
        assert CROSS_ENCODER_DEGRADATION.severity == ErrorSeverity.WARNING
    
    def test_qdrant_degradation(self):
        """Test Qdrant degradation mode."""
        assert QDRANT_DEGRADATION.mode_name == "qdrant_unavailable"
        assert QDRANT_DEGRADATION.severity == ErrorSeverity.CRITICAL
    
    def test_llm_degradation(self):
        """Test LLM degradation mode."""
        assert LLM_DEGRADATION.mode_name == "llm_unavailable"
        assert LLM_DEGRADATION.severity == ErrorSeverity.CRITICAL


class TestErrorContext:
    """Test error context builder."""
    
    def test_empty_context(self):
        """Test empty context."""
        context = ErrorContext()
        assert context.build() == {}
    
    def test_add_single_item(self):
        """Test adding single context item."""
        context = ErrorContext()
        context.add("key", "value")
        assert context.build() == {"key": "value"}
    
    def test_add_multiple_items(self):
        """Test adding multiple context items."""
        context = ErrorContext()
        context.add("key1", "value1").add("key2", "value2")
        result = context.build()
        assert result == {"key1": "value1", "key2": "value2"}
    
    def test_add_all(self):
        """Test adding multiple items at once."""
        context = ErrorContext()
        items = {"key1": "value1", "key2": "value2"}
        context.add_all(items)
        assert context.build() == items
    
    def test_chaining(self):
        """Test method chaining."""
        context = ErrorContext()
        result = context.add("a", 1).add("b", 2).add_all({"c": 3}).build()
        assert result == {"a": 1, "b": 2, "c": 3}


class TestErrorLogging:
    """Test error logging utilities."""
    
    @patch('src.utils.errors.logger')
    def test_log_error_basic(self, mock_logger):
        """Test basic error logging."""
        error = ValueError("Test error")
        log_error(error)
        
        mock_logger.error.assert_called_once()
    
    @patch('src.utils.errors.logger')
    def test_log_error_with_context(self, mock_logger):
        """Test error logging with context."""
        error = ValueError("Test error")
        context = {"key": "value"}
        log_error(error, context=context, component="TestComponent")
        
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args
        assert "TestComponent" in call_args[0][0]
    
    @patch('src.utils.errors.logger')
    def test_log_error_with_severity(self, mock_logger):
        """Test error logging with different severity."""
        error = ValueError("Test error")
        log_error(error, severity=ErrorSeverity.WARNING)
        
        mock_logger.warning.assert_called_once()
    
    @patch('src.utils.errors.logger')
    def test_log_error_with_context_helper(self, mock_logger):
        """Test log_error_with_context helper."""
        error = ValueError("Test error")
        log_error_with_context(
            error,
            component="TestComponent",
            operation="test_operation",
            extra_key="extra_value"
        )
        
        mock_logger.error.assert_called_once()


class TestGlobalDegradationManager:
    """Test global degradation manager."""
    
    def test_get_degradation_manager(self):
        """Test getting global degradation manager."""
        manager1 = get_degradation_manager()
        manager2 = get_degradation_manager()
        
        # Should return same instance
        assert manager1 is manager2
    
    def test_global_manager_state_persists(self):
        """Test global manager state persists across calls."""
        manager = get_degradation_manager()
        
        # Clear any existing state
        for mode in manager.get_active_degradations():
            manager.exit_degraded_mode(mode.mode_name)
        
        # Add degradation
        mode = DegradationMode("test", "Test", "Fallback")
        manager.enter_degraded_mode(mode)
        
        # Get manager again and check state
        manager2 = get_degradation_manager()
        assert manager2.is_degraded("test")
        
        # Cleanup
        manager.exit_degraded_mode("test")
