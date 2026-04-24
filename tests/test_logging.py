"""Tests for logging infrastructure"""

import json
import logging
from io import StringIO
from src.utils.logging import setup_logging, get_logger, StructuredFormatter


class TestStructuredFormatter:
    """Test StructuredFormatter"""
    
    def test_format_basic_message(self):
        """Test formatting a basic log message"""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert log_data["level"] == "INFO"
        assert log_data["logger"] == "test"
        assert log_data["message"] == "Test message"
        assert "timestamp" in log_data
    
    def test_format_with_extra_fields(self):
        """Test formatting with extra fields"""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Error occurred",
            args=(),
            exc_info=None
        )
        record.component = "VectorRetriever"
        record.error_type = "StorageError"
        record.context = {"query": "test query"}
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert log_data["component"] == "VectorRetriever"
        assert log_data["error_type"] == "StorageError"
        assert log_data["context"] == {"query": "test query"}
    
    def test_format_with_exception(self):
        """Test formatting with exception info"""
        formatter = StructuredFormatter()
        
        try:
            raise ValueError("Test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()
            
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=1,
                msg="Exception occurred",
                args=(),
                exc_info=exc_info
            )
            
            formatted = formatter.format(record)
            log_data = json.loads(formatted)
            
            assert "exception" in log_data
            assert log_data["exception"]["type"] == "ValueError"
            assert log_data["exception"]["message"] == "Test error"
            assert "stack_trace" in log_data["exception"]


class TestLoggingSetup:
    """Test logging setup"""
    
    def test_setup_logging_default(self):
        """Test default logging setup"""
        setup_logging()
        
        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO
    
    def test_setup_logging_debug_level(self):
        """Test logging setup with DEBUG level"""
        setup_logging(level="DEBUG")
        
        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG
    
    def test_get_logger(self):
        """Test getting a logger instance"""
        logger = get_logger("test_module")
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"
    
    def test_structured_logging_output(self):
        """Test that structured logging produces JSON output"""
        setup_logging(structured=True)
        logger = get_logger("test")
        
        # Verify logger is configured
        root_logger = logging.getLogger()
        assert len(root_logger.handlers) > 0
        
        # Verify handler has StructuredFormatter
        handler = root_logger.handlers[0]
        assert isinstance(handler.formatter, StructuredFormatter)
