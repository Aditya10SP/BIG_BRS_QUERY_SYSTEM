# Error Handling Guide

This document describes the comprehensive error handling system for the Graph RAG Layer.

## Overview

The error handling system provides:

1. **Custom Error Classes** - Specific exception types for different failure scenarios
2. **Retry Logic with Exponential Backoff** - Automatic retry for transient failures
3. **Graceful Degradation** - Fallback strategies when components fail
4. **Error Context Tracking** - Rich context information for debugging

## Error Classes

### Base Error Class

All custom errors inherit from `GraphRAGError`:

```python
from src.utils.errors import GraphRAGError, ErrorContext

# Create error with context
context = ErrorContext().add("file_path", "/path/to/file").add("line", 42)
raise GraphRAGError(
    "Something went wrong",
    context=context.build(),
    cause=original_exception
)
```

### Specific Error Types

#### ParsingError
Raised when document parsing fails.

**Scenarios:**
- Invalid file format (not .docx or .pdf)
- Corrupted file content
- Unsupported document structure
- Missing required metadata

**Example:**
```python
from src.utils.errors import ParsingError, ErrorContext

context = ErrorContext().add("file_path", file_path).add("file_type", "docx")
raise ParsingError(
    f"Corrupted or invalid .docx file: {str(e)}",
    context=context.build(),
    cause=e
)
```

#### StorageError
Raised when storage operations fail.

**Scenarios:**
- Qdrant connection failure
- Neo4j connection failure
- PostgreSQL connection failure
- Write operation timeout
- Insufficient storage space

**Example:**
```python
from src.utils.errors import StorageError

raise StorageError(
    "Failed to connect to Qdrant",
    context={"url": qdrant_url, "timeout": 30}
)
```

#### RetrievalError
Raised when retrieval operations fail.

**Scenarios:**
- Empty query string
- Invalid query parameters
- No results found above threshold
- Graph traversal timeout
- Cypher query syntax error

**Example:**
```python
from src.utils.errors import RetrievalError

raise RetrievalError(
    "Graph traversal timeout",
    context={"query": query, "max_depth": 3, "timeout": 30}
)
```

#### LLMError
Raised when LLM operations fail.

**Scenarios:**
- Ollama service unavailable
- Model not found
- Context too large for model
- Generation timeout
- Rate limiting

**Example:**
```python
from src.utils.errors import LLMError

raise LLMError(
    "Ollama service unavailable",
    context={"base_url": base_url, "model": model}
)
```

#### ValidationError
Raised when validation fails.

**Scenarios:**
- Missing required configuration
- Invalid configuration values
- Malformed API request
- Invalid file format

**Example:**
```python
from src.utils.errors import ValidationError

raise ValidationError(
    "Missing required configuration: OLLAMA_BASE_URL",
    context={"config_key": "OLLAMA_BASE_URL"}
)
```

## Retry Logic with Exponential Backoff

### Basic Usage

Use the `@retry_with_backoff` decorator to automatically retry functions:

```python
from src.utils.errors import retry_with_backoff, RetryConfig

@retry_with_backoff(RetryConfig(max_attempts=3))
def call_external_api():
    # API call that might fail
    response = requests.get("https://api.example.com/data")
    return response.json()
```

### Custom Retry Configuration

```python
from src.utils.errors import retry_with_backoff, RetryConfig

config = RetryConfig(
    max_attempts=5,           # Total attempts (including initial)
    initial_delay=1.0,        # Initial delay in seconds
    max_delay=60.0,           # Maximum delay between retries
    exponential_base=2.0,     # Exponential backoff multiplier
    jitter=True,              # Add random jitter to prevent thundering herd
    retryable_exceptions=(    # Exception types to retry
        ConnectionError,
        TimeoutError,
        StorageError
    )
)

@retry_with_backoff(config)
def my_function():
    # Your code here
    pass
```

### Retry Callback

Execute custom logic on each retry:

```python
def on_retry_callback(exception, attempt_number):
    print(f"Retry attempt {attempt_number} after error: {exception}")

@retry_with_backoff(
    RetryConfig(max_attempts=3),
    on_retry=on_retry_callback
)
def my_function():
    # Your code here
    pass
```

### How It Works

The retry logic uses exponential backoff:

```
delay = min(initial_delay * (exponential_base ** attempt), max_delay)
```

With jitter enabled:
```
delay *= (0.5 + random.random())  # Adds 0-50% variation
```

**Example delays with default config:**
- Attempt 1: Immediate
- Attempt 2: ~1.0s (1.0 * 2^0 with jitter)
- Attempt 3: ~2.0s (1.0 * 2^1 with jitter)

## Graceful Degradation

### Overview

Graceful degradation allows the system to continue operating with reduced functionality when components fail.

### Using Degradation Manager

```python
from src.utils.errors import get_degradation_manager, BM25_DEGRADATION

degradation_manager = get_degradation_manager()

try:
    # Try BM25 search
    results = bm25_index.search(query)
except Exception as e:
    # Enter degraded mode
    degradation_manager.enter_degraded_mode(BM25_DEGRADATION)
    
    # Fall back to vector-only search
    results = vector_only_search(query)
```

### Checking Degradation Status

```python
# Check if any degradation is active
if degradation_manager.is_degraded():
    print("System is in degraded mode")

# Check specific degradation
if degradation_manager.is_degraded("bm25_unavailable"):
    print("BM25 index is unavailable, using vector-only search")

# Get all active degradations
active = degradation_manager.get_active_degradations()
for mode in active:
    print(f"Active: {mode.mode_name} - {mode.description}")
```

### Exiting Degraded Mode

```python
try:
    # Try to use BM25 again
    results = bm25_index.search(query)
    
    # Success! Exit degraded mode
    if degradation_manager.is_degraded("bm25_unavailable"):
        degradation_manager.exit_degraded_mode("bm25_unavailable")
except Exception:
    # Still failing, stay in degraded mode
    pass
```

### Predefined Degradation Modes

#### BM25_DEGRADATION
- **Mode:** `bm25_unavailable`
- **Severity:** WARNING
- **Fallback:** Fall back to vector-only search

#### NEO4J_DEGRADATION
- **Mode:** `neo4j_unavailable`
- **Severity:** ERROR
- **Fallback:** Fall back to vector-only mode (no graph retrieval)

#### CROSS_ENCODER_DEGRADATION
- **Mode:** `cross_encoder_unavailable`
- **Severity:** WARNING
- **Fallback:** Use original retrieval scores without reranking

#### QDRANT_DEGRADATION
- **Mode:** `qdrant_unavailable`
- **Severity:** CRITICAL
- **Fallback:** System cannot perform vector search (critical failure)

#### LLM_DEGRADATION
- **Mode:** `llm_unavailable`
- **Severity:** CRITICAL
- **Fallback:** Cannot generate responses or perform LLM-based operations

### Custom Degradation Modes

```python
from src.utils.errors import DegradationMode, ErrorSeverity

custom_mode = DegradationMode(
    mode_name="custom_service_unavailable",
    description="Custom service is unavailable",
    fallback_behavior="Use cached results",
    severity=ErrorSeverity.WARNING
)

degradation_manager.enter_degraded_mode(custom_mode)
```

## Error Context

### Building Error Context

Use `ErrorContext` to build rich context information:

```python
from src.utils.errors import ErrorContext

context = ErrorContext()
context.add("file_path", "/path/to/file.txt")
context.add("line_number", 42)
context.add("operation", "parse_document")

# Or chain calls
context = (ErrorContext()
    .add("file_path", "/path/to/file.txt")
    .add("line_number", 42)
    .add("operation", "parse_document"))

# Or add multiple at once
context = ErrorContext()
context.add_all({
    "file_path": "/path/to/file.txt",
    "line_number": 42,
    "operation": "parse_document"
})

# Get the context dictionary
context_dict = context.build()
```

## Error Logging

### Basic Error Logging

```python
from src.utils.errors import log_error, ErrorSeverity

try:
    # Some operation
    pass
except Exception as e:
    log_error(
        e,
        severity=ErrorSeverity.ERROR,
        context={"operation": "my_operation"},
        component="MyComponent"
    )
```

### Convenience Function

```python
from src.utils.errors import log_error_with_context

try:
    # Some operation
    pass
except Exception as e:
    log_error_with_context(
        e,
        component="MyComponent",
        operation="my_operation",
        file_path="/path/to/file",
        extra_key="extra_value"
    )
```

### Error Severity Levels

- **DEBUG**: Detailed information for debugging
- **INFO**: Informational messages
- **WARNING**: Warning messages (degraded functionality)
- **ERROR**: Error messages (operation failed)
- **CRITICAL**: Critical errors (system failure)

## Best Practices

### 1. Use Specific Error Types

```python
# Good
raise ParsingError("Failed to parse document", context={"file": path})

# Avoid
raise Exception("Failed to parse document")
```

### 2. Include Rich Context

```python
# Good
context = ErrorContext()
context.add("file_path", file_path)
context.add("file_type", file_type)
context.add("file_size", os.path.getsize(file_path))
raise ParsingError("Corrupted file", context=context.build())

# Avoid
raise ParsingError("Corrupted file")
```

### 3. Chain Exceptions

```python
# Good
try:
    # Some operation
    pass
except ValueError as e:
    raise ParsingError("Parse failed", cause=e) from e

# Avoid
try:
    # Some operation
    pass
except ValueError:
    raise ParsingError("Parse failed")
```

### 4. Use Retry for Transient Failures

```python
# Good - Retry transient failures
@retry_with_backoff(RetryConfig(max_attempts=3))
def call_external_service():
    return requests.get("https://api.example.com")

# Avoid - No retry for network calls
def call_external_service():
    return requests.get("https://api.example.com")
```

### 5. Implement Graceful Degradation

```python
# Good - Graceful degradation
try:
    results = bm25_search(query)
except Exception as e:
    degradation_manager.enter_degraded_mode(BM25_DEGRADATION)
    results = vector_only_search(query)

# Avoid - Hard failure
results = bm25_search(query)  # Crashes if BM25 unavailable
```

### 6. Log Errors with Context

```python
# Good
try:
    process_document(file_path)
except Exception as e:
    log_error_with_context(
        e,
        component="DocumentProcessor",
        operation="process_document",
        file_path=file_path,
        file_size=os.path.getsize(file_path)
    )
    raise

# Avoid
try:
    process_document(file_path)
except Exception as e:
    logger.error(f"Error: {e}")
    raise
```

## Examples

### Complete Example: Document Parser with Error Handling

```python
from src.utils.errors import (
    ParsingError,
    ErrorContext,
    log_error_with_context
)

def parse_document(file_path: str, file_type: str):
    """Parse document with comprehensive error handling."""
    
    # Validate inputs
    if not file_path:
        context = ErrorContext().add("file_path", file_path)
        raise ParsingError("File path cannot be empty", context=context.build())
    
    # Check file exists
    if not os.path.exists(file_path):
        context = ErrorContext().add("file_path", file_path)
        raise ParsingError("File not found", context=context.build())
    
    # Parse with error handling
    try:
        if file_type == "docx":
            return parse_docx(file_path)
        elif file_type == "pdf":
            return parse_pdf(file_path)
        else:
            context = ErrorContext().add("file_type", file_type)
            raise ParsingError("Unsupported file type", context=context.build())
    
    except ParsingError:
        raise
    
    except Exception as e:
        log_error_with_context(
            e,
            component="DocumentParser",
            operation="parse_document",
            file_path=file_path,
            file_type=file_type
        )
        context = ErrorContext().add("file_path", file_path).add("file_type", file_type)
        raise ParsingError(
            f"Failed to parse {file_type} file",
            context=context.build(),
            cause=e
        ) from e
```

### Complete Example: Retrieval with Retry and Degradation

```python
from src.utils.errors import (
    RetrievalError,
    retry_with_backoff,
    RetryConfig,
    get_degradation_manager,
    BM25_DEGRADATION,
    log_error_with_context
)

class VectorRetriever:
    def retrieve(self, query: str, top_k: int = 10):
        """Retrieve with retry and graceful degradation."""
        
        # Vector search with retry
        vector_results = self._vector_search_with_retry(query)
        
        # BM25 search with graceful degradation
        bm25_results = self._bm25_search_with_degradation(query)
        
        # Fuse results
        return self._fuse_results(vector_results, bm25_results, top_k)
    
    @retry_with_backoff(RetryConfig(max_attempts=3))
    def _vector_search_with_retry(self, query: str):
        """Vector search with automatic retry."""
        embedding = self.embedding_generator.generate(query)
        return self.vector_store.search(embedding, top_k=100)
    
    def _bm25_search_with_degradation(self, query: str):
        """BM25 search with graceful degradation."""
        degradation_manager = get_degradation_manager()
        
        try:
            results = self.bm25_index.search(query, top_k=100)
            
            # Exit degraded mode if we were in it
            if degradation_manager.is_degraded("bm25_unavailable"):
                degradation_manager.exit_degraded_mode("bm25_unavailable")
            
            return results
        
        except Exception as e:
            log_error_with_context(
                e,
                component="VectorRetriever",
                operation="_bm25_search",
                query=query
            )
            
            # Enter degraded mode
            if not degradation_manager.is_degraded("bm25_unavailable"):
                degradation_manager.enter_degraded_mode(BM25_DEGRADATION)
            
            # Return empty results (fall back to vector-only)
            return []
```

## Testing Error Handling

### Testing Custom Errors

```python
import pytest
from src.utils.errors import ParsingError, ErrorContext

def test_parsing_error_with_context():
    """Test ParsingError includes context."""
    context = ErrorContext().add("file_path", "/test/file.txt")
    
    with pytest.raises(ParsingError) as exc_info:
        raise ParsingError("Parse failed", context=context.build())
    
    error = exc_info.value
    assert error.context["file_path"] == "/test/file.txt"
```

### Testing Retry Logic

```python
from unittest.mock import Mock
from src.utils.errors import retry_with_backoff, RetryConfig

def test_retry_succeeds_after_failure():
    """Test function succeeds after retry."""
    mock_func = Mock(side_effect=[
        ConnectionError("First attempt"),
        "success"
    ])
    
    @retry_with_backoff(RetryConfig(max_attempts=3, initial_delay=0.01))
    def test_func():
        return mock_func()
    
    result = test_func()
    assert result == "success"
    assert mock_func.call_count == 2
```

### Testing Graceful Degradation

```python
from src.utils.errors import GracefulDegradation, DegradationMode

def test_graceful_degradation():
    """Test entering and exiting degraded mode."""
    manager = GracefulDegradation()
    mode = DegradationMode("test", "Test mode", "Fallback")
    
    # Enter degraded mode
    manager.enter_degraded_mode(mode)
    assert manager.is_degraded("test")
    
    # Exit degraded mode
    manager.exit_degraded_mode("test")
    assert not manager.is_degraded("test")
```

## Troubleshooting

### Issue: Retry not working

**Problem:** Function is not retrying on failure.

**Solution:** Check that the exception type is in `retryable_exceptions`:

```python
config = RetryConfig(
    retryable_exceptions=(
        ConnectionError,
        TimeoutError,
        YourCustomError  # Add your exception type
    )
)
```

### Issue: Degradation mode not exiting

**Problem:** System stays in degraded mode even after recovery.

**Solution:** Explicitly exit degraded mode when operation succeeds:

```python
try:
    result = operation()
    # Success! Exit degraded mode
    if degradation_manager.is_degraded("my_mode"):
        degradation_manager.exit_degraded_mode("my_mode")
    return result
except Exception:
    # Still failing
    pass
```

### Issue: Context not appearing in logs

**Problem:** Error context is not showing up in log output.

**Solution:** Use the `extra` parameter in logging:

```python
logger.error(
    "Error occurred",
    extra={
        "context": error.context,
        "component": "MyComponent"
    }
)
```
