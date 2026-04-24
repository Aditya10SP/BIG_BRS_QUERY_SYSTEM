"""Prometheus metrics for monitoring"""

from prometheus_client import Counter, Histogram, Gauge, Info
import time
from functools import wraps
from typing import Callable, Any

# Application info
app_info = Info("graph_rag_app", "Graph RAG Layer application information")
app_info.info({
    "version": "0.1.0",
    "service": "graph-rag-layer"
})

# Request metrics
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
)

# Query pipeline metrics
query_pipeline_requests_total = Counter(
    "query_pipeline_requests_total",
    "Total query pipeline requests",
    ["mode", "status"]
)

query_pipeline_duration_seconds = Histogram(
    "query_pipeline_duration_seconds",
    "Query pipeline duration in seconds",
    ["mode", "step"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
)

query_pipeline_step_duration_seconds = Histogram(
    "query_pipeline_step_duration_seconds",
    "Individual query pipeline step duration",
    ["step"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
)

# Ingestion pipeline metrics
ingestion_pipeline_requests_total = Counter(
    "ingestion_pipeline_requests_total",
    "Total ingestion pipeline requests",
    ["status"]
)

ingestion_pipeline_duration_seconds = Histogram(
    "ingestion_pipeline_duration_seconds",
    "Ingestion pipeline duration in seconds",
    ["step"],
    buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0)
)

documents_ingested_total = Counter(
    "documents_ingested_total",
    "Total documents ingested",
    ["file_type"]
)

chunks_created_total = Counter(
    "chunks_created_total",
    "Total chunks created",
    ["chunk_type"]
)

entities_extracted_total = Counter(
    "entities_extracted_total",
    "Total entities extracted",
    ["entity_type"]
)

# Retrieval metrics
vector_search_duration_seconds = Histogram(
    "vector_search_duration_seconds",
    "Vector search duration in seconds",
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0)
)

graph_search_duration_seconds = Histogram(
    "graph_search_duration_seconds",
    "Graph search duration in seconds",
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0)
)

retrieved_chunks_total = Counter(
    "retrieved_chunks_total",
    "Total chunks retrieved",
    ["source"]
)

# LLM metrics
llm_requests_total = Counter(
    "llm_requests_total",
    "Total LLM requests",
    ["model", "status"]
)

llm_request_duration_seconds = Histogram(
    "llm_request_duration_seconds",
    "LLM request duration in seconds",
    ["model"],
    buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0)
)

llm_tokens_used_total = Counter(
    "llm_tokens_used_total",
    "Total tokens used in LLM requests",
    ["model", "type"]  # type: prompt or completion
)

# Faithfulness metrics
faithfulness_score = Histogram(
    "faithfulness_score",
    "Faithfulness validation scores",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
)

low_faithfulness_warnings_total = Counter(
    "low_faithfulness_warnings_total",
    "Total low faithfulness warnings"
)

# Storage metrics
storage_operations_total = Counter(
    "storage_operations_total",
    "Total storage operations",
    ["storage", "operation", "status"]
)

storage_operation_duration_seconds = Histogram(
    "storage_operation_duration_seconds",
    "Storage operation duration in seconds",
    ["storage", "operation"],
    buckets=(0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0)
)

# Error metrics
errors_total = Counter(
    "errors_total",
    "Total errors",
    ["component", "error_type"]
)

# System metrics
active_connections = Gauge(
    "active_connections",
    "Number of active connections",
    ["service"]
)

# Celery worker metrics
celery_tasks_total = Counter(
    "celery_tasks_total",
    "Total Celery tasks",
    ["task_name", "status"]
)

celery_task_duration_seconds = Histogram(
    "celery_task_duration_seconds",
    "Celery task duration in seconds",
    ["task_name"],
    buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0, 3600.0)
)


def track_time(metric: Histogram, labels: dict = None):
    """
    Decorator to track execution time of a function.
    
    Args:
        metric: Prometheus Histogram metric to record to
        labels: Optional labels to add to the metric
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                if labels:
                    metric.labels(**labels).observe(duration)
                else:
                    metric.observe(duration)
        return wrapper
    return decorator


def track_async_time(metric: Histogram, labels: dict = None):
    """
    Decorator to track execution time of an async function.
    
    Args:
        metric: Prometheus Histogram metric to record to
        labels: Optional labels to add to the metric
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                if labels:
                    metric.labels(**labels).observe(duration)
                else:
                    metric.observe(duration)
        return wrapper
    return decorator


def increment_counter(metric: Counter, labels: dict = None):
    """
    Increment a counter metric.
    
    Args:
        metric: Prometheus Counter metric to increment
        labels: Optional labels to add to the metric
    """
    if labels:
        metric.labels(**labels).inc()
    else:
        metric.inc()


def record_histogram(metric: Histogram, value: float, labels: dict = None):
    """
    Record a value to a histogram metric.
    
    Args:
        metric: Prometheus Histogram metric to record to
        value: Value to record
        labels: Optional labels to add to the metric
    """
    if labels:
        metric.labels(**labels).observe(value)
    else:
        metric.observe(value)


def set_gauge(metric: Gauge, value: float, labels: dict = None):
    """
    Set a gauge metric value.
    
    Args:
        metric: Prometheus Gauge metric to set
        value: Value to set
        labels: Optional labels to add to the metric
    """
    if labels:
        metric.labels(**labels).set(value)
    else:
        metric.set(value)
