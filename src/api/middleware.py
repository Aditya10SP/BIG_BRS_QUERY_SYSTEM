"""FastAPI middleware for monitoring and observability"""

import time
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from src.utils.logging import get_logger, set_request_id, get_request_id
from src.utils.metrics import (
    http_requests_total,
    http_request_duration_seconds,
    increment_counter,
    record_histogram
)


logger = get_logger(__name__)


class RequestTrackingMiddleware(BaseHTTPMiddleware):
    """Middleware for request tracking and metrics collection"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and collect metrics.
        
        Args:
            request: FastAPI request
            call_next: Next middleware/handler
            
        Returns:
            Response from handler
        """
        # Generate and set request ID
        request_id = request.headers.get("X-Request-ID")
        request_id = set_request_id(request_id)
        
        # Start timer
        start_time = time.time()
        
        # Log request
        logger.info(
            "Request started",
            extra={
                "method": request.method,
                "path": request.url.path,
                "client": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent")
            }
        )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Record metrics
            increment_counter(
                http_requests_total,
                labels={
                    "method": request.method,
                    "endpoint": request.url.path,
                    "status": str(response.status_code)
                }
            )
            
            record_histogram(
                http_request_duration_seconds,
                duration,
                labels={
                    "method": request.method,
                    "endpoint": request.url.path
                }
            )
            
            # Log response
            logger.info(
                "Request completed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_seconds": duration
                }
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            # Calculate duration
            duration = time.time() - start_time
            
            # Record error metrics
            increment_counter(
                http_requests_total,
                labels={
                    "method": request.method,
                    "endpoint": request.url.path,
                    "status": "500"
                }
            )
            
            record_histogram(
                http_request_duration_seconds,
                duration,
                labels={
                    "method": request.method,
                    "endpoint": request.url.path
                }
            )
            
            # Log error
            logger.error(
                "Request failed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "duration_seconds": duration,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                },
                exc_info=True
            )
            
            # Re-raise exception
            raise


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting application metrics"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Collect application-level metrics.
        
        Args:
            request: FastAPI request
            call_next: Next middleware/handler
            
        Returns:
            Response from handler
        """
        # Skip metrics collection for metrics endpoint
        if request.url.path == "/metrics":
            return await call_next(request)
        
        # Process request
        response = await call_next(request)
        
        # TODO: Add custom application metrics here
        # For example:
        # - Active user sessions
        # - Cache hit rates
        # - Queue depths
        
        return response
