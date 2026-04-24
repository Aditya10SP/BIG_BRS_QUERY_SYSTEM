"""Celery application for async ingestion tasks"""

import os
from celery import Celery

from src.utils.logging import setup_logging, get_logger


# Set up logging
log_level = os.getenv("LOG_LEVEL", "INFO")
structured = os.getenv("STRUCTURED_LOGGING", "true").lower() == "true"
setup_logging(level=log_level, structured=structured)

logger = get_logger(__name__)

# Create Celery app
celery_app = Celery(
    "graph_rag_worker",
    broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0"),
    include=["src.worker.tasks"]
)

# Configure Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max per task
    task_soft_time_limit=3300,  # 55 minutes soft limit
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=50,
)

logger.info("Celery app initialized", extra={"broker": celery_app.conf.broker_url})
