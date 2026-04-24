"""FastAPI application entry point"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from config.system_config import SystemConfig
from src.utils.logging import setup_logging, get_logger
from src.api.routes import router as api_router
from src.api.middleware import RequestTrackingMiddleware, MetricsMiddleware


# Initialize logger
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Graph RAG Layer application")
    
    try:
        # Load and validate configuration
        config = SystemConfig.from_env()
        app.state.config = config
        logger.info("Configuration loaded and validated successfully")
        
        # Initialize ingestion jobs storage
        app.state.ingestion_jobs = {}
        
        # Initialize pipeline components
        logger.info("Initializing pipeline components...")
        
        try:
            from src.pipeline.query_pipeline import QueryPipeline
            from src.pipeline.ingestion_pipeline import IngestionPipeline
            from src.query.query_router import QueryRouter
            from src.retrieval.vector_retriever import VectorRetriever
            from src.retrieval.graph_retriever import GraphRetriever
            from src.retrieval.result_fusion import ResultFusion
            from src.retrieval.cross_encoder_reranker import CrossEncoderReranker
            from src.retrieval.context_assembler import ContextAssembler
            from src.query.llm_generator import LLMGenerator
            from src.query.faithfulness_validator import FaithfulnessValidator
            from src.embedding.embedding_generator import EmbeddingGenerator
            from src.indexing.bm25_indexer import BM25Indexer
            from src.storage.vector_store import VectorStore
            from src.storage.database_manager import DatabaseManager
            from src.storage.graph_populator import GraphPopulator
            from src.parsing.document_parser import DocumentParser
            from src.chunking.hierarchical_chunker import HierarchicalChunker
            from src.extraction.entity_extractor import EntityExtractor
            from src.extraction.entity_resolver import EntityResolver
            from src.extraction.conflict_detector import ConflictDetector
            
            # Initialize storage components
            logger.info("Initializing storage components...")
            embedding_generator = EmbeddingGenerator()
            vector_store = VectorStore(url=config.qdrant_url)
            doc_store = DatabaseManager(connection_string=config.postgres_connection_string)
            doc_store.initialize()
            
            # Initialize BM25 index
            bm25_index = BM25Indexer()
            
            # Initialize query pipeline components
            logger.info("Initializing query pipeline...")
            query_router = QueryRouter(
                ollama_base_url=config.ollama_base_url,
                llm_model=config.llm_model
            )
            
            vector_retriever = VectorRetriever(
                vector_store=vector_store,
                bm25_index=bm25_index,
                embedding_generator=embedding_generator,
                doc_store=doc_store
            )
            
            graph_retriever = GraphRetriever(
                neo4j_uri=config.neo4j_uri,
                neo4j_user=config.neo4j_user,
                neo4j_password=config.neo4j_password
            )
            
            result_fusion = ResultFusion()
            reranker = CrossEncoderReranker()
            context_assembler = ContextAssembler(max_tokens=config.max_context_tokens)
            
            llm_generator = LLMGenerator(
                base_url=config.ollama_base_url,
                model=config.llm_model
            )
            
            faithfulness_validator = FaithfulnessValidator(
                base_url=config.ollama_base_url,
                model=config.llm_model
            )
            
            # Create query pipeline
            app.state.query_pipeline = QueryPipeline(
                query_router=query_router,
                vector_retriever=vector_retriever,
                graph_retriever=graph_retriever,
                result_fusion=result_fusion,
                reranker=reranker,
                context_assembler=context_assembler,
                llm_generator=llm_generator,
                faithfulness_validator=faithfulness_validator
            )
            
            # Initialize ingestion pipeline components
            logger.info("Initializing ingestion pipeline...")
            parser = DocumentParser()
            chunker = HierarchicalChunker(
                parent_size=config.parent_chunk_size,
                child_size=config.child_chunk_size
            )
            
            entity_extractor = EntityExtractor(
                ollama_base_url=config.ollama_base_url,
                llm_model=config.llm_model
            )
            
            entity_resolver = EntityResolver(
                similarity_threshold=config.entity_similarity_threshold
            )
            
            conflict_detector = ConflictDetector(
                ollama_base_url=config.ollama_base_url,
                llm_model=config.llm_model
            )
            
            graph_populator = GraphPopulator(
                neo4j_uri=config.neo4j_uri,
                neo4j_user=config.neo4j_user,
                neo4j_password=config.neo4j_password
            )
            
            # Create ingestion pipeline
            app.state.ingestion_pipeline = IngestionPipeline(
                parser=parser,
                chunker=chunker,
                embedding_generator=embedding_generator,
                database_manager=doc_store,
                vector_store=vector_store,
                bm25_indexer=bm25_index,
                entity_extractor=entity_extractor,
                entity_resolver=entity_resolver,
                conflict_detector=conflict_detector,
                graph_populator=graph_populator
            )
            
            logger.info("✅ Pipeline components initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipelines: {e}")
            logger.warning(
                "Pipeline components not initialized. "
                "Query and ingestion endpoints will fail until pipelines are configured."
            )
        
        yield
        
    except Exception as e:
        logger.error(
            "Failed to start application",
            extra={
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
        )
        raise
    finally:
        # Shutdown
        logger.info("Shutting down Graph RAG Layer application")
        
        # Close storage connections
        try:
            if hasattr(app.state, 'query_pipeline'):
                if hasattr(app.state.query_pipeline, 'graph_retriever'):
                    app.state.query_pipeline.graph_retriever.close()
                    logger.info("Closed graph retriever connection")
            
            if hasattr(app.state, 'ingestion_pipeline'):
                if hasattr(app.state.ingestion_pipeline, 'graph_populator'):
                    app.state.ingestion_pipeline.graph_populator.close()
                    logger.info("Closed graph populator connection")
                
                if hasattr(app.state.ingestion_pipeline, 'database_manager'):
                    app.state.ingestion_pipeline.database_manager.close()
                    logger.info("Closed database connection")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# Create FastAPI application
app = FastAPI(
    title="Graph RAG Layer",
    description="Hybrid retrieval system for banking documents combining vector search and knowledge graphs",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add monitoring middleware
app.add_middleware(RequestTrackingMiddleware)
app.add_middleware(MetricsMiddleware)

# Include API routes
app.include_router(api_router, prefix="/api/v1", tags=["api"])

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Graph RAG Layer",
        "version": "0.1.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint with detailed service status.
    
    Returns:
        dict: Health status of the application and its dependencies
    """
    health_status = {
        "status": "healthy",
        "version": "0.1.0",
        "services": {
            "api": "up"
        }
    }
    
    # TODO: Add actual health checks for storage services
    # For now, return basic status
    
    try:
        # Check if config is loaded
        if hasattr(app.state, "config"):
            health_status["config"] = "loaded"
        else:
            health_status["config"] = "not_loaded"
            health_status["status"] = "degraded"
        
        # TODO: Check Qdrant connection
        # health_status["services"]["qdrant"] = "up" or "down"
        
        # TODO: Check Neo4j connection
        # health_status["services"]["neo4j"] = "up" or "down"
        
        # TODO: Check PostgreSQL connection
        # health_status["services"]["postgres"] = "up" or "down"
        
        # TODO: Check Redis connection (for worker)
        # health_status["services"]["redis"] = "up" or "down"
        
    except Exception as e:
        logger.error(
            "Health check failed",
            extra={
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
        )
        health_status["status"] = "unhealthy"
        health_status["error"] = str(e)
    
    return health_status


@app.get("/health/live")
async def liveness_check():
    """
    Kubernetes liveness probe endpoint.
    
    Returns:
        dict: Simple liveness status
    """
    return {"status": "alive"}


@app.get("/health/ready")
async def readiness_check():
    """
    Kubernetes readiness probe endpoint.
    
    Returns:
        dict: Readiness status based on dependencies
    """
    # TODO: Check if all required services are available
    # For now, return ready if config is loaded
    
    if hasattr(app.state, "config"):
        return {"status": "ready"}
    else:
        return {"status": "not_ready", "reason": "configuration not loaded"}


if __name__ == "__main__":
    import uvicorn
    
    # Set up logging
    log_level = os.getenv("LOG_LEVEL", "INFO")
    structured = os.getenv("STRUCTURED_LOGGING", "true").lower() == "true"
    log_file = os.getenv("LOG_FILE", None)
    
    setup_logging(level=log_level, structured=structured, log_file=log_file)
    
    # Run application
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_config=None  # Use our custom logging
    )
