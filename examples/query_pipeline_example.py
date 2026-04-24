"""Example usage of QueryPipeline for end-to-end query processing."""

import os
import logging
from dotenv import load_dotenv

from config.system_config import SystemConfig
from src.query.query_router import QueryRouter
from src.retrieval.vector_retriever import VectorRetriever
from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.result_fusion import ResultFusion
from src.retrieval.cross_encoder_reranker import CrossEncoderReranker
from src.retrieval.context_assembler import ContextAssembler
from src.query.llm_generator import LLMGenerator
from src.query.faithfulness_validator import FaithfulnessValidator
from src.pipeline.query_pipeline import QueryPipeline
from src.storage.vector_store import VectorStore
from src.indexing.bm25_indexer import BM25Indexer
from src.embedding.embedding_generator import EmbeddingGenerator
from src.storage.database_manager import DatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate QueryPipeline usage."""
    
    # Load environment variables
    load_dotenv()
    
    # Initialize system configuration
    config = SystemConfig(
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        llm_model=os.getenv("LLM_MODEL", "llama2"),
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "password"),
        postgres_connection_string=os.getenv(
            "POSTGRES_CONNECTION_STRING",
            "postgresql://user:password@localhost:5432/graph_rag"
        )
    )
    
    logger.info("Initializing QueryPipeline components...")
    
    # Initialize storage components
    vector_store = VectorStore(
        url=config.qdrant_url,
        collection_name="banking_docs",
        embedding_dim=config.embedding_dimension
    )
    
    embedding_generator = EmbeddingGenerator(model_name=config.embedding_model)
    
    bm25_index = BM25Indexer()
    
    doc_store = DatabaseManager(connection_string=config.postgres_connection_string)
    
    # Initialize query components
    query_router = QueryRouter(
        ollama_base_url=config.ollama_base_url,
        llm_model=config.llm_model,
        confidence_threshold=0.7
    )
    
    vector_retriever = VectorRetriever(
        vector_store=vector_store,
        bm25_index=bm25_index,
        embedding_generator=embedding_generator,
        doc_store=doc_store,
        similarity_threshold=config.similarity_threshold
    )
    
    graph_retriever = GraphRetriever(
        neo4j_uri=config.neo4j_uri,
        neo4j_user=config.neo4j_user,
        neo4j_password=config.neo4j_password,
        max_depth=config.max_graph_depth
    )
    
    result_fusion = ResultFusion(
        vector_weight=0.6,
        graph_weight=0.4
    )
    
    reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    
    context_assembler = ContextAssembler(
        max_tokens=config.max_context_tokens
    )
    
    llm_generator = LLMGenerator(
        base_url=config.ollama_base_url,
        model=config.llm_model
    )
    
    faithfulness_validator = FaithfulnessValidator(
        base_url=config.ollama_base_url,
        model=config.llm_model,
        faithfulness_threshold=config.faithfulness_threshold
    )
    
    # Initialize QueryPipeline
    pipeline = QueryPipeline(
        query_router=query_router,
        vector_retriever=vector_retriever,
        graph_retriever=graph_retriever,
        result_fusion=result_fusion,
        reranker=reranker,
        context_assembler=context_assembler,
        llm_generator=llm_generator,
        faithfulness_validator=faithfulness_validator
    )
    
    logger.info("QueryPipeline initialized successfully")
    
    # Example queries
    queries = [
        "What is NEFT?",  # Factual query (VECTOR mode)
        "What systems depend on NEFT?",  # Relational query (GRAPH mode)
        "How does NEFT integrate with Core Banking and what are the transaction limits?"  # Complex query (HYBRID mode)
    ]
    
    for query_text in queries:
        logger.info(f"\n{'='*80}")
        logger.info(f"Query: {query_text}")
        logger.info(f"{'='*80}")
        
        # Execute query
        response = pipeline.query(
            query_text=query_text,
            top_k=10,
            rerank_top_k=5
        )
        
        # Display results
        if response.error:
            logger.error(f"Query failed: {response.error}")
        else:
            logger.info(f"\nRetrieval Mode: {response.retrieval_mode}")
            logger.info(f"Faithfulness Score: {response.faithfulness_score:.2f}")
            
            logger.info(f"\nAnswer:\n{response.answer}")
            
            logger.info(f"\nCitations ({len(response.citations)}):")
            for citation_id, citation in response.citations.items():
                logger.info(f"  [{citation_id}] {citation['breadcrumbs']}")
            
            if response.warnings:
                logger.warning(f"\nWarnings:")
                for warning in response.warnings:
                    logger.warning(f"  - {warning}")
            
            # Display metrics
            if response.metrics:
                logger.info(f"\nQuery Metrics:")
                logger.info(f"  Total Time: {response.metrics.total_time:.2f}s")
                logger.info(f"  Routing Time: {response.metrics.routing_time:.3f}s")
                logger.info(f"  Retrieval Time: {response.metrics.retrieval_time:.3f}s")
                logger.info(f"  Fusion Time: {response.metrics.fusion_time:.3f}s")
                logger.info(f"  Reranking Time: {response.metrics.reranking_time:.3f}s")
                logger.info(f"  Assembly Time: {response.metrics.assembly_time:.3f}s")
                logger.info(f"  Generation Time: {response.metrics.generation_time:.3f}s")
                logger.info(f"  Validation Time: {response.metrics.validation_time:.3f}s")
                logger.info(f"  Chunks Retrieved: {response.metrics.chunks_retrieved}")
                logger.info(f"  Chunks Reranked: {response.metrics.chunks_reranked}")
                logger.info(f"  Context Tokens: {response.metrics.context_tokens}")
    
    # Cleanup
    graph_retriever.close()
    logger.info("\nQueryPipeline example completed")


if __name__ == "__main__":
    main()
