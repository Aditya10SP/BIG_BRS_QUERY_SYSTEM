"""Example usage of IngestionPipeline for document ingestion."""

import logging
from pathlib import Path

from config.system_config import SystemConfig
from src.parsing.document_parser import DocumentParser
from src.chunking.hierarchical_chunker import HierarchicalChunker
from src.embedding.embedding_generator import EmbeddingGenerator
from src.storage.database_manager import DatabaseManager
from src.storage.vector_store import VectorStore
from src.indexing.bm25_indexer import BM25Indexer
from src.extraction.entity_extractor import EntityExtractor
from src.extraction.entity_resolver import EntityResolver
from src.extraction.conflict_detector import ConflictDetector
from src.storage.graph_populator import GraphPopulator
from src.pipeline.ingestion_pipeline import IngestionPipeline, IngestionStatus


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate IngestionPipeline usage."""
    
    # Load configuration
    config = SystemConfig.from_env()
    
    logger.info("Initializing components...")
    
    # Initialize all components
    parser = DocumentParser()
    chunker = HierarchicalChunker(
        parent_size=config.parent_chunk_size,
        child_size=config.child_chunk_size,
        overlap=config.chunk_overlap
    )
    embedding_generator = EmbeddingGenerator(model_name=config.embedding_model)
    
    # Initialize storage components
    database_manager = DatabaseManager(config.postgres_connection_string)
    database_manager.initialize()
    
    vector_store = VectorStore(
        url=config.qdrant_url,
        collection_name="banking_docs",
        vector_size=config.embedding_dimension
    )
    
    bm25_indexer = BM25Indexer()
    
    # Initialize extraction components
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
    
    # Create schema
    graph_populator.create_schema()
    
    # Initialize pipeline
    logger.info("Initializing IngestionPipeline...")
    pipeline = IngestionPipeline(
        parser=parser,
        chunker=chunker,
        embedding_generator=embedding_generator,
        database_manager=database_manager,
        vector_store=vector_store,
        bm25_indexer=bm25_indexer,
        entity_extractor=entity_extractor,
        entity_resolver=entity_resolver,
        conflict_detector=conflict_detector,
        graph_populator=graph_populator
    )
    
    # Example 1: Ingest a single document
    logger.info("\n=== Example 1: Single Document Ingestion ===")
    
    # Replace with actual document path
    doc_path = "path/to/your/document.docx"
    
    if Path(doc_path).exists():
        result = pipeline.ingest(doc_path, "docx")
        
        logger.info(f"Ingestion Status: {result.status.value}")
        logger.info(f"Document ID: {result.doc_id}")
        logger.info(f"Number of chunks: {result.num_chunks}")
        logger.info(f"Number of entities: {result.num_entities}")
        logger.info(f"Number of relationships: {result.num_relationships}")
        
        if result.status == IngestionStatus.COMPLETED:
            logger.info("✓ Ingestion completed successfully!")
            logger.info(f"Metadata: {result.metadata}")
        else:
            logger.error(f"✗ Ingestion failed: {result.error}")
    else:
        logger.warning(f"Document not found: {doc_path}")
    
    # Example 2: Batch ingestion
    logger.info("\n=== Example 2: Batch Document Ingestion ===")
    
    documents = [
        {"file_path": "doc1.docx", "file_type": "docx", "doc_id": "doc1"},
        {"file_path": "doc2.pdf", "file_type": "pdf", "doc_id": "doc2"},
        {"file_path": "doc3.docx", "file_type": "docx", "doc_id": "doc3"}
    ]
    
    # Filter to only existing documents
    existing_docs = [
        doc for doc in documents if Path(doc["file_path"]).exists()
    ]
    
    if existing_docs:
        results = pipeline.ingest_batch(existing_docs)
        
        # Summary
        successful = sum(1 for r in results if r.status == IngestionStatus.COMPLETED)
        failed = sum(1 for r in results if r.status == IngestionStatus.FAILED)
        
        logger.info(f"\nBatch Ingestion Summary:")
        logger.info(f"  Total: {len(results)}")
        logger.info(f"  Successful: {successful}")
        logger.info(f"  Failed: {failed}")
        
        # Details for each document
        for result in results:
            logger.info(f"\n  {result.doc_id}:")
            logger.info(f"    Status: {result.status.value}")
            logger.info(f"    Chunks: {result.num_chunks}")
            logger.info(f"    Entities: {result.num_entities}")
            if result.error:
                logger.info(f"    Error: {result.error}")
    else:
        logger.warning("No documents found for batch ingestion")
    
    # Example 3: Check ingestion status
    logger.info("\n=== Example 3: Check Ingestion Status ===")
    
    doc_id = "doc1"
    status = pipeline.get_status(doc_id)
    
    if status:
        logger.info(f"Status for {doc_id}:")
        logger.info(f"  Current status: {status.status.value}")
        logger.info(f"  Message: {status.message}")
        logger.info(f"  Started at: {status.started_at}")
        logger.info(f"  Completed at: {status.completed_at}")
    else:
        logger.info(f"No status found for {doc_id}")
    
    # Cleanup
    logger.info("\n=== Cleanup ===")
    database_manager.close()
    vector_store.close()
    graph_populator.close()
    
    logger.info("Example completed!")


if __name__ == "__main__":
    main()
