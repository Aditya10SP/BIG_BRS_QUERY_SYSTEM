"""Example usage of GraphRetriever for graph-based retrieval."""

import os
import logging
from src.retrieval.graph_retriever import GraphRetriever, GraphResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Demonstrate GraphRetriever usage with various query patterns."""
    
    # Initialize GraphRetriever with Neo4j connection
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
    
    retriever = GraphRetriever(
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        max_depth=3
    )
    
    try:
        # Example 1: Dependency query
        logger.info("=" * 80)
        logger.info("Example 1: Dependency Query")
        logger.info("=" * 80)
        
        query1 = "What systems depend on NEFT?"
        result1 = retriever.retrieve(query1)
        
        logger.info(f"Query: {query1}")
        logger.info(f"Found {len(result1.nodes)} nodes and {len(result1.relationships)} relationships")
        
        for node in result1.nodes[:5]:  # Show first 5 nodes
            logger.info(f"  Node: {node.node_type} - {node.properties.get('name', 'N/A')}")
        
        for rel in result1.relationships[:5]:  # Show first 5 relationships
            logger.info(f"  Relationship: {rel.source_id} -{rel.rel_type}-> {rel.target_id}")
        
        # Example 2: Integration query
        logger.info("\n" + "=" * 80)
        logger.info("Example 2: Integration Query")
        logger.info("=" * 80)
        
        query2 = "How does NEFT integrate with Core Banking?"
        result2 = retriever.retrieve(query2)
        
        logger.info(f"Query: {query2}")
        logger.info(f"Found {len(result2.nodes)} nodes and {len(result2.relationships)} relationships")
        
        for node in result2.nodes[:5]:
            logger.info(f"  Node: {node.node_type} - {node.properties.get('name', 'N/A')}")
        
        # Example 3: Workflow query
        logger.info("\n" + "=" * 80)
        logger.info("Example 3: Workflow Query")
        logger.info("=" * 80)
        
        query3 = "Show the payment authorization workflow"
        result3 = retriever.retrieve(query3)
        
        logger.info(f"Query: {query3}")
        logger.info(f"Found {len(result3.nodes)} nodes and {len(result3.relationships)} relationships")
        
        # Example 4: Conflict query
        logger.info("\n" + "=" * 80)
        logger.info("Example 4: Conflict Query")
        logger.info("=" * 80)
        
        query4 = "What conflicts exist for NEFT?"
        result4 = retriever.retrieve(query4)
        
        logger.info(f"Query: {query4}")
        logger.info(f"Found {len(result4.nodes)} nodes and {len(result4.relationships)} relationships")
        
        for rel in result4.relationships:
            if rel.rel_type == "CONFLICTS_WITH":
                logger.info(f"  Conflict: {rel.source_id} <-> {rel.target_id}")
                logger.info(f"    Properties: {rel.properties}")
        
        # Example 5: Comparison query
        logger.info("\n" + "=" * 80)
        logger.info("Example 5: Comparison Query")
        logger.info("=" * 80)
        
        query5 = "Compare NEFT and RTGS"
        result5 = retriever.retrieve(query5)
        
        logger.info(f"Query: {query5}")
        logger.info(f"Found {len(result5.nodes)} nodes and {len(result5.relationships)} relationships")
        
        # Example 6: Custom depth limit
        logger.info("\n" + "=" * 80)
        logger.info("Example 6: Custom Depth Limit")
        logger.info("=" * 80)
        
        query6 = "What are the dependencies of NEFT?"
        result6 = retriever.retrieve(query6, max_depth=2)
        
        logger.info(f"Query: {query6} (max_depth=2)")
        logger.info(f"Found {len(result6.nodes)} nodes and {len(result6.relationships)} relationships")
        
        # Example 7: Associated chunks
        logger.info("\n" + "=" * 80)
        logger.info("Example 7: Associated Chunks")
        logger.info("=" * 80)
        
        query7 = "What is NEFT?"
        result7 = retriever.retrieve(query7)
        
        logger.info(f"Query: {query7}")
        logger.info(f"Found {len(result7.chunks)} associated chunks")
        
        for chunk in result7.chunks[:3]:  # Show first 3 chunks
            logger.info(f"  Chunk: {chunk['chunk_id']}")
            logger.info(f"    Breadcrumbs: {chunk['breadcrumbs']}")
            logger.info(f"    Text preview: {chunk['text'][:100]}...")
        
    finally:
        # Close connection
        retriever.close()
        logger.info("\nGraphRetriever connection closed")


if __name__ == "__main__":
    main()
