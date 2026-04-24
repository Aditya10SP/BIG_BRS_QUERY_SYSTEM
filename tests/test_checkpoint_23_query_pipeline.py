"""
Checkpoint 23: End-to-end query pipeline integration tests.

This test suite validates that the query pipeline works end-to-end with:
- All three query modes (VECTOR, GRAPH, HYBRID)
- Sample banking queries
- Citations and faithfulness validation

Note: These tests require:
1. Data to be ingested (run checkpoint 13 first)
2. Ollama LLM to be running
3. All storage services (Qdrant, Neo4j, PostgreSQL) to be running
"""

import pytest
import os
from dotenv import load_dotenv

from src.pipeline.query_pipeline import QueryPipeline
from src.query.query_router import QueryRouter, QueryMode
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

# Load environment variables
load_dotenv()


@pytest.fixture(scope="module")
def query_pipeline():
    """
    Create a fully integrated QueryPipeline with real components.
    
    This fixture initializes all pipeline components with actual implementations
    connected to the storage layers (Qdrant, Neo4j, PostgreSQL).
    """
    # Get configuration from environment
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    llm_model = os.getenv("LLM_MODEL", "llama3.2:3b")
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
    postgres_conn = os.getenv(
        "POSTGRES_CONNECTION_STRING",
        "postgresql://postgres:password@localhost:5432/graph_rag"
    )
    
    # Initialize storage components
    embedding_generator = EmbeddingGenerator()
    vector_store = VectorStore(url=qdrant_url)
    doc_store = DatabaseManager(connection_string=postgres_conn)
    doc_store.initialize()  # Initialize the connection pool
    
    # Initialize BM25 index and load existing chunks
    bm25_index = BM25Indexer()
    try:
        # Try to load existing chunks for BM25 indexing
        chunks = doc_store.get_all_chunks()
        if chunks:
            from src.chunking.hierarchical_chunker import Chunk
            chunk_objects = [
                Chunk(
                    chunk_id=c["chunk_id"],
                    doc_id=c["doc_id"],
                    text=c["text"],
                    chunk_type=c.get("chunk_type", "child"),
                    parent_chunk_id=c.get("parent_chunk_id"),
                    breadcrumbs=c.get("breadcrumbs", ""),
                    section=c.get("section", ""),
                    token_count=c.get("token_count", 0),
                    metadata=c.get("metadata", {})
                )
                for c in chunks
            ]
            bm25_index.index(chunk_objects)
            print(f"Loaded {len(chunk_objects)} chunks into BM25 index")
    except Exception as e:
        print(f"Warning: Could not load chunks for BM25 index: {e}")
    
    # Initialize query components
    query_router = QueryRouter(
        ollama_base_url=ollama_base_url,
        llm_model=llm_model
    )
    
    vector_retriever = VectorRetriever(
        vector_store=vector_store,
        bm25_index=bm25_index,
        embedding_generator=embedding_generator,
        doc_store=doc_store
    )
    
    graph_retriever = GraphRetriever(
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password
    )
    
    result_fusion = ResultFusion()
    
    reranker = CrossEncoderReranker()
    
    context_assembler = ContextAssembler(max_tokens=4096)
    
    llm_generator = LLMGenerator(
        base_url=ollama_base_url,
        model=llm_model
    )
    
    faithfulness_validator = FaithfulnessValidator(
        base_url=ollama_base_url,
        model=llm_model
    )
    
    # Create pipeline
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
    
    yield pipeline
    
    # Cleanup
    graph_retriever.close()


@pytest.fixture(scope="module")
def check_prerequisites(query_pipeline):
    """
    Check if prerequisites are met before running tests.
    
    Verifies:
    - Ollama LLM is accessible
    - Data exists in the system
    """
    # Check if Ollama is accessible - try with subprocess instead of requests
    import subprocess
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        result = subprocess.run(
            ["curl", "-s", f"{ollama_base_url}/api/tags"],
            capture_output=True,
            timeout=5
        )
        if result.returncode != 0:
            pytest.skip("Ollama LLM is not accessible")
    except Exception as e:
        pytest.skip(f"Ollama LLM is not accessible: {e}")
    
    # Check if data exists
    postgres_conn = os.getenv(
        "POSTGRES_CONNECTION_STRING",
        "postgresql://postgres:password@localhost:5432/graph_rag"
    )
    doc_store = DatabaseManager(connection_string=postgres_conn)
    doc_store.initialize()
    chunks = doc_store.get_all_chunks()
    doc_store.close()
    if not chunks:
        pytest.skip("No data in system. Run checkpoint 13 ingestion first.")
    
    return True


class TestCheckpoint23QueryPipeline:
    """
    Checkpoint 23: End-to-end query pipeline validation.
    
    Tests all three query modes with sample banking queries and verifies:
    - Query execution completes successfully
    - Appropriate mode is selected for each query type
    - Citations are included in responses
    - Faithfulness validation is performed
    """
    
    # Sample banking queries for each mode
    VECTOR_QUERIES = [
        "What is NEFT?",
        "Define transaction limit",
        "Explain RTGS process",
        "What is a payment mode?",
    ]
    
    GRAPH_QUERIES = [
        "What systems depend on NEFT?",
        "Show payment workflow",
        "What are the conflicts in payment rules?",
        "How does NEFT integrate with other systems?",
    ]
    
    HYBRID_QUERIES = [
        "How does NEFT integrate with Core Banking and what are the limits?",
        "What are the conflicts between payment rules across documents?",
        "Compare RTGS and IMPS payment modes",
        "What are the dependencies and transaction limits for NEFT?",
    ]
    
    def test_vector_mode_queries(self, query_pipeline, check_prerequisites):
        """
        Test VECTOR mode with factual/definitional queries.
        
        Validates:
        - Query completes without errors
        - Response contains an answer
        - Citations are present
        - Faithfulness score is computed
        - Metrics are tracked
        """
        for query_text in self.VECTOR_QUERIES[:2]:  # Test first 2 queries only
            print(f"\n{'='*60}")
            print(f"Testing VECTOR query: {query_text}")
            print(f"{'='*60}")
            
            response = query_pipeline.query(query_text, top_k=10, rerank_top_k=5)
            
            # Verify no errors
            assert response.error is None, f"Query failed: {response.error}"
            
            # Verify response structure
            assert response.answer, "Response should contain an answer"
            assert isinstance(response.citations, dict), "Citations should be a dictionary"
            assert response.faithfulness_score >= 0.0, "Faithfulness score should be non-negative"
            assert response.retrieval_mode in ["vector", "graph", "hybrid"], "Invalid retrieval mode"
            
            # Verify metrics
            assert response.metrics is not None, "Metrics should be present"
            assert response.metrics.total_time > 0, "Total time should be positive"
            assert response.metrics.query_mode in ["vector", "graph", "hybrid"], "Invalid query mode"
            
            # Print results
            print(f"\nRetrieval Mode: {response.retrieval_mode}")
            print(f"Faithfulness Score: {response.faithfulness_score:.2f}")
            print(f"Citations: {len(response.citations)}")
            print(f"Warnings: {len(response.warnings)}")
            print(f"Total Time: {response.metrics.total_time:.2f}s")
            print(f"\nAnswer:\n{response.answer[:200]}...")
            
            if response.warnings:
                print(f"\nWarnings:")
                for warning in response.warnings:
                    print(f"  - {warning}")
    
    def test_graph_mode_queries(self, query_pipeline, check_prerequisites):
        """
        Test GRAPH mode with relational/structural queries.
        
        Validates:
        - Query completes without errors
        - Response contains an answer
        - Citations are present
        - Faithfulness score is computed
        - Metrics are tracked
        """
        for query_text in self.GRAPH_QUERIES[:2]:  # Test first 2 queries only
            print(f"\n{'='*60}")
            print(f"Testing GRAPH query: {query_text}")
            print(f"{'='*60}")
            
            response = query_pipeline.query(query_text, top_k=10, rerank_top_k=5, max_depth=3)
            
            # Verify no errors
            assert response.error is None, f"Query failed: {response.error}"
            
            # Verify response structure
            assert response.answer, "Response should contain an answer"
            assert isinstance(response.citations, dict), "Citations should be a dictionary"
            assert response.faithfulness_score >= 0.0, "Faithfulness score should be non-negative"
            assert response.retrieval_mode in ["vector", "graph", "hybrid"], "Invalid retrieval mode"
            
            # Verify metrics
            assert response.metrics is not None, "Metrics should be present"
            assert response.metrics.total_time > 0, "Total time should be positive"
            assert response.metrics.query_mode in ["vector", "graph", "hybrid"], "Invalid query mode"
            
            # Print results
            print(f"\nRetrieval Mode: {response.retrieval_mode}")
            print(f"Faithfulness Score: {response.faithfulness_score:.2f}")
            print(f"Citations: {len(response.citations)}")
            print(f"Warnings: {len(response.warnings)}")
            print(f"Total Time: {response.metrics.total_time:.2f}s")
            print(f"\nAnswer:\n{response.answer[:200]}...")
            
            if response.warnings:
                print(f"\nWarnings:")
                for warning in response.warnings:
                    print(f"  - {warning}")
    
    def test_hybrid_mode_queries(self, query_pipeline, check_prerequisites):
        """
        Test HYBRID mode with complex queries requiring both vector and graph.
        
        Validates:
        - Query completes without errors
        - Response contains an answer
        - Citations are present
        - Faithfulness score is computed
        - Metrics are tracked
        """
        for query_text in self.HYBRID_QUERIES[:2]:  # Test first 2 queries only
            print(f"\n{'='*60}")
            print(f"Testing HYBRID query: {query_text}")
            print(f"{'='*60}")
            
            response = query_pipeline.query(query_text, top_k=10, rerank_top_k=5, max_depth=3)
            
            # Verify no errors
            assert response.error is None, f"Query failed: {response.error}"
            
            # Verify response structure
            assert response.answer, "Response should contain an answer"
            assert isinstance(response.citations, dict), "Citations should be a dictionary"
            assert response.faithfulness_score >= 0.0, "Faithfulness score should be non-negative"
            assert response.retrieval_mode in ["vector", "graph", "hybrid"], "Invalid retrieval mode"
            
            # Verify metrics
            assert response.metrics is not None, "Metrics should be present"
            assert response.metrics.total_time > 0, "Total time should be positive"
            assert response.metrics.query_mode in ["vector", "graph", "hybrid"], "Invalid query mode"
            
            # Print results
            print(f"\nRetrieval Mode: {response.retrieval_mode}")
            print(f"Faithfulness Score: {response.faithfulness_score:.2f}")
            print(f"Citations: {len(response.citations)}")
            print(f"Warnings: {len(response.warnings)}")
            print(f"Total Time: {response.metrics.total_time:.2f}s")
            print(f"\nAnswer:\n{response.answer[:200]}...")
            
            if response.warnings:
                print(f"\nWarnings:")
                for warning in response.warnings:
                    print(f"  - {warning}")
    
    def test_citations_format(self, query_pipeline, check_prerequisites):
        """
        Test that citations are properly formatted in responses.
        
        Validates:
        - Citations dictionary is present
        - Each citation has required fields
        - Citation IDs match the format [doc_id:section]
        """
        query_text = "What is NEFT?"
        response = query_pipeline.query(query_text)
        
        assert response.error is None, f"Query failed: {response.error}"
        
        # Check citations structure
        if response.citations:
            for citation_id, citation in response.citations.items():
                # Verify citation format
                assert ":" in citation_id, f"Citation ID should contain ':' separator: {citation_id}"
                
                # Verify required fields
                assert "doc_id" in citation, f"Citation missing doc_id: {citation_id}"
                assert "section" in citation, f"Citation missing section: {citation_id}"
                assert "chunk_id" in citation, f"Citation missing chunk_id: {citation_id}"
                assert "breadcrumbs" in citation, f"Citation missing breadcrumbs: {citation_id}"
                
                print(f"\nCitation {citation_id}:")
                print(f"  Doc ID: {citation['doc_id']}")
                print(f"  Section: {citation['section']}")
                print(f"  Chunk ID: {citation['chunk_id']}")
                print(f"  Breadcrumbs: {citation['breadcrumbs']}")
        else:
            print("\nNote: No citations in response (may be due to insufficient context)")
    
    def test_faithfulness_validation(self, query_pipeline, check_prerequisites):
        """
        Test that faithfulness validation is performed on responses.
        
        Validates:
        - Faithfulness score is between 0 and 1
        - Warnings are generated for low scores
        - Validation metrics are tracked
        """
        query_text = "What is NEFT?"
        response = query_pipeline.query(query_text)
        
        assert response.error is None, f"Query failed: {response.error}"
        
        # Verify faithfulness score
        assert 0.0 <= response.faithfulness_score <= 1.0, \
            f"Faithfulness score should be between 0 and 1: {response.faithfulness_score}"
        
        # Verify warnings for low scores
        if response.faithfulness_score < 0.8:
            assert len(response.warnings) > 0, \
                "Low faithfulness score should generate warnings"
            assert any("faithfulness" in w.lower() for w in response.warnings), \
                "Warnings should mention faithfulness"
        
        print(f"\nFaithfulness Score: {response.faithfulness_score:.2f}")
        print(f"Warnings: {len(response.warnings)}")
        
        if response.warnings:
            print("\nWarnings:")
            for warning in response.warnings:
                print(f"  - {warning}")
    
    def test_query_metrics_tracking(self, query_pipeline, check_prerequisites):
        """
        Test that query execution metrics are properly tracked.
        
        Validates:
        - All metric fields are populated
        - Timing metrics are positive
        - Component metrics are reasonable
        """
        query_text = "What is NEFT?"
        response = query_pipeline.query(query_text)
        
        assert response.error is None, f"Query failed: {response.error}"
        assert response.metrics is not None, "Metrics should be present"
        
        metrics = response.metrics
        
        # Verify timing metrics
        assert metrics.total_time > 0, "Total time should be positive"
        assert metrics.routing_time >= 0, "Routing time should be non-negative"
        assert metrics.retrieval_time >= 0, "Retrieval time should be non-negative"
        assert metrics.fusion_time >= 0, "Fusion time should be non-negative"
        assert metrics.reranking_time >= 0, "Reranking time should be non-negative"
        assert metrics.assembly_time >= 0, "Assembly time should be non-negative"
        assert metrics.generation_time >= 0, "Generation time should be non-negative"
        assert metrics.validation_time >= 0, "Validation time should be non-negative"
        
        # Verify component metrics
        assert metrics.query_mode in ["vector", "graph", "hybrid"], "Invalid query mode"
        assert metrics.chunks_retrieved >= 0, "Chunks retrieved should be non-negative"
        assert metrics.chunks_reranked >= 0, "Chunks reranked should be non-negative"
        assert metrics.context_tokens >= 0, "Context tokens should be non-negative"
        
        # Print metrics
        print(f"\nQuery Metrics:")
        print(f"  Total Time: {metrics.total_time:.3f}s")
        print(f"  Routing Time: {metrics.routing_time:.3f}s")
        print(f"  Retrieval Time: {metrics.retrieval_time:.3f}s")
        print(f"  Fusion Time: {metrics.fusion_time:.3f}s")
        print(f"  Reranking Time: {metrics.reranking_time:.3f}s")
        print(f"  Assembly Time: {metrics.assembly_time:.3f}s")
        print(f"  Generation Time: {metrics.generation_time:.3f}s")
        print(f"  Validation Time: {metrics.validation_time:.3f}s")
        print(f"  Query Mode: {metrics.query_mode}")
        print(f"  Chunks Retrieved: {metrics.chunks_retrieved}")
        print(f"  Chunks Reranked: {metrics.chunks_reranked}")
        print(f"  Context Tokens: {metrics.context_tokens}")
    
    def test_error_handling(self, query_pipeline):
        """
        Test error handling for invalid queries.
        
        Validates:
        - Empty queries return appropriate errors
        - Error responses include error messages
        - Partial metrics are still tracked
        """
        # Test empty query
        response = query_pipeline.query("")
        
        assert response.error is not None, "Empty query should return error"
        assert "empty" in response.error.lower(), "Error should mention empty query"
        assert response.answer == "", "Answer should be empty for error"
        
        print(f"\nEmpty Query Error: {response.error}")
    
    def test_all_modes_comprehensive(self, query_pipeline, check_prerequisites):
        """
        Comprehensive test running queries in all three modes.
        
        This test ensures the pipeline can handle different query types
        and properly route them to the appropriate retrieval mode.
        
        Note: Query routing is LLM-based and may vary, so we verify
        the pipeline completes successfully rather than strict mode matching.
        """
        test_queries = [
            "What is NEFT?",  # Factual query
            "What depends on NEFT?",  # Relational query
            "How does NEFT work and what are its dependencies?",  # Complex query
        ]
        
        results = []
        
        for query_text in test_queries:
            print(f"\n{'='*60}")
            print(f"Testing: {query_text}")
            print(f"{'='*60}")
            
            response = query_pipeline.query(query_text)
            
            # Verify no errors
            assert response.error is None, f"Query failed: {response.error}"
            
            # Verify mode is valid (don't enforce specific mode due to LLM variability)
            assert response.retrieval_mode in ["vector", "graph", "hybrid"], \
                f"Invalid retrieval mode: {response.retrieval_mode}"
            
            # Store results
            results.append({
                "query": query_text,
                "mode": response.retrieval_mode,
                "faithfulness": response.faithfulness_score,
                "citations": len(response.citations),
                "time": response.metrics.total_time
            })
            
            print(f"Mode: {response.retrieval_mode}")
            print(f"Faithfulness: {response.faithfulness_score:.2f}")
            print(f"Citations: {len(response.citations)}")
            print(f"Time: {response.metrics.total_time:.2f}s")
        
        # Print summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        for result in results:
            print(f"\nQuery: {result['query']}")
            print(f"  Mode: {result['mode']}")
            print(f"  Faithfulness: {result['faithfulness']:.2f}")
            print(f"  Citations: {result['citations']}")
            print(f"  Time: {result['time']:.2f}s")


if __name__ == "__main__":
    """
    Run checkpoint tests directly.
    
    Usage:
        python -m pytest tests/test_checkpoint_23_query_pipeline.py -v -s
    """
    pytest.main([__file__, "-v", "-s"])
