"""Tests for QueryPipeline orchestration."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from src.pipeline.query_pipeline import QueryPipeline, QueryResponse, QueryMetrics
from src.query.query_router import QueryMode
from src.retrieval.vector_retriever import RetrievedChunk
from src.retrieval.graph_retriever import GraphResult, GraphNode, GraphRelationship
from src.retrieval.result_fusion import FusedResults
from src.retrieval.context_assembler import AssembledContext, Citation
from src.query.llm_generator import GeneratedResponse
from src.query.faithfulness_validator import ValidationResult


@pytest.fixture
def mock_query_router():
    """Create mock QueryRouter."""
    router = Mock()
    router.route = Mock(return_value=(QueryMode.VECTOR, 0.9))
    return router


@pytest.fixture
def mock_vector_retriever():
    """Create mock VectorRetriever."""
    retriever = Mock()
    retriever.retrieve = Mock(return_value=[
        RetrievedChunk(
            chunk_id="chunk1",
            text="NEFT is a payment system",
            breadcrumbs="Doc > Section",
            doc_id="doc1",
            section="section1",
            score=0.95,
            retrieval_source="vector"
        )
    ])
    return retriever


@pytest.fixture
def mock_graph_retriever():
    """Create mock GraphRetriever."""
    retriever = Mock()
    retriever.retrieve = Mock(return_value=GraphResult(
        nodes=[
            GraphNode(
                node_id="node1",
                node_type="System",
                properties={"name": "NEFT"}
            )
        ],
        relationships=[],
        chunks=[]
    ))
    return retriever


@pytest.fixture
def mock_result_fusion():
    """Create mock ResultFusion."""
    fusion = Mock()
    fusion.fuse = Mock(return_value=FusedResults(
        chunks=[
            RetrievedChunk(
                chunk_id="chunk1",
                text="NEFT is a payment system",
                breadcrumbs="Doc > Section",
                doc_id="doc1",
                section="section1",
                score=0.95,
                retrieval_source="vector"
            )
        ],
        graph_facts=["System A DEPENDS_ON System B"],
        combined_score={"chunk1": 0.95}
    ))
    fusion._extract_graph_facts = Mock(return_value=["System A DEPENDS_ON System B"])
    return fusion


@pytest.fixture
def mock_reranker():
    """Create mock CrossEncoderReranker."""
    reranker = Mock()
    reranker.rerank = Mock(return_value=[
        RetrievedChunk(
            chunk_id="chunk1",
            text="NEFT is a payment system",
            breadcrumbs="Doc > Section",
            doc_id="doc1",
            section="section1",
            score=0.98,
            retrieval_source="vector"
        )
    ])
    return reranker


@pytest.fixture
def mock_context_assembler():
    """Create mock ContextAssembler."""
    assembler = Mock()
    assembler.assemble = Mock(return_value=AssembledContext(
        context_text="Query: test\n\nRelevant Document Excerpts:\n\n[doc1:section1] (Doc > Section)\nNEFT is a payment system\n\n",
        citations={
            "doc1:section1": Citation(
                citation_id="doc1:section1",
                doc_id="doc1",
                section="section1",
                chunk_id="chunk1",
                breadcrumbs="Doc > Section"
            )
        },
        token_count=50
    ))
    return assembler


@pytest.fixture
def mock_llm_generator():
    """Create mock LLMGenerator."""
    generator = Mock()
    generator.generate = Mock(return_value=GeneratedResponse(
        answer="NEFT is a payment system [doc1:section1]",
        citations_used=["doc1:section1"],
        model="test-model",
        timestamp=datetime.now()
    ))
    return generator


@pytest.fixture
def mock_faithfulness_validator():
    """Create mock FaithfulnessValidator."""
    validator = Mock()
    validator.validate = Mock(return_value=ValidationResult(
        faithfulness_score=0.95,
        total_claims=1,
        supported_claims=1,
        unsupported_claims=[],
        warnings=[]
    ))
    return validator


@pytest.fixture
def query_pipeline(
    mock_query_router,
    mock_vector_retriever,
    mock_graph_retriever,
    mock_result_fusion,
    mock_reranker,
    mock_context_assembler,
    mock_llm_generator,
    mock_faithfulness_validator
):
    """Create QueryPipeline with all mocked components."""
    return QueryPipeline(
        query_router=mock_query_router,
        vector_retriever=mock_vector_retriever,
        graph_retriever=mock_graph_retriever,
        result_fusion=mock_result_fusion,
        reranker=mock_reranker,
        context_assembler=mock_context_assembler,
        llm_generator=mock_llm_generator,
        faithfulness_validator=mock_faithfulness_validator
    )


class TestQueryPipeline:
    """Test suite for QueryPipeline."""
    
    def test_initialization(self, query_pipeline):
        """Test QueryPipeline initialization."""
        assert query_pipeline.query_router is not None
        assert query_pipeline.vector_retriever is not None
        assert query_pipeline.graph_retriever is not None
        assert query_pipeline.result_fusion is not None
        assert query_pipeline.reranker is not None
        assert query_pipeline.context_assembler is not None
        assert query_pipeline.llm_generator is not None
        assert query_pipeline.faithfulness_validator is not None
    
    def test_query_empty_string(self, query_pipeline):
        """Test query with empty string returns error."""
        response = query_pipeline.query("")
        
        assert response.error == "Query cannot be empty"
        assert response.answer == ""
    
    def test_query_vector_mode_success(self, query_pipeline, mock_query_router):
        """Test successful query execution in VECTOR mode."""
        mock_query_router.route.return_value = (QueryMode.VECTOR, 0.9)
        
        response = query_pipeline.query("What is NEFT?")
        
        # Verify response structure
        assert response.error is None
        assert response.answer == "NEFT is a payment system [doc1:section1]"
        assert response.retrieval_mode == "vector"
        assert response.faithfulness_score == 0.95
        assert len(response.citations) == 1
        assert "doc1:section1" in response.citations
        
        # Verify metrics
        assert response.metrics is not None
        assert response.metrics.query_mode == "vector"
        assert response.metrics.total_time > 0
        assert response.metrics.chunks_retrieved > 0
        assert response.metrics.chunks_reranked > 0
    
    def test_query_graph_mode_success(self, query_pipeline, mock_query_router):
        """Test successful query execution in GRAPH mode."""
        mock_query_router.route.return_value = (QueryMode.GRAPH, 0.85)
        
        response = query_pipeline.query("What depends on NEFT?")
        
        # Verify response
        assert response.error is None
        assert response.retrieval_mode == "graph"
        assert response.metrics.query_mode == "graph"
    
    def test_query_hybrid_mode_success(self, query_pipeline, mock_query_router):
        """Test successful query execution in HYBRID mode with parallel retrieval."""
        mock_query_router.route.return_value = (QueryMode.HYBRID, 0.75)
        
        response = query_pipeline.query("How does NEFT integrate with Core Banking?")
        
        # Verify response
        assert response.error is None
        assert response.retrieval_mode == "hybrid"
        assert response.metrics.query_mode == "hybrid"
        
        # Verify both retrievers were called (parallel execution)
        query_pipeline.vector_retriever.retrieve.assert_called_once()
        query_pipeline.graph_retriever.retrieve.assert_called_once()
    
    def test_query_pipeline_step_ordering(self, query_pipeline):
        """Test that pipeline steps execute in correct order."""
        # Track call order
        call_order = []
        
        query_pipeline.query_router.route.side_effect = lambda q: (
            call_order.append("route") or (QueryMode.VECTOR, 0.9)
        )
        query_pipeline.vector_retriever.retrieve.side_effect = lambda query, top_k: (
            call_order.append("retrieve") or []
        )
        query_pipeline.reranker.rerank.side_effect = lambda query, results, top_k: (
            call_order.append("rerank") or []
        )
        query_pipeline.context_assembler.assemble.side_effect = lambda query, chunks, graph_facts: (
            call_order.append("assemble") or AssembledContext("", {}, 0)
        )
        query_pipeline.llm_generator.generate.side_effect = lambda query, context: (
            call_order.append("generate") or GeneratedResponse("", [], "model", datetime.now())
        )
        query_pipeline.faithfulness_validator.validate.side_effect = lambda response, context: (
            call_order.append("validate") or ValidationResult(1.0, 0, 0, [], [])
        )
        
        query_pipeline.query("test query")
        
        # Verify order: route -> retrieve -> rerank -> assemble -> generate -> validate
        assert call_order == ["route", "retrieve", "rerank", "assemble", "generate", "validate"]
    
    def test_query_with_custom_parameters(self, query_pipeline):
        """Test query with custom top_k and rerank_top_k parameters."""
        response = query_pipeline.query(
            "What is NEFT?",
            top_k=20,
            rerank_top_k=10,
            max_depth=5
        )
        
        # Verify parameters were passed correctly
        query_pipeline.vector_retriever.retrieve.assert_called_with(
            query="What is NEFT?",
            top_k=20
        )
        
        # Verify reranker was called with correct top_k
        assert query_pipeline.reranker.rerank.call_args[1]["top_k"] == 10
    
    def test_query_routing_failure_handling(self, query_pipeline, mock_query_router):
        """Test error handling when query routing fails."""
        mock_query_router.route.side_effect = Exception("Routing failed")
        
        response = query_pipeline.query("test query")
        
        assert response.error is not None
        assert "Routing failed" in response.error
        assert response.answer == ""
    
    def test_query_retrieval_failure_handling(self, query_pipeline, mock_vector_retriever):
        """Test error handling when retrieval fails."""
        mock_vector_retriever.retrieve.side_effect = Exception("Retrieval failed")
        
        response = query_pipeline.query("test query")
        
        assert response.error is not None
        assert "Retrieval failed" in response.error
    
    def test_query_generation_failure_handling(self, query_pipeline, mock_llm_generator):
        """Test error handling when LLM generation fails."""
        mock_llm_generator.generate.side_effect = Exception("Generation failed")
        
        response = query_pipeline.query("test query")
        
        assert response.error is not None
        assert "Generation failed" in response.error
    
    def test_query_metrics_tracking(self, query_pipeline):
        """Test that query metrics are properly tracked."""
        response = query_pipeline.query("What is NEFT?")
        
        # Verify all metric fields are populated
        assert response.metrics is not None
        assert response.metrics.total_time > 0
        assert response.metrics.routing_time >= 0
        assert response.metrics.retrieval_time >= 0
        assert response.metrics.fusion_time >= 0
        assert response.metrics.reranking_time >= 0
        assert response.metrics.assembly_time >= 0
        assert response.metrics.generation_time >= 0
        assert response.metrics.validation_time >= 0
        assert response.metrics.query_mode != ""
        assert response.metrics.chunks_retrieved >= 0
        assert response.metrics.chunks_reranked >= 0
        assert response.metrics.context_tokens >= 0
    
    def test_query_low_faithfulness_warning(self, query_pipeline, mock_faithfulness_validator):
        """Test that low faithfulness score generates warnings."""
        mock_faithfulness_validator.validate.return_value = ValidationResult(
            faithfulness_score=0.6,
            total_claims=5,
            supported_claims=3,
            unsupported_claims=["Claim 1", "Claim 2"],
            warnings=["Low faithfulness score (0.60 < 0.80)"]
        )
        
        response = query_pipeline.query("test query")
        
        assert response.faithfulness_score == 0.6
        assert len(response.warnings) > 0
        assert any("faithfulness" in w.lower() for w in response.warnings)
    
    def test_parallel_retrieval_in_hybrid_mode(self, query_pipeline, mock_query_router):
        """Test that HYBRID mode executes vector and graph retrieval in parallel."""
        mock_query_router.route.return_value = (QueryMode.HYBRID, 0.75)
        
        # Add delays to simulate parallel execution
        import time
        
        def slow_vector_retrieve(*args, **kwargs):
            time.sleep(0.1)
            return []
        
        def slow_graph_retrieve(*args, **kwargs):
            time.sleep(0.1)
            return GraphResult()
        
        query_pipeline.vector_retriever.retrieve = Mock(side_effect=slow_vector_retrieve)
        query_pipeline.graph_retriever.retrieve = Mock(side_effect=slow_graph_retrieve)
        
        start_time = time.time()
        query_pipeline.query("test query")
        elapsed_time = time.time() - start_time
        
        # If parallel, should take ~0.1s, if sequential would take ~0.2s
        # Allow some overhead, but should be significantly less than 0.2s
        assert elapsed_time < 0.18, "Retrieval should execute in parallel"
    
    def test_citations_format_in_response(self, query_pipeline):
        """Test that citations are properly formatted in response."""
        response = query_pipeline.query("What is NEFT?")
        
        assert "doc1:section1" in response.citations
        citation = response.citations["doc1:section1"]
        
        assert citation["doc_id"] == "doc1"
        assert citation["section"] == "section1"
        assert citation["chunk_id"] == "chunk1"
        assert citation["breadcrumbs"] == "Doc > Section"
    
    def test_vector_mode_no_graph_retrieval(self, query_pipeline, mock_query_router):
        """Test that VECTOR mode doesn't call graph retriever."""
        mock_query_router.route.return_value = (QueryMode.VECTOR, 0.9)
        
        query_pipeline.query("What is NEFT?")
        
        # Vector retriever should be called
        query_pipeline.vector_retriever.retrieve.assert_called_once()
        
        # Graph retriever should NOT be called
        query_pipeline.graph_retriever.retrieve.assert_not_called()
    
    def test_graph_mode_no_vector_retrieval(self, query_pipeline, mock_query_router):
        """Test that GRAPH mode doesn't call vector retriever."""
        mock_query_router.route.return_value = (QueryMode.GRAPH, 0.85)
        
        query_pipeline.query("What depends on NEFT?")
        
        # Graph retriever should be called
        query_pipeline.graph_retriever.retrieve.assert_called_once()
        
        # Vector retriever should NOT be called
        query_pipeline.vector_retriever.retrieve.assert_not_called()
    
    def test_error_response_includes_metrics(self, query_pipeline, mock_vector_retriever):
        """Test that error responses still include partial metrics."""
        mock_vector_retriever.retrieve.side_effect = Exception("Retrieval failed")
        
        response = query_pipeline.query("test query")
        
        assert response.error is not None
        assert response.metrics is not None
        assert response.metrics.total_time > 0


class TestQueryPipelineIntegration:
    """Integration tests for QueryPipeline with real component interactions."""
    
    def test_end_to_end_vector_mode(self, query_pipeline):
        """Test complete end-to-end execution in VECTOR mode."""
        response = query_pipeline.query("What is NEFT?", top_k=10, rerank_top_k=5)
        
        # Verify complete response
        assert response.error is None
        assert response.answer != ""
        assert response.retrieval_mode == "vector"
        assert response.faithfulness_score > 0
        assert len(response.citations) > 0
        assert response.metrics.total_time > 0
        
        # Verify all components were called
        query_pipeline.query_router.route.assert_called_once()
        query_pipeline.vector_retriever.retrieve.assert_called_once()
        query_pipeline.reranker.rerank.assert_called_once()
        query_pipeline.context_assembler.assemble.assert_called_once()
        query_pipeline.llm_generator.generate.assert_called_once()
        query_pipeline.faithfulness_validator.validate.assert_called_once()
