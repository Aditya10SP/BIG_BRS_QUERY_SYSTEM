"""Query pipeline orchestration for end-to-end query processing."""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.query.query_router import QueryRouter, QueryMode
from src.retrieval.vector_retriever import VectorRetriever, RetrievedChunk
from src.retrieval.graph_retriever import GraphRetriever, GraphResult
from src.retrieval.result_fusion import ResultFusion, FusedResults
from src.retrieval.cross_encoder_reranker import CrossEncoderReranker
from src.retrieval.context_assembler import ContextAssembler, AssembledContext
from src.query.llm_generator import LLMGenerator, GeneratedResponse
from src.query.faithfulness_validator import FaithfulnessValidator, ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class QueryMetrics:
    """
    Metrics for query execution tracking.
    
    Attributes:
        total_time: Total query execution time in seconds
        routing_time: Time spent on query routing
        retrieval_time: Time spent on retrieval (vector/graph/both)
        fusion_time: Time spent on result fusion
        reranking_time: Time spent on reranking
        assembly_time: Time spent on context assembly
        generation_time: Time spent on LLM generation
        validation_time: Time spent on faithfulness validation
        query_mode: Selected query mode (VECTOR/GRAPH/HYBRID)
        chunks_retrieved: Number of chunks retrieved
        chunks_reranked: Number of chunks after reranking
        context_tokens: Number of tokens in assembled context
    """
    total_time: float = 0.0
    routing_time: float = 0.0
    retrieval_time: float = 0.0
    fusion_time: float = 0.0
    reranking_time: float = 0.0
    assembly_time: float = 0.0
    generation_time: float = 0.0
    validation_time: float = 0.0
    query_mode: str = ""
    chunks_retrieved: int = 0
    chunks_reranked: int = 0
    context_tokens: int = 0


@dataclass
class QueryResponse:
    """
    Complete query response with answer, citations, and metadata.
    
    Attributes:
        answer: Generated answer text with inline citations
        citations: Dictionary of citation_id to Citation objects
        faithfulness_score: Faithfulness validation score (0-1)
        retrieval_mode: Query mode used (VECTOR/GRAPH/HYBRID)
        warnings: List of warning messages (e.g., low faithfulness)
        metrics: Query execution metrics
        error: Error message if query failed (None if successful)
    """
    answer: str
    citations: Dict[str, Any] = field(default_factory=dict)
    faithfulness_score: float = 0.0
    retrieval_mode: str = ""
    warnings: list = field(default_factory=list)
    metrics: Optional[QueryMetrics] = None
    error: Optional[str] = None


class QueryPipeline:
    """
    Orchestrates end-to-end query processing pipeline.
    
    This class implements the complete query pipeline:
    1. Query routing (classify intent)
    2. Retrieval (vector/graph/hybrid with parallel execution)
    3. Result fusion (deduplicate and merge)
    4. Reranking (cross-encoder scoring)
    5. Context assembly (format with citations)
    6. LLM generation (grounded response)
    7. Faithfulness validation (verify grounding)
    
    The pipeline includes:
    - Parallel execution for HYBRID mode retrievals
    - Comprehensive error handling at each step
    - Detailed logging for debugging
    - Query execution metrics tracking
    
    Attributes:
        query_router: QueryRouter for intent classification
        vector_retriever: VectorRetriever for semantic search
        graph_retriever: GraphRetriever for graph traversal
        result_fusion: ResultFusion for merging results
        reranker: CrossEncoderReranker for relevance scoring
        context_assembler: ContextAssembler for formatting
        llm_generator: LLMGenerator for response generation
        faithfulness_validator: FaithfulnessValidator for validation
    """
    
    def __init__(
        self,
        query_router: QueryRouter,
        vector_retriever: VectorRetriever,
        graph_retriever: GraphRetriever,
        result_fusion: ResultFusion,
        reranker: CrossEncoderReranker,
        context_assembler: ContextAssembler,
        llm_generator: LLMGenerator,
        faithfulness_validator: FaithfulnessValidator
    ):
        """
        Initialize QueryPipeline with all components.
        
        Args:
            query_router: QueryRouter for intent classification
            vector_retriever: VectorRetriever for semantic search
            graph_retriever: GraphRetriever for graph traversal
            result_fusion: ResultFusion for merging results
            reranker: CrossEncoderReranker for relevance scoring
            context_assembler: ContextAssembler for formatting
            llm_generator: LLMGenerator for response generation
            faithfulness_validator: FaithfulnessValidator for validation
        """
        self.query_router = query_router
        self.vector_retriever = vector_retriever
        self.graph_retriever = graph_retriever
        self.result_fusion = result_fusion
        self.reranker = reranker
        self.context_assembler = context_assembler
        self.llm_generator = llm_generator
        self.faithfulness_validator = faithfulness_validator
        
        logger.info("QueryPipeline initialized with all components")
    
    def query(
        self,
        query_text: str,
        top_k: int = 10,
        rerank_top_k: int = 5,
        max_depth: Optional[int] = None
    ) -> QueryResponse:
        """
        Execute complete query pipeline and return response.
        
        This method orchestrates all pipeline steps:
        1. Classify query intent (routing)
        2. Retrieve relevant information (vector/graph/hybrid)
        3. Fuse and deduplicate results
        4. Rerank by relevance
        5. Assemble context with citations
        6. Generate grounded response
        7. Validate faithfulness
        
        For HYBRID mode, vector and graph retrieval execute in parallel.
        
        Args:
            query_text: User query string
            top_k: Number of chunks to retrieve (default: 10)
            rerank_top_k: Number of chunks after reranking (default: 5)
            max_depth: Maximum graph traversal depth (optional, uses retriever default)
        
        Returns:
            QueryResponse with answer, citations, and metadata
            If any step fails, returns QueryResponse with error field set
        """
        if not query_text or not query_text.strip():
            return QueryResponse(
                answer="",
                error="Query cannot be empty"
            )
        
        logger.info(f"Starting query pipeline for: '{query_text}'")
        
        # Initialize metrics tracking
        metrics = QueryMetrics()
        start_time = time.time()
        
        try:
            # Step 1: Query routing
            logger.info("Step 1: Query routing")
            routing_start = time.time()
            
            query_mode, confidence = self.query_router.route(query_text)
            metrics.query_mode = query_mode.value
            metrics.routing_time = time.time() - routing_start
            
            logger.info(
                f"Query routed to {query_mode.value} mode (confidence: {confidence:.2f})",
                extra={"query_mode": query_mode.value, "confidence": confidence}
            )
            
            # Step 2: Retrieval (mode-dependent, parallel for HYBRID)
            logger.info("Step 2: Retrieval")
            retrieval_start = time.time()
            
            vector_results, graph_results = self._execute_retrieval(
                query_text, query_mode, top_k, max_depth
            )
            
            metrics.retrieval_time = time.time() - retrieval_start
            logger.info(
                f"Retrieval complete: {len(vector_results)} vector chunks, "
                f"{len(graph_results.chunks) if graph_results else 0} graph chunks"
            )
            
            # Step 3: Result fusion (for HYBRID mode)
            logger.info("Step 3: Result fusion")
            fusion_start = time.time()
            
            fused_results = self._fuse_results(
                vector_results, graph_results, query_mode
            )
            
            metrics.fusion_time = time.time() - fusion_start
            metrics.chunks_retrieved = len(fused_results.chunks)
            
            logger.info(
                f"Fusion complete: {len(fused_results.chunks)} chunks, "
                f"{len(fused_results.graph_facts)} graph facts"
            )
            
            # Step 4: Reranking
            logger.info("Step 4: Reranking")
            reranking_start = time.time()
            
            reranked_chunks = self.reranker.rerank(
                query=query_text,
                results=fused_results,
                top_k=rerank_top_k
            )
            
            metrics.reranking_time = time.time() - reranking_start
            metrics.chunks_reranked = len(reranked_chunks)
            
            logger.info(f"Reranking complete: {len(reranked_chunks)} top chunks")
            
            # Step 5: Context assembly
            logger.info("Step 5: Context assembly")
            assembly_start = time.time()
            
            assembled_context = self.context_assembler.assemble(
                query=query_text,
                chunks=reranked_chunks,
                graph_facts=fused_results.graph_facts
            )
            
            metrics.assembly_time = time.time() - assembly_start
            metrics.context_tokens = assembled_context.token_count
            
            logger.info(
                f"Context assembly complete: {assembled_context.token_count} tokens, "
                f"{len(assembled_context.citations)} citations"
            )
            
            # Step 6: LLM generation
            logger.info("Step 6: LLM generation")
            generation_start = time.time()
            
            generated_response = self.llm_generator.generate(
                query=query_text,
                context=assembled_context
            )
            
            metrics.generation_time = time.time() - generation_start
            
            logger.info(
                f"Generation complete: {len(generated_response.answer)} chars, "
                f"{len(generated_response.citations_used)} citations used"
            )
            
            # Step 7: Faithfulness validation
            logger.info("Step 7: Faithfulness validation")
            validation_start = time.time()
            
            validation_result = self.faithfulness_validator.validate(
                response=generated_response,
                context=assembled_context
            )
            
            metrics.validation_time = time.time() - validation_start
            
            logger.info(
                f"Validation complete: score={validation_result.faithfulness_score:.2f}, "
                f"warnings={len(validation_result.warnings)}"
            )
            
            # Calculate total time
            metrics.total_time = time.time() - start_time
            
            # Log final metrics
            logger.info(
                "Query pipeline complete",
                extra={
                    "total_time": f"{metrics.total_time:.2f}s",
                    "query_mode": metrics.query_mode,
                    "chunks_retrieved": metrics.chunks_retrieved,
                    "chunks_reranked": metrics.chunks_reranked,
                    "context_tokens": metrics.context_tokens,
                    "faithfulness_score": validation_result.faithfulness_score
                }
            )
            
            # Build response
            return QueryResponse(
                answer=generated_response.answer,
                citations={
                    cid: {
                        "doc_id": citation.doc_id,
                        "section": citation.section,
                        "chunk_id": citation.chunk_id,
                        "breadcrumbs": citation.breadcrumbs
                    }
                    for cid, citation in assembled_context.citations.items()
                },
                faithfulness_score=validation_result.faithfulness_score,
                retrieval_mode=query_mode.value,
                warnings=validation_result.warnings,
                metrics=metrics,
                error=None
            )
        
        except Exception as e:
            # Log error with context
            logger.error(
                f"Query pipeline failed: {str(e)}",
                extra={
                    "query": query_text,
                    "error_type": type(e).__name__
                },
                exc_info=True
            )
            
            # Calculate partial metrics
            metrics.total_time = time.time() - start_time
            
            # Return error response
            return QueryResponse(
                answer="",
                error=f"Query processing failed: {str(e)}",
                metrics=metrics
            )
    
    def _execute_retrieval(
        self,
        query_text: str,
        query_mode: QueryMode,
        top_k: int,
        max_depth: Optional[int]
    ) -> tuple:
        """
        Execute retrieval based on query mode with parallel execution for HYBRID.
        
        Args:
            query_text: User query string
            query_mode: Selected query mode (VECTOR/GRAPH/HYBRID)
            top_k: Number of chunks to retrieve
            max_depth: Maximum graph traversal depth
        
        Returns:
            Tuple of (vector_results, graph_results)
            Empty lists/results for unused modes
        """
        vector_results = []
        graph_results = GraphResult()
        
        if query_mode == QueryMode.VECTOR:
            # Vector-only retrieval
            logger.debug("Executing VECTOR mode retrieval")
            vector_results = self.vector_retriever.retrieve(
                query=query_text,
                top_k=top_k
            )
        
        elif query_mode == QueryMode.GRAPH:
            # Graph-only retrieval
            logger.debug("Executing GRAPH mode retrieval")
            graph_results = self.graph_retriever.retrieve(
                query=query_text,
                max_depth=max_depth
            )
        
        elif query_mode == QueryMode.HYBRID:
            # Parallel execution for HYBRID mode
            logger.debug("Executing HYBRID mode retrieval (parallel)")
            
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Submit both retrieval tasks
                vector_future = executor.submit(
                    self.vector_retriever.retrieve,
                    query_text,
                    top_k
                )
                graph_future = executor.submit(
                    self.graph_retriever.retrieve,
                    query_text,
                    max_depth
                )
                
                # Collect results as they complete
                for future in as_completed([vector_future, graph_future]):
                    try:
                        result = future.result()
                        if future == vector_future:
                            vector_results = result
                            logger.debug(f"Vector retrieval complete: {len(vector_results)} chunks")
                        else:
                            graph_results = result
                            logger.debug(
                                f"Graph retrieval complete: {len(graph_results.nodes)} nodes, "
                                f"{len(graph_results.relationships)} relationships"
                            )
                    except Exception as e:
                        logger.error(f"Retrieval task failed: {str(e)}")
                        # Continue with partial results
        
        return vector_results, graph_results
    
    def _fuse_results(
        self,
        vector_results: list,
        graph_results: GraphResult,
        query_mode: QueryMode
    ) -> FusedResults:
        """
        Fuse vector and graph results based on query mode.
        
        For VECTOR mode: Only vector results
        For GRAPH mode: Only graph results
        For HYBRID mode: Fuse both with deduplication
        
        Args:
            vector_results: List of RetrievedChunk from vector retrieval
            graph_results: GraphResult from graph retrieval
            query_mode: Selected query mode
        
        Returns:
            FusedResults with appropriate content for the mode
        """
        if query_mode == QueryMode.VECTOR:
            # Vector-only: no graph facts
            logger.debug("Fusion: VECTOR mode (no graph facts)")
            return FusedResults(
                chunks=vector_results,
                graph_facts=[],
                combined_score={chunk.chunk_id: chunk.score for chunk in vector_results}
            )
        
        elif query_mode == QueryMode.GRAPH:
            # Graph-only: convert graph chunks to RetrievedChunk format
            logger.debug("Fusion: GRAPH mode (graph facts only)")
            
            # Extract graph facts
            graph_facts = self.result_fusion._extract_graph_facts(
                graph_results.nodes,
                graph_results.relationships
            )
            
            # Convert graph chunks to RetrievedChunk
            chunks = []
            for chunk_data in graph_results.chunks:
                chunk = RetrievedChunk(
                    chunk_id=chunk_data.get("chunk_id", ""),
                    text=chunk_data.get("text", ""),
                    breadcrumbs=chunk_data.get("breadcrumbs", ""),
                    doc_id=chunk_data.get("doc_id", ""),
                    section=chunk_data.get("section", ""),
                    score=1.0,  # Default score for graph chunks
                    retrieval_source="graph"
                )
                chunks.append(chunk)
            
            return FusedResults(
                chunks=chunks,
                graph_facts=graph_facts,
                combined_score={chunk.chunk_id: 1.0 for chunk in chunks}
            )
        
        else:  # HYBRID mode
            # Full fusion with deduplication
            logger.debug("Fusion: HYBRID mode (full fusion)")
            return self.result_fusion.fuse(vector_results, graph_results)
