"""Vector-based retrieval using hybrid vector + BM25 search with RRF fusion."""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from src.storage.vector_store import VectorStore
from src.indexing.bm25_indexer import BM25Indexer
from src.embedding.embedding_generator import EmbeddingGenerator
from src.storage.database_manager import DatabaseManager
from src.utils.errors import (
    RetrievalError,
    ErrorContext,
    get_degradation_manager,
    BM25_DEGRADATION,
    log_error_with_context
)

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """
    Represents a retrieved chunk with text, metadata, and relevance scores.
    
    Attributes:
        chunk_id: Unique chunk identifier
        text: Full text content of the chunk
        breadcrumbs: Hierarchical context path (e.g., "Doc > Section > Subsection")
        doc_id: Parent document identifier
        section: Section name
        score: Combined relevance score (from RRF fusion)
        retrieval_source: Source of retrieval ('vector', 'bm25', or 'both')
        vector_score: Original vector similarity score (if applicable)
        bm25_score: Original BM25 score (if applicable)
    """
    chunk_id: str
    text: str
    breadcrumbs: str
    doc_id: str
    section: str
    score: float
    retrieval_source: str
    vector_score: float = 0.0
    bm25_score: float = 0.0


class VectorRetriever:
    """
    Retrieves relevant chunks using hybrid vector + BM25 search with RRF fusion.
    
    This class implements a hybrid retrieval strategy that combines:
    1. Vector similarity search (semantic matching)
    2. BM25 keyword search (exact term matching)
    3. Reciprocal Rank Fusion (RRF) for score combination
    
    The retriever performs both searches in parallel for efficiency and fuses
    the results using RRF to produce a unified ranking. Results are filtered
    by similarity threshold and limited to top-k.
    
    Attributes:
        vector_store: VectorStore instance for semantic search
        bm25_index: BM25Indexer instance for keyword search
        embedding_generator: EmbeddingGenerator for query embeddings
        doc_store: DatabaseManager for retrieving full chunk text
        similarity_threshold: Minimum similarity score for vector results (default: 0.7)
        rrf_k: RRF parameter for rank fusion (default: 60)
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Indexer,
        embedding_generator: EmbeddingGenerator,
        doc_store: DatabaseManager,
        similarity_threshold: float = 0.7,
        rrf_k: int = 60
    ):
        """
        Initialize VectorRetriever with storage clients.
        
        Args:
            vector_store: VectorStore instance for semantic search
            bm25_index: BM25Indexer instance for keyword search
            embedding_generator: EmbeddingGenerator for query embeddings
            doc_store: DatabaseManager for retrieving full chunk text
            similarity_threshold: Minimum similarity score for vector results (default: 0.7)
            rrf_k: RRF parameter for rank fusion (default: 60)
        """
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.embedding_generator = embedding_generator
        self.doc_store = doc_store
        self.similarity_threshold = similarity_threshold
        self.rrf_k = rrf_k
        
        logger.info(
            "VectorRetriever initialized",
            extra={
                "similarity_threshold": similarity_threshold,
                "rrf_k": rrf_k
            }
        )
    
    def retrieve(self, query: str, top_k: int = 10) -> List[RetrievedChunk]:
        """
        Retrieve chunks using hybrid vector + BM25 search with RRF fusion.
        
        This method:
        1. Runs vector search and BM25 search in parallel
        2. Applies similarity threshold filtering to vector results
        3. Combines rankings using Reciprocal Rank Fusion (RRF)
        4. Retrieves full chunk text and metadata from document store
        5. Returns top-k results sorted by combined score
        
        Args:
            query: Search query string
            top_k: Maximum number of results to return (default: 10)
        
        Returns:
            List of RetrievedChunk objects sorted by relevance score (descending)
            Limited to top_k results
        
        Raises:
            ValueError: If query is empty
            Exception: If retrieval operations fail
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        logger.info(f"Retrieving chunks for query: '{query}' (top_k={top_k})")
        
        try:
            # Run vector and BM25 search in parallel
            vector_results, bm25_results = self._parallel_search(query)
            
            logger.debug(
                f"Parallel search completed: {len(vector_results)} vector results, "
                f"{len(bm25_results)} BM25 results"
            )
            
            # Apply similarity threshold to vector results
            filtered_vector_results = [
                (chunk_id, score) 
                for chunk_id, score in vector_results 
                if score >= self.similarity_threshold
            ]
            
            logger.debug(
                f"Filtered vector results by threshold {self.similarity_threshold}: "
                f"{len(filtered_vector_results)} results"
            )
            
            # Combine rankings using RRF
            fused_results = self._reciprocal_rank_fusion(
                filtered_vector_results,
                bm25_results
            )
            
            logger.debug(f"RRF fusion produced {len(fused_results)} combined results")
            
            # Limit to top-k
            top_results = fused_results[:top_k]
            
            # Retrieve full chunk data from document store
            retrieved_chunks = self._fetch_chunk_data(top_results)
            
            logger.info(
                f"Retrieved {len(retrieved_chunks)} chunks for query",
                extra={
                    "query": query,
                    "top_k": top_k,
                    "results_count": len(retrieved_chunks)
                }
            )
            
            return retrieved_chunks
            
        except Exception as e:
            logger.error(
                f"Failed to retrieve chunks: {str(e)}",
                extra={"query": query, "top_k": top_k}
            )
            raise
    
    def _parallel_search(
        self, 
        query: str
    ) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        """
        Execute vector and BM25 search in parallel.
        
        Args:
            query: Search query string
        
        Returns:
            Tuple of (vector_results, bm25_results)
            Each result is a list of (chunk_id, score) tuples
        """
        vector_results = []
        bm25_results = []
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both search tasks
            vector_future = executor.submit(self._vector_search, query)
            bm25_future = executor.submit(self._bm25_search, query)
            
            # Collect results as they complete
            for future in as_completed([vector_future, bm25_future]):
                try:
                    result = future.result()
                    if future == vector_future:
                        vector_results = result
                    else:
                        bm25_results = result
                except Exception as e:
                    logger.error(f"Search task failed: {str(e)}")
                    # Continue with partial results
        
        return vector_results, bm25_results
    
    def _vector_search(self, query: str) -> List[Tuple[str, float]]:
        """
        Perform vector similarity search.
        
        Args:
            query: Search query string
        
        Returns:
            List of (chunk_id, similarity_score) tuples
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.generate(query)
            
            # Search vector store (no threshold here, we'll filter later)
            results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=100,  # Get more results for better fusion
                score_threshold=None  # Don't filter yet
            )
            
            # Extract chunk_id and score
            vector_results = [(chunk_id, score) for chunk_id, score, _ in results]
            
            logger.debug(f"Vector search returned {len(vector_results)} results")
            return vector_results
            
        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            return []
    
    def _bm25_search(self, query: str) -> List[Tuple[str, float]]:
        """
        Perform BM25 keyword search with graceful degradation.
        
        Args:
            query: Search query string
        
        Returns:
            List of (chunk_id, bm25_score) tuples
        """
        try:
            # Search BM25 index
            results = self.bm25_index.search(
                query=query,
                top_k=100,  # Get more results for better fusion
                score_threshold=None  # No threshold for BM25
            )
            
            # Exit degraded mode if we were in it
            degradation_manager = get_degradation_manager()
            if degradation_manager.is_degraded("bm25_unavailable"):
                degradation_manager.exit_degraded_mode("bm25_unavailable")
            
            logger.debug(f"BM25 search returned {len(results)} results")
            return results
            
        except Exception as e:
            log_error_with_context(
                e,
                component="VectorRetriever",
                operation="_bm25_search",
                query=query
            )
            
            # Enter degraded mode - fall back to vector-only search
            degradation_manager = get_degradation_manager()
            if not degradation_manager.is_degraded("bm25_unavailable"):
                degradation_manager.enter_degraded_mode(BM25_DEGRADATION)
            
            logger.warning(
                "BM25 search failed, falling back to vector-only search",
                extra={"error": str(e)}
            )
            return []
    
    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Tuple[str, float]],
        bm25_results: List[Tuple[str, float]]
    ) -> List[Tuple[str, float, str, float, float]]:
        """
        Combine rankings using Reciprocal Rank Fusion (RRF).
        
        RRF formula: score(chunk) = sum(1 / (k + rank_i))
        where rank_i is the rank of the chunk in result list i
        
        Args:
            vector_results: List of (chunk_id, vector_score) tuples
            bm25_results: List of (chunk_id, bm25_score) tuples
        
        Returns:
            List of (chunk_id, rrf_score, source, vector_score, bm25_score) tuples
            sorted by rrf_score descending
        """
        # Build rank maps
        vector_ranks = {chunk_id: rank for rank, (chunk_id, _) in enumerate(vector_results)}
        bm25_ranks = {chunk_id: rank for rank, (chunk_id, _) in enumerate(bm25_results)}
        
        # Build score maps for tracking original scores
        vector_scores = {chunk_id: score for chunk_id, score in vector_results}
        bm25_scores = {chunk_id: score for chunk_id, score in bm25_results}
        
        # Get all unique chunk IDs
        all_chunk_ids = set(vector_ranks.keys()) | set(bm25_ranks.keys())
        
        # Compute RRF scores
        rrf_scores = {}
        sources = {}
        
        for chunk_id in all_chunk_ids:
            score = 0.0
            source_parts = []
            
            # Add vector contribution
            if chunk_id in vector_ranks:
                score += 1.0 / (self.rrf_k + vector_ranks[chunk_id])
                source_parts.append("vector")
            
            # Add BM25 contribution
            if chunk_id in bm25_ranks:
                score += 1.0 / (self.rrf_k + bm25_ranks[chunk_id])
                source_parts.append("bm25")
            
            rrf_scores[chunk_id] = score
            sources[chunk_id] = "+".join(source_parts) if len(source_parts) > 1 else source_parts[0]
        
        # Create result list with all scores
        results = [
            (
                chunk_id,
                rrf_scores[chunk_id],
                sources[chunk_id],
                vector_scores.get(chunk_id, 0.0),
                bm25_scores.get(chunk_id, 0.0)
            )
            for chunk_id in all_chunk_ids
        ]
        
        # Sort by RRF score descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def _fetch_chunk_data(
        self,
        fused_results: List[Tuple[str, float, str, float, float]]
    ) -> List[RetrievedChunk]:
        """
        Fetch full chunk data from document store.
        
        Args:
            fused_results: List of (chunk_id, rrf_score, source, vector_score, bm25_score) tuples
        
        Returns:
            List of RetrievedChunk objects with full text and metadata
        """
        retrieved_chunks = []
        
        for chunk_id, rrf_score, source, vector_score, bm25_score in fused_results:
            try:
                # Fetch chunk from database
                chunk_data = self.doc_store.get_chunk_by_id(chunk_id)
                
                if chunk_data is None:
                    logger.warning(f"Chunk {chunk_id} not found in document store")
                    continue
                
                # Create RetrievedChunk
                retrieved_chunk = RetrievedChunk(
                    chunk_id=chunk_id,
                    text=chunk_data.get("text", ""),
                    breadcrumbs=chunk_data.get("breadcrumbs", ""),
                    doc_id=chunk_data.get("doc_id", ""),
                    section=chunk_data.get("section", ""),
                    score=rrf_score,
                    retrieval_source=source,
                    vector_score=vector_score,
                    bm25_score=bm25_score
                )
                
                retrieved_chunks.append(retrieved_chunk)
                
            except Exception as e:
                logger.error(f"Failed to fetch chunk {chunk_id}: {str(e)}")
                continue
        
        return retrieved_chunks
