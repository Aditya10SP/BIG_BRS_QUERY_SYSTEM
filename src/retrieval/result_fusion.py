"""Result fusion for combining vector and graph retrieval results."""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any

from src.retrieval.vector_retriever import RetrievedChunk
from src.retrieval.graph_retriever import GraphResult, GraphNode, GraphRelationship

logger = logging.getLogger(__name__)


@dataclass
class FusedResults:
    """
    Represents fused results from vector and graph retrieval.
    
    Attributes:
        chunks: List of deduplicated RetrievedChunk objects
        graph_facts: List of formatted graph facts as strings
        combined_score: Dictionary mapping chunk_id to combined score (0.6 vector + 0.4 graph)
    """
    chunks: List[RetrievedChunk] = field(default_factory=list)
    graph_facts: List[str] = field(default_factory=list)
    combined_score: Dict[str, float] = field(default_factory=dict)


class ResultFusion:
    """
    Merges and deduplicates results from vector and graph retrieval.
    
    This class implements result fusion for HYBRID mode queries by:
    1. Deduplicating chunks by chunk_id (removing duplicates from both sources)
    2. Extracting and formatting graph facts from nodes and relationships
    3. Combining scores using weighted average (0.6 vector + 0.4 graph)
    4. Preserving both vector similarity and graph centrality context
    
    The fusion process ensures that:
    - Chunks appearing in both vector and graph results appear only once
    - Graph relationships are converted to human-readable facts
    - Scores are combined to reflect both semantic relevance and graph importance
    
    Attributes:
        vector_weight: Weight for vector similarity scores (default: 0.6)
        graph_weight: Weight for graph centrality scores (default: 0.4)
    """
    
    def __init__(self, vector_weight: float = 0.6, graph_weight: float = 0.4):
        """
        Initialize ResultFusion with score weights.
        
        Args:
            vector_weight: Weight for vector similarity scores (default: 0.6)
            graph_weight: Weight for graph centrality scores (default: 0.4)
        
        Raises:
            ValueError: If weights don't sum to 1.0
        """
        if abs(vector_weight + graph_weight - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {vector_weight + graph_weight}")
        
        self.vector_weight = vector_weight
        self.graph_weight = graph_weight
        
        logger.info(
            "ResultFusion initialized",
            extra={
                "vector_weight": vector_weight,
                "graph_weight": graph_weight
            }
        )
    
    def fuse(
        self,
        vector_results: List[RetrievedChunk],
        graph_results: GraphResult
    ) -> FusedResults:
        """
        Merge vector chunks and graph results, deduplicating overlaps.
        
        This method:
        1. Deduplicates chunks by chunk_id (vector + graph chunks)
        2. Extracts graph facts from nodes and relationships
        3. Computes combined scores for all chunks
        4. Preserves metadata from both sources
        
        Args:
            vector_results: List of RetrievedChunk from vector retrieval
            graph_results: GraphResult from graph retrieval
        
        Returns:
            FusedResults with deduplicated chunks, graph facts, and combined scores
        """
        logger.info(
            f"Fusing results: {len(vector_results)} vector chunks, "
            f"{len(graph_results.chunks)} graph chunks, "
            f"{len(graph_results.nodes)} nodes, "
            f"{len(graph_results.relationships)} relationships"
        )
        
        # Step 1: Deduplicate chunks by chunk_id
        chunks_dict: Dict[str, RetrievedChunk] = {}
        
        # Add vector results
        for chunk in vector_results:
            chunks_dict[chunk.chunk_id] = chunk
        
        # Add graph chunks (convert from dict to RetrievedChunk if needed)
        for chunk_data in graph_results.chunks:
            chunk_id = chunk_data.get("chunk_id")
            if chunk_id and chunk_id not in chunks_dict:
                # Create RetrievedChunk from graph chunk data
                retrieved_chunk = RetrievedChunk(
                    chunk_id=chunk_id,
                    text=chunk_data.get("text", ""),
                    breadcrumbs=chunk_data.get("breadcrumbs", ""),
                    doc_id=chunk_data.get("doc_id", ""),
                    section=chunk_data.get("section", ""),
                    score=0.0,  # Will be computed in combined_score
                    retrieval_source="graph"
                )
                chunks_dict[chunk_id] = retrieved_chunk
        
        logger.debug(f"Deduplicated to {len(chunks_dict)} unique chunks")
        
        # Step 2: Extract graph facts from nodes and relationships
        graph_facts = self._extract_graph_facts(
            graph_results.nodes,
            graph_results.relationships
        )
        
        logger.debug(f"Extracted {len(graph_facts)} graph facts")
        
        # Step 3: Compute combined scores
        combined_scores = self._compute_combined_scores(
            list(chunks_dict.values()),
            graph_results
        )
        
        # Step 4: Update chunk scores with combined scores
        for chunk in chunks_dict.values():
            if chunk.chunk_id in combined_scores:
                chunk.score = combined_scores[chunk.chunk_id]
        
        # Sort chunks by combined score (descending)
        sorted_chunks = sorted(
            chunks_dict.values(),
            key=lambda c: combined_scores.get(c.chunk_id, 0.0),
            reverse=True
        )
        
        logger.info(
            f"Fusion complete: {len(sorted_chunks)} chunks, {len(graph_facts)} facts"
        )
        
        return FusedResults(
            chunks=sorted_chunks,
            graph_facts=graph_facts,
            combined_score=combined_scores
        )
    
    def _extract_graph_facts(
        self,
        nodes: List[GraphNode],
        relationships: List[GraphRelationship]
    ) -> List[str]:
        """
        Extract and format graph facts from nodes and relationships.
        
        Converts graph structure into human-readable facts like:
        - "System A DEPENDS_ON System B"
        - "Workflow X NEXT_STEP Workflow Y"
        - "Entity E1 CONFLICTS_WITH Entity E2 (reason: different limits)"
        
        Args:
            nodes: List of GraphNode objects
            relationships: List of GraphRelationship objects
        
        Returns:
            List of formatted graph fact strings
        """
        facts = []
        
        # Build node lookup for relationship formatting
        node_lookup = {node.node_id: node for node in nodes}
        
        # Format relationships as facts
        for rel in relationships:
            source_node = node_lookup.get(rel.source_id)
            target_node = node_lookup.get(rel.target_id)
            
            if not source_node or not target_node:
                logger.warning(
                    f"Relationship {rel.rel_id} references missing nodes: "
                    f"source={rel.source_id}, target={rel.target_id}"
                )
                continue
            
            # Get entity names
            source_name = source_node.properties.get("name") or source_node.properties.get("canonical_name") or rel.source_id
            target_name = target_node.properties.get("name") or target_node.properties.get("canonical_name") or rel.target_id
            
            # Format basic fact
            fact = f"{source_name} {rel.rel_type} {target_name}"
            
            # Add relationship properties if present
            if rel.properties:
                # Special handling for CONFLICTS_WITH
                if rel.rel_type == "CONFLICTS_WITH":
                    conflict_type = rel.properties.get("conflict_type", "")
                    explanation = rel.properties.get("explanation", "")
                    if conflict_type or explanation:
                        detail = conflict_type or explanation
                        fact += f" (reason: {detail})"
                
                # Special handling for other relationship types with metadata
                elif "metadata" in rel.properties:
                    metadata = rel.properties["metadata"]
                    if isinstance(metadata, dict) and metadata:
                        # Format first key-value pair
                        key, value = next(iter(metadata.items()))
                        fact += f" ({key}: {value})"
            
            facts.append(fact)
        
        return facts
    
    def _compute_combined_scores(
        self,
        chunks: List[RetrievedChunk],
        graph_results: GraphResult
    ) -> Dict[str, float]:
        """
        Compute combined scores using weighted average (0.6 vector + 0.4 graph).
        
        For chunks from vector retrieval: Use vector score
        For chunks from graph retrieval: Compute graph centrality score
        For chunks from both: Combine both scores
        
        Args:
            chunks: List of all chunks (from both sources)
            graph_results: GraphResult containing graph structure
        
        Returns:
            Dictionary mapping chunk_id to combined score
        """
        combined_scores = {}
        
        # Build chunk_id to graph centrality map
        graph_chunk_ids = {chunk_data.get("chunk_id") for chunk_data in graph_results.chunks}
        
        # Compute graph centrality scores (simple: 1.0 if in graph, 0.0 otherwise)
        # In a more sophisticated implementation, this could use PageRank or degree centrality
        graph_centrality = {}
        for chunk_data in graph_results.chunks:
            chunk_id = chunk_data.get("chunk_id")
            if chunk_id:
                # Simple centrality: 1.0 for all graph chunks
                # Could be enhanced with actual centrality metrics
                graph_centrality[chunk_id] = 1.0
        
        # Compute combined scores for all chunks
        for chunk in chunks:
            vector_score = chunk.score if chunk.retrieval_source in ["vector", "bm25", "vector+bm25", "both"] else 0.0
            graph_score = graph_centrality.get(chunk.chunk_id, 0.0)
            
            # Weighted combination
            combined_score = (self.vector_weight * vector_score) + (self.graph_weight * graph_score)
            
            combined_scores[chunk.chunk_id] = combined_score
        
        return combined_scores
