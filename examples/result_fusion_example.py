"""Example usage of ResultFusion for combining vector and graph retrieval results."""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.retrieval.result_fusion import ResultFusion, FusedResults
from src.retrieval.vector_retriever import RetrievedChunk
from src.retrieval.graph_retriever import GraphResult, GraphNode, GraphRelationship


def main():
    """Demonstrate ResultFusion usage."""
    
    print("=" * 80)
    print("Result Fusion Example")
    print("=" * 80)
    print()
    
    # Initialize ResultFusion with default weights (0.6 vector, 0.4 graph)
    fusion = ResultFusion()
    print(f"Initialized ResultFusion with weights: vector={fusion.vector_weight}, graph={fusion.graph_weight}")
    print()
    
    # Example 1: Vector-only results
    print("Example 1: Fusing vector-only results")
    print("-" * 80)
    
    vector_results = [
        RetrievedChunk(
            chunk_id="chunk1",
            text="NEFT (National Electronic Funds Transfer) is a payment system that enables electronic transfer of funds between banks.",
            breadcrumbs="Payment Systems Guide > NEFT > Overview",
            doc_id="doc_payment_systems",
            section="NEFT Overview",
            score=0.92,
            retrieval_source="vector",
            vector_score=0.92,
            bm25_score=0.85
        ),
        RetrievedChunk(
            chunk_id="chunk2",
            text="NEFT transactions are processed in batches at specific settlement times throughout the day.",
            breadcrumbs="Payment Systems Guide > NEFT > Processing",
            doc_id="doc_payment_systems",
            section="NEFT Processing",
            score=0.87,
            retrieval_source="vector",
            vector_score=0.87,
            bm25_score=0.80
        )
    ]
    
    graph_results_empty = GraphResult()
    
    fused1 = fusion.fuse(vector_results, graph_results_empty)
    
    print(f"Vector chunks: {len(vector_results)}")
    print(f"Graph chunks: {len(graph_results_empty.chunks)}")
    print(f"Fused chunks: {len(fused1.chunks)}")
    print(f"Graph facts: {len(fused1.graph_facts)}")
    print()
    
    for i, chunk in enumerate(fused1.chunks, 1):
        print(f"  {i}. {chunk.chunk_id} (score: {chunk.score:.3f}, combined: {fused1.combined_score[chunk.chunk_id]:.3f})")
        print(f"     {chunk.text[:80]}...")
    print()
    
    # Example 2: Graph-only results
    print("Example 2: Fusing graph-only results")
    print("-" * 80)
    
    vector_results_empty = []
    
    graph_results = GraphResult(
        nodes=[
            GraphNode(
                node_id="entity_neft",
                node_type="System",
                properties={"name": "NEFT", "canonical_name": "NEFT"}
            ),
            GraphNode(
                node_id="entity_core_banking",
                node_type="System",
                properties={"name": "Core Banking System", "canonical_name": "Core Banking"}
            ),
            GraphNode(
                node_id="entity_rtgs",
                node_type="System",
                properties={"name": "RTGS", "canonical_name": "RTGS"}
            )
        ],
        relationships=[
            GraphRelationship(
                rel_id="rel1",
                rel_type="DEPENDS_ON",
                source_id="entity_neft",
                target_id="entity_core_banking",
                properties={}
            ),
            GraphRelationship(
                rel_id="rel2",
                rel_type="INTEGRATES_WITH",
                source_id="entity_neft",
                target_id="entity_rtgs",
                properties={}
            )
        ],
        chunks=[
            {
                "chunk_id": "chunk3",
                "text": "NEFT system depends on Core Banking System for account validation and transaction processing.",
                "breadcrumbs": "System Architecture > Dependencies",
                "doc_id": "doc_architecture",
                "section": "System Dependencies"
            }
        ]
    )
    
    fused2 = fusion.fuse(vector_results_empty, graph_results)
    
    print(f"Vector chunks: {len(vector_results_empty)}")
    print(f"Graph chunks: {len(graph_results.chunks)}")
    print(f"Fused chunks: {len(fused2.chunks)}")
    print(f"Graph facts: {len(fused2.graph_facts)}")
    print()
    
    print("Graph facts extracted:")
    for i, fact in enumerate(fused2.graph_facts, 1):
        print(f"  {i}. {fact}")
    print()
    
    print("Chunks:")
    for i, chunk in enumerate(fused2.chunks, 1):
        print(f"  {i}. {chunk.chunk_id} (combined score: {fused2.combined_score[chunk.chunk_id]:.3f})")
        print(f"     {chunk.text[:80]}...")
    print()
    
    # Example 3: Hybrid results with deduplication
    print("Example 3: Fusing hybrid results with deduplication")
    print("-" * 80)
    
    vector_results_hybrid = [
        RetrievedChunk(
            chunk_id="chunk1",
            text="NEFT enables electronic fund transfers between banks across India.",
            breadcrumbs="Payment Systems > NEFT",
            doc_id="doc1",
            section="NEFT",
            score=0.90,
            retrieval_source="vector",
            vector_score=0.90,
            bm25_score=0.85
        ),
        RetrievedChunk(
            chunk_id="chunk2",
            text="RTGS is used for high-value transactions requiring immediate settlement.",
            breadcrumbs="Payment Systems > RTGS",
            doc_id="doc1",
            section="RTGS",
            score=0.85,
            retrieval_source="vector",
            vector_score=0.85,
            bm25_score=0.80
        )
    ]
    
    graph_results_hybrid = GraphResult(
        nodes=[
            GraphNode(
                node_id="entity_neft",
                node_type="System",
                properties={"name": "NEFT", "canonical_name": "NEFT"}
            ),
            GraphNode(
                node_id="entity_rtgs",
                node_type="System",
                properties={"name": "RTGS", "canonical_name": "RTGS"}
            ),
            GraphNode(
                node_id="entity_rule1",
                node_type="Rule",
                properties={"name": "Transaction Limit Rule A"}
            ),
            GraphNode(
                node_id="entity_rule2",
                node_type="Rule",
                properties={"name": "Transaction Limit Rule B"}
            )
        ],
        relationships=[
            GraphRelationship(
                rel_id="rel1",
                rel_type="USES",
                source_id="entity_neft",
                target_id="entity_rtgs",
                properties={}
            ),
            GraphRelationship(
                rel_id="rel2",
                rel_type="CONFLICTS_WITH",
                source_id="entity_rule1",
                target_id="entity_rule2",
                properties={
                    "conflict_type": "property",
                    "explanation": "different transaction limits"
                }
            )
        ],
        chunks=[
            {
                "chunk_id": "chunk1",  # Duplicate - should be deduplicated
                "text": "NEFT enables electronic fund transfers between banks across India.",
                "breadcrumbs": "Payment Systems > NEFT",
                "doc_id": "doc1",
                "section": "NEFT"
            },
            {
                "chunk_id": "chunk3",  # New chunk from graph
                "text": "Transaction limits vary between NEFT and RTGS systems.",
                "breadcrumbs": "Rules > Transaction Limits",
                "doc_id": "doc2",
                "section": "Limits"
            }
        ]
    )
    
    fused3 = fusion.fuse(vector_results_hybrid, graph_results_hybrid)
    
    print(f"Vector chunks: {len(vector_results_hybrid)}")
    print(f"Graph chunks: {len(graph_results_hybrid.chunks)}")
    print(f"Fused chunks (after deduplication): {len(fused3.chunks)}")
    print(f"Graph facts: {len(fused3.graph_facts)}")
    print()
    
    print("Graph facts extracted:")
    for i, fact in enumerate(fused3.graph_facts, 1):
        print(f"  {i}. {fact}")
    print()
    
    print("Deduplicated chunks (sorted by combined score):")
    for i, chunk in enumerate(fused3.chunks, 1):
        vector_score = chunk.score if chunk.retrieval_source == "vector" else 0.0
        combined = fused3.combined_score[chunk.chunk_id]
        print(f"  {i}. {chunk.chunk_id}")
        print(f"     Source: {chunk.retrieval_source}")
        print(f"     Vector score: {vector_score:.3f}, Combined score: {combined:.3f}")
        print(f"     {chunk.text[:80]}...")
        print()
    
    # Example 4: Custom weights
    print("Example 4: Custom fusion weights (0.7 vector, 0.3 graph)")
    print("-" * 80)
    
    fusion_custom = ResultFusion(vector_weight=0.7, graph_weight=0.3)
    
    fused4 = fusion_custom.fuse(vector_results_hybrid, graph_results_hybrid)
    
    print("Comparing scores with different weights:")
    print()
    print(f"{'Chunk ID':<10} {'Default (0.6/0.4)':<20} {'Custom (0.7/0.3)':<20}")
    print("-" * 50)
    
    for chunk_id in fused3.combined_score.keys():
        default_score = fused3.combined_score[chunk_id]
        custom_score = fused4.combined_score[chunk_id]
        print(f"{chunk_id:<10} {default_score:<20.3f} {custom_score:<20.3f}")
    
    print()
    print("=" * 80)
    print("Result Fusion Example Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
