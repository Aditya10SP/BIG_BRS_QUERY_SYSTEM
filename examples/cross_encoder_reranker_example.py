"""Example usage of CrossEncoderReranker for reranking retrieved results."""

import logging
from src.retrieval.cross_encoder_reranker import CrossEncoderReranker
from src.retrieval.vector_retriever import RetrievedChunk
from src.retrieval.result_fusion import FusedResults

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate CrossEncoderReranker usage."""
    
    print("CrossEncoderReranker Example")
    print("=" * 50)
    
    # Create sample retrieved chunks (simulating results from vector/graph retrieval)
    sample_chunks = [
        RetrievedChunk(
            chunk_id="chunk1",
            text="NEFT (National Electronic Funds Transfer) is a payment system that enables electronic transfer of funds between banks in India. It operates in hourly batches and is suitable for non-urgent transactions.",
            breadcrumbs="Banking Systems > Payment Methods > NEFT",
            doc_id="banking_doc_1",
            section="payment_methods",
            score=0.75,  # Original retrieval score
            retrieval_source="vector",
            vector_score=0.75,
            bm25_score=0.60
        ),
        RetrievedChunk(
            chunk_id="chunk2",
            text="RTGS (Real Time Gross Settlement) is used for high-value transactions above 2 lakhs and provides real-time settlement. It operates during business hours and ensures immediate fund transfer.",
            breadcrumbs="Banking Systems > Payment Methods > RTGS", 
            doc_id="banking_doc_1",
            section="payment_methods",
            score=0.65,
            retrieval_source="vector",
            vector_score=0.65,
            bm25_score=0.55
        ),
        RetrievedChunk(
            chunk_id="chunk3",
            text="UPI (Unified Payments Interface) enables instant money transfer between bank accounts through mobile applications. It supports 24x7 operations and handles both P2P and P2M transactions.",
            breadcrumbs="Banking Systems > Payment Methods > UPI",
            doc_id="banking_doc_1",
            section="payment_methods", 
            score=0.70,
            retrieval_source="hybrid",
            vector_score=0.60,
            bm25_score=0.80
        ),
        RetrievedChunk(
            chunk_id="chunk4",
            text="Core Banking System (CBS) manages customer accounts, loan processing, and transaction history. It provides centralized banking operations and integrates with various payment systems.",
            breadcrumbs="Banking Systems > Core Systems > CBS",
            doc_id="banking_doc_2",
            section="core_systems",
            score=0.55,
            retrieval_source="vector", 
            vector_score=0.55,
            bm25_score=0.40
        ),
        RetrievedChunk(
            chunk_id="chunk5",
            text="IMPS (Immediate Payment Service) allows instant interbank electronic fund transfers. It operates 24x7 including holidays and weekends, making it suitable for urgent transactions.",
            breadcrumbs="Banking Systems > Payment Methods > IMPS",
            doc_id="banking_doc_1", 
            section="payment_methods",
            score=0.60,
            retrieval_source="bm25",
            vector_score=0.45,
            bm25_score=0.75
        )
    ]
    
    # Create FusedResults object
    fused_results = FusedResults(
        chunks=sample_chunks,
        graph_facts=[
            "NEFT DEPENDS_ON Core Banking System",
            "RTGS INTEGRATES_WITH Core Banking System", 
            "UPI USES Core Banking System"
        ],
        combined_score={chunk.chunk_id: chunk.score for chunk in sample_chunks}
    )
    
    print(f"Initial retrieval results ({len(sample_chunks)} chunks):")
    print("-" * 50)
    for i, chunk in enumerate(sample_chunks, 1):
        print(f"{i}. [{chunk.chunk_id}] Score: {chunk.score:.3f}")
        print(f"   Text: {chunk.text[:100]}...")
        print(f"   Source: {chunk.retrieval_source}")
        print()
    
    # Test queries
    test_queries = [
        "What is NEFT and how does it work?",
        "Tell me about real-time payment systems", 
        "How do mobile payment systems work?",
        "What are the core banking operations?"
    ]
    
    try:
        # Initialize CrossEncoderReranker
        print("Initializing CrossEncoderReranker...")
        print("Note: This will download the cross-encoder model on first run")
        
        reranker = CrossEncoderReranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            max_length=512
        )
        
        print("✓ CrossEncoderReranker initialized successfully")
        print()
        
        # Test reranking for each query
        for query_idx, query in enumerate(test_queries, 1):
            print(f"Query {query_idx}: {query}")
            print("-" * 50)
            
            # Rerank results
            reranked_chunks = reranker.rerank(
                query=query,
                results=fused_results,
                top_k=3  # Return top 3 results
            )
            
            print(f"Reranked results (top {len(reranked_chunks)}):")
            for i, chunk in enumerate(reranked_chunks, 1):
                print(f"{i}. [{chunk.chunk_id}] Cross-encoder score: {chunk.score:.3f}")
                print(f"   Original score: {fused_results.combined_score[chunk.chunk_id]:.3f}")
                print(f"   Text: {chunk.text[:100]}...")
                print(f"   Improvement: {chunk.score - fused_results.combined_score[chunk.chunk_id]:+.3f}")
                print()
            
            print("=" * 70)
            print()
    
    except Exception as e:
        print(f"Error during reranking: {str(e)}")
        print("This might be due to model download issues or insufficient resources.")
        print("In production, you would handle this gracefully with fallback scoring.")
        
        # Demonstrate fallback behavior
        print("\nDemonstrating fallback behavior (using original scores):")
        for query in test_queries[:1]:  # Just show one example
            print(f"Query: {query}")
            print("Fallback results (original ranking):")
            
            # Sort by original scores
            sorted_chunks = sorted(sample_chunks, key=lambda x: x.score, reverse=True)[:3]
            
            for i, chunk in enumerate(sorted_chunks, 1):
                print(f"{i}. [{chunk.chunk_id}] Score: {chunk.score:.3f}")
                print(f"   Text: {chunk.text[:100]}...")
                print()


if __name__ == "__main__":
    main()