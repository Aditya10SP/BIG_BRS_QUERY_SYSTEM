"""Example usage of ContextAssembler for creating structured context with citations."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval.context_assembler import ContextAssembler
from src.retrieval.vector_retriever import RetrievedChunk


def main():
    """Demonstrate ContextAssembler functionality."""
    print("=== ContextAssembler Example ===\n")
    
    # Initialize ContextAssembler
    assembler = ContextAssembler(max_tokens=2000)
    
    # Sample retrieved chunks (would come from vector/graph retrieval)
    chunks = [
        RetrievedChunk(
            chunk_id="chunk_neft_1",
            text="NEFT (National Electronic Funds Transfer) is a nationwide payment system that enables bank customers to transfer funds from any bank branch to any individual having an account with any other bank branch in the country. NEFT operates on a Deferred Net Settlement (DNS) basis which settles transactions in batches.",
            breadcrumbs="Banking Systems > Payment Systems > NEFT > Overview",
            doc_id="banking_systems_v2",
            section="Payment Systems",
            score=0.95,
            retrieval_source="vector"
        ),
        RetrievedChunk(
            chunk_id="chunk_neft_limits",
            text="Transaction limits for NEFT are as follows: For retail customers, the maximum limit is Rs. 10 lakhs per transaction. For corporate customers, there is no upper limit. The minimum amount for NEFT transactions is Re. 1.",
            breadcrumbs="Banking Rules > Transaction Limits > NEFT Limits",
            doc_id="banking_rules_v3",
            section="Transaction Limits",
            score=0.88,
            retrieval_source="bm25"
        ),
        RetrievedChunk(
            chunk_id="chunk_rtgs_comparison",
            text="RTGS (Real Time Gross Settlement) differs from NEFT in that RTGS processes transactions in real-time on a gross settlement basis, while NEFT uses batch processing. RTGS has a minimum transaction limit of Rs. 2 lakhs, making it suitable for high-value transactions.",
            breadcrumbs="Banking Systems > Payment Systems > RTGS > Comparison",
            doc_id="banking_systems_v2",
            section="Payment Systems",
            score=0.82,
            retrieval_source="vector"
        )
    ]
    
    # Sample graph facts (would come from graph retrieval)
    graph_facts = [
        "NEFT DEPENDS_ON Core Banking System",
        "NEFT INTEGRATES_WITH RTGS",
        "Transaction Limit Rule APPLIES_TO NEFT",
        "NEFT USES Deferred Net Settlement"
    ]
    
    # User query
    query = "What is NEFT and what are its transaction limits compared to RTGS?"
    
    print(f"Query: {query}\n")
    print(f"Input: {len(chunks)} chunks, {len(graph_facts)} graph facts\n")
    
    # Assemble context
    context = assembler.assemble(query, chunks, graph_facts)
    
    print("=== Assembled Context ===")
    print(f"Token count: {context.token_count}")
    print(f"Citations: {len(context.citations)}")
    print()
    print(context.context_text)
    
    print("\n=== Citations Details ===")
    for citation_id, citation in context.citations.items():
        print(f"Citation ID: {citation_id}")
        print(f"  Document: {citation.doc_id}")
        print(f"  Section: {citation.section}")
        print(f"  Chunk: {citation.chunk_id}")
        print(f"  Breadcrumbs: {citation.breadcrumbs}")
        print()
    
    # Demonstrate truncation with smaller limit
    print("\n=== Truncation Example ===")
    small_assembler = ContextAssembler(max_tokens=300)
    truncated_context = small_assembler.assemble(query, chunks, graph_facts)
    
    print(f"Original token count: {context.token_count}")
    print(f"Truncated token count: {truncated_context.token_count}")
    print(f"Original citations: {len(context.citations)}")
    print(f"Truncated citations: {len(truncated_context.citations)}")
    print()
    print("Truncated context:")
    print(truncated_context.context_text)


if __name__ == "__main__":
    main()