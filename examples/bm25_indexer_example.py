"""Example demonstrating BM25Indexer usage for keyword-based search."""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.indexing.bm25_indexer import BM25Indexer
from src.chunking.hierarchical_chunker import Chunk


def main():
    """Demonstrate BM25Indexer functionality."""
    
    print("=" * 80)
    print("BM25 Indexer Example - Keyword-Based Search for Banking Documents")
    print("=" * 80)
    print()
    
    # Create sample banking document chunks
    chunks = [
        Chunk(
            chunk_id="chunk_1",
            doc_id="banking_doc_1",
            text=(
                "NEFT (National Electronic Funds Transfer) is a nationwide payment system "
                "facilitating one-to-one funds transfer. NEFT operates in hourly batches "
                "and is available 24x7 throughout the year."
            ),
            chunk_type="child",
            parent_chunk_id="parent_1",
            breadcrumbs="Banking Systems > Payment Methods > NEFT",
            section="NEFT Payment System",
            token_count=45,
            metadata={"page": 1}
        ),
        Chunk(
            chunk_id="chunk_2",
            doc_id="banking_doc_1",
            text=(
                "The transaction limit for NEFT is 2 lakhs per transaction. "
                "There is no maximum limit on the number of transactions per day."
            ),
            chunk_type="child",
            parent_chunk_id="parent_1",
            breadcrumbs="Banking Systems > Payment Methods > NEFT",
            section="NEFT Transaction Limits",
            token_count=28,
            metadata={"page": 1}
        ),
        Chunk(
            chunk_id="chunk_3",
            doc_id="banking_doc_1",
            text=(
                "RTGS (Real Time Gross Settlement) is used for high-value transactions. "
                "RTGS transactions are processed in real-time on a continuous basis. "
                "The minimum transaction amount for RTGS is 2 lakhs."
            ),
            chunk_type="child",
            parent_chunk_id="parent_2",
            breadcrumbs="Banking Systems > Payment Methods > RTGS",
            section="RTGS Payment System",
            token_count=42,
            metadata={"page": 2}
        ),
        Chunk(
            chunk_id="chunk_4",
            doc_id="banking_doc_2",
            text=(
                "IMPS (Immediate Payment Service) provides instant interbank electronic fund transfer. "
                "IMPS is available 24x7 including bank holidays. "
                "IMPS supports both mobile and internet banking channels."
            ),
            chunk_type="child",
            parent_chunk_id="parent_3",
            breadcrumbs="Banking Systems > Instant Payments > IMPS",
            section="IMPS Payment System",
            token_count=38,
            metadata={"page": 3}
        ),
        Chunk(
            chunk_id="chunk_5",
            doc_id="banking_doc_2",
            text=(
                "UPI (Unified Payments Interface) is a real-time payment system developed by NPCI. "
                "UPI allows instant money transfer between bank accounts using mobile phones. "
                "UPI supports multiple bank accounts in a single mobile application."
            ),
            chunk_type="child",
            parent_chunk_id="parent_4",
            breadcrumbs="Banking Systems > Instant Payments > UPI",
            section="UPI Payment System",
            token_count=42,
            metadata={"page": 4}
        ),
    ]
    
    print(f"Created {len(chunks)} sample banking document chunks")
    print()
    
    # Initialize BM25 indexer
    print("Initializing BM25 Indexer...")
    indexer = BM25Indexer(k1=1.5, b=0.75)
    print(f"  - k1 (term frequency saturation): {indexer.k1}")
    print(f"  - b (length normalization): {indexer.b}")
    print()
    
    # Build index
    print("Building BM25 index from chunks...")
    indexer.index(chunks)
    print(f"  - Index size: {indexer.get_index_size()} documents")
    print()
    
    # Example 1: Search for specific acronym
    print("-" * 80)
    print("Example 1: Search for 'NEFT' acronym")
    print("-" * 80)
    query = "NEFT"
    results = indexer.search(query, top_k=3)
    
    print(f"Query: '{query}'")
    print(f"Found {len(results)} results:\n")
    
    for i, (chunk_id, score) in enumerate(results, 1):
        chunk = next(c for c in chunks if c.chunk_id == chunk_id)
        print(f"{i}. Chunk ID: {chunk_id}")
        print(f"   Score: {score:.4f}")
        print(f"   Section: {chunk.section}")
        print(f"   Text: {chunk.text[:100]}...")
        print()
    
    # Example 2: Search for transaction limits
    print("-" * 80)
    print("Example 2: Search for 'transaction limit'")
    print("-" * 80)
    query = "transaction limit"
    results = indexer.search(query, top_k=3)
    
    print(f"Query: '{query}'")
    print(f"Found {len(results)} results:\n")
    
    for i, (chunk_id, score) in enumerate(results, 1):
        chunk = next(c for c in chunks if c.chunk_id == chunk_id)
        print(f"{i}. Chunk ID: {chunk_id}")
        print(f"   Score: {score:.4f}")
        print(f"   Section: {chunk.section}")
        print(f"   Text: {chunk.text[:100]}...")
        print()
    
    # Example 3: Search with multiple acronyms
    print("-" * 80)
    print("Example 3: Search for multiple payment systems")
    print("-" * 80)
    query = "NEFT RTGS IMPS"
    results = indexer.search(query, top_k=5)
    
    print(f"Query: '{query}'")
    print(f"Found {len(results)} results:\n")
    
    for i, (chunk_id, score) in enumerate(results, 1):
        chunk = next(c for c in chunks if c.chunk_id == chunk_id)
        print(f"{i}. Chunk ID: {chunk_id}")
        print(f"   Score: {score:.4f}")
        print(f"   Section: {chunk.section}")
        print(f"   Breadcrumbs: {chunk.breadcrumbs}")
        print()
    
    # Example 4: Search with score threshold
    print("-" * 80)
    print("Example 4: Search with score threshold")
    print("-" * 80)
    query = "real-time payment"
    threshold = 2.0
    results = indexer.search(query, top_k=5, score_threshold=threshold)
    
    print(f"Query: '{query}'")
    print(f"Score threshold: {threshold}")
    print(f"Found {len(results)} results above threshold:\n")
    
    for i, (chunk_id, score) in enumerate(results, 1):
        chunk = next(c for c in chunks if c.chunk_id == chunk_id)
        print(f"{i}. Chunk ID: {chunk_id}")
        print(f"   Score: {score:.4f}")
        print(f"   Section: {chunk.section}")
        print(f"   Text: {chunk.text[:100]}...")
        print()
    
    # Example 5: Demonstrate acronym preservation
    print("-" * 80)
    print("Example 5: Acronym preservation in tokenization")
    print("-" * 80)
    
    test_texts = [
        "NEFT and RTGS are payment systems",
        "The ISO20022 standard is used",
        "UPI provides instant transfers",
        "Real-time payment processing"
    ]
    
    print("Tokenization examples:\n")
    for text in test_texts:
        tokens = indexer._tokenize(text)
        print(f"Text: '{text}'")
        print(f"Tokens: {tokens}")
        print()
    
    print("=" * 80)
    print("BM25 Indexer Example Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
