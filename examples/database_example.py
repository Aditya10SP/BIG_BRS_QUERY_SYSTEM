"""Example usage of DatabaseManager for storing documents and chunks"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.database_manager import DatabaseManager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def main():
    """Demonstrate DatabaseManager usage."""
    
    # Get connection string from environment
    connection_string = os.getenv(
        "POSTGRES_CONNECTION_STRING",
        "postgresql://postgres:postgres@localhost:5432/graph_rag"
    )
    
    print("Initializing DatabaseManager...")
    db_manager = DatabaseManager(connection_string)
    db_manager.initialize()
    
    try:
        # Create a sample document
        print("\n1. Creating document...")
        db_manager.create_document(
            doc_id="example_doc_1",
            title="Banking Functional Specification Document",
            file_path="/docs/banking_fsd.pdf",
            file_type="pdf",
            metadata={
                "version": "2.0",
                "author": "Banking Team",
                "department": "IT",
                "tags": ["banking", "payments", "neft"]
            }
        )
        print("   ✓ Document created: example_doc_1")
        
        # Create parent chunk
        print("\n2. Creating parent chunk...")
        db_manager.create_chunk(
            chunk_id="parent_chunk_1",
            doc_id="example_doc_1",
            text="""
            Payment Systems Overview
            
            This section describes the various payment systems supported by the bank,
            including NEFT, RTGS, IMPS, and UPI. Each system has specific use cases,
            transaction limits, and processing times.
            """,
            chunk_type="parent",
            breadcrumbs="Banking FSD > Payment Systems",
            section="Payment Systems",
            token_count=256,
            metadata={"page": 10, "level": 1}
        )
        print("   ✓ Parent chunk created: parent_chunk_1")
        
        # Create child chunks
        print("\n3. Creating child chunks...")
        child_chunks = [
            {
                "chunk_id": "child_chunk_1",
                "text": "NEFT (National Electronic Funds Transfer) is a nationwide payment system "
                        "facilitating one-to-one funds transfer. Transaction limit: Rs 2 lakhs per transaction.",
                "breadcrumbs": "Banking FSD > Payment Systems > NEFT",
                "metadata": {"page": 11, "level": 2, "system": "NEFT"}
            },
            {
                "chunk_id": "child_chunk_2",
                "text": "RTGS (Real Time Gross Settlement) is used for high-value transactions. "
                        "Minimum amount: Rs 2 lakhs. Transactions are processed in real-time.",
                "breadcrumbs": "Banking FSD > Payment Systems > RTGS",
                "metadata": {"page": 12, "level": 2, "system": "RTGS"}
            },
            {
                "chunk_id": "child_chunk_3",
                "text": "IMPS (Immediate Payment Service) enables instant interbank electronic fund transfer. "
                        "Available 24x7 including holidays. Transaction limit: Rs 5 lakhs.",
                "breadcrumbs": "Banking FSD > Payment Systems > IMPS",
                "metadata": {"page": 13, "level": 2, "system": "IMPS"}
            }
        ]
        
        for chunk_data in child_chunks:
            db_manager.create_chunk(
                chunk_id=chunk_data["chunk_id"],
                doc_id="example_doc_1",
                text=chunk_data["text"],
                chunk_type="child",
                parent_chunk_id="parent_chunk_1",
                breadcrumbs=chunk_data["breadcrumbs"],
                section="Payment Systems",
                token_count=len(chunk_data["text"].split()),
                metadata=chunk_data["metadata"]
            )
            print(f"   ✓ Child chunk created: {chunk_data['chunk_id']}")
        
        # Query document
        print("\n4. Querying document...")
        doc = db_manager.get_document_by_id("example_doc_1")
        print(f"   Document: {doc['title']}")
        print(f"   File: {doc['file_path']}")
        print(f"   Tags: {doc['metadata']['tags']}")
        
        # Query all chunks for document
        print("\n5. Querying all chunks for document...")
        chunks = db_manager.get_chunks_by_doc_id("example_doc_1")
        print(f"   Total chunks: {len(chunks)}")
        for chunk in chunks:
            print(f"   - {chunk['chunk_id']} ({chunk['chunk_type']}): {chunk['text'][:50]}...")
        
        # Query chunks by section
        print("\n6. Querying chunks by section...")
        section_chunks = db_manager.get_chunks_by_section("example_doc_1", "Payment Systems")
        print(f"   Chunks in 'Payment Systems' section: {len(section_chunks)}")
        
        # Update a chunk
        print("\n7. Updating chunk...")
        db_manager.update_chunk(
            "child_chunk_1",
            text="NEFT (National Electronic Funds Transfer) is a nationwide payment system "
                 "facilitating one-to-one funds transfer. UPDATED: Transaction limit increased to Rs 5 lakhs.",
            metadata={"page": 11, "level": 2, "system": "NEFT", "updated": True}
        )
        print("   ✓ Chunk updated: child_chunk_1")
        
        # Verify update
        updated_chunk = db_manager.get_chunk_by_id("child_chunk_1")
        print(f"   Updated text: {updated_chunk['text'][:80]}...")
        
        # Cleanup
        print("\n8. Cleaning up...")
        db_manager.delete_document("example_doc_1")
        print("   ✓ Document and all chunks deleted")
        
        print("\n✅ Example completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise
    
    finally:
        # Close database connections
        db_manager.close()
        print("\nDatabase connections closed.")


if __name__ == "__main__":
    main()
