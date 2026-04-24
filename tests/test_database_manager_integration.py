"""Integration tests for DatabaseManager with real PostgreSQL"""

import pytest
import os
from src.storage.database_manager import DatabaseManager


@pytest.fixture(scope="module")
def db_connection_string():
    """Get database connection string from environment."""
    return os.getenv(
        "POSTGRES_CONNECTION_STRING",
        "postgresql://postgres:postgres@localhost:5432/graph_rag"
    )


@pytest.fixture
def db_manager(db_connection_string):
    """Create a DatabaseManager instance with real database."""
    manager = DatabaseManager(db_connection_string)
    manager.initialize()
    
    yield manager
    
    # Cleanup
    with manager.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chunks")
            cur.execute("DELETE FROM documents")
            conn.commit()
    
    manager.close()


def test_end_to_end_document_workflow(db_manager):
    """Test complete workflow: create document, add chunks, query, update, delete."""
    # Create document
    db_manager.create_document(
        doc_id="integration_doc_1",
        title="Integration Test Document",
        file_path="/test/path/doc.pdf",
        file_type="pdf",
        metadata={"test": True, "version": 1}
    )
    
    # Verify document exists
    doc = db_manager.get_document_by_id("integration_doc_1")
    assert doc is not None
    assert doc["title"] == "Integration Test Document"
    
    # Create parent chunk
    db_manager.create_chunk(
        chunk_id="parent_chunk_1",
        doc_id="integration_doc_1",
        text="This is a parent chunk containing section-level content.",
        chunk_type="parent",
        breadcrumbs="Integration Test Document > Section 1",
        section="Section 1",
        token_count=50,
        metadata={"level": 1}
    )
    
    # Create child chunks
    for i in range(3):
        db_manager.create_chunk(
            chunk_id=f"child_chunk_{i+1}",
            doc_id="integration_doc_1",
            text=f"This is child chunk {i+1} with detailed content.",
            chunk_type="child",
            parent_chunk_id="parent_chunk_1",
            breadcrumbs=f"Integration Test Document > Section 1 > Subsection {i+1}",
            section="Section 1",
            token_count=20 + i,
            metadata={"level": 2, "index": i}
        )
    
    # Query all chunks for document
    chunks = db_manager.get_chunks_by_doc_id("integration_doc_1")
    assert len(chunks) == 4  # 1 parent + 3 children
    
    # Query chunks by section
    section_chunks = db_manager.get_chunks_by_section("integration_doc_1", "Section 1")
    assert len(section_chunks) == 4
    
    # Update a chunk
    updated = db_manager.update_chunk(
        "child_chunk_1",
        text="Updated child chunk text",
        metadata={"level": 2, "index": 0, "updated": True}
    )
    assert updated is True
    
    # Verify update
    chunk = db_manager.get_chunk_by_id("child_chunk_1")
    assert chunk["text"] == "Updated child chunk text"
    assert chunk["metadata"]["updated"] is True
    
    # Delete document (should cascade to chunks)
    deleted = db_manager.delete_document("integration_doc_1")
    assert deleted is True
    
    # Verify everything is deleted
    doc = db_manager.get_document_by_id("integration_doc_1")
    assert doc is None
    
    chunks = db_manager.get_chunks_by_doc_id("integration_doc_1")
    assert len(chunks) == 0


def test_multiple_documents_isolation(db_manager):
    """Test that operations on one document don't affect others."""
    # Create two documents
    db_manager.create_document(doc_id="doc_a", title="Document A")
    db_manager.create_document(doc_id="doc_b", title="Document B")
    
    # Add chunks to each
    db_manager.create_chunk(
        chunk_id="chunk_a1",
        doc_id="doc_a",
        text="Chunk from document A",
        chunk_type="child",
        section="Section A"
    )
    db_manager.create_chunk(
        chunk_id="chunk_b1",
        doc_id="doc_b",
        text="Chunk from document B",
        chunk_type="child",
        section="Section B"
    )
    
    # Verify isolation
    chunks_a = db_manager.get_chunks_by_doc_id("doc_a")
    chunks_b = db_manager.get_chunks_by_doc_id("doc_b")
    
    assert len(chunks_a) == 1
    assert len(chunks_b) == 1
    assert chunks_a[0]["chunk_id"] == "chunk_a1"
    assert chunks_b[0]["chunk_id"] == "chunk_b1"
    
    # Delete one document
    db_manager.delete_document("doc_a")
    
    # Verify other document is unaffected
    doc_b = db_manager.get_document_by_id("doc_b")
    assert doc_b is not None
    
    chunks_b = db_manager.get_chunks_by_doc_id("doc_b")
    assert len(chunks_b) == 1
    
    # Cleanup
    db_manager.delete_document("doc_b")


def test_large_text_storage(db_manager):
    """Test storing and retrieving large text chunks."""
    # Create document
    db_manager.create_document(doc_id="large_doc", title="Large Document")
    
    # Create chunk with large text (simulate 2048 token parent chunk)
    large_text = "This is a test sentence. " * 200  # ~1000 words
    
    db_manager.create_chunk(
        chunk_id="large_chunk",
        doc_id="large_doc",
        text=large_text,
        chunk_type="parent",
        token_count=2048
    )
    
    # Retrieve and verify
    chunk = db_manager.get_chunk_by_id("large_chunk")
    assert chunk is not None
    assert chunk["text"] == large_text
    assert chunk["token_count"] == 2048
    
    # Cleanup
    db_manager.delete_document("large_doc")


def test_special_characters_in_text(db_manager):
    """Test handling of special characters and unicode."""
    # Create document
    db_manager.create_document(doc_id="special_doc", title="Special Characters Test")
    
    # Text with special characters
    special_text = """
    This text contains special characters:
    - Quotes: "double" and 'single'
    - Unicode: café, naïve, 日本語
    - Symbols: @#$%^&*()
    - Newlines and tabs
    """
    
    db_manager.create_chunk(
        chunk_id="special_chunk",
        doc_id="special_doc",
        text=special_text,
        chunk_type="child"
    )
    
    # Retrieve and verify
    chunk = db_manager.get_chunk_by_id("special_chunk")
    assert chunk is not None
    assert chunk["text"] == special_text
    
    # Cleanup
    db_manager.delete_document("special_doc")


def test_json_metadata_complex(db_manager):
    """Test storing complex JSON metadata."""
    # Create document with complex metadata
    complex_metadata = {
        "authors": ["Author 1", "Author 2"],
        "tags": ["banking", "fsd", "neft"],
        "version": 2.5,
        "nested": {
            "key1": "value1",
            "key2": [1, 2, 3]
        },
        "boolean": True,
        "null_value": None
    }
    
    db_manager.create_document(
        doc_id="complex_doc",
        title="Complex Metadata Document",
        metadata=complex_metadata
    )
    
    # Retrieve and verify
    doc = db_manager.get_document_by_id("complex_doc")
    assert doc is not None
    assert doc["metadata"]["authors"] == ["Author 1", "Author 2"]
    assert doc["metadata"]["nested"]["key2"] == [1, 2, 3]
    assert doc["metadata"]["boolean"] is True
    
    # Cleanup
    db_manager.delete_document("complex_doc")


def test_concurrent_operations(db_manager):
    """Test that connection pooling handles concurrent operations."""
    import concurrent.futures
    
    # Create base document
    db_manager.create_document(doc_id="concurrent_doc", title="Concurrent Test")
    
    def create_chunk(i):
        db_manager.create_chunk(
            chunk_id=f"concurrent_chunk_{i}",
            doc_id="concurrent_doc",
            text=f"Chunk {i}",
            chunk_type="child"
        )
        return i
    
    # Create chunks concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(create_chunk, i) for i in range(10)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    assert len(results) == 10
    
    # Verify all chunks were created
    chunks = db_manager.get_chunks_by_doc_id("concurrent_doc")
    assert len(chunks) == 10
    
    # Cleanup
    db_manager.delete_document("concurrent_doc")
