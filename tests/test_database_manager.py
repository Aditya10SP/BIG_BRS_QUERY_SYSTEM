"""Unit tests for DatabaseManager"""

import pytest
import psycopg2
from psycopg2.extras import RealDictCursor
from src.storage.database_manager import DatabaseManager


@pytest.fixture
def db_manager():
    """Create a DatabaseManager instance for testing."""
    # Use test database connection string
    connection_string = "postgresql://postgres:postgres@localhost:5432/graph_rag_test"
    manager = DatabaseManager(connection_string, min_connections=1, max_connections=2)
    
    try:
        manager.initialize()
        yield manager
    finally:
        # Clean up: drop all tables
        with manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS chunks CASCADE")
                cur.execute("DROP TABLE IF EXISTS documents CASCADE")
                conn.commit()
        manager.close()


@pytest.fixture
def clean_db(db_manager):
    """Ensure database is clean before each test."""
    with db_manager.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chunks")
            cur.execute("DELETE FROM documents")
            conn.commit()
    return db_manager


def test_initialize_creates_schema(db_manager):
    """Test that initialize creates the required tables."""
    with db_manager.get_connection() as conn:
        with conn.cursor() as cur:
            # Check documents table exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'documents'
                )
            """)
            assert cur.fetchone()[0] is True
            
            # Check chunks table exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'chunks'
                )
            """)
            assert cur.fetchone()[0] is True


def test_create_document(clean_db):
    """Test creating a document."""
    clean_db.create_document(
        doc_id="doc1",
        title="Test Document",
        file_path="/path/to/doc.pdf",
        file_type="pdf",
        metadata={"author": "Test Author"}
    )
    
    # Verify document was created
    doc = clean_db.get_document_by_id("doc1")
    assert doc is not None
    assert doc["doc_id"] == "doc1"
    assert doc["title"] == "Test Document"
    assert doc["file_path"] == "/path/to/doc.pdf"
    assert doc["file_type"] == "pdf"
    assert doc["metadata"]["author"] == "Test Author"


def test_create_document_upsert(clean_db):
    """Test that creating a document with same ID updates it."""
    # Create initial document
    clean_db.create_document(
        doc_id="doc1",
        title="Original Title",
        file_path="/path/to/doc.pdf",
        file_type="pdf"
    )
    
    # Create again with different title
    clean_db.create_document(
        doc_id="doc1",
        title="Updated Title",
        file_path="/path/to/doc.pdf",
        file_type="pdf"
    )
    
    # Verify document was updated
    doc = clean_db.get_document_by_id("doc1")
    assert doc["title"] == "Updated Title"


def test_create_chunk(clean_db):
    """Test creating a chunk."""
    # Create parent document first
    clean_db.create_document(
        doc_id="doc1",
        title="Test Document"
    )
    
    # Create chunk
    clean_db.create_chunk(
        chunk_id="chunk1",
        doc_id="doc1",
        text="This is test chunk text.",
        chunk_type="child",
        parent_chunk_id="parent1",
        breadcrumbs="Doc > Section > Subsection",
        section="Section 1",
        token_count=10,
        metadata={"page": 1}
    )
    
    # Verify chunk was created
    chunk = clean_db.get_chunk_by_id("chunk1")
    assert chunk is not None
    assert chunk["chunk_id"] == "chunk1"
    assert chunk["doc_id"] == "doc1"
    assert chunk["text"] == "This is test chunk text."
    assert chunk["chunk_type"] == "child"
    assert chunk["parent_chunk_id"] == "parent1"
    assert chunk["breadcrumbs"] == "Doc > Section > Subsection"
    assert chunk["section"] == "Section 1"
    assert chunk["token_count"] == 10
    assert chunk["metadata"]["page"] == 1


def test_get_chunk_by_id_not_found(clean_db):
    """Test getting a non-existent chunk returns None."""
    chunk = clean_db.get_chunk_by_id("nonexistent")
    assert chunk is None


def test_get_chunks_by_doc_id(clean_db):
    """Test retrieving all chunks for a document."""
    # Create document
    clean_db.create_document(doc_id="doc1", title="Test Document")
    
    # Create multiple chunks
    clean_db.create_chunk(
        chunk_id="chunk1",
        doc_id="doc1",
        text="Chunk 1 text",
        chunk_type="parent",
        section="Section 1"
    )
    clean_db.create_chunk(
        chunk_id="chunk2",
        doc_id="doc1",
        text="Chunk 2 text",
        chunk_type="child",
        section="Section 1"
    )
    clean_db.create_chunk(
        chunk_id="chunk3",
        doc_id="doc1",
        text="Chunk 3 text",
        chunk_type="child",
        section="Section 2"
    )
    
    # Get all chunks for document
    chunks = clean_db.get_chunks_by_doc_id("doc1")
    assert len(chunks) == 3
    assert {c["chunk_id"] for c in chunks} == {"chunk1", "chunk2", "chunk3"}


def test_get_chunks_by_section(clean_db):
    """Test filtering chunks by document and section."""
    # Create document
    clean_db.create_document(doc_id="doc1", title="Test Document")
    
    # Create chunks in different sections
    clean_db.create_chunk(
        chunk_id="chunk1",
        doc_id="doc1",
        text="Chunk 1 text",
        chunk_type="child",
        section="Section 1"
    )
    clean_db.create_chunk(
        chunk_id="chunk2",
        doc_id="doc1",
        text="Chunk 2 text",
        chunk_type="child",
        section="Section 1"
    )
    clean_db.create_chunk(
        chunk_id="chunk3",
        doc_id="doc1",
        text="Chunk 3 text",
        chunk_type="child",
        section="Section 2"
    )
    
    # Get chunks for Section 1
    chunks = clean_db.get_chunks_by_section("doc1", "Section 1")
    assert len(chunks) == 2
    assert {c["chunk_id"] for c in chunks} == {"chunk1", "chunk2"}
    
    # Get chunks for Section 2
    chunks = clean_db.get_chunks_by_section("doc1", "Section 2")
    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == "chunk3"


def test_update_chunk_text(clean_db):
    """Test updating chunk text."""
    # Create document and chunk
    clean_db.create_document(doc_id="doc1", title="Test Document")
    clean_db.create_chunk(
        chunk_id="chunk1",
        doc_id="doc1",
        text="Original text",
        chunk_type="child"
    )
    
    # Update text
    result = clean_db.update_chunk("chunk1", text="Updated text")
    assert result is True
    
    # Verify update
    chunk = clean_db.get_chunk_by_id("chunk1")
    assert chunk["text"] == "Updated text"


def test_update_chunk_metadata(clean_db):
    """Test updating chunk metadata."""
    # Create document and chunk
    clean_db.create_document(doc_id="doc1", title="Test Document")
    clean_db.create_chunk(
        chunk_id="chunk1",
        doc_id="doc1",
        text="Test text",
        chunk_type="child",
        metadata={"version": 1}
    )
    
    # Update metadata
    result = clean_db.update_chunk("chunk1", metadata={"version": 2, "updated": True})
    assert result is True
    
    # Verify update
    chunk = clean_db.get_chunk_by_id("chunk1")
    assert chunk["metadata"]["version"] == 2
    assert chunk["metadata"]["updated"] is True


def test_update_chunk_not_found(clean_db):
    """Test updating non-existent chunk returns False."""
    result = clean_db.update_chunk("nonexistent", text="New text")
    assert result is False


def test_delete_document_cascade(clean_db):
    """Test that deleting a document also deletes its chunks."""
    # Create document and chunks
    clean_db.create_document(doc_id="doc1", title="Test Document")
    clean_db.create_chunk(
        chunk_id="chunk1",
        doc_id="doc1",
        text="Chunk 1",
        chunk_type="child"
    )
    clean_db.create_chunk(
        chunk_id="chunk2",
        doc_id="doc1",
        text="Chunk 2",
        chunk_type="child"
    )
    
    # Delete document
    result = clean_db.delete_document("doc1")
    assert result is True
    
    # Verify document is deleted
    doc = clean_db.get_document_by_id("doc1")
    assert doc is None
    
    # Verify chunks are also deleted (cascade)
    chunk1 = clean_db.get_chunk_by_id("chunk1")
    chunk2 = clean_db.get_chunk_by_id("chunk2")
    assert chunk1 is None
    assert chunk2 is None


def test_delete_document_not_found(clean_db):
    """Test deleting non-existent document returns False."""
    result = clean_db.delete_document("nonexistent")
    assert result is False


def test_connection_pooling(db_manager):
    """Test that connection pooling works correctly."""
    # Get multiple connections
    connections = []
    for _ in range(3):
        with db_manager.get_connection() as conn:
            connections.append(id(conn))
    
    # Connections should be reused from pool
    assert len(set(connections)) <= db_manager.max_connections


def test_foreign_key_constraint(clean_db):
    """Test that foreign key constraint prevents orphan chunks."""
    # Try to create chunk without parent document
    with pytest.raises(psycopg2.IntegrityError):
        clean_db.create_chunk(
            chunk_id="chunk1",
            doc_id="nonexistent_doc",
            text="Orphan chunk",
            chunk_type="child"
        )


def test_mask_password():
    """Test password masking in connection strings."""
    conn_str = "postgresql://user:secret123@localhost:5432/db"
    masked = DatabaseManager._mask_password(conn_str)
    assert "secret123" not in masked
    assert "****" in masked
    assert "user" in masked
    assert "localhost" in masked
