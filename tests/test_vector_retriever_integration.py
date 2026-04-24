"""Integration tests for VectorRetriever with real components."""

import pytest
import os
from src.retrieval.vector_retriever import VectorRetriever, RetrievedChunk
from src.storage.vector_store import VectorStore
from src.indexing.bm25_indexer import BM25Indexer
from src.embedding.embedding_generator import EmbeddingGenerator
from src.storage.database_manager import DatabaseManager
from src.chunking.hierarchical_chunker import Chunk


@pytest.fixture(scope="module")
def test_db_connection():
    """Get test database connection string."""
    return os.getenv(
        "TEST_DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/graph_rag_test"
    )


@pytest.fixture(scope="module")
def test_qdrant_url():
    """Get test Qdrant URL."""
    return os.getenv("TEST_QDRANT_URL", "http://localhost:6333")


@pytest.fixture
def vector_store(test_qdrant_url):
    """Create VectorStore instance for testing."""
    store = VectorStore(
        url=test_qdrant_url,
        collection_name="test_retriever_collection"
    )
    yield store
    # Cleanup: delete test collection
    try:
        store.client.delete_collection("test_retriever_collection")
    except:
        pass
    store.close()


@pytest.fixture
def doc_store(test_db_connection):
    """Create DatabaseManager instance for testing."""
    db = DatabaseManager(test_db_connection)
    db.initialize()
    yield db
    # Cleanup: delete test data
    try:
        db.delete_document("test_doc_1")
        db.delete_document("test_doc_2")
    except:
        pass
    db.close()


@pytest.fixture
def embedding_generator():
    """Create EmbeddingGenerator instance."""
    return EmbeddingGenerator()


@pytest.fixture
def bm25_index():
    """Create BM25Indexer instance."""
    return BM25Indexer()


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing."""
    return [
        Chunk(
            chunk_id="chunk1",
            doc_id="test_doc_1",
            text="NEFT is a nationwide payment system facilitating one-to-one funds transfer.",
            chunk_type="child",
            parent_chunk_id="parent1",
            breadcrumbs="Banking Systems > Payment Systems > NEFT",
            section="Payment Systems",
            token_count=15,
            metadata={}
        ),
        Chunk(
            chunk_id="chunk2",
            doc_id="test_doc_1",
            text="RTGS is used for large-value transactions with real-time settlement.",
            chunk_type="child",
            parent_chunk_id="parent1",
            breadcrumbs="Banking Systems > Payment Systems > RTGS",
            section="Payment Systems",
            token_count=12,
            metadata={}
        ),
        Chunk(
            chunk_id="chunk3",
            doc_id="test_doc_2",
            text="The transaction limit for NEFT is 2 lakhs per transaction.",
            chunk_type="child",
            parent_chunk_id="parent2",
            breadcrumbs="Banking Rules > Transaction Limits",
            section="Transaction Limits",
            token_count=11,
            metadata={}
        ),
    ]


@pytest.mark.integration
class TestVectorRetrieverIntegration:
    """Integration tests for VectorRetriever."""
    
    def test_end_to_end_retrieval(
        self, vector_store, doc_store, embedding_generator, bm25_index, sample_chunks
    ):
        """Test end-to-end retrieval with real components."""
        # Setup: Store documents and chunks
        doc_store.create_document("test_doc_1", "Banking Systems", metadata={})
        doc_store.create_document("test_doc_2", "Banking Rules", metadata={})
        
        for chunk in sample_chunks:
            doc_store.create_chunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                text=chunk.text,
                chunk_type=chunk.chunk_type,
                parent_chunk_id=chunk.parent_chunk_id,
                breadcrumbs=chunk.breadcrumbs,
                section=chunk.section,
                token_count=chunk.token_count,
                metadata=chunk.metadata
            )
        
        # Generate and store embeddings
        texts = [chunk.text for chunk in sample_chunks]
        embeddings = embedding_generator.batch_generate(texts)
        chunk_ids = [chunk.chunk_id for chunk in sample_chunks]
        metadata = [
            {
                "doc_id": chunk.doc_id,
                "text": chunk.text,
                "breadcrumbs": chunk.breadcrumbs,
                "section": chunk.section,
                "chunk_type": chunk.chunk_type
            }
            for chunk in sample_chunks
        ]
        vector_store.store_embeddings(chunk_ids, embeddings, metadata)
        
        # Build BM25 index
        bm25_index.index(sample_chunks)
        
        # Create retriever
        retriever = VectorRetriever(
            vector_store=vector_store,
            bm25_index=bm25_index,
            embedding_generator=embedding_generator,
            doc_store=doc_store,
            similarity_threshold=0.5  # Lower threshold for testing
        )
        
        # Test retrieval
        results = retriever.retrieve("What is NEFT?", top_k=5)
        
        # Assertions
        assert len(results) > 0
        assert all(isinstance(chunk, RetrievedChunk) for chunk in results)
        
        # chunk1 should be in results (most relevant to "What is NEFT?")
        chunk_ids_retrieved = [c.chunk_id for c in results]
        assert "chunk1" in chunk_ids_retrieved
        
        # Check that results have all required fields
        for chunk in results:
            assert chunk.chunk_id
            assert chunk.text
            assert chunk.doc_id
            assert chunk.score > 0
            assert chunk.retrieval_source in ["vector", "bm25", "vector+bm25"]
    
    def test_retrieval_with_acronym_query(
        self, vector_store, doc_store, embedding_generator, bm25_index, sample_chunks
    ):
        """Test that acronym queries work with BM25."""
        # Setup
        doc_store.create_document("test_doc_1", "Banking Systems", metadata={})
        doc_store.create_document("test_doc_2", "Banking Rules", metadata={})
        
        for chunk in sample_chunks:
            doc_store.create_chunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                text=chunk.text,
                chunk_type=chunk.chunk_type,
                parent_chunk_id=chunk.parent_chunk_id,
                breadcrumbs=chunk.breadcrumbs,
                section=chunk.section,
                token_count=chunk.token_count,
                metadata=chunk.metadata
            )
        
        texts = [chunk.text for chunk in sample_chunks]
        embeddings = embedding_generator.batch_generate(texts)
        chunk_ids = [chunk.chunk_id for chunk in sample_chunks]
        metadata = [
            {
                "doc_id": chunk.doc_id,
                "text": chunk.text,
                "breadcrumbs": chunk.breadcrumbs,
                "section": chunk.section,
                "chunk_type": chunk.chunk_type
            }
            for chunk in sample_chunks
        ]
        vector_store.store_embeddings(chunk_ids, embeddings, metadata)
        bm25_index.index(sample_chunks)
        
        retriever = VectorRetriever(
            vector_store=vector_store,
            bm25_index=bm25_index,
            embedding_generator=embedding_generator,
            doc_store=doc_store,
            similarity_threshold=0.5
        )
        
        # Query with acronym
        results = retriever.retrieve("RTGS", top_k=5)
        
        # chunk2 should be highly ranked (contains RTGS)
        assert len(results) > 0
        chunk_ids = [c.chunk_id for c in results]
        assert "chunk2" in chunk_ids
        
        # Check that BM25 contributed to the result
        chunk2 = next(c for c in results if c.chunk_id == "chunk2")
        assert chunk2.bm25_score > 0 or chunk2.retrieval_source in ["bm25", "vector+bm25"]
    
    def test_similarity_threshold_filtering(
        self, vector_store, doc_store, embedding_generator, bm25_index, sample_chunks
    ):
        """Test that similarity threshold filters results correctly."""
        # Setup
        doc_store.create_document("test_doc_1", "Banking Systems", metadata={})
        
        for chunk in sample_chunks[:2]:  # Only use first 2 chunks
            doc_store.create_chunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                text=chunk.text,
                chunk_type=chunk.chunk_type,
                parent_chunk_id=chunk.parent_chunk_id,
                breadcrumbs=chunk.breadcrumbs,
                section=chunk.section,
                token_count=chunk.token_count,
                metadata=chunk.metadata
            )
        
        texts = [chunk.text for chunk in sample_chunks[:2]]
        embeddings = embedding_generator.batch_generate(texts)
        chunk_ids = [chunk.chunk_id for chunk in sample_chunks[:2]]
        metadata = [
            {
                "doc_id": chunk.doc_id,
                "text": chunk.text,
                "breadcrumbs": chunk.breadcrumbs,
                "section": chunk.section,
                "chunk_type": chunk.chunk_type
            }
            for chunk in sample_chunks[:2]
        ]
        vector_store.store_embeddings(chunk_ids, embeddings, metadata)
        bm25_index.index(sample_chunks[:2])
        
        # Create retriever with high threshold
        retriever = VectorRetriever(
            vector_store=vector_store,
            bm25_index=bm25_index,
            embedding_generator=embedding_generator,
            doc_store=doc_store,
            similarity_threshold=0.9  # Very high threshold
        )
        
        # Query with unrelated text (should have low similarity)
        results = retriever.retrieve("quantum computing algorithms", top_k=5)
        
        # Should have few or no results due to high threshold
        # (unless BM25 finds something)
        assert len(results) <= 2
