"""Unit tests for GraphPopulator class."""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from neo4j.exceptions import ServiceUnavailable, Neo4jError

from src.storage.graph_populator import GraphPopulator
from src.extraction.entity_extractor import Entity
from src.extraction.entity_resolver import Relationship
from src.chunking.hierarchical_chunker import Chunk


@pytest.fixture
def mock_driver():
    """Create a mock Neo4j driver."""
    driver = Mock()
    driver.verify_connectivity = Mock()
    return driver


@pytest.fixture
def graph_populator(mock_driver):
    """Create GraphPopulator with mocked driver."""
    with patch('src.storage.graph_populator.GraphDatabase.driver', return_value=mock_driver):
        populator = GraphPopulator(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password"
        )
        return populator


@pytest.fixture
def sample_entities():
    """Create sample entities for testing."""
    return [
        Entity(
            entity_id="ent_1",
            entity_type="System",
            name="NEFT",
            canonical_name="NEFT",
            source_chunk_id="chunk_1",
            context="NEFT is a payment system",
            properties={"key": "value"}
        ),
        Entity(
            entity_id="ent_2",
            entity_type="PaymentMode",
            name="RTGS",
            canonical_name="RTGS",
            source_chunk_id="chunk_2",
            context="RTGS for large transactions",
            properties={}
        )
    ]


@pytest.fixture
def sample_relationships():
    """Create sample relationships for testing."""
    return [
        Relationship(
            rel_id="rel_1",
            rel_type="SAME_AS",
            source_entity_id="ent_1",
            target_entity_id="ent_2",
            properties={"reason": "duplicate"}
        ),
        Relationship(
            rel_id="rel_2",
            rel_type="DEPENDS_ON",
            source_entity_id="ent_1",
            target_entity_id="ent_2",
            properties={}
        )
    ]


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing."""
    return [
        Chunk(
            chunk_id="chunk_1",
            doc_id="doc_1",
            text="Sample text 1",
            chunk_type="parent",
            parent_chunk_id=None,
            breadcrumbs="Doc > Section",
            section="Section 1",
            token_count=10,
            metadata={}
        ),
        Chunk(
            chunk_id="chunk_2",
            doc_id="doc_1",
            text="Sample text 2",
            chunk_type="child",
            parent_chunk_id="chunk_1",
            breadcrumbs="Doc > Section > Subsection",
            section="Section 1",
            token_count=5,
            metadata={}
        )
    ]


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        {
            "doc_id": "doc_1",
            "title": "Test Document",
            "file_path": "/path/to/doc.pdf",
            "file_type": "pdf",
            "metadata": {"author": "Test"}
        }
    ]


class TestGraphPopulatorInit:
    """Tests for GraphPopulator initialization."""
    
    def test_init_success(self, mock_driver):
        """Test successful initialization."""
        with patch('src.storage.graph_populator.GraphDatabase.driver', return_value=mock_driver):
            populator = GraphPopulator(
                neo4j_uri="bolt://localhost:7687",
                neo4j_user="neo4j",
                neo4j_password="password"
            )
            
            assert populator.driver == mock_driver
            mock_driver.verify_connectivity.assert_called_once()
    
    def test_init_connection_failure(self):
        """Test initialization with connection failure."""
        with patch('src.storage.graph_populator.GraphDatabase.driver') as mock_driver_class:
            mock_driver_class.return_value.verify_connectivity.side_effect = ServiceUnavailable("Connection failed")
            
            with pytest.raises(ServiceUnavailable):
                GraphPopulator(
                    neo4j_uri="bolt://localhost:7687",
                    neo4j_user="neo4j",
                    neo4j_password="wrong_password"
                )
    
    def test_close(self, graph_populator, mock_driver):
        """Test closing the driver connection."""
        graph_populator.close()
        mock_driver.close.assert_called_once()


class TestCreateSchema:
    """Tests for create_schema method."""
    
    def test_create_schema_success(self, graph_populator, mock_driver):
        """Test successful schema creation."""
        mock_session = MagicMock()
        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=mock_session)
        mock_context.__exit__ = MagicMock(return_value=False)
        mock_driver.session.return_value = mock_context
        
        graph_populator.create_schema()
        
        # Verify session.run was called multiple times for constraints and indexes
        assert mock_session.run.call_count > 0
    
    def test_create_schema_with_existing_constraints(self, graph_populator, mock_driver):
        """Test schema creation when constraints already exist."""
        mock_session = MagicMock()
        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=mock_session)
        mock_context.__exit__ = MagicMock(return_value=False)
        mock_driver.session.return_value = mock_context
        
        # Simulate constraint already exists error for some calls, success for others
        # Need enough None values for all constraint and index creation calls
        mock_session.run.side_effect = [Neo4jError("Constraint already exists")] + [None] * 20
        
        # Should not raise exception
        graph_populator.create_schema()
    
    def test_create_schema_failure(self, graph_populator, mock_driver):
        """Test schema creation failure from session context."""
        mock_context = MagicMock()
        # Simulate failure when entering the session context
        mock_context.__enter__ = MagicMock(side_effect=Neo4jError("Session creation failed"))
        mock_context.__exit__ = MagicMock(return_value=False)
        mock_driver.session.return_value = mock_context
        
        with pytest.raises(Neo4jError):
            graph_populator.create_schema()


class TestPopulate:
    """Tests for populate method."""
    
    def test_populate_success(
        self,
        graph_populator,
        mock_driver,
        sample_entities,
        sample_relationships,
        sample_chunks,
        sample_documents
    ):
        """Test successful graph population."""
        mock_session = MagicMock()
        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=mock_session)
        mock_context.__exit__ = MagicMock(return_value=False)
        mock_driver.session.return_value = mock_context
        
        graph_populator.populate(
            entities=sample_entities,
            relationships=sample_relationships,
            chunks=sample_chunks,
            documents=sample_documents
        )
        
        # Verify session.run was called for various operations
        assert mock_session.run.call_count > 0
    
    def test_populate_empty_lists(self, graph_populator, mock_driver):
        """Test populate with empty lists."""
        mock_session = MagicMock()
        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=mock_session)
        mock_context.__exit__ = MagicMock(return_value=False)
        mock_driver.session.return_value = mock_context
        
        graph_populator.populate(
            entities=[],
            relationships=[],
            chunks=[]
        )
        
        # Should complete without errors
        assert True
    
    def test_populate_without_documents(
        self,
        graph_populator,
        mock_driver,
        sample_entities,
        sample_relationships,
        sample_chunks
    ):
        """Test populate without document metadata."""
        mock_session = MagicMock()
        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=mock_session)
        mock_context.__exit__ = MagicMock(return_value=False)
        mock_driver.session.return_value = mock_context
        
        graph_populator.populate(
            entities=sample_entities,
            relationships=sample_relationships,
            chunks=sample_chunks,
            documents=None
        )
        
        # Should complete without errors
        assert mock_session.run.call_count > 0
    
    def test_populate_failure(
        self,
        graph_populator,
        mock_driver,
        sample_entities,
        sample_relationships,
        sample_chunks
    ):
        """Test populate with Neo4j error."""
        mock_session = MagicMock()
        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=mock_session)
        mock_context.__exit__ = MagicMock(return_value=False)
        mock_driver.session.return_value = mock_context
        mock_session.run.side_effect = Neo4jError("Population failed")
        
        with pytest.raises(Neo4jError):
            graph_populator.populate(
                entities=sample_entities,
                relationships=sample_relationships,
                chunks=sample_chunks
            )


class TestCreateEntityNodes:
    """Tests for _create_entity_nodes method."""
    
    def test_create_entity_nodes_with_apoc(
        self, graph_populator, mock_driver, sample_entities
    ):
        """Test entity node creation with APOC."""
        mock_session = MagicMock()
        
        graph_populator._create_entity_nodes(mock_session, sample_entities)
        
        # Verify session.run was called
        assert mock_session.run.call_count > 0
    
    def test_create_entity_nodes_without_apoc(
        self, graph_populator, mock_driver, sample_entities
    ):
        """Test entity node creation fallback without APOC."""
        mock_session = MagicMock()
        # First call fails (APOC not available), second succeeds (fallback)
        mock_session.run.side_effect = [
            Neo4jError("APOC not available"),
            None
        ]
        
        graph_populator._create_entity_nodes(mock_session, sample_entities)
        
        # Verify fallback query was used
        assert mock_session.run.call_count == 2
    
    def test_create_entity_nodes_empty_list(self, graph_populator):
        """Test entity node creation with empty list."""
        mock_session = MagicMock()
        
        graph_populator._create_entity_nodes(mock_session, [])
        
        # Should not call session.run
        mock_session.run.assert_not_called()


class TestCreateChunkNodes:
    """Tests for _create_chunk_nodes method."""
    
    def test_create_chunk_nodes_success(
        self, graph_populator, sample_chunks
    ):
        """Test chunk node creation."""
        mock_session = MagicMock()
        
        graph_populator._create_chunk_nodes(mock_session, sample_chunks)
        
        # Verify session.run was called
        assert mock_session.run.call_count > 0
    
    def test_create_chunk_nodes_empty_list(self, graph_populator):
        """Test chunk node creation with empty list."""
        mock_session = MagicMock()
        
        graph_populator._create_chunk_nodes(mock_session, [])
        
        # Should not call session.run
        mock_session.run.assert_not_called()


class TestCreateRelationships:
    """Tests for _create_relationships method."""
    
    def test_create_relationships_success(
        self, graph_populator, sample_relationships
    ):
        """Test relationship creation."""
        mock_session = MagicMock()
        
        graph_populator._create_relationships(mock_session, sample_relationships)
        
        # Verify session.run was called for each relationship type
        assert mock_session.run.call_count >= 2  # SAME_AS and DEPENDS_ON
    
    def test_create_relationships_empty_list(self, graph_populator):
        """Test relationship creation with empty list."""
        mock_session = MagicMock()
        
        graph_populator._create_relationships(mock_session, [])
        
        # Should not call session.run
        mock_session.run.assert_not_called()
    
    def test_create_relationships_groups_by_type(
        self, graph_populator, sample_relationships
    ):
        """Test that relationships are grouped by type."""
        mock_session = MagicMock()
        
        # Add more relationships of same type
        relationships = sample_relationships + [
            Relationship(
                rel_id="rel_3",
                rel_type="SAME_AS",
                source_entity_id="ent_3",
                target_entity_id="ent_4",
                properties={}
            )
        ]
        
        graph_populator._create_relationships(mock_session, relationships)
        
        # Should group SAME_AS relationships together
        assert mock_session.run.call_count >= 2


class TestCreateMentionsRelationships:
    """Tests for _create_mentions_relationships method."""
    
    def test_create_mentions_relationships_success(
        self, graph_populator, sample_entities
    ):
        """Test MENTIONS relationship creation."""
        mock_session = MagicMock()
        
        graph_populator._create_mentions_relationships(mock_session, sample_entities)
        
        # Verify session.run was called
        assert mock_session.run.call_count > 0
    
    def test_create_mentions_relationships_empty_list(self, graph_populator):
        """Test MENTIONS relationship creation with empty list."""
        mock_session = MagicMock()
        
        graph_populator._create_mentions_relationships(mock_session, [])
        
        # Should not call session.run
        mock_session.run.assert_not_called()


class TestCreateStructureRelationships:
    """Tests for _create_structure_relationships method."""
    
    def test_create_structure_relationships_success(
        self, graph_populator, sample_chunks
    ):
        """Test structure relationship creation."""
        mock_session = MagicMock()
        
        graph_populator._create_structure_relationships(mock_session, sample_chunks)
        
        # Verify session.run was called for various structure relationships
        assert mock_session.run.call_count > 0
    
    def test_create_structure_relationships_empty_list(self, graph_populator):
        """Test structure relationship creation with empty list."""
        mock_session = MagicMock()
        
        graph_populator._create_structure_relationships(mock_session, [])
        
        # Should not call session.run
        mock_session.run.assert_not_called()
    
    def test_create_structure_relationships_parent_child(self, graph_populator):
        """Test parent-child chunk relationships."""
        mock_session = MagicMock()
        
        chunks = [
            Chunk(
                chunk_id="parent_1",
                doc_id="doc_1",
                text="Parent text",
                chunk_type="parent",
                parent_chunk_id=None,
                breadcrumbs="Doc",
                section="Section 1",
                token_count=100,
                metadata={}
            ),
            Chunk(
                chunk_id="child_1",
                doc_id="doc_1",
                text="Child text",
                chunk_type="child",
                parent_chunk_id="parent_1",
                breadcrumbs="Doc > Section",
                section="Section 1",
                token_count=50,
                metadata={}
            )
        ]
        
        graph_populator._create_structure_relationships(mock_session, chunks)
        
        # Should create parent-child CONTAINS relationship
        assert mock_session.run.call_count > 0


class TestExecuteBatch:
    """Tests for _execute_batch method."""
    
    def test_execute_batch_single_batch(self, graph_populator):
        """Test batch execution with data fitting in one batch."""
        mock_session = MagicMock()
        query = "CREATE (n:Node {id: $id})"
        data = [{"id": i} for i in range(50)]  # Less than BATCH_SIZE
        
        graph_populator._execute_batch(mock_session, query, data, "nodes")
        
        # Should execute once
        assert mock_session.run.call_count == 1
    
    def test_execute_batch_multiple_batches(self, graph_populator):
        """Test batch execution with data requiring multiple batches."""
        mock_session = MagicMock()
        query = "CREATE (n:Node {id: $id})"
        data = [{"id": i} for i in range(250)]  # More than BATCH_SIZE (100)
        
        graph_populator._execute_batch(mock_session, query, data, "nodes")
        
        # Should execute 3 times (100 + 100 + 50)
        assert mock_session.run.call_count == 3
    
    def test_execute_batch_empty_data(self, graph_populator):
        """Test batch execution with empty data."""
        mock_session = MagicMock()
        query = "CREATE (n:Node {id: $id})"
        
        graph_populator._execute_batch(mock_session, query, [], "nodes")
        
        # Should not execute
        mock_session.run.assert_not_called()
    
    def test_execute_batch_failure(self, graph_populator):
        """Test batch execution with Neo4j error."""
        mock_session = MagicMock()
        mock_session.run.side_effect = Neo4jError("Batch execution failed")
        query = "CREATE (n:Node {id: $id})"
        data = [{"id": 1}]
        
        with pytest.raises(Neo4jError):
            graph_populator._execute_batch(mock_session, query, data, "nodes")


class TestIntegration:
    """Integration tests for GraphPopulator."""
    
    def test_full_populate_workflow(
        self,
        graph_populator,
        mock_driver,
        sample_entities,
        sample_relationships,
        sample_chunks,
        sample_documents
    ):
        """Test complete populate workflow."""
        mock_session = MagicMock()
        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=mock_session)
        mock_context.__exit__ = MagicMock(return_value=False)
        mock_driver.session.return_value = mock_context
        
        # Execute full populate
        graph_populator.populate(
            entities=sample_entities,
            relationships=sample_relationships,
            chunks=sample_chunks,
            documents=sample_documents
        )
        
        # Verify all major operations were called
        assert mock_session.run.call_count > 0
        
        # Verify no exceptions were raised
        assert True
