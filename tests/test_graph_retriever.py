"""Tests for GraphRetriever class."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from src.retrieval.graph_retriever import (
    GraphRetriever,
    GraphNode,
    GraphRelationship,
    GraphResult
)


@pytest.fixture
def mock_neo4j_driver():
    """Create a mock Neo4j driver."""
    driver = Mock()
    driver.verify_connectivity = Mock()
    return driver


@pytest.fixture
def graph_retriever(mock_neo4j_driver):
    """Create GraphRetriever with mocked Neo4j driver."""
    with patch('src.retrieval.graph_retriever.GraphDatabase.driver', return_value=mock_neo4j_driver):
        retriever = GraphRetriever(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password",
            max_depth=3
        )
        return retriever


class TestGraphRetriever:
    """Test suite for GraphRetriever class."""
    
    def test_initialization(self, mock_neo4j_driver):
        """Test GraphRetriever initialization."""
        with patch('src.retrieval.graph_retriever.GraphDatabase.driver', return_value=mock_neo4j_driver):
            retriever = GraphRetriever(
                neo4j_uri="bolt://localhost:7687",
                neo4j_user="neo4j",
                neo4j_password="password",
                max_depth=3
            )
            
            assert retriever.neo4j_uri == "bolt://localhost:7687"
            assert retriever.max_depth == 3
            assert retriever.driver is not None
            mock_neo4j_driver.verify_connectivity.assert_called_once()
    
    def test_close(self, graph_retriever):
        """Test closing Neo4j connection."""
        graph_retriever.close()
        graph_retriever.driver.close.assert_called_once()
    
    def test_extract_entity_mentions_acronyms(self, graph_retriever):
        """Test entity extraction with acronyms."""
        query = "What is NEFT and how does RTGS work?"
        entities = graph_retriever._extract_entity_mentions(query)
        
        assert "NEFT" in entities
        assert "RTGS" in entities
    
    def test_extract_entity_mentions_quoted(self, graph_retriever):
        """Test entity extraction with quoted strings."""
        query = 'What is "Core Banking System"?'
        entities = graph_retriever._extract_entity_mentions(query)
        
        assert "Core Banking System" in entities
    
    def test_extract_entity_mentions_capitalized(self, graph_retriever):
        """Test entity extraction with capitalized words."""
        query = "How does Payment Gateway integrate with Core Banking?"
        entities = graph_retriever._extract_entity_mentions(query)
        
        # Should extract multi-word capitalized phrases
        assert any("Payment" in e or "Gateway" in e for e in entities)
        assert any("Core" in e or "Banking" in e for e in entities)
    
    def test_detect_query_pattern_dependency(self, graph_retriever):
        """Test dependency pattern detection."""
        queries = [
            "What depends on NEFT?",
            "What are the dependencies of RTGS?",
            "What systems require Core Banking?",
            "What is the impact of changing NEFT?"
        ]
        
        for query in queries:
            pattern = graph_retriever._detect_query_pattern(query)
            assert pattern == "dependency", f"Failed for query: {query}"
    
    def test_detect_query_pattern_integration(self, graph_retriever):
        """Test integration pattern detection."""
        queries = [
            "How does NEFT integrate with Core Banking?",
            "What are the connections between RTGS and IMPS?",
            "Show integration points for Payment Gateway"
        ]
        
        for query in queries:
            pattern = graph_retriever._detect_query_pattern(query)
            assert pattern == "integration", f"Failed for query: {query}"
    
    def test_detect_query_pattern_workflow(self, graph_retriever):
        """Test workflow pattern detection."""
        queries = [
            "Show the payment workflow",
            "What are the steps in KYC process?",
            "Describe the authorization flow"
        ]
        
        for query in queries:
            pattern = graph_retriever._detect_query_pattern(query)
            assert pattern == "workflow", f"Failed for query: {query}"
    
    def test_detect_query_pattern_conflict(self, graph_retriever):
        """Test conflict pattern detection."""
        queries = [
            "What conflicts exist for NEFT?",
            "Show contradictions in payment rules",
            "Are there inconsistent rules in the documentation?"
        ]
        
        for query in queries:
            pattern = graph_retriever._detect_query_pattern(query)
            assert pattern == "conflict", f"Failed for query: {query}"
    
    def test_detect_query_pattern_comparison(self, graph_retriever):
        """Test comparison pattern detection."""
        queries = [
            "Compare NEFT and RTGS",
            "What are the differences between IMPS and UPI?",
            "NEFT versus RTGS"
        ]
        
        for query in queries:
            pattern = graph_retriever._detect_query_pattern(query)
            assert pattern == "comparison", f"Failed for query: {query}"
    
    def test_detect_query_pattern_general(self, graph_retriever):
        """Test general pattern detection (fallback)."""
        query = "Tell me about NEFT"
        pattern = graph_retriever._detect_query_pattern(query)
        assert pattern == "general"
    
    def test_retrieve_empty_query(self, graph_retriever):
        """Test retrieve with empty query raises ValueError."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            graph_retriever.retrieve("")
    
    def test_retrieve_with_depth_override(self, graph_retriever):
        """Test retrieve with custom max_depth."""
        # Mock session and result
        mock_session = MagicMock()
        mock_result = []
        mock_session.run.return_value = mock_result
        
        # Properly mock the context manager
        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=mock_session)
        mock_context.__exit__ = MagicMock(return_value=False)
        graph_retriever.driver.session = MagicMock(return_value=mock_context)
        
        result = graph_retriever.retrieve("What is NEFT?", max_depth=2)
        
        assert isinstance(result, GraphResult)
        # Verify that the query was executed
        mock_session.run.assert_called()
    
    def test_process_graph_result_with_nodes(self, graph_retriever):
        """Test processing graph result with nodes."""
        # Create mock Neo4j nodes
        mock_node1 = MagicMock()
        mock_node1.get = lambda key: "entity_1" if key == "entity_id" else None
        mock_node1.labels = ["Entity", "System"]
        mock_node1.keys = lambda: ["entity_id", "name"]
        mock_node1.__getitem__ = lambda self, key: {"entity_id": "entity_1", "name": "NEFT"}.get(key)
        mock_node1.__iter__ = lambda self: iter(["entity_id", "name"])
        
        mock_node2 = MagicMock()
        mock_node2.get = lambda key: "entity_2" if key == "entity_id" else None
        mock_node2.labels = ["Entity", "System"]
        mock_node2.keys = lambda: ["entity_id", "name"]
        mock_node2.__getitem__ = lambda self, key: {"entity_id": "entity_2", "name": "RTGS"}.get(key)
        mock_node2.__iter__ = lambda self: iter(["entity_id", "name"])
        
        # Create mock result
        mock_record = {
            "center_node": mock_node1,
            "path_nodes": [mock_node1, mock_node2],
            "path_rels": []
        }
        mock_result = [mock_record]
        
        result = graph_retriever._process_graph_result(mock_result)
        
        assert len(result.nodes) == 2
        assert any(node.node_id == "entity_1" for node in result.nodes)
        assert any(node.node_id == "entity_2" for node in result.nodes)
    
    def test_process_graph_result_with_relationships(self, graph_retriever):
        """Test processing graph result with relationships."""
        # Create mock Neo4j nodes
        mock_node1 = MagicMock()
        mock_node1.get = lambda key: "entity_1" if key == "entity_id" else None
        mock_node1.id = 1
        mock_node1.labels = ["Entity"]
        mock_node1.keys = lambda: ["entity_id"]
        mock_node1.__getitem__ = lambda self, key: {"entity_id": "entity_1"}.get(key)
        mock_node1.__iter__ = lambda self: iter(["entity_id"])
        
        mock_node2 = MagicMock()
        mock_node2.get = lambda key: "entity_2" if key == "entity_id" else None
        mock_node2.id = 2
        mock_node2.labels = ["Entity"]
        mock_node2.keys = lambda: ["entity_id"]
        mock_node2.__getitem__ = lambda self, key: {"entity_id": "entity_2"}.get(key)
        mock_node2.__iter__ = lambda self: iter(["entity_id"])
        
        # Create mock relationship
        mock_rel = MagicMock()
        mock_rel.get = lambda key: "rel_1" if key == "rel_id" else None
        mock_rel.id = 100
        mock_rel.type = "DEPENDS_ON"
        mock_rel.start_node = mock_node1
        mock_rel.end_node = mock_node2
        mock_rel.keys = lambda: ["rel_id"]
        mock_rel.__getitem__ = lambda self, key: {"rel_id": "rel_1"}.get(key)
        mock_rel.__iter__ = lambda self: iter(["rel_id"])
        
        # Create mock result
        mock_record = {
            "path_nodes": [mock_node1, mock_node2],
            "path_rels": [mock_rel]
        }
        mock_result = [mock_record]
        
        result = graph_retriever._process_graph_result(mock_result)
        
        assert len(result.relationships) == 1
        assert result.relationships[0].rel_type == "DEPENDS_ON"
        assert result.relationships[0].source_id == "entity_1"
        assert result.relationships[0].target_id == "entity_2"
    
    def test_retrieve_chunks(self, graph_retriever):
        """Test retrieving chunks from Neo4j."""
        # Mock session and result
        mock_session = MagicMock()
        mock_record = {
            "chunk_id": "chunk_1",
            "text": "Sample chunk text",
            "breadcrumbs": "Doc > Section",
            "doc_id": "doc_1",
            "section": "Section 1"
        }
        mock_session.run.return_value = [mock_record]
        
        # Properly mock the context manager
        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=mock_session)
        mock_context.__exit__ = MagicMock(return_value=False)
        graph_retriever.driver.session = MagicMock(return_value=mock_context)
        
        chunks = graph_retriever._retrieve_chunks(["chunk_1"])
        
        assert len(chunks) == 1
        assert chunks[0]["chunk_id"] == "chunk_1"
        assert chunks[0]["text"] == "Sample chunk text"
        assert chunks[0]["breadcrumbs"] == "Doc > Section"
    
    def test_retrieve_chunks_empty_list(self, graph_retriever):
        """Test retrieving chunks with empty list."""
        chunks = graph_retriever._retrieve_chunks([])
        assert chunks == []


class TestGraphDataClasses:
    """Test suite for graph data classes."""
    
    def test_graph_node_creation(self):
        """Test GraphNode creation."""
        node = GraphNode(
            node_id="entity_1",
            node_type="System",
            properties={"name": "NEFT", "canonical_name": "NEFT"}
        )
        
        assert node.node_id == "entity_1"
        assert node.node_type == "System"
        assert node.properties["name"] == "NEFT"
    
    def test_graph_relationship_creation(self):
        """Test GraphRelationship creation."""
        rel = GraphRelationship(
            rel_id="rel_1",
            rel_type="DEPENDS_ON",
            source_id="entity_1",
            target_id="entity_2",
            properties={"weight": 1.0}
        )
        
        assert rel.rel_id == "rel_1"
        assert rel.rel_type == "DEPENDS_ON"
        assert rel.source_id == "entity_1"
        assert rel.target_id == "entity_2"
        assert rel.properties["weight"] == 1.0
    
    def test_graph_result_creation(self):
        """Test GraphResult creation."""
        node1 = GraphNode(node_id="entity_1", node_type="System")
        node2 = GraphNode(node_id="entity_2", node_type="System")
        rel = GraphRelationship(
            rel_id="rel_1",
            rel_type="DEPENDS_ON",
            source_id="entity_1",
            target_id="entity_2"
        )
        chunk = {"chunk_id": "chunk_1", "text": "Sample text"}
        
        result = GraphResult(
            nodes=[node1, node2],
            relationships=[rel],
            chunks=[chunk],
            impact_radius=5,
            circular_dependencies=[["A", "B", "C"]]
        )
        
        assert len(result.nodes) == 2
        assert len(result.relationships) == 1
        assert len(result.chunks) == 1
        assert result.impact_radius == 5
        assert len(result.circular_dependencies) == 1
    
    def test_graph_result_default_values(self):
        """Test GraphResult with default empty lists."""
        result = GraphResult()
        
        assert result.nodes == []
        assert result.relationships == []
        assert result.chunks == []
        assert result.impact_radius is None
        assert result.circular_dependencies == []


class TestGraphRetrieverDepthLimiting:
    """Test suite for depth limiting functionality."""
    
    def test_max_depth_in_dependency_query(self, graph_retriever):
        """Test that max_depth is applied in dependency queries."""
        mock_session = MagicMock()
        mock_session.run.return_value = []
        
        # Properly mock the context manager
        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=mock_session)
        mock_context.__exit__ = MagicMock(return_value=False)
        graph_retriever.driver.session = MagicMock(return_value=mock_context)
        
        graph_retriever._execute_dependency_query(["NEFT"], depth=2)
        
        # Verify that the Cypher query contains the depth limit
        call_args = mock_session.run.call_args
        cypher_query = call_args[0][0]
        assert "*1..2" in cypher_query
    
    def test_max_depth_in_integration_query(self, graph_retriever):
        """Test that max_depth is applied in integration queries."""
        mock_session = MagicMock()
        mock_session.run.return_value = []
        
        # Properly mock the context manager
        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=mock_session)
        mock_context.__exit__ = MagicMock(return_value=False)
        graph_retriever.driver.session = MagicMock(return_value=mock_context)
        
        graph_retriever._execute_integration_query(["NEFT", "RTGS"], depth=3)
        
        # Verify that the Cypher query contains the depth limit
        call_args = mock_session.run.call_args
        cypher_query = call_args[0][0]
        assert "*1..3" in cypher_query
    
    def test_max_depth_in_workflow_query(self, graph_retriever):
        """Test that max_depth is applied in workflow queries."""
        mock_session = MagicMock()
        mock_session.run.return_value = []
        
        # Properly mock the context manager
        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=mock_session)
        mock_context.__exit__ = MagicMock(return_value=False)
        graph_retriever.driver.session = MagicMock(return_value=mock_context)
        
        graph_retriever._execute_workflow_query(["Payment Flow"], depth=2)
        
        # Verify that the Cypher query contains the depth limit
        call_args = mock_session.run.call_args
        cypher_query = call_args[0][0]
        assert "*0..2" in cypher_query


class TestDependencyQueryFeatures:
    """Test suite for advanced dependency query features (Requirements 18.1-18.5)."""
    
    def test_dependency_query_forward_traversal(self, graph_retriever):
        """Test forward dependency traversal (what depends on X)."""
        # Mock session with forward dependency result
        mock_session = MagicMock()
        
        # Create mock nodes
        mock_node_neft = MagicMock()
        mock_node_neft.get = lambda key: "neft_1" if key == "entity_id" else None
        mock_node_neft.labels = ["Entity", "System"]
        mock_node_neft.__getitem__ = lambda self, key: {"entity_id": "neft_1", "name": "NEFT"}.get(key)
        mock_node_neft.__iter__ = lambda self: iter(["entity_id", "name"])
        
        mock_node_dependent = MagicMock()
        mock_node_dependent.get = lambda key: "system_1" if key == "entity_id" else None
        mock_node_dependent.labels = ["Entity", "System"]
        mock_node_dependent.__getitem__ = lambda self, key: {"entity_id": "system_1", "name": "Payment Gateway"}.get(key)
        mock_node_dependent.__iter__ = lambda self: iter(["entity_id", "name"])
        
        # Create mock relationship
        mock_rel = MagicMock()
        mock_rel.id = 100
        mock_rel.type = "DEPENDS_ON"
        mock_rel.start_node = mock_node_dependent
        mock_rel.end_node = mock_node_neft
        mock_rel.__iter__ = lambda self: iter([])
        
        mock_record = {
            "center_node": mock_node_neft,
            "path_nodes": [mock_node_dependent, mock_node_neft],
            "path_rels": [mock_rel]
        }
        
        # Mock all three queries: main query, impact radius, circular deps
        mock_session.run.side_effect = [
            [mock_record],  # Main dependency query
            [{"impact_radius": 2}],  # Impact radius query
            []  # Circular dependency query
        ]
        
        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=mock_session)
        mock_context.__exit__ = MagicMock(return_value=False)
        graph_retriever.driver.session = MagicMock(return_value=mock_context)
        
        result = graph_retriever._execute_dependency_query(["NEFT"], depth=3)
        
        # Verify forward dependency is captured
        assert len(result.nodes) >= 1
        assert any(node.node_id == "neft_1" for node in result.nodes)
        assert len(result.relationships) >= 1
        assert result.relationships[0].rel_type == "DEPENDS_ON"
    
    def test_dependency_query_backward_traversal(self, graph_retriever):
        """Test backward dependency traversal (what X depends on)."""
        # Mock session with backward dependency result
        mock_session = MagicMock()
        
        # Create mock nodes
        mock_node_gateway = MagicMock()
        mock_node_gateway.get = lambda key: "gateway_1" if key == "entity_id" else None
        mock_node_gateway.labels = ["Entity", "System"]
        mock_node_gateway.__getitem__ = lambda self, key: {"entity_id": "gateway_1", "name": "Payment Gateway"}.get(key)
        mock_node_gateway.__iter__ = lambda self: iter(["entity_id", "name"])
        
        mock_node_neft = MagicMock()
        mock_node_neft.get = lambda key: "neft_1" if key == "entity_id" else None
        mock_node_neft.labels = ["Entity", "System"]
        mock_node_neft.__getitem__ = lambda self, key: {"entity_id": "neft_1", "name": "NEFT"}.get(key)
        mock_node_neft.__iter__ = lambda self: iter(["entity_id", "name"])
        
        # Create mock relationship
        mock_rel = MagicMock()
        mock_rel.id = 101
        mock_rel.type = "DEPENDS_ON"
        mock_rel.start_node = mock_node_gateway
        mock_rel.end_node = mock_node_neft
        mock_rel.__iter__ = lambda self: iter([])
        
        mock_record = {
            "center_node": mock_node_gateway,
            "path_nodes": [mock_node_gateway, mock_node_neft],
            "path_rels": [mock_rel]
        }
        
        # Mock all three queries
        mock_session.run.side_effect = [
            [mock_record],  # Main dependency query
            [{"impact_radius": 2}],  # Impact radius query
            []  # Circular dependency query
        ]
        
        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=mock_session)
        mock_context.__exit__ = MagicMock(return_value=False)
        graph_retriever.driver.session = MagicMock(return_value=mock_context)
        
        result = graph_retriever._execute_dependency_query(["Payment Gateway"], depth=3)
        
        # Verify backward dependency is captured
        assert len(result.nodes) >= 1
        assert len(result.relationships) >= 1
        assert result.relationships[0].rel_type == "DEPENDS_ON"
    
    def test_compute_impact_radius(self, graph_retriever):
        """Test impact radius computation (Requirement 18.4)."""
        mock_session = MagicMock()
        mock_session.run.return_value = [{"impact_radius": 5}]
        
        impact_radius = graph_retriever._compute_impact_radius(["NEFT"], depth=3, session=mock_session)
        
        assert impact_radius == 5
        mock_session.run.assert_called_once()
    
    def test_compute_impact_radius_empty_entities(self, graph_retriever):
        """Test impact radius with empty entity list."""
        mock_session = MagicMock()
        
        impact_radius = graph_retriever._compute_impact_radius([], depth=3, session=mock_session)
        
        assert impact_radius == 0
        mock_session.run.assert_not_called()
    
    def test_compute_impact_radius_no_results(self, graph_retriever):
        """Test impact radius when no results found."""
        mock_session = MagicMock()
        mock_session.run.return_value.single.return_value = None
        
        impact_radius = graph_retriever._compute_impact_radius(["NonExistent"], depth=3, session=mock_session)
        
        assert impact_radius == 0
    
    def test_detect_circular_dependencies(self, graph_retriever):
        """Test circular dependency detection (Requirement 18.5)."""
        mock_session = MagicMock()
        
        # Mock circular dependency: A -> B -> C -> A
        mock_session.run.return_value = [
            {"cycle_entities": ["NEFT", "Core Banking", "Payment Gateway", "NEFT"]}
        ]
        
        circular_deps = graph_retriever._detect_circular_dependencies(["NEFT"], depth=3, session=mock_session)
        
        assert len(circular_deps) == 1
        assert "NEFT" in circular_deps[0]
        assert "Core Banking" in circular_deps[0]
        assert "Payment Gateway" in circular_deps[0]
    
    def test_detect_circular_dependencies_none_found(self, graph_retriever):
        """Test circular dependency detection when none exist."""
        mock_session = MagicMock()
        mock_session.run.return_value = []
        
        circular_deps = graph_retriever._detect_circular_dependencies(["NEFT"], depth=3, session=mock_session)
        
        assert len(circular_deps) == 0
    
    def test_detect_circular_dependencies_empty_entities(self, graph_retriever):
        """Test circular dependency detection with empty entity list."""
        mock_session = MagicMock()
        
        circular_deps = graph_retriever._detect_circular_dependencies([], depth=3, session=mock_session)
        
        assert len(circular_deps) == 0
        mock_session.run.assert_not_called()
    
    def test_dependency_query_includes_impact_radius(self, graph_retriever):
        """Test that dependency query result includes impact radius."""
        mock_session = MagicMock()
        
        # Mock all three queries
        mock_session.run.side_effect = [
            [],  # Main dependency query
            [{"impact_radius": 3}],  # Impact radius query
            []  # Circular dependency query
        ]
        
        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=mock_session)
        mock_context.__exit__ = MagicMock(return_value=False)
        graph_retriever.driver.session = MagicMock(return_value=mock_context)
        
        result = graph_retriever._execute_dependency_query(["NEFT"], depth=3)
        
        assert result.impact_radius == 3
    
    def test_dependency_query_includes_circular_dependencies(self, graph_retriever):
        """Test that dependency query result includes circular dependencies."""
        mock_session = MagicMock()
        
        # Mock all three queries
        mock_session.run.side_effect = [
            [],  # Main dependency query
            [{"impact_radius": 2}],  # Impact radius query
            [{"cycle_entities": ["A", "B", "C", "A"]}]  # Circular dependency query
        ]
        
        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=mock_session)
        mock_context.__exit__ = MagicMock(return_value=False)
        graph_retriever.driver.session = MagicMock(return_value=mock_context)
        
        result = graph_retriever._execute_dependency_query(["A"], depth=3)
        
        assert len(result.circular_dependencies) == 1
        assert "A" in result.circular_dependencies[0]
        assert "B" in result.circular_dependencies[0]
        assert "C" in result.circular_dependencies[0]
    
    def test_dependency_chains_include_intermediate_nodes(self, graph_retriever):
        """Test that dependency chains include all intermediate nodes (Requirement 18.3)."""
        mock_session = MagicMock()
        
        # Create a chain: A -> B -> C
        mock_node_a = MagicMock()
        mock_node_a.get = lambda key: "a_1" if key == "entity_id" else None
        mock_node_a.labels = ["Entity", "System"]
        mock_node_a.__getitem__ = lambda self, key: {"entity_id": "a_1", "name": "System A"}.get(key)
        mock_node_a.__iter__ = lambda self: iter(["entity_id", "name"])
        
        mock_node_b = MagicMock()
        mock_node_b.get = lambda key: "b_1" if key == "entity_id" else None
        mock_node_b.labels = ["Entity", "System"]
        mock_node_b.__getitem__ = lambda self, key: {"entity_id": "b_1", "name": "System B"}.get(key)
        mock_node_b.__iter__ = lambda self: iter(["entity_id", "name"])
        
        mock_node_c = MagicMock()
        mock_node_c.get = lambda key: "c_1" if key == "entity_id" else None
        mock_node_c.labels = ["Entity", "System"]
        mock_node_c.__getitem__ = lambda self, key: {"entity_id": "c_1", "name": "System C"}.get(key)
        mock_node_c.__iter__ = lambda self: iter(["entity_id", "name"])
        
        mock_record = {
            "center_node": mock_node_a,
            "path_nodes": [mock_node_a, mock_node_b, mock_node_c],
            "path_rels": []
        }
        
        # Mock all three queries
        mock_session.run.side_effect = [
            [mock_record],  # Main dependency query
            [{"impact_radius": 3}],  # Impact radius query
            []  # Circular dependency query
        ]
        
        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=mock_session)
        mock_context.__exit__ = MagicMock(return_value=False)
        graph_retriever.driver.session = MagicMock(return_value=mock_context)
        
        result = graph_retriever._execute_dependency_query(["System A"], depth=3)
        
        # Verify all intermediate nodes are included
        assert len(result.nodes) == 3
        node_ids = [node.node_id for node in result.nodes]
        assert "a_1" in node_ids
        assert "b_1" in node_ids
        assert "c_1" in node_ids



class TestCrossDocumentComparison:
    """Test suite for cross-document comparison functionality (Requirements 19.1-19.5)."""
    
    def test_comparison_query_multi_document_retrieval(self, graph_retriever):
        """Test multi-document retrieval for comparison (Requirement 19.1)."""
        mock_session = MagicMock()
        
        # Create mock entities from different documents
        mock_entity1 = MagicMock()
        mock_entity1.get = lambda key: {"entity_id": "e1", "doc_id": "doc1", "name": "NEFT"}.get(key)
        mock_entity1.labels = ["Entity", "System"]
        mock_entity1.__getitem__ = lambda self, key: {"entity_id": "e1", "name": "NEFT", "entity_type": "System"}.get(key)
        mock_entity1.__iter__ = lambda self: iter(["entity_id", "name", "entity_type"])
        
        mock_entity2 = MagicMock()
        mock_entity2.get = lambda key: {"entity_id": "e2", "doc_id": "doc2", "name": "RTGS"}.get(key)
        mock_entity2.labels = ["Entity", "System"]
        mock_entity2.__getitem__ = lambda self, key: {"entity_id": "e2", "name": "RTGS", "entity_type": "System"}.get(key)
        mock_entity2.__iter__ = lambda self: iter(["entity_id", "name", "entity_type"])
        
        # Create mock chunks from different documents
        mock_chunk1 = MagicMock()
        mock_chunk1.get = lambda key: {"chunk_id": "c1", "doc_id": "doc1", "text": "NEFT info", "breadcrumbs": "Doc1 > Section1", "section": "Section1"}.get(key)
        
        mock_chunk2 = MagicMock()
        mock_chunk2.get = lambda key: {"chunk_id": "c2", "doc_id": "doc2", "text": "RTGS info", "breadcrumbs": "Doc2 > Section1", "section": "Section1"}.get(key)
        
        # Mock the entity query result
        entity_result = [
            {"entity_id": "e1", "name": "NEFT", "entity_type": "System", "canonical_id": "e1", "canonical_name": "NEFT"},
            {"entity_id": "e2", "name": "RTGS", "entity_type": "System", "canonical_id": "e2", "canonical_name": "RTGS"}
        ]
        
        # Mock the main comparison query result
        comparison_result = [{
            "e1": mock_entity1,
            "e2": mock_entity2,
            "paths1": [],
            "paths2": [],
            "chunks1": [mock_chunk1],
            "chunks2": [mock_chunk2]
        }]
        
        # Mock common entities query
        common_entities_result = []
        
        # Mock chunks retrieval
        chunks_result = [
            {"chunk_id": "c1", "text": "NEFT info", "breadcrumbs": "Doc1 > Section1", "doc_id": "doc1", "section": "Section1"},
            {"chunk_id": "c2", "text": "RTGS info", "breadcrumbs": "Doc2 > Section1", "doc_id": "doc2", "section": "Section1"}
        ]
        
        # Set up mock session to return different results for different queries
        mock_session.run.side_effect = [
            entity_result,  # Entity lookup query
            comparison_result,  # Main comparison query
            common_entities_result,  # Common entities query
            chunks_result  # Chunks retrieval
        ]
        
        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=mock_session)
        mock_context.__exit__ = MagicMock(return_value=False)
        graph_retriever.driver.session = MagicMock(return_value=mock_context)
        
        result = graph_retriever._execute_comparison_query(["NEFT", "RTGS"], depth=3)
        
        # Verify multi-document retrieval
        assert len(result.document_groups) >= 1
        assert isinstance(result.document_groups, dict)
    
    def test_comparison_query_document_grouping(self, graph_retriever):
        """Test document grouping logic (Requirement 19.2)."""
        mock_session = MagicMock()
        
        # Create mock entities and chunks
        mock_entity1 = MagicMock()
        mock_entity1.get = lambda key: {"entity_id": "e1", "name": "NEFT", "entity_type": "System"}.get(key)
        mock_entity1.labels = ["Entity", "System"]
        mock_entity1.__getitem__ = lambda self, key: {"entity_id": "e1", "name": "NEFT", "entity_type": "System"}.get(key)
        mock_entity1.__iter__ = lambda self: iter(["entity_id", "name", "entity_type"])
        
        mock_chunk1 = MagicMock()
        mock_chunk1.get = lambda key: {"chunk_id": "c1", "doc_id": "doc1", "text": "Text 1", "breadcrumbs": "Doc1", "section": "S1"}.get(key)
        
        mock_chunk2 = MagicMock()
        mock_chunk2.get = lambda key: {"chunk_id": "c2", "doc_id": "doc2", "text": "Text 2", "breadcrumbs": "Doc2", "section": "S1"}.get(key)
        
        entity_result = [
            {"entity_id": "e1", "name": "NEFT", "entity_type": "System", "canonical_id": "e1", "canonical_name": "NEFT"}
        ]
        
        comparison_result = [{
            "e1": mock_entity1,
            "e2": mock_entity1,
            "paths1": [],
            "paths2": [],
            "chunks1": [mock_chunk1],
            "chunks2": [mock_chunk2]
        }]
        
        chunks_result = [
            {"chunk_id": "c1", "text": "Text 1", "breadcrumbs": "Doc1", "doc_id": "doc1", "section": "S1"},
            {"chunk_id": "c2", "text": "Text 2", "breadcrumbs": "Doc2", "doc_id": "doc2", "section": "S1"}
        ]
        
        mock_session.run.side_effect = [
            entity_result,
            comparison_result,
            [],  # Common entities
            chunks_result  # Chunks retrieval
        ]
        
        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=mock_session)
        mock_context.__exit__ = MagicMock(return_value=False)
        graph_retriever.driver.session = MagicMock(return_value=mock_context)
        
        result = graph_retriever._execute_comparison_query(["NEFT", "NEFT"], depth=3)
        
        # Verify results are grouped by document_id
        assert isinstance(result.document_groups, dict)
        for doc_id, doc_group in result.document_groups.items():
            assert "doc_id" in doc_group
            assert "entities" in doc_group
            assert "chunks" in doc_group
            assert doc_group["doc_id"] == doc_id
    
    def test_identify_common_entities_using_same_as(self, graph_retriever):
        """Test common entity identification using SAME_AS relationships (Requirement 19.3)."""
        mock_session = MagicMock()
        
        # Mock result showing entity appears in multiple documents via SAME_AS
        mock_result = [
            {
                "canonical_id": "neft_canonical",
                "canonical_name": "NEFT",
                "docs": ["doc1", "doc2", "doc3"]
            }
        ]
        mock_session.run.return_value = mock_result
        
        common_entities = graph_retriever._identify_common_entities(mock_session, ["doc1", "doc2", "doc3"])
        
        assert len(common_entities) == 1
        assert "neft_canonical" in common_entities
        mock_session.run.assert_called_once()
    
    def test_identify_common_entities_single_document(self, graph_retriever):
        """Test common entity identification with single document returns empty."""
        mock_session = MagicMock()
        
        common_entities = graph_retriever._identify_common_entities(mock_session, ["doc1"])
        
        assert len(common_entities) == 0
        mock_session.run.assert_not_called()
    
    def test_identify_common_entities_no_common(self, graph_retriever):
        """Test common entity identification when no entities are common."""
        mock_session = MagicMock()
        mock_session.run.return_value = []
        
        common_entities = graph_retriever._identify_common_entities(mock_session, ["doc1", "doc2"])
        
        assert len(common_entities) == 0
    
    def test_highlight_differences_property_differences(self, graph_retriever):
        """Test difference highlighting for entity properties (Requirement 19.4)."""
        document_groups = {
            "doc1": {
                "doc_id": "doc1",
                "entities": [
                    {
                        "entity_id": "e1",
                        "name": "NEFT",
                        "type": "System",
                        "properties": {
                            "entity_id": "e1",
                            "name": "NEFT",
                            "limit": "2 lakhs"
                        }
                    }
                ],
                "chunks": [],
                "relationships": []
            },
            "doc2": {
                "doc_id": "doc2",
                "entities": [
                    {
                        "entity_id": "e2",
                        "name": "NEFT",
                        "type": "System",
                        "properties": {
                            "entity_id": "e2",
                            "name": "NEFT",
                            "limit": "5 lakhs"
                        }
                    }
                ],
                "chunks": [],
                "relationships": []
            }
        }
        
        differences = graph_retriever._highlight_differences(document_groups, {}, {})
        
        # Should detect difference in 'limit' property
        assert len(differences) > 0
        assert any("limit" in diff for diff in differences)
        assert any("2 lakhs" in diff and "5 lakhs" in diff for diff in differences)
    
    def test_highlight_differences_no_differences(self, graph_retriever):
        """Test difference highlighting when entities are identical."""
        document_groups = {
            "doc1": {
                "doc_id": "doc1",
                "entities": [
                    {
                        "entity_id": "e1",
                        "name": "NEFT",
                        "type": "System",
                        "properties": {
                            "entity_id": "e1",
                            "name": "NEFT",
                            "limit": "2 lakhs"
                        }
                    }
                ],
                "chunks": [],
                "relationships": []
            },
            "doc2": {
                "doc_id": "doc2",
                "entities": [
                    {
                        "entity_id": "e2",
                        "name": "NEFT",
                        "type": "System",
                        "properties": {
                            "entity_id": "e2",
                            "name": "NEFT",
                            "limit": "2 lakhs"
                        }
                    }
                ],
                "chunks": [],
                "relationships": []
            }
        }
        
        differences = graph_retriever._highlight_differences(document_groups, {}, {})
        
        # Should not detect differences (entity_id is skipped)
        assert len(differences) == 0
    
    def test_highlight_differences_single_document(self, graph_retriever):
        """Test difference highlighting with single document returns empty."""
        document_groups = {
            "doc1": {
                "doc_id": "doc1",
                "entities": [],
                "chunks": [],
                "relationships": []
            }
        }
        
        differences = graph_retriever._highlight_differences(document_groups, {}, {})
        
        assert len(differences) == 0
    
    def test_single_entity_comparison_across_documents(self, graph_retriever):
        """Test comparing a single entity across multiple documents (Requirement 19.5)."""
        mock_session = MagicMock()
        
        # Create mock entity and chunks from different documents
        mock_entity = MagicMock()
        mock_entity.get = lambda key: {"entity_id": "e1", "name": "NEFT", "entity_type": "System"}.get(key)
        mock_entity.labels = ["Entity", "System"]
        mock_entity.__getitem__ = lambda self, key: {"entity_id": "e1", "name": "NEFT", "entity_type": "System"}.get(key)
        mock_entity.__iter__ = lambda self: iter(["entity_id", "name", "entity_type"])
        
        mock_chunk1 = MagicMock()
        mock_chunk1.get = lambda key: {"chunk_id": "c1", "doc_id": "doc1", "text": "NEFT in doc1", "breadcrumbs": "Doc1", "section": "S1"}.get(key)
        
        mock_chunk2 = MagicMock()
        mock_chunk2.get = lambda key: {"chunk_id": "c2", "doc_id": "doc2", "text": "NEFT in doc2", "breadcrumbs": "Doc2", "section": "S1"}.get(key)
        
        # Mock query results
        main_result = [
            {"entity": mock_entity, "chunk": mock_chunk1, "paths": []},
            {"entity": mock_entity, "chunk": mock_chunk2, "paths": []}
        ]
        
        chunks_result = [
            {"chunk_id": "c1", "text": "NEFT in doc1", "breadcrumbs": "Doc1", "doc_id": "doc1", "section": "S1"},
            {"chunk_id": "c2", "text": "NEFT in doc2", "breadcrumbs": "Doc2", "doc_id": "doc2", "section": "S1"}
        ]
        
        mock_session.run.side_effect = [
            main_result,  # Main query
            [],  # Common entities query
            chunks_result  # Chunks retrieval
        ]
        
        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=mock_session)
        mock_context.__exit__ = MagicMock(return_value=False)
        graph_retriever.driver.session = MagicMock(return_value=mock_context)
        
        result = graph_retriever._execute_single_entity_comparison("NEFT", depth=3)
        
        # Verify entity appears in multiple documents
        assert len(result.document_groups) >= 1
        assert isinstance(result.document_groups, dict)
    
    def test_comparison_query_with_single_entity(self, graph_retriever):
        """Test comparison query delegates to single entity comparison for one entity."""
        mock_session = MagicMock()
        
        mock_entity = MagicMock()
        mock_entity.get = lambda key: {"entity_id": "e1", "name": "NEFT", "entity_type": "System"}.get(key)
        mock_entity.labels = ["Entity", "System"]
        mock_entity.__getitem__ = lambda self, key: {"entity_id": "e1", "name": "NEFT", "entity_type": "System"}.get(key)
        mock_entity.__iter__ = lambda self: iter(["entity_id", "name", "entity_type"])
        
        mock_chunk = MagicMock()
        mock_chunk.get = lambda key: {"chunk_id": "c1", "doc_id": "doc1", "text": "Text", "breadcrumbs": "Doc1", "section": "S1"}.get(key)
        
        mock_session.run.side_effect = [
            [{"entity": mock_entity, "chunk": mock_chunk, "paths": []}],
            []
        ]
        
        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=mock_session)
        mock_context.__exit__ = MagicMock(return_value=False)
        graph_retriever.driver.session = MagicMock(return_value=mock_context)
        
        result = graph_retriever._execute_comparison_query(["NEFT"], depth=3)
        
        # Should handle single entity comparison
        assert isinstance(result, GraphResult)
    
    def test_comparison_result_includes_all_fields(self, graph_retriever):
        """Test that comparison result includes all required fields."""
        mock_session = MagicMock()
        
        entity_result = [
            {"entity_id": "e1", "name": "NEFT", "entity_type": "System", "canonical_id": "e1", "canonical_name": "NEFT"}
        ]
        
        comparison_result = [{
            "e1": MagicMock(get=lambda k: "e1" if k == "entity_id" else None, labels=["Entity"], 
                           __getitem__=lambda s, k: {"entity_id": "e1", "name": "NEFT"}.get(k),
                           __iter__=lambda s: iter(["entity_id", "name"])),
            "e2": MagicMock(get=lambda k: "e2" if k == "entity_id" else None, labels=["Entity"],
                           __getitem__=lambda s, k: {"entity_id": "e2", "name": "RTGS"}.get(k),
                           __iter__=lambda s: iter(["entity_id", "name"])),
            "paths1": [],
            "paths2": [],
            "chunks1": [],
            "chunks2": []
        }]
        
        mock_session.run.side_effect = [
            entity_result,
            comparison_result,
            []
        ]
        
        mock_context = MagicMock()
        mock_context.__enter__ = MagicMock(return_value=mock_session)
        mock_context.__exit__ = MagicMock(return_value=False)
        graph_retriever.driver.session = MagicMock(return_value=mock_context)
        
        result = graph_retriever._execute_comparison_query(["NEFT", "RTGS"], depth=3)
        
        # Verify all comparison-specific fields are present
        assert hasattr(result, "document_groups")
        assert hasattr(result, "common_entities")
        assert hasattr(result, "differences")
        assert isinstance(result.document_groups, dict)
        assert isinstance(result.common_entities, list)
        assert isinstance(result.differences, list)
