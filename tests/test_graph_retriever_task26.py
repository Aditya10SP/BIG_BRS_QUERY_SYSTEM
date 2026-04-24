"""Tests for Task 26: Conflict and Process Queries functionality."""

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


class TestConflictQuerySupport:
    """Test suite for conflict query support (Task 26.1)."""
    
    def test_detect_query_pattern_rule(self, graph_retriever):
        """Test rule pattern detection."""
        queries = [
            "What rules apply to NEFT?",
            "Show risk policies for payment systems",
            "Which regulations apply to Core Banking?"
        ]
        
        for query in queries:
            pattern = graph_retriever._detect_query_pattern(query)
            assert pattern == "rule", f"Failed for query: {query}"
    
    def test_conflict_categorization_property(self, graph_retriever):
        """Test conflict categorization for property conflicts."""
        # Create mock entities
        e1 = MagicMock()
        e1.get = lambda key, default=None: {"entity_id": "e1", "entity_type": "System", "name": "NEFT"}.get(key, default)
        
        e2 = MagicMock()
        e2.get = lambda key, default=None: {"entity_id": "e2", "entity_type": "System", "name": "NEFT"}.get(key, default)
        
        # Create mock relationship without explicit conflict_type
        rel = MagicMock()
        rel.get = lambda key, default=None: None
        rel.__iter__ = lambda self: iter([])
        
        conflict_type = graph_retriever._categorize_conflict(e1, e2, rel)
        
        # Should default to property conflict
        assert conflict_type == "property"
    
    def test_conflict_categorization_rule(self, graph_retriever):
        """Test conflict categorization for rule conflicts."""
        # Create mock entities with Rule type
        e1 = MagicMock()
        e1.get = lambda key, default=None: {"entity_id": "e1", "entity_type": "Rule", "name": "Rule A"}.get(key, default)
        
        e2 = MagicMock()
        e2.get = lambda key, default=None: {"entity_id": "e2", "entity_type": "Rule", "name": "Rule B"}.get(key, default)
        
        rel = MagicMock()
        rel.get = lambda key, default=None: None
        rel.__iter__ = lambda self: iter([])
        
        conflict_type = graph_retriever._categorize_conflict(e1, e2, rel)
        
        assert conflict_type == "rule"
    
    def test_conflict_categorization_workflow(self, graph_retriever):
        """Test conflict categorization for workflow conflicts."""
        # Create mock entities with Workflow type
        e1 = MagicMock()
        e1.get = lambda key, default=None: {"entity_id": "e1", "entity_type": "Workflow", "name": "Flow A"}.get(key, default)
        
        e2 = MagicMock()
        e2.get = lambda key, default=None: {"entity_id": "e2", "entity_type": "System", "name": "System B"}.get(key, default)
        
        rel = MagicMock()
        rel.get = lambda key, default=None: None
        rel.__iter__ = lambda self: iter([])
        
        conflict_type = graph_retriever._categorize_conflict(e1, e2, rel)
        
        assert conflict_type == "workflow"
    
    def test_conflict_severity_critical(self, graph_retriever):
        """Test conflict severity computation for critical conflicts."""
        # System + Rule conflict = critical
        e1 = MagicMock()
        e1.get = lambda key, default=None: {"entity_id": "e1", "entity_type": "System", "name": "NEFT"}.get(key, default)
        
        e2 = MagicMock()
        e2.get = lambda key, default=None: {"entity_id": "e2", "entity_type": "System", "name": "RTGS"}.get(key, default)
        
        rel = MagicMock()
        
        severity = graph_retriever._compute_conflict_severity(e1, e2, rel, "rule")
        
        assert severity == "critical"
    
    def test_conflict_severity_high(self, graph_retriever):
        """Test conflict severity computation for high severity conflicts."""
        # PaymentMode + workflow conflict = high (4 + 2 = 6, which is >= 5)
        e1 = MagicMock()
        e1.get = lambda key, default=None: {"entity_id": "e1", "entity_type": "PaymentMode", "name": "NEFT"}.get(key, default)
        
        e2 = MagicMock()
        e2.get = lambda key, default=None: {"entity_id": "e2", "entity_type": "PaymentMode", "name": "RTGS"}.get(key, default)
        
        rel = MagicMock()
        
        severity = graph_retriever._compute_conflict_severity(e1, e2, rel, "workflow")
        
        # PaymentMode (4) + workflow (2) = 6, which is >= 5 but < 7, so "high"
        assert severity == "high"
    
    def test_conflict_severity_medium(self, graph_retriever):
        """Test conflict severity computation for medium severity conflicts."""
        # Workflow + property conflict = medium
        e1 = MagicMock()
        e1.get = lambda key, default=None: {"entity_id": "e1", "entity_type": "Workflow", "name": "Flow A"}.get(key, default)
        
        e2 = MagicMock()
        e2.get = lambda key, default=None: {"entity_id": "e2", "entity_type": "Workflow", "name": "Flow B"}.get(key, default)
        
        rel = MagicMock()
        
        severity = graph_retriever._compute_conflict_severity(e1, e2, rel, "property")
        
        assert severity == "medium"
    
    def test_conflict_severity_low(self, graph_retriever):
        """Test conflict severity computation for low severity conflicts."""
        # Entity + property conflict = low (1 + 1 = 2, which is < 3)
        e1 = MagicMock()
        e1.get = lambda key, default=None: {"entity_id": "e1", "entity_type": "Entity", "name": "Entity A"}.get(key, default)
        
        e2 = MagicMock()
        e2.get = lambda key, default=None: {"entity_id": "e2", "entity_type": "Entity", "name": "Entity B"}.get(key, default)
        
        rel = MagicMock()
        
        severity = graph_retriever._compute_conflict_severity(e1, e2, rel, "property")
        
        # Entity (1) + property (1) = 2, which is < 3, so "low"
        assert severity == "low"
    
    def test_severity_to_score(self, graph_retriever):
        """Test severity to score conversion."""
        assert graph_retriever._severity_to_score("critical") == 4
        assert graph_retriever._severity_to_score("high") == 3
        assert graph_retriever._severity_to_score("medium") == 2
        assert graph_retriever._severity_to_score("low") == 1
        assert graph_retriever._severity_to_score("unknown") == 0
    
    def test_process_conflict_result_sorting(self, graph_retriever):
        """Test that conflicts are sorted by severity."""
        # Create mock result with multiple conflicts
        e1 = MagicMock()
        e1.get = lambda key, default=None: {"entity_id": "e1", "entity_type": "Field", "name": "Field A"}.get(key, default)
        e1.labels = ["Entity"]
        e1.__getitem__ = lambda self, k: {"entity_id": "e1", "entity_type": "Field", "name": "Field A"}.get(k)
        e1.__iter__ = lambda self: iter(["entity_id", "entity_type", "name"])
        
        e2 = MagicMock()
        e2.get = lambda key, default=None: {"entity_id": "e2", "entity_type": "System", "name": "NEFT"}.get(key, default)
        e2.labels = ["Entity"]
        e2.__getitem__ = lambda self, k: {"entity_id": "e2", "entity_type": "System", "name": "NEFT"}.get(k)
        e2.__iter__ = lambda self: iter(["entity_id", "entity_type", "name"])
        
        rel1 = MagicMock()
        rel1.get = lambda key, default=None: None
        rel1.id = 1
        rel1.type = "CONFLICTS_WITH"
        rel1.start_node = e1
        rel1.end_node = e2
        rel1.__iter__ = lambda self: iter([])
        
        rel2 = MagicMock()
        rel2.get = lambda key, default=None: None
        rel2.id = 2
        rel2.type = "CONFLICTS_WITH"
        rel2.start_node = e2
        rel2.end_node = e1
        rel2.__iter__ = lambda self: iter([])
        
        mock_result = [
            {"e1": e1, "e2": e2, "r": rel1, "chunks1": [], "chunks2": []},
            {"e1": e2, "e2": e1, "r": rel2, "chunks1": [], "chunks2": []}
        ]
        
        result = graph_retriever._process_conflict_result(mock_result)
        
        # Verify conflicts are sorted by severity (high to low)
        assert len(result.conflicts) >= 1
        for i in range(len(result.conflicts) - 1):
            assert result.conflicts[i]["severity_score"] >= result.conflicts[i + 1]["severity_score"]


class TestProcessChainTraversal:
    """Test suite for process chain traversal (Task 26.2)."""
    
    def test_construct_process_chains_simple(self, graph_retriever):
        """Test constructing a simple linear process chain."""
        # Create mock workflow nodes
        wf_node = MagicMock()
        wf_node.get = lambda key: {"entity_id": "wf1", "name": "Payment Flow"}.get(key)
        
        # Create mock nodes
        node1 = GraphNode(node_id="step1", node_type="Workflow", properties={"name": "Step 1"})
        node2 = GraphNode(node_id="step2", node_type="Workflow", properties={"name": "Step 2"})
        node3 = GraphNode(node_id="step3", node_type="Workflow", properties={"name": "Step 3"})
        
        all_nodes = {
            "wf1": GraphNode(node_id="wf1", node_type="Workflow", properties={"name": "Payment Flow"}),
            "step1": node1,
            "step2": node2,
            "step3": node3
        }
        
        # Create relationships
        rel1 = GraphRelationship(rel_id="r1", rel_type="NEXT_STEP", source_id="wf1", target_id="step1")
        rel2 = GraphRelationship(rel_id="r2", rel_type="NEXT_STEP", source_id="step1", target_id="step2")
        rel3 = GraphRelationship(rel_id="r3", rel_type="NEXT_STEP", source_id="step2", target_id="step3")
        
        relationships = [rel1, rel2, rel3]
        
        chains = graph_retriever._construct_process_chains([wf_node], relationships, all_nodes)
        
        assert len(chains) >= 1
        # Verify chain has steps
        assert chains[0]["step_count"] >= 1
    
    def test_construct_process_chains_branching(self, graph_retriever):
        """Test constructing a branching process chain."""
        wf_node = MagicMock()
        wf_node.get = lambda key: {"entity_id": "wf1", "name": "Payment Flow"}.get(key)
        
        all_nodes = {
            "wf1": GraphNode(node_id="wf1", node_type="Workflow", properties={"name": "Payment Flow"}),
            "step1": GraphNode(node_id="step1", node_type="Workflow", properties={"name": "Step 1"}),
            "step2a": GraphNode(node_id="step2a", node_type="Workflow", properties={"name": "Step 2A"}),
            "step2b": GraphNode(node_id="step2b", node_type="Workflow", properties={"name": "Step 2B"})
        }
        
        # Create branching relationships
        rel1 = GraphRelationship(rel_id="r1", rel_type="NEXT_STEP", source_id="wf1", target_id="step1")
        rel2a = GraphRelationship(rel_id="r2a", rel_type="NEXT_STEP", source_id="step1", target_id="step2a")
        rel2b = GraphRelationship(rel_id="r2b", rel_type="NEXT_STEP", source_id="step1", target_id="step2b")
        
        relationships = [rel1, rel2a, rel2b]
        
        chains = graph_retriever._construct_process_chains([wf_node], relationships, all_nodes)
        
        # Should have multiple chains due to branching
        assert len(chains) >= 1
        # At least one chain should have branches
        assert any(chain["has_branches"] for chain in chains)
    
    def test_detect_process_gaps_incomplete_chain(self, graph_retriever):
        """Test detecting gaps in incomplete process chains."""
        process_chains = [
            {
                "workflow_id": "wf1",
                "workflow_name": "Payment Flow",
                "steps": [
                    {"step_id": "step1", "step_name": "Step 1", "step_type": "Workflow", "properties": {}}
                ],
                "step_count": 1,
                "is_complete": False,
                "has_branches": False,
                "branch_points": []
            }
        ]
        
        all_nodes = {
            "step1": GraphNode(node_id="step1", node_type="Workflow", properties={"name": "Step 1"})
        }
        
        gaps = graph_retriever._detect_process_gaps(process_chains, all_nodes, {})
        
        # Should detect incomplete chain
        assert len(gaps) >= 1
        assert any("incomplete" in gap.lower() for gap in gaps)
    
    def test_detect_process_gaps_orphaned_steps(self, graph_retriever):
        """Test detecting orphaned workflow steps."""
        process_chains = [
            {
                "workflow_id": "wf1",
                "workflow_name": "Payment Flow",
                "steps": [
                    {"step_id": "step1", "step_name": "Step 1", "step_type": "Workflow", "properties": {}}
                ],
                "step_count": 1,
                "is_complete": True,
                "has_branches": False,
                "branch_points": []
            }
        ]
        
        all_nodes = {
            "step1": GraphNode(node_id="step1", node_type="Workflow", properties={"name": "Step 1"}),
            "orphan": GraphNode(node_id="orphan", node_type="Workflow", properties={"name": "Orphaned Step"})
        }
        
        gaps = graph_retriever._detect_process_gaps(process_chains, all_nodes, {})
        
        # Should detect orphaned step
        assert len(gaps) >= 1
        assert any("orphan" in gap.lower() for gap in gaps)


class TestRiskRuleAnalysis:
    """Test suite for risk rule analysis (Task 26.3)."""
    
    def test_compute_rule_specificity_few_entities(self, graph_retriever):
        """Test rule specificity with few applicable entities (more specific)."""
        rule_dict = {
            "conditions": "IF amount > 1000 AND type = 'NEFT'",
            "scope": "specific"
        }
        
        applicable_entities = [
            {"entity_id": "e1", "entity_name": "NEFT", "entity_type": "System"}
        ]
        
        specificity = graph_retriever._compute_rule_specificity(rule_dict, applicable_entities)
        
        # Should have high specificity (few entities, specific scope, multiple conditions)
        assert specificity > 10.0
    
    def test_compute_rule_specificity_many_entities(self, graph_retriever):
        """Test rule specificity with many applicable entities (less specific)."""
        rule_dict = {
            "conditions": "IF amount > 0",
            "scope": "general"
        }
        
        applicable_entities = [
            {"entity_id": f"e{i}", "entity_name": f"Entity {i}", "entity_type": "Entity"}
            for i in range(10)
        ]
        
        specificity = graph_retriever._compute_rule_specificity(rule_dict, applicable_entities)
        
        # Should have lower specificity (many entities, general scope)
        assert specificity < 10.0
    
    def test_detect_rule_overlaps(self, graph_retriever):
        """Test detecting overlapping rules."""
        entity_to_rules = {
            "entity1": ["rule1", "rule2", "rule3"],  # 3 rules apply to entity1
            "entity2": ["rule1"],  # Only 1 rule applies to entity2
            "entity3": ["rule2", "rule3"]  # 2 rules apply to entity3
        }
        
        rules = [
            {"rule_id": "rule1", "rule_name": "Rule 1"},
            {"rule_id": "rule2", "rule_name": "Rule 2"},
            {"rule_id": "rule3", "rule_name": "Rule 3"}
        ]
        
        overlaps = graph_retriever._detect_rule_overlaps(entity_to_rules, rules)
        
        # rule1, rule2, rule3 should all have overlaps on entity1
        assert "rule1" in overlaps
        assert "rule2" in overlaps
        assert "rule3" in overlaps
        
        # rule1 should overlap with rule2 and rule3 on entity1
        assert len(overlaps["rule1"]) >= 2
    
    def test_detect_rule_overlaps_no_overlaps(self, graph_retriever):
        """Test detecting rule overlaps when none exist."""
        entity_to_rules = {
            "entity1": ["rule1"],
            "entity2": ["rule2"],
            "entity3": ["rule3"]
        }
        
        rules = [
            {"rule_id": "rule1", "rule_name": "Rule 1"},
            {"rule_id": "rule2", "rule_name": "Rule 2"},
            {"rule_id": "rule3", "rule_name": "Rule 3"}
        ]
        
        overlaps = graph_retriever._detect_rule_overlaps(entity_to_rules, rules)
        
        # No overlaps should be detected
        assert len(overlaps) == 0
    
    def test_rule_query_result_includes_rankings(self, graph_retriever):
        """Test that rule query results include rankings."""
        mock_session = MagicMock()
        
        # Create mock rule
        mock_rule = MagicMock()
        mock_rule.get = lambda key: {"entity_id": "rule1", "name": "Rule 1", "priority": 1}.get(key)
        mock_rule.labels = ["Entity"]
        mock_rule.__getitem__ = lambda self, k: {
            "entity_id": "rule1",
            "name": "Rule 1",
            "entity_type": "Rule",
            "conditions": "IF amount > 1000",
            "actions": "REJECT",
            "scope": "specific",
            "priority": 1
        }.get(k)
        mock_rule.__iter__ = lambda self: iter(["entity_id", "name", "entity_type", "conditions", "actions", "scope", "priority"])
        
        mock_result = [{
            "rule": mock_rule,
            "applicable_entities": [],
            "applies_to_rels": []
        }]
        
        result = graph_retriever._process_rule_result(mock_result, mock_session)
        
        # Verify rules have rankings
        assert len(result.rules) >= 1
        assert "rank" in result.rules[0]
        assert "specificity" in result.rules[0]
        assert "priority" in result.rules[0]


class TestGraphResultDataClass:
    """Test suite for GraphResult data class with new fields."""
    
    def test_graph_result_with_conflicts(self):
        """Test GraphResult with conflicts field."""
        conflicts = [
            {
                "entity1_id": "e1",
                "entity2_id": "e2",
                "conflict_type": "property",
                "severity": "high"
            }
        ]
        
        result = GraphResult(conflicts=conflicts)
        
        assert len(result.conflicts) == 1
        assert result.conflicts[0]["conflict_type"] == "property"
        assert result.conflicts[0]["severity"] == "high"
    
    def test_graph_result_with_process_chains(self):
        """Test GraphResult with process_chains field."""
        process_chains = [
            {
                "workflow_id": "wf1",
                "workflow_name": "Payment Flow",
                "steps": [],
                "step_count": 3,
                "is_complete": True,
                "has_branches": False
            }
        ]
        
        result = GraphResult(process_chains=process_chains)
        
        assert len(result.process_chains) == 1
        assert result.process_chains[0]["workflow_name"] == "Payment Flow"
        assert result.process_chains[0]["is_complete"] is True
    
    def test_graph_result_with_process_gaps(self):
        """Test GraphResult with process_gaps field."""
        process_gaps = [
            "Incomplete process chain in workflow 'Payment Flow': last step 'Step 3' has no NEXT_STEP relationship"
        ]
        
        result = GraphResult(process_gaps=process_gaps)
        
        assert len(result.process_gaps) == 1
        assert "incomplete" in result.process_gaps[0].lower()
    
    def test_graph_result_with_rules(self):
        """Test GraphResult with rules field."""
        rules = [
            {
                "rule_id": "rule1",
                "rule_name": "Transaction Limit Rule",
                "conditions": "IF amount > 1000",
                "actions": "REJECT",
                "scope": "NEFT transactions",
                "specificity": 15.0,
                "rank": 1
            }
        ]
        
        result = GraphResult(rules=rules)
        
        assert len(result.rules) == 1
        assert result.rules[0]["rule_name"] == "Transaction Limit Rule"
        assert result.rules[0]["rank"] == 1
