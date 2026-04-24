"""Tests for ResultFusion class."""

import pytest
from src.retrieval.result_fusion import ResultFusion, FusedResults
from src.retrieval.vector_retriever import RetrievedChunk
from src.retrieval.graph_retriever import GraphResult, GraphNode, GraphRelationship


class TestResultFusion:
    """Test suite for ResultFusion class."""
    
    def test_initialization_default_weights(self):
        """Test ResultFusion initialization with default weights."""
        fusion = ResultFusion()
        assert fusion.vector_weight == 0.6
        assert fusion.graph_weight == 0.4
    
    def test_initialization_custom_weights(self):
        """Test ResultFusion initialization with custom weights."""
        fusion = ResultFusion(vector_weight=0.7, graph_weight=0.3)
        assert fusion.vector_weight == 0.7
        assert fusion.graph_weight == 0.3
    
    def test_initialization_invalid_weights(self):
        """Test ResultFusion initialization with invalid weights that don't sum to 1.0."""
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            ResultFusion(vector_weight=0.5, graph_weight=0.6)
    
    def test_fuse_empty_results(self):
        """Test fusing empty vector and graph results."""
        fusion = ResultFusion()
        vector_results = []
        graph_results = GraphResult()
        
        fused = fusion.fuse(vector_results, graph_results)
        
        assert isinstance(fused, FusedResults)
        assert len(fused.chunks) == 0
        assert len(fused.graph_facts) == 0
        assert len(fused.combined_score) == 0
    
    def test_fuse_vector_only_results(self):
        """Test fusing with only vector results."""
        fusion = ResultFusion()
        
        vector_results = [
            RetrievedChunk(
                chunk_id="chunk1",
                text="Test chunk 1",
                breadcrumbs="Doc > Section",
                doc_id="doc1",
                section="section1",
                score=0.9,
                retrieval_source="vector"
            ),
            RetrievedChunk(
                chunk_id="chunk2",
                text="Test chunk 2",
                breadcrumbs="Doc > Section",
                doc_id="doc1",
                section="section2",
                score=0.8,
                retrieval_source="vector"
            )
        ]
        graph_results = GraphResult()
        
        fused = fusion.fuse(vector_results, graph_results)
        
        assert len(fused.chunks) == 2
        assert len(fused.graph_facts) == 0
        assert len(fused.combined_score) == 2
        
        # Check combined scores (0.6 * vector_score + 0.4 * 0.0)
        assert fused.combined_score["chunk1"] == pytest.approx(0.54, rel=0.01)  # 0.6 * 0.9
        assert fused.combined_score["chunk2"] == pytest.approx(0.48, rel=0.01)  # 0.6 * 0.8
        
        # Check chunks are sorted by combined score
        assert fused.chunks[0].chunk_id == "chunk1"
        assert fused.chunks[1].chunk_id == "chunk2"
    
    def test_fuse_graph_only_results(self):
        """Test fusing with only graph results."""
        fusion = ResultFusion()
        
        vector_results = []
        graph_results = GraphResult(
            nodes=[
                GraphNode(
                    node_id="entity1",
                    node_type="System",
                    properties={"name": "NEFT", "canonical_name": "NEFT"}
                ),
                GraphNode(
                    node_id="entity2",
                    node_type="System",
                    properties={"name": "Core Banking", "canonical_name": "Core Banking"}
                )
            ],
            relationships=[
                GraphRelationship(
                    rel_id="rel1",
                    rel_type="DEPENDS_ON",
                    source_id="entity1",
                    target_id="entity2",
                    properties={}
                )
            ],
            chunks=[
                {
                    "chunk_id": "chunk1",
                    "text": "NEFT depends on Core Banking",
                    "breadcrumbs": "Doc > Section",
                    "doc_id": "doc1",
                    "section": "section1"
                }
            ]
        )
        
        fused = fusion.fuse(vector_results, graph_results)
        
        assert len(fused.chunks) == 1
        assert len(fused.graph_facts) == 1
        assert fused.graph_facts[0] == "NEFT DEPENDS_ON Core Banking"
        
        # Check combined score (0.6 * 0.0 + 0.4 * 1.0)
        assert fused.combined_score["chunk1"] == pytest.approx(0.4, rel=0.01)
    
    def test_fuse_deduplicates_chunks(self):
        """Test that fusion deduplicates chunks appearing in both sources."""
        fusion = ResultFusion()
        
        vector_results = [
            RetrievedChunk(
                chunk_id="chunk1",
                text="Test chunk 1",
                breadcrumbs="Doc > Section",
                doc_id="doc1",
                section="section1",
                score=0.9,
                retrieval_source="vector"
            ),
            RetrievedChunk(
                chunk_id="chunk2",
                text="Test chunk 2",
                breadcrumbs="Doc > Section",
                doc_id="doc1",
                section="section2",
                score=0.8,
                retrieval_source="vector"
            )
        ]
        
        graph_results = GraphResult(
            nodes=[],
            relationships=[],
            chunks=[
                {
                    "chunk_id": "chunk1",  # Duplicate
                    "text": "Test chunk 1",
                    "breadcrumbs": "Doc > Section",
                    "doc_id": "doc1",
                    "section": "section1"
                },
                {
                    "chunk_id": "chunk3",  # New chunk
                    "text": "Test chunk 3",
                    "breadcrumbs": "Doc > Section",
                    "doc_id": "doc1",
                    "section": "section3"
                }
            ]
        )
        
        fused = fusion.fuse(vector_results, graph_results)
        
        # Should have 3 unique chunks (chunk1, chunk2, chunk3)
        assert len(fused.chunks) == 3
        chunk_ids = {chunk.chunk_id for chunk in fused.chunks}
        assert chunk_ids == {"chunk1", "chunk2", "chunk3"}
    
    def test_fuse_combines_scores_correctly(self):
        """Test that fusion combines vector and graph scores with correct weights."""
        fusion = ResultFusion(vector_weight=0.6, graph_weight=0.4)
        
        vector_results = [
            RetrievedChunk(
                chunk_id="chunk1",
                text="Test chunk 1",
                breadcrumbs="Doc > Section",
                doc_id="doc1",
                section="section1",
                score=0.8,
                retrieval_source="vector"
            )
        ]
        
        graph_results = GraphResult(
            nodes=[],
            relationships=[],
            chunks=[
                {
                    "chunk_id": "chunk1",
                    "text": "Test chunk 1",
                    "breadcrumbs": "Doc > Section",
                    "doc_id": "doc1",
                    "section": "section1"
                }
            ]
        )
        
        fused = fusion.fuse(vector_results, graph_results)
        
        # Combined score: 0.6 * 0.8 (vector) + 0.4 * 1.0 (graph) = 0.48 + 0.4 = 0.88
        assert fused.combined_score["chunk1"] == pytest.approx(0.88, rel=0.01)
    
    def test_extract_graph_facts_basic_relationship(self):
        """Test extracting basic graph facts from relationships."""
        fusion = ResultFusion()
        
        nodes = [
            GraphNode(
                node_id="entity1",
                node_type="System",
                properties={"name": "NEFT"}
            ),
            GraphNode(
                node_id="entity2",
                node_type="System",
                properties={"name": "Core Banking"}
            )
        ]
        
        relationships = [
            GraphRelationship(
                rel_id="rel1",
                rel_type="DEPENDS_ON",
                source_id="entity1",
                target_id="entity2",
                properties={}
            )
        ]
        
        facts = fusion._extract_graph_facts(nodes, relationships)
        
        assert len(facts) == 1
        assert facts[0] == "NEFT DEPENDS_ON Core Banking"
    
    def test_extract_graph_facts_conflict_relationship(self):
        """Test extracting conflict relationships with metadata."""
        fusion = ResultFusion()
        
        nodes = [
            GraphNode(
                node_id="entity1",
                node_type="Rule",
                properties={"name": "Rule A"}
            ),
            GraphNode(
                node_id="entity2",
                node_type="Rule",
                properties={"name": "Rule B"}
            )
        ]
        
        relationships = [
            GraphRelationship(
                rel_id="rel1",
                rel_type="CONFLICTS_WITH",
                source_id="entity1",
                target_id="entity2",
                properties={
                    "conflict_type": "property",
                    "explanation": "different limits"
                }
            )
        ]
        
        facts = fusion._extract_graph_facts(nodes, relationships)
        
        assert len(facts) == 1
        assert "Rule A CONFLICTS_WITH Rule B" in facts[0]
        assert "reason:" in facts[0]
    
    def test_extract_graph_facts_multiple_relationships(self):
        """Test extracting multiple graph facts."""
        fusion = ResultFusion()
        
        nodes = [
            GraphNode(node_id="e1", node_type="System", properties={"name": "System A"}),
            GraphNode(node_id="e2", node_type="System", properties={"name": "System B"}),
            GraphNode(node_id="e3", node_type="System", properties={"name": "System C"})
        ]
        
        relationships = [
            GraphRelationship(
                rel_id="rel1",
                rel_type="DEPENDS_ON",
                source_id="e1",
                target_id="e2",
                properties={}
            ),
            GraphRelationship(
                rel_id="rel2",
                rel_type="INTEGRATES_WITH",
                source_id="e2",
                target_id="e3",
                properties={}
            )
        ]
        
        facts = fusion._extract_graph_facts(nodes, relationships)
        
        assert len(facts) == 2
        assert "System A DEPENDS_ON System B" in facts
        assert "System B INTEGRATES_WITH System C" in facts
    
    def test_extract_graph_facts_missing_nodes(self):
        """Test extracting facts when relationship references missing nodes."""
        fusion = ResultFusion()
        
        nodes = [
            GraphNode(node_id="e1", node_type="System", properties={"name": "System A"})
        ]
        
        relationships = [
            GraphRelationship(
                rel_id="rel1",
                rel_type="DEPENDS_ON",
                source_id="e1",
                target_id="e2",  # Missing node
                properties={}
            )
        ]
        
        facts = fusion._extract_graph_facts(nodes, relationships)
        
        # Should skip relationship with missing node
        assert len(facts) == 0
    
    def test_fuse_sorts_by_combined_score(self):
        """Test that fused results are sorted by combined score in descending order."""
        fusion = ResultFusion()
        
        vector_results = [
            RetrievedChunk(
                chunk_id="chunk1",
                text="Low score chunk",
                breadcrumbs="Doc > Section",
                doc_id="doc1",
                section="section1",
                score=0.5,
                retrieval_source="vector"
            ),
            RetrievedChunk(
                chunk_id="chunk2",
                text="High score chunk",
                breadcrumbs="Doc > Section",
                doc_id="doc1",
                section="section2",
                score=0.9,
                retrieval_source="vector"
            ),
            RetrievedChunk(
                chunk_id="chunk3",
                text="Medium score chunk",
                breadcrumbs="Doc > Section",
                doc_id="doc1",
                section="section3",
                score=0.7,
                retrieval_source="vector"
            )
        ]
        
        graph_results = GraphResult()
        
        fused = fusion.fuse(vector_results, graph_results)
        
        # Check chunks are sorted by score (descending)
        assert fused.chunks[0].chunk_id == "chunk2"  # 0.9 * 0.6 = 0.54
        assert fused.chunks[1].chunk_id == "chunk3"  # 0.7 * 0.6 = 0.42
        assert fused.chunks[2].chunk_id == "chunk1"  # 0.5 * 0.6 = 0.30
    
    def test_fuse_preserves_chunk_metadata(self):
        """Test that fusion preserves all chunk metadata."""
        fusion = ResultFusion()
        
        vector_results = [
            RetrievedChunk(
                chunk_id="chunk1",
                text="Test chunk",
                breadcrumbs="Doc Title > Section > Subsection",
                doc_id="doc123",
                section="section_abc",
                score=0.85,
                retrieval_source="vector",
                vector_score=0.85,
                bm25_score=0.75
            )
        ]
        
        graph_results = GraphResult()
        
        fused = fusion.fuse(vector_results, graph_results)
        
        chunk = fused.chunks[0]
        assert chunk.chunk_id == "chunk1"
        assert chunk.text == "Test chunk"
        assert chunk.breadcrumbs == "Doc Title > Section > Subsection"
        assert chunk.doc_id == "doc123"
        assert chunk.section == "section_abc"
        assert chunk.retrieval_source == "vector"
        assert chunk.vector_score == 0.85
        assert chunk.bm25_score == 0.75
