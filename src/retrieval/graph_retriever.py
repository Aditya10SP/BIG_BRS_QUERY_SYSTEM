"""Graph-based retrieval using Neo4j knowledge graph traversal."""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import ServiceUnavailable, Neo4jError

from src.utils.cache import CypherQueryCache

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """
    Represents a node in the graph result.
    
    Attributes:
        node_id: Unique node identifier (entity_id, chunk_id, etc.)
        node_type: Type of node (System, PaymentMode, Workflow, Rule, Field, Chunk, Document, Section)
        properties: Dictionary of node properties
    """
    node_id: str
    node_type: str
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphRelationship:
    """
    Represents a relationship in the graph result.
    
    Attributes:
        rel_id: Unique relationship identifier
        rel_type: Type of relationship (DEPENDS_ON, INTEGRATES_WITH, NEXT_STEP, etc.)
        source_id: Source node identifier
        target_id: Target node identifier
        properties: Dictionary of relationship properties
    """
    rel_id: str
    rel_type: str
    source_id: str
    target_id: str
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphResult:
    """
    Represents the result of a graph retrieval operation.
    
    Attributes:
        nodes: List of GraphNode objects in the result
        relationships: List of GraphRelationship objects in the result
        chunks: List of associated chunk dictionaries with text content
        impact_radius: Number of affected systems (for dependency queries)
        circular_dependencies: List of circular dependency chains detected
        document_groups: Dictionary grouping results by document_id (for comparison queries)
        common_entities: List of entity IDs that appear across multiple documents
        differences: List of difference descriptions between documents
        conflicts: List of conflict dictionaries with categorization and severity (for conflict queries)
        process_chains: List of process chain dictionaries (for workflow queries)
        process_gaps: List of gap descriptions for incomplete workflows
        rules: List of rule dictionaries with applicability info (for rule queries)
    """
    nodes: List[GraphNode] = field(default_factory=list)
    relationships: List[GraphRelationship] = field(default_factory=list)
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    impact_radius: Optional[int] = None
    circular_dependencies: List[List[str]] = field(default_factory=list)
    document_groups: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    common_entities: List[str] = field(default_factory=list)
    differences: List[str] = field(default_factory=list)
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    process_chains: List[Dict[str, Any]] = field(default_factory=list)
    process_gaps: List[str] = field(default_factory=list)
    rules: List[Dict[str, Any]] = field(default_factory=list)


class GraphRetriever:
    """
    Retrieves relevant subgraphs using Cypher queries on Neo4j knowledge graph.
    
    This class implements graph-based retrieval for relational and structural queries.
    It supports multiple query patterns:
    1. Dependency queries (forward/backward traversal)
    2. Integration queries (system connections)
    3. Workflow queries (process chains)
    4. Conflict queries (contradictions)
    5. Comparison queries (entity comparisons)
    
    The retriever extracts entities from queries, detects query patterns, generates
    appropriate Cypher queries, and retrieves relevant subgraphs with depth limiting.
    
    Attributes:
        driver: Neo4j driver instance
        max_depth: Maximum traversal depth in hops (default: 3)
    """
    
    # Query pattern keywords for classification
    DEPENDENCY_KEYWORDS = {
        "depend", "depends", "dependency", "dependencies", "require", "requires",
        "prerequisite", "needed", "impact", "affect", "affects"
    }
    
    INTEGRATION_KEYWORDS = {
        "integrate", "integrates", "integration", "connect", "connects", "connection",
        "interface", "communicate", "communicates"
    }
    
    WORKFLOW_KEYWORDS = {
        "workflow", "process", "flow", "step", "steps", "procedure", "sequence"
    }
    
    CONFLICT_KEYWORDS = {
        "conflict", "conflicts", "contradiction", "contradicts", "inconsistent",
        "inconsistency", "disagree", "disagrees"
    }
    
    COMPARISON_KEYWORDS = {
        "compare", "comparison", "difference", "differences", "similar", "similarity",
        "versus", "vs", "between"
    }
    
    RULE_KEYWORDS = {
        "rule", "rules", "policy", "policies", "regulation", "regulations",
        "risk", "compliance", "applies", "applicable"
    }
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, max_depth: int = 3, enable_cache: bool = True, cache_size: int = 1000):
        """
        Initialize GraphRetriever with Neo4j connection.
        
        Args:
            neo4j_uri: Neo4j connection URI (e.g., "bolt://localhost:7687")
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            max_depth: Maximum traversal depth in hops (default: 3)
            enable_cache: Whether to enable Cypher query caching (default: True)
            cache_size: Maximum number of query results to cache (default: 1000)
        
        Raises:
            ServiceUnavailable: If connection to Neo4j fails
        """
        self.neo4j_uri = neo4j_uri
        self.max_depth = max_depth
        self.driver: Optional[Driver] = None
        self.enable_cache = enable_cache
        
        # Initialize cache if enabled
        if enable_cache:
            self.cache = CypherQueryCache(max_size=cache_size)
            logger.info(f"Cypher query cache enabled with size={cache_size}")
        else:
            self.cache = None
            logger.info("Cypher query cache disabled")
        
        logger.info(f"Initializing GraphRetriever with Neo4j at {neo4j_uri}, max_depth={max_depth}")
        
        try:
            self.driver = GraphDatabase.driver(
                neo4j_uri,
                auth=(neo4j_user, neo4j_password)
            )
            # Verify connectivity
            self.driver.verify_connectivity()
            logger.info("Connected to Neo4j successfully")
        except ServiceUnavailable as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def close(self) -> None:
        """Close Neo4j driver connection."""
        if self.driver:
            self.driver.close()
            logger.info("Closed Neo4j driver connection")
    
    def retrieve(self, query: str, max_depth: Optional[int] = None) -> GraphResult:
        """
        Extract entities from query and retrieve relevant subgraph.
        
        This method:
        1. Extracts entity mentions from the query
        2. Detects query pattern (dependency, integration, workflow, conflict, comparison)
        3. Generates appropriate Cypher query
        4. Executes graph traversal with depth limiting
        5. Retrieves associated chunk text via MENTIONS relationships
        
        Args:
            query: Search query string
            max_depth: Override default max traversal depth (optional)
        
        Returns:
            GraphResult with nodes, relationships, and associated chunks
        
        Raises:
            ValueError: If query is empty
            Neo4jError: If graph query execution fails
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        depth = max_depth if max_depth is not None else self.max_depth
        
        logger.info(f"Retrieving graph for query: '{query}' (max_depth={depth})")
        
        try:
            # Extract entity mentions from query
            entity_names = self._extract_entity_mentions(query)
            
            logger.debug(f"Extracted entity mentions: {entity_names}")
            
            # Detect query pattern
            pattern = self._detect_query_pattern(query)
            
            logger.debug(f"Detected query pattern: {pattern}")
            
            # Generate and execute Cypher query based on pattern
            if pattern == "dependency":
                result = self._execute_dependency_query(entity_names, depth)
            elif pattern == "integration":
                result = self._execute_integration_query(entity_names, depth)
            elif pattern == "workflow":
                result = self._execute_workflow_query(entity_names, depth)
            elif pattern == "conflict":
                result = self._execute_conflict_query(entity_names, depth)
            elif pattern == "comparison":
                result = self._execute_comparison_query(entity_names, depth)
            elif pattern == "rule":
                result = self._execute_rule_query(entity_names, depth)
            else:
                # Default: general entity retrieval
                result = self._execute_general_query(entity_names, depth)
            
            logger.info(
                f"Retrieved graph: {len(result.nodes)} nodes, "
                f"{len(result.relationships)} relationships, "
                f"{len(result.chunks)} chunks"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to retrieve graph: {str(e)}", extra={"query": query})
            raise
    
    def _extract_entity_mentions(self, query: str) -> List[str]:
        """
        Extract entity mentions from query using simple heuristics.
        
        Looks for:
        - Capitalized words (potential entity names)
        - Acronyms (all caps, 2-6 letters)
        - Quoted strings
        
        Args:
            query: Query string
        
        Returns:
            List of potential entity names
        """
        entity_names = []
        
        # Extract quoted strings
        quoted = re.findall(r'"([^"]+)"', query)
        entity_names.extend(quoted)
        
        # Extract acronyms (2-6 uppercase letters)
        acronyms = re.findall(r'\b[A-Z]{2,6}\b', query)
        entity_names.extend(acronyms)
        
        # Extract capitalized words (potential proper nouns)
        # Skip common words
        common_words = {"What", "How", "Why", "When", "Where", "Which", "Who", "Does", "Is", "Are"}
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        entity_names.extend([word for word in capitalized if word not in common_words])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for name in entity_names:
            name_lower = name.lower()
            if name_lower not in seen:
                seen.add(name_lower)
                unique_entities.append(name)
        
        return unique_entities
    
    def _detect_query_pattern(self, query: str) -> str:
        """
        Detect query pattern based on keywords.
        
        Args:
            query: Query string
        
        Returns:
            Pattern name: "dependency", "integration", "workflow", "conflict", "comparison", "rule", or "general"
        """
        query_lower = query.lower()
        
        # Check for each pattern
        if any(keyword in query_lower for keyword in self.DEPENDENCY_KEYWORDS):
            return "dependency"
        
        if any(keyword in query_lower for keyword in self.INTEGRATION_KEYWORDS):
            return "integration"
        
        if any(keyword in query_lower for keyword in self.WORKFLOW_KEYWORDS):
            return "workflow"
        
        if any(keyword in query_lower for keyword in self.CONFLICT_KEYWORDS):
            return "conflict"
        
        if any(keyword in query_lower for keyword in self.COMPARISON_KEYWORDS):
            return "comparison"
        
        if any(keyword in query_lower for keyword in self.RULE_KEYWORDS):
            return "rule"
        
        return "general"
    
    def _execute_dependency_query(self, entity_names: List[str], depth: int) -> GraphResult:
        """
        Execute dependency query (forward and backward traversal).
        
        Finds:
        - Forward dependencies: What depends on X?
        - Backward dependencies: What does X depend on?
        - Impact radius: Number of affected systems
        - Circular dependencies: Detects and reports circular dependency chains
        
        Args:
            entity_names: List of entity names to query
            depth: Maximum traversal depth
        
        Returns:
            GraphResult with dependency subgraph, impact radius, and circular dependencies
        """
        if not entity_names:
            return GraphResult()
        
        with self.driver.session() as session:
            # Query for both forward and backward dependencies
            cypher = f"""
            UNWIND $entity_names AS entity_name
            MATCH (e)
            WHERE (e.name = entity_name OR e.canonical_name = entity_name)
              AND (e:System OR e:Entity)
            
            // Forward dependencies (what depends on this entity)
            OPTIONAL MATCH forward_path = (dependent)-[:DEPENDS_ON*1..{depth}]->(e)
            WHERE dependent:System OR dependent:Entity
            
            // Backward dependencies (what this entity depends on)
            OPTIONAL MATCH backward_path = (e)-[:DEPENDS_ON*1..{depth}]->(dependency)
            WHERE dependency:System OR dependency:Entity
            
            WITH e, 
                 collect(DISTINCT forward_path) AS forward_paths,
                 collect(DISTINCT backward_path) AS backward_paths
            
            // Combine all paths
            WITH e, forward_paths + backward_paths AS all_paths
            UNWIND all_paths AS path
            WITH e, path
            WHERE path IS NOT NULL
            
            RETURN e AS center_node,
                   nodes(path) AS path_nodes,
                   relationships(path) AS path_rels
            """
            
            result = session.run(cypher, entity_names=entity_names)
            graph_result = self._process_graph_result(result)
            
            # Compute impact radius (number of unique systems affected)
            impact_radius = self._compute_impact_radius(entity_names, depth, session)
            graph_result.impact_radius = impact_radius
            
            # Detect circular dependencies
            circular_deps = self._detect_circular_dependencies(entity_names, depth, session)
            graph_result.circular_dependencies = circular_deps
            
            logger.info(
                f"Dependency query completed: impact_radius={impact_radius}, "
                f"circular_dependencies={len(circular_deps)}"
            )
            
            return graph_result
    
    def _execute_integration_query(self, entity_names: List[str], depth: int) -> GraphResult:
        """
        Execute integration query (system connections).
        
        Finds paths between entities using INTEGRATES_WITH and DEPENDS_ON relationships.
        
        Args:
            entity_names: List of entity names to query
            depth: Maximum traversal depth
        
        Returns:
            GraphResult with integration subgraph
        """
        if not entity_names:
            return GraphResult()
        
        with self.driver.session() as session:
            if len(entity_names) >= 2:
                # Query for paths between two entities
                cypher = f"""
                MATCH (e1:Entity)
                WHERE e1.name = $entity1 OR e1.canonical_name = $entity1
                
                MATCH (e2:Entity)
                WHERE e2.name = $entity2 OR e2.canonical_name = $entity2
                
                MATCH path = (e1)-[:INTEGRATES_WITH|DEPENDS_ON*1..{depth}]-(e2)
                
                RETURN nodes(path) AS path_nodes,
                       relationships(path) AS path_rels
                ORDER BY length(path)
                LIMIT 10
                """
                
                result = session.run(cypher, entity1=entity_names[0], entity2=entity_names[1])
            else:
                # Query for all integrations of a single entity
                cypher = f"""
                MATCH (e:Entity)
                WHERE e.name = $entity_name OR e.canonical_name = $entity_name
                
                MATCH path = (e)-[:INTEGRATES_WITH|DEPENDS_ON*1..{depth}]-(related:Entity)
                
                RETURN e AS center_node,
                       nodes(path) AS path_nodes,
                       relationships(path) AS path_rels
                LIMIT 20
                """
                
                result = session.run(cypher, entity_name=entity_names[0])
            
            return self._process_graph_result(result)
    
    def _execute_workflow_query(self, entity_names: List[str], depth: int) -> GraphResult:
        """
        Execute workflow query (process chains).
        
        Finds workflow sequences using NEXT_STEP relationships with support for:
        - Complete process chains from start to end
        - Branching workflows with conditional paths
        - Gap detection for incomplete chains
        
        Args:
            entity_names: List of entity names to query
            depth: Maximum traversal depth
        
        Returns:
            GraphResult with workflow subgraph, process chains, and gap information
        """
        if not entity_names:
            return GraphResult()
        
        with self.driver.session() as session:
            cypher = f"""
            UNWIND $entity_names AS entity_name
            MATCH (wf:Entity)
            WHERE (wf.entity_type = 'Workflow' OR wf.name CONTAINS 'Flow' OR wf.name CONTAINS 'Process')
              AND (wf.name = entity_name OR wf.canonical_name = entity_name OR wf.name CONTAINS entity_name)
            
            // Get workflow steps
            OPTIONAL MATCH path = (wf)-[:NEXT_STEP*0..{depth}]->(step:Entity)
            
            RETURN wf AS center_node,
                   nodes(path) AS path_nodes,
                   relationships(path) AS path_rels
            ORDER BY length(path)
            """
            
            result = session.run(cypher, entity_names=entity_names)
            
            # Process result with process chain analysis
            graph_result = self._process_workflow_result(result, entity_names, depth, session)
            
            logger.info(
                f"Workflow query completed: {len(graph_result.process_chains)} process chains, "
                f"{len(graph_result.process_gaps)} gaps detected"
            )
            
            return graph_result
    
    def _process_workflow_result(
        self, 
        result, 
        entity_names: List[str], 
        depth: int, 
        session: Session
    ) -> GraphResult:
        """
        Process workflow query result with process chain construction and gap detection.
        
        Args:
            result: Neo4j query result
            entity_names: Original entity names queried
            depth: Maximum traversal depth
            session: Neo4j session for additional queries
        
        Returns:
            GraphResult with process chains and gaps
        """
        # First, process the basic graph result
        nodes_dict: Dict[str, GraphNode] = {}
        relationships_dict: Dict[str, GraphRelationship] = {}
        chunk_ids = set()
        workflow_nodes = []
        
        for record in result:
            # Process center node
            if "center_node" in record and record["center_node"]:
                center_node = record["center_node"]
                self._add_node_to_dict(center_node, nodes_dict)
                workflow_nodes.append(center_node)
            
            # Process path nodes
            if "path_nodes" in record and record["path_nodes"]:
                for node in record["path_nodes"]:
                    if node:
                        self._add_node_to_dict(node, nodes_dict)
                        if "source_chunk_id" in dict(node):
                            chunk_ids.add(dict(node)["source_chunk_id"])
            
            # Process path relationships
            if "path_rels" in record and record["path_rels"]:
                for rel in record["path_rels"]:
                    if rel:
                        self._add_relationship_to_dict(rel, relationships_dict)
        
        # Construct process chains
        process_chains = self._construct_process_chains(
            workflow_nodes, 
            list(relationships_dict.values()),
            nodes_dict
        )
        
        # Detect gaps in process chains
        process_gaps = self._detect_process_gaps(
            process_chains,
            nodes_dict,
            relationships_dict
        )
        
        # Retrieve chunks
        chunks = self._retrieve_chunks(list(chunk_ids))
        
        return GraphResult(
            nodes=list(nodes_dict.values()),
            relationships=list(relationships_dict.values()),
            chunks=chunks,
            process_chains=process_chains,
            process_gaps=process_gaps
        )
    
    def _construct_process_chains(
        self,
        workflow_nodes: List,
        relationships: List[GraphRelationship],
        all_nodes: Dict[str, GraphNode]
    ) -> List[Dict[str, Any]]:
        """
        Construct complete process chains from workflow nodes and NEXT_STEP relationships.
        
        Supports branching workflows where a step can have multiple next steps.
        
        Args:
            workflow_nodes: List of workflow starting nodes
            relationships: List of all relationships (filtered to NEXT_STEP)
            all_nodes: Dictionary of all nodes
        
        Returns:
            List of process chain dictionaries with steps and metadata
        """
        process_chains = []
        
        # Filter to NEXT_STEP relationships
        next_step_rels = [r for r in relationships if r.rel_type == "NEXT_STEP"]
        
        # Build adjacency map for NEXT_STEP relationships
        adjacency = {}
        for rel in next_step_rels:
            if rel.source_id not in adjacency:
                adjacency[rel.source_id] = []
            adjacency[rel.source_id].append({
                "target_id": rel.target_id,
                "properties": rel.properties
            })
        
        # For each workflow node, construct chains
        for wf_node in workflow_nodes:
            wf_id = wf_node.get("entity_id")
            if not wf_id:
                continue
            
            # Find all paths from this workflow node
            chains = self._find_all_paths(wf_id, adjacency, all_nodes, max_depth=20)
            
            for chain in chains:
                process_chain = {
                    "workflow_id": wf_id,
                    "workflow_name": wf_node.get("name"),
                    "steps": chain["steps"],
                    "step_count": len(chain["steps"]),
                    "is_complete": chain["is_complete"],
                    "has_branches": chain["has_branches"],
                    "branch_points": chain["branch_points"]
                }
                process_chains.append(process_chain)
        
        return process_chains
    
    def _find_all_paths(
        self,
        start_id: str,
        adjacency: Dict[str, List[Dict[str, Any]]],
        all_nodes: Dict[str, GraphNode],
        max_depth: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Find all paths from a starting node using DFS.
        
        Detects branching and marks complete vs incomplete chains.
        
        Args:
            start_id: Starting node ID
            adjacency: Adjacency map of NEXT_STEP relationships
            all_nodes: Dictionary of all nodes
            max_depth: Maximum path depth
        
        Returns:
            List of path dictionaries
        """
        paths = []
        visited = set()
        
        def dfs(node_id: str, path: List[Dict[str, Any]], depth: int):
            if depth > max_depth:
                return
            
            # Avoid cycles
            if node_id in visited:
                return
            
            visited.add(node_id)
            
            # Add current node to path
            node = all_nodes.get(node_id)
            if node:
                step_info = {
                    "step_id": node_id,
                    "step_name": node.properties.get("name", "Unknown"),
                    "step_type": node.node_type,
                    "properties": node.properties
                }
                path.append(step_info)
            
            # Get next steps
            next_steps = adjacency.get(node_id, [])
            
            if not next_steps:
                # End of chain - record this path
                paths.append({
                    "steps": path.copy(),
                    "is_complete": True,  # Reached an end node
                    "has_branches": any(len(adjacency.get(s["step_id"], [])) > 1 for s in path),
                    "branch_points": [s["step_name"] for s in path if len(adjacency.get(s["step_id"], [])) > 1]
                })
            elif len(next_steps) == 1:
                # Single next step - continue
                dfs(next_steps[0]["target_id"], path, depth + 1)
            else:
                # Multiple next steps - branching
                for next_step in next_steps:
                    # Create a new path for each branch
                    branch_path = path.copy()
                    dfs(next_step["target_id"], branch_path, depth + 1)
            
            visited.remove(node_id)
        
        # Start DFS from the starting node
        dfs(start_id, [], 0)
        
        # If no paths found, create a single-node path
        if not paths and start_id in all_nodes:
            node = all_nodes[start_id]
            paths.append({
                "steps": [{
                    "step_id": start_id,
                    "step_name": node.properties.get("name", "Unknown"),
                    "step_type": node.node_type,
                    "properties": node.properties
                }],
                "is_complete": False,  # No next steps found
                "has_branches": False,
                "branch_points": []
            })
        
        return paths
    
    def _detect_process_gaps(
        self,
        process_chains: List[Dict[str, Any]],
        all_nodes: Dict[str, GraphNode],
        all_relationships: Dict[str, GraphRelationship]
    ) -> List[str]:
        """
        Detect gaps in process chains where NEXT_STEP relationships are missing.
        
        A gap is detected when:
        - A process chain is marked as incomplete (no end node)
        - A step references another step but no NEXT_STEP relationship exists
        - A workflow has orphaned steps (not connected to main chain)
        
        Args:
            process_chains: List of process chains
            all_nodes: Dictionary of all nodes
            all_relationships: Dictionary of all relationships
        
        Returns:
            List of gap descriptions
        """
        gaps = []
        
        # Check for incomplete chains
        for chain in process_chains:
            if not chain["is_complete"]:
                last_step = chain["steps"][-1] if chain["steps"] else None
                if last_step:
                    gap_msg = (
                        f"Incomplete process chain in workflow '{chain['workflow_name']}': "
                        f"last step '{last_step['step_name']}' has no NEXT_STEP relationship"
                    )
                    gaps.append(gap_msg)
                    logger.debug(f"Gap detected: {gap_msg}")
        
        # Check for orphaned workflow steps
        # Find all workflow-related nodes
        workflow_step_nodes = [
            node for node in all_nodes.values()
            if node.node_type == "Workflow" or "workflow" in node.properties.get("name", "").lower()
        ]
        
        # Find nodes that are in chains
        nodes_in_chains = set()
        for chain in process_chains:
            for step in chain["steps"]:
                nodes_in_chains.add(step["step_id"])
        
        # Find orphaned nodes
        for node in workflow_step_nodes:
            if node.node_id not in nodes_in_chains:
                # Check if this node has any NEXT_STEP relationships
                has_next_step = any(
                    r.source_id == node.node_id or r.target_id == node.node_id
                    for r in all_relationships.values()
                    if r.rel_type == "NEXT_STEP"
                )
                
                if not has_next_step:
                    gap_msg = (
                        f"Orphaned workflow step: '{node.properties.get('name', node.node_id)}' "
                        f"is not connected to any process chain"
                    )
                    gaps.append(gap_msg)
                    logger.debug(f"Gap detected: {gap_msg}")
        
        return gaps
    
    def _execute_rule_query(self, entity_names: List[str], depth: int) -> GraphResult:
        """
        Execute risk rule analysis query.
        
        Retrieves Rule nodes and their applicability using APPLIES_TO relationships.
        Includes:
        - Rule retrieval with conditions, actions, and scope
        - APPLIES_TO traversal to find applicable entities
        - Rule overlap detection (multiple rules for same entity)
        - Rule ranking by specificity and priority
        
        Args:
            entity_names: List of entity names (rules or entities to check rules for)
            depth: Maximum traversal depth
        
        Returns:
            GraphResult with rules, applicability info, and rankings
        """
        with self.driver.session() as session:
            if not entity_names:
                # Return all rules if no specific entities mentioned
                cypher = f"""
                MATCH (rule:Entity)
                WHERE rule.entity_type = 'Rule'
                
                // Get entities this rule applies to
                OPTIONAL MATCH (rule)-[r:APPLIES_TO*1..{depth}]->(entity:Entity)
                
                RETURN rule,
                       collect(DISTINCT entity) AS applicable_entities,
                       collect(DISTINCT r) AS applies_to_rels
                LIMIT 50
                """
                
                result = session.run(cypher)
            else:
                # Check if entity_names are rules or entities to find rules for
                cypher = f"""
                UNWIND $entity_names AS entity_name
                
                // Try to match as a rule first
                OPTIONAL MATCH (rule:Entity {{entity_type: 'Rule'}})
                WHERE rule.name = entity_name OR rule.canonical_name = entity_name
                
                // Try to match as an entity that rules apply to
                OPTIONAL MATCH (entity:Entity)
                WHERE (entity.name = entity_name OR entity.canonical_name = entity_name)
                  AND entity.entity_type <> 'Rule'
                
                // Get rules that apply to this entity
                OPTIONAL MATCH (applicable_rule:Entity {{entity_type: 'Rule'}})-[r:APPLIES_TO*1..{depth}]->(entity)
                
                // Get entities that the matched rule applies to
                OPTIONAL MATCH (rule)-[r2:APPLIES_TO*1..{depth}]->(target:Entity)
                
                WITH COALESCE(rule, applicable_rule) AS final_rule,
                     CASE WHEN rule IS NOT NULL THEN collect(DISTINCT target)
                          ELSE [entity]
                     END AS applicable_entities,
                     CASE WHEN rule IS NOT NULL THEN collect(DISTINCT r2)
                          ELSE collect(DISTINCT r)
                     END AS applies_to_rels
                
                WHERE final_rule IS NOT NULL
                
                RETURN final_rule AS rule,
                       applicable_entities,
                       applies_to_rels
                """
                
                result = session.run(cypher, entity_names=entity_names)
            
            # Process result with rule analysis
            graph_result = self._process_rule_result(result, session)
            
            logger.info(
                f"Rule query completed: {len(graph_result.rules)} rules found, "
                f"{len(graph_result.nodes)} nodes, {len(graph_result.relationships)} relationships"
            )
            
            return graph_result
    
    def _process_rule_result(self, result, session: Session) -> GraphResult:
        """
        Process rule query result with overlap detection and ranking.
        
        Args:
            result: Neo4j query result
            session: Neo4j session for additional queries
        
        Returns:
            GraphResult with rules, rankings, and overlap information
        """
        nodes_dict: Dict[str, GraphNode] = {}
        relationships_dict: Dict[str, GraphRelationship] = {}
        rules = []
        chunk_ids = set()
        entity_to_rules = {}  # Track which rules apply to which entities
        
        for record in result:
            rule = record.get("rule")
            applicable_entities = record.get("applicable_entities", [])
            applies_to_rels = record.get("applies_to_rels", [])
            
            if not rule:
                continue
            
            # Add rule node
            self._add_node_to_dict(rule, nodes_dict)
            
            # Collect chunk ID
            if "source_chunk_id" in dict(rule):
                chunk_ids.add(dict(rule)["source_chunk_id"])
            
            # Process applicable entities
            applicable_entity_list = []
            for entity in applicable_entities:
                if entity:
                    self._add_node_to_dict(entity, nodes_dict)
                    entity_id = entity.get("entity_id")
                    entity_name = entity.get("name")
                    
                    if entity_id:
                        applicable_entity_list.append({
                            "entity_id": entity_id,
                            "entity_name": entity_name,
                            "entity_type": entity.get("entity_type")
                        })
                        
                        # Track for overlap detection
                        if entity_id not in entity_to_rules:
                            entity_to_rules[entity_id] = []
                        entity_to_rules[entity_id].append(rule.get("entity_id"))
                    
                    if "source_chunk_id" in dict(entity):
                        chunk_ids.add(dict(entity)["source_chunk_id"])
            
            # Process relationships
            for rel in applies_to_rels:
                if rel:
                    self._add_relationship_to_dict(rel, relationships_dict)
            
            # Extract rule properties
            rule_dict = dict(rule)
            rule_entry = {
                "rule_id": rule.get("entity_id"),
                "rule_name": rule.get("name"),
                "conditions": rule_dict.get("conditions", ""),
                "actions": rule_dict.get("actions", ""),
                "scope": rule_dict.get("scope", ""),
                "priority": rule_dict.get("priority", 0),
                "specificity": self._compute_rule_specificity(rule_dict, applicable_entity_list),
                "applicable_entities": applicable_entity_list,
                "properties": rule_dict
            }
            
            rules.append(rule_entry)
        
        # Detect rule overlaps
        overlaps = self._detect_rule_overlaps(entity_to_rules, rules)
        
        # Add overlap information to rules
        for rule in rules:
            rule["overlaps"] = overlaps.get(rule["rule_id"], [])
        
        # Rank rules by specificity and priority
        rules.sort(key=lambda r: (r["specificity"], r["priority"]), reverse=True)
        
        # Add ranking
        for i, rule in enumerate(rules):
            rule["rank"] = i + 1
        
        # Retrieve chunks
        chunks = self._retrieve_chunks(list(chunk_ids))
        
        return GraphResult(
            nodes=list(nodes_dict.values()),
            relationships=list(relationships_dict.values()),
            chunks=chunks,
            rules=rules
        )
    
    def _compute_rule_specificity(
        self, 
        rule_dict: Dict[str, Any], 
        applicable_entities: List[Dict[str, Any]]
    ) -> float:
        """
        Compute rule specificity score.
        
        More specific rules have:
        - Fewer applicable entities (more targeted)
        - More detailed conditions
        - Specific entity types
        
        Args:
            rule_dict: Rule properties dictionary
            applicable_entities: List of entities this rule applies to
        
        Returns:
            Specificity score (higher = more specific)
        """
        specificity = 0.0
        
        # Factor 1: Inverse of number of applicable entities
        # Fewer entities = more specific
        if applicable_entities:
            specificity += 10.0 / len(applicable_entities)
        else:
            specificity += 1.0  # No entities = low specificity
        
        # Factor 2: Condition complexity (more conditions = more specific)
        conditions = rule_dict.get("conditions", "")
        if conditions:
            # Count condition clauses (simple heuristic: count "AND", "OR", "IF")
            condition_count = conditions.count("AND") + conditions.count("OR") + conditions.count("IF") + 1
            specificity += condition_count * 2.0
        
        # Factor 3: Scope specificity
        scope = rule_dict.get("scope", "")
        if scope:
            if "specific" in scope.lower() or "targeted" in scope.lower():
                specificity += 5.0
            elif "general" in scope.lower() or "broad" in scope.lower():
                specificity += 1.0
            else:
                specificity += 3.0
        
        # Factor 4: Entity type specificity
        # Rules applying to specific types (System, PaymentMode) are more specific
        # than rules applying to generic Entity types
        entity_types = set(e["entity_type"] for e in applicable_entities if e.get("entity_type"))
        if "System" in entity_types or "PaymentMode" in entity_types:
            specificity += 3.0
        
        return specificity
    
    def _detect_rule_overlaps(
        self,
        entity_to_rules: Dict[str, List[str]],
        rules: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Detect overlapping rules (multiple rules applying to same entity).
        
        Args:
            entity_to_rules: Map of entity IDs to rule IDs
            rules: List of rule dictionaries
        
        Returns:
            Dictionary mapping rule IDs to list of overlapping rules
        """
        overlaps = {}
        rule_by_id = {r["rule_id"]: r for r in rules}
        
        # For each entity with multiple rules, mark those rules as overlapping
        for entity_id, rule_ids in entity_to_rules.items():
            if len(rule_ids) > 1:
                # These rules overlap on this entity
                for rule_id in rule_ids:
                    if rule_id not in overlaps:
                        overlaps[rule_id] = []
                    
                    # Add other rules as overlaps
                    for other_rule_id in rule_ids:
                        if other_rule_id != rule_id:
                            other_rule = rule_by_id.get(other_rule_id)
                            if other_rule:
                                overlap_info = {
                                    "rule_id": other_rule_id,
                                    "rule_name": other_rule["rule_name"],
                                    "entity_id": entity_id,
                                    "conflict_type": "overlap"
                                }
                                
                                # Avoid duplicates
                                if overlap_info not in overlaps[rule_id]:
                                    overlaps[rule_id].append(overlap_info)
        
        return overlaps
    
    def _execute_conflict_query(self, entity_names: List[str], depth: int) -> GraphResult:
        """
        Execute conflict query (contradictions).
        
        Finds CONFLICTS_WITH relationships and categorizes them by type with severity ranking.
        
        Conflict types:
        - property: Same entity with different property values
        - rule: Contradictory rules for same scenario
        - workflow: Different process flows for same operation
        
        Severity is ranked based on:
        - Entity importance (System > PaymentMode > Workflow > Rule > Field)
        - Conflict scope (number of affected entities)
        - Conflict type (rule > workflow > property)
        
        Args:
            entity_names: List of entity names to query
            depth: Maximum traversal depth (not used for conflicts, but kept for consistency)
        
        Returns:
            GraphResult with conflict subgraph, categorized conflicts, and severity rankings
        """
        with self.driver.session() as session:
            if not entity_names:
                # Return all conflicts if no specific entities mentioned
                cypher = """
                MATCH (e1:Entity)-[r:CONFLICTS_WITH]-(e2:Entity)
                
                // Get source chunks for both sides
                OPTIONAL MATCH (e1)<-[:MENTIONS]-(chunk1:Chunk)
                OPTIONAL MATCH (e2)<-[:MENTIONS]-(chunk2:Chunk)
                
                RETURN e1, r, e2,
                       collect(DISTINCT chunk1) AS chunks1,
                       collect(DISTINCT chunk2) AS chunks2
                LIMIT 50
                """
                
                result = session.run(cypher)
            else:
                cypher = """
                UNWIND $entity_names AS entity_name
                MATCH (e:Entity)
                WHERE e.name = entity_name OR e.canonical_name = entity_name
                
                // Get conflicts
                MATCH (e)-[r:CONFLICTS_WITH]-(conflicting:Entity)
                
                // Get source chunks for both sides
                OPTIONAL MATCH (e)<-[:MENTIONS]-(chunk1:Chunk)
                OPTIONAL MATCH (conflicting)<-[:MENTIONS]-(chunk2:Chunk)
                
                RETURN e AS center_node, r, conflicting,
                       collect(DISTINCT chunk1) AS chunks1,
                       collect(DISTINCT chunk2) AS chunks2
                """
                
                result = session.run(cypher, entity_names=entity_names)
            
            # Process result and categorize conflicts
            graph_result = self._process_conflict_result(result)
            
            logger.info(
                f"Conflict query completed: {len(graph_result.conflicts)} conflicts found, "
                f"{len(graph_result.nodes)} nodes, {len(graph_result.relationships)} relationships"
            )
            
            return graph_result
    
    def _process_conflict_result(self, result) -> GraphResult:
        """
        Process conflict query result with categorization and severity ranking.
        
        Args:
            result: Neo4j query result
        
        Returns:
            GraphResult with categorized and ranked conflicts
        """
        nodes_dict: Dict[str, GraphNode] = {}
        relationships_dict: Dict[str, GraphRelationship] = {}
        conflicts = []
        chunk_ids = set()
        
        for record in result:
            # Process nodes
            e1 = record.get("e1") or record.get("center_node")
            e2 = record.get("e2") or record.get("conflicting")
            r = record.get("r")
            chunks1 = record.get("chunks1", [])
            chunks2 = record.get("chunks2", [])
            
            if e1:
                self._add_node_to_dict(e1, nodes_dict)
            if e2:
                self._add_node_to_dict(e2, nodes_dict)
            if r:
                self._add_relationship_to_dict(r, relationships_dict)
            
            # Collect chunk IDs
            for chunk in chunks1:
                if chunk and chunk.get("chunk_id"):
                    chunk_ids.add(chunk.get("chunk_id"))
            for chunk in chunks2:
                if chunk and chunk.get("chunk_id"):
                    chunk_ids.add(chunk.get("chunk_id"))
            
            # Create conflict entry with categorization
            if e1 and e2 and r:
                conflict_type = self._categorize_conflict(e1, e2, r)
                severity = self._compute_conflict_severity(e1, e2, r, conflict_type)
                
                conflict_entry = {
                    "entity1_id": e1.get("entity_id"),
                    "entity1_name": e1.get("name"),
                    "entity1_type": e1.get("entity_type"),
                    "entity2_id": e2.get("entity_id"),
                    "entity2_name": e2.get("name"),
                    "entity2_type": e2.get("entity_type"),
                    "conflict_type": conflict_type,
                    "severity": severity,
                    "severity_score": self._severity_to_score(severity),
                    "metadata": dict(r) if hasattr(r, "__iter__") else {},
                    "source_chunks": {
                        "entity1": [c.get("chunk_id") for c in chunks1 if c and c.get("chunk_id")],
                        "entity2": [c.get("chunk_id") for c in chunks2 if c and c.get("chunk_id")]
                    }
                }
                
                conflicts.append(conflict_entry)
        
        # Sort conflicts by severity (high to low)
        conflicts.sort(key=lambda x: x["severity_score"], reverse=True)
        
        # Retrieve chunks
        chunks = self._retrieve_chunks(list(chunk_ids))
        
        return GraphResult(
            nodes=list(nodes_dict.values()),
            relationships=list(relationships_dict.values()),
            chunks=chunks,
            conflicts=conflicts
        )
    
    def _categorize_conflict(self, e1, e2, relationship) -> str:
        """
        Categorize conflict by type based on entity types and relationship metadata.
        
        Args:
            e1: First entity
            e2: Second entity
            relationship: CONFLICTS_WITH relationship
        
        Returns:
            Conflict type: "property", "rule", or "workflow"
        """
        # Check relationship metadata for explicit conflict type
        if hasattr(relationship, "get"):
            conflict_type = relationship.get("conflict_type")
            if conflict_type:
                return conflict_type
        elif hasattr(relationship, "__iter__"):
            rel_dict = dict(relationship)
            conflict_type = rel_dict.get("conflict_type")
            if conflict_type:
                return conflict_type
        
        # Infer from entity types
        e1_type = e1.get("entity_type", "")
        e2_type = e2.get("entity_type", "")
        
        if e1_type == "Rule" or e2_type == "Rule":
            return "rule"
        
        if e1_type == "Workflow" or e2_type == "Workflow":
            return "workflow"
        
        # Default to property conflict
        return "property"
    
    def _compute_conflict_severity(self, e1, e2, relationship, conflict_type: str) -> str:
        """
        Compute conflict severity based on entity importance, scope, and type.
        
        Severity levels: "critical", "high", "medium", "low"
        
        Args:
            e1: First entity
            e2: Second entity
            relationship: CONFLICTS_WITH relationship
            conflict_type: Type of conflict
        
        Returns:
            Severity level string
        """
        # Entity type importance ranking
        type_importance = {
            "System": 5,
            "PaymentMode": 4,
            "Workflow": 3,
            "Rule": 3,
            "Field": 2,
            "Entity": 1
        }
        
        # Conflict type severity
        type_severity = {
            "rule": 3,
            "workflow": 2,
            "property": 1
        }
        
        # Get entity importance
        e1_importance = type_importance.get(e1.get("entity_type", "Entity"), 1)
        e2_importance = type_importance.get(e2.get("entity_type", "Entity"), 1)
        max_importance = max(e1_importance, e2_importance)
        
        # Get conflict type severity
        conflict_severity = type_severity.get(conflict_type, 1)
        
        # Compute total score
        total_score = max_importance + conflict_severity
        
        # Map to severity level
        if total_score >= 7:
            return "critical"
        elif total_score >= 5:
            return "high"
        elif total_score >= 3:
            return "medium"
        else:
            return "low"
    
    def _severity_to_score(self, severity: str) -> int:
        """
        Convert severity level to numeric score for sorting.
        
        Args:
            severity: Severity level string
        
        Returns:
            Numeric score (higher = more severe)
        """
        severity_scores = {
            "critical": 4,
            "high": 3,
            "medium": 2,
            "low": 1
        }
        return severity_scores.get(severity, 0)
    
    def _execute_comparison_query(self, entity_names: List[str], depth: int) -> GraphResult:
        """
        Execute comparison query (cross-document entity comparisons).
        
        This method implements comprehensive cross-document comparison by:
        1. Retrieving relevant chunks from multiple documents
        2. Grouping results by document for side-by-side comparison
        3. Identifying common entities using SAME_AS relationships
        4. Highlighting differences in entity properties and relationships
        
        Args:
            entity_names: List of entity names to compare across documents
            depth: Maximum traversal depth
        
        Returns:
            GraphResult with comparison data including document_groups, common_entities, and differences
        """
        if len(entity_names) < 2:
            # Need at least 2 entities for comparison, but we can also compare
            # a single entity across multiple documents
            if len(entity_names) == 1:
                return self._execute_single_entity_comparison(entity_names[0], depth)
            return self._execute_general_query(entity_names, depth)
        
        with self.driver.session() as session:
            # First, find the entities and their canonical forms via SAME_AS
            cypher_entities = """
            UNWIND $entity_names AS entity_name
            MATCH (e:Entity)
            WHERE e.name = entity_name OR e.canonical_name = entity_name
            
            // Find canonical entity via SAME_AS relationships
            OPTIONAL MATCH (e)-[:SAME_AS*0..2]-(canonical:Entity)
            
            WITH e, 
                 CASE WHEN canonical IS NOT NULL THEN canonical ELSE e END AS canonical_entity
            
            RETURN DISTINCT e.entity_id AS entity_id,
                   e.name AS name,
                   e.entity_type AS entity_type,
                   canonical_entity.entity_id AS canonical_id,
                   canonical_entity.name AS canonical_name
            """
            
            entity_result = session.run(cypher_entities, entity_names=entity_names)
            entity_map = {}
            canonical_map = {}
            
            for record in entity_result:
                entity_id = record["entity_id"]
                canonical_id = record["canonical_id"] or entity_id
                entity_map[entity_id] = {
                    "name": record["name"],
                    "type": record["entity_type"],
                    "canonical_id": canonical_id,
                    "canonical_name": record["canonical_name"] or record["name"]
                }
                canonical_map[canonical_id] = canonical_map.get(canonical_id, []) + [entity_id]
            
            # Now get relationships and properties for both entities
            cypher = f"""
            MATCH (e1:Entity)
            WHERE e1.name = $entity1 OR e1.canonical_name = $entity1
            
            MATCH (e2:Entity)
            WHERE e2.name = $entity2 OR e2.canonical_name = $entity2
            
            // Get all relationships and properties for e1
            OPTIONAL MATCH path1 = (e1)-[r1*1..{depth}]-(related1:Entity)
            
            // Get all relationships and properties for e2
            OPTIONAL MATCH path2 = (e2)-[r2*1..{depth}]-(related2:Entity)
            
            // Get chunks mentioning these entities
            OPTIONAL MATCH (e1)<-[:MENTIONS]-(chunk1:Chunk)
            OPTIONAL MATCH (e2)<-[:MENTIONS]-(chunk2:Chunk)
            
            WITH e1, e2, 
                 collect(DISTINCT path1) AS paths1,
                 collect(DISTINCT path2) AS paths2,
                 collect(DISTINCT chunk1) AS chunks1,
                 collect(DISTINCT chunk2) AS chunks2
            
            RETURN e1, e2, paths1, paths2, chunks1, chunks2
            """
            
            result = session.run(cypher, entity1=entity_names[0], entity2=entity_names[1])
            
            # Process the result
            graph_result = GraphResult()
            document_groups = {}
            all_nodes = {}
            all_relationships = {}
            
            for record in result:
                e1 = record["e1"]
                e2 = record["e2"]
                paths1 = record["paths1"]
                paths2 = record["paths2"]
                chunks1 = record["chunks1"]
                chunks2 = record["chunks2"]
                
                # Add entities to nodes
                if e1:
                    self._add_node_to_dict(e1, all_nodes)
                if e2:
                    self._add_node_to_dict(e2, all_nodes)
                
                # Process paths for e1
                for path in paths1:
                    if path:
                        for node in path.nodes:
                            self._add_node_to_dict(node, all_nodes)
                        for rel in path.relationships:
                            self._add_relationship_to_dict(rel, all_relationships)
                
                # Process paths for e2
                for path in paths2:
                    if path:
                        for node in path.nodes:
                            self._add_node_to_dict(node, all_nodes)
                        for rel in path.relationships:
                            self._add_relationship_to_dict(rel, all_relationships)
                
                # Group chunks by document
                for chunk in chunks1:
                    if chunk:
                        doc_id = chunk.get("doc_id")
                        if doc_id:
                            if doc_id not in document_groups:
                                document_groups[doc_id] = {
                                    "doc_id": doc_id,
                                    "entities": [],
                                    "chunks": [],
                                    "relationships": []
                                }
                            
                            chunk_dict = {
                                "chunk_id": chunk.get("chunk_id"),
                                "text": chunk.get("text"),
                                "breadcrumbs": chunk.get("breadcrumbs"),
                                "section": chunk.get("section")
                            }
                            document_groups[doc_id]["chunks"].append(chunk_dict)
                            
                            # Add entity to this document group
                            entity_id = e1.get("entity_id")
                            if entity_id and entity_id not in [e["entity_id"] for e in document_groups[doc_id]["entities"]]:
                                document_groups[doc_id]["entities"].append({
                                    "entity_id": entity_id,
                                    "name": e1.get("name"),
                                    "type": e1.get("entity_type"),
                                    "properties": dict(e1)
                                })
                
                for chunk in chunks2:
                    if chunk:
                        doc_id = chunk.get("doc_id")
                        if doc_id:
                            if doc_id not in document_groups:
                                document_groups[doc_id] = {
                                    "doc_id": doc_id,
                                    "entities": [],
                                    "chunks": [],
                                    "relationships": []
                                }
                            
                            chunk_dict = {
                                "chunk_id": chunk.get("chunk_id"),
                                "text": chunk.get("text"),
                                "breadcrumbs": chunk.get("breadcrumbs"),
                                "section": chunk.get("section")
                            }
                            document_groups[doc_id]["chunks"].append(chunk_dict)
                            
                            # Add entity to this document group
                            entity_id = e2.get("entity_id")
                            if entity_id and entity_id not in [e["entity_id"] for e in document_groups[doc_id]["entities"]]:
                                document_groups[doc_id]["entities"].append({
                                    "entity_id": entity_id,
                                    "name": e2.get("name"),
                                    "type": e2.get("entity_type"),
                                    "properties": dict(e2)
                                })
            
            # Identify common entities using SAME_AS relationships
            common_entities = self._identify_common_entities(session, list(document_groups.keys()))
            
            # Highlight differences between documents
            differences = self._highlight_differences(document_groups, all_nodes, all_relationships)
            
            # Retrieve all chunks
            chunk_ids = set()
            for doc_group in document_groups.values():
                for chunk in doc_group["chunks"]:
                    chunk_ids.add(chunk["chunk_id"])
            
            chunks = self._retrieve_chunks(list(chunk_ids))
            
            graph_result.nodes = list(all_nodes.values())
            graph_result.relationships = list(all_relationships.values())
            graph_result.chunks = chunks
            graph_result.document_groups = document_groups
            graph_result.common_entities = common_entities
            graph_result.differences = differences
            
            logger.info(
                f"Comparison query completed: {len(document_groups)} documents, "
                f"{len(common_entities)} common entities, {len(differences)} differences"
            )
            
            return graph_result
    
    def _execute_single_entity_comparison(self, entity_name: str, depth: int) -> GraphResult:
        """
        Compare a single entity across multiple documents.
        
        Args:
            entity_name: Entity name to compare across documents
            depth: Maximum traversal depth
        
        Returns:
            GraphResult with cross-document comparison data
        """
        with self.driver.session() as session:
            # Find all mentions of this entity across documents
            cypher = f"""
            MATCH (e:Entity)
            WHERE e.name = $entity_name OR e.canonical_name = $entity_name
            
            // Find all entities linked via SAME_AS (same entity in different documents)
            OPTIONAL MATCH (e)-[:SAME_AS*0..2]-(same_entity:Entity)
            
            WITH collect(DISTINCT COALESCE(same_entity, e)) AS all_entities
            UNWIND all_entities AS entity
            
            // Get chunks mentioning this entity
            MATCH (entity)<-[:MENTIONS]-(chunk:Chunk)
            
            // Get relationships for this entity
            OPTIONAL MATCH path = (entity)-[r*1..{depth}]-(related:Entity)
            
            RETURN entity, chunk, 
                   collect(DISTINCT path) AS paths
            """
            
            result = session.run(cypher, entity_name=entity_name)
            
            # Process results and group by document
            document_groups = {}
            all_nodes = {}
            all_relationships = {}
            
            for record in result:
                entity = record["entity"]
                chunk = record["chunk"]
                paths = record["paths"]
                
                if entity:
                    self._add_node_to_dict(entity, all_nodes)
                
                # Process paths
                for path in paths:
                    if path:
                        for node in path.nodes:
                            self._add_node_to_dict(node, all_nodes)
                        for rel in path.relationships:
                            self._add_relationship_to_dict(rel, all_relationships)
                
                # Group by document
                if chunk:
                    doc_id = chunk.get("doc_id")
                    if doc_id:
                        if doc_id not in document_groups:
                            document_groups[doc_id] = {
                                "doc_id": doc_id,
                                "entities": [],
                                "chunks": [],
                                "relationships": []
                            }
                        
                        chunk_dict = {
                            "chunk_id": chunk.get("chunk_id"),
                            "text": chunk.get("text"),
                            "breadcrumbs": chunk.get("breadcrumbs"),
                            "section": chunk.get("section")
                        }
                        document_groups[doc_id]["chunks"].append(chunk_dict)
                        
                        # Add entity to this document group
                        entity_id = entity.get("entity_id")
                        if entity_id and entity_id not in [e["entity_id"] for e in document_groups[doc_id]["entities"]]:
                            document_groups[doc_id]["entities"].append({
                                "entity_id": entity_id,
                                "name": entity.get("name"),
                                "type": entity.get("entity_type"),
                                "properties": dict(entity)
                            })
            
            # Identify common entities
            common_entities = self._identify_common_entities(session, list(document_groups.keys()))
            
            # Highlight differences
            differences = self._highlight_differences(document_groups, all_nodes, all_relationships)
            
            # Retrieve chunks
            chunk_ids = set()
            for doc_group in document_groups.values():
                for chunk in doc_group["chunks"]:
                    chunk_ids.add(chunk["chunk_id"])
            
            chunks = self._retrieve_chunks(list(chunk_ids))
            
            return GraphResult(
                nodes=list(all_nodes.values()),
                relationships=list(all_relationships.values()),
                chunks=chunks,
                document_groups=document_groups,
                common_entities=common_entities,
                differences=differences
            )
    
    def _identify_common_entities(self, session: Session, doc_ids: List[str]) -> List[str]:
        """
        Identify entities that appear across multiple documents using SAME_AS relationships.
        
        Args:
            session: Neo4j session
            doc_ids: List of document IDs to check
        
        Returns:
            List of canonical entity IDs that appear in multiple documents
        """
        if len(doc_ids) < 2:
            return []
        
        cypher = """
        UNWIND $doc_ids AS doc_id
        
        // Find all entities mentioned in chunks from these documents
        MATCH (chunk:Chunk {doc_id: doc_id})-[:MENTIONS]->(entity:Entity)
        
        // Find canonical entity via SAME_AS
        OPTIONAL MATCH (entity)-[:SAME_AS*0..2]-(canonical:Entity)
        
        WITH doc_id, 
             CASE WHEN canonical IS NOT NULL THEN canonical ELSE entity END AS canonical_entity
        
        // Group by canonical entity and count documents
        WITH canonical_entity.entity_id AS canonical_id,
             canonical_entity.name AS canonical_name,
             collect(DISTINCT doc_id) AS docs
        
        WHERE size(docs) > 1
        
        RETURN canonical_id, canonical_name, docs
        """
        
        result = session.run(cypher, doc_ids=doc_ids)
        
        common_entities = []
        for record in result:
            canonical_id = record["canonical_id"]
            if canonical_id:
                common_entities.append(canonical_id)
                logger.debug(
                    f"Common entity found: {record['canonical_name']} "
                    f"(appears in {len(record['docs'])} documents)"
                )
        
        return common_entities
    
    def _highlight_differences(
        self, 
        document_groups: Dict[str, Dict[str, Any]], 
        all_nodes: Dict[str, GraphNode],
        all_relationships: Dict[str, GraphRelationship]
    ) -> List[str]:
        """
        Highlight differences between documents in entity properties and relationships.
        
        Args:
            document_groups: Dictionary of document groups
            all_nodes: Dictionary of all nodes
            all_relationships: Dictionary of all relationships
        
        Returns:
            List of difference descriptions
        """
        differences = []
        
        if len(document_groups) < 2:
            return differences
        
        doc_ids = list(document_groups.keys())
        
        # Compare entities across documents
        # Build entity map by name for comparison
        entity_by_name = {}
        for doc_id, doc_group in document_groups.items():
            for entity in doc_group["entities"]:
                name = entity["name"]
                if name not in entity_by_name:
                    entity_by_name[name] = {}
                entity_by_name[name][doc_id] = entity
        
        # Find entities that appear in multiple documents with different properties
        for entity_name, doc_entities in entity_by_name.items():
            if len(doc_entities) > 1:
                # Compare properties
                doc_list = list(doc_entities.keys())
                for i in range(len(doc_list)):
                    for j in range(i + 1, len(doc_list)):
                        doc1 = doc_list[i]
                        doc2 = doc_list[j]
                        entity1 = doc_entities[doc1]
                        entity2 = doc_entities[doc2]
                        
                        # Compare properties
                        props1 = entity1.get("properties", {})
                        props2 = entity2.get("properties", {})
                        
                        # Find differing properties
                        all_keys = set(props1.keys()) | set(props2.keys())
                        for key in all_keys:
                            if key in ["entity_id", "source_chunk_id", "context"]:
                                continue  # Skip metadata fields
                            
                            val1 = props1.get(key)
                            val2 = props2.get(key)
                            
                            if val1 != val2:
                                diff_msg = (
                                    f"Entity '{entity_name}' has different '{key}': "
                                    f"'{val1}' in {doc1} vs '{val2}' in {doc2}"
                                )
                                differences.append(diff_msg)
                                logger.debug(f"Difference found: {diff_msg}")
        
        # Compare relationship counts
        rel_by_doc = {}
        for rel in all_relationships.values():
            # Try to determine which document this relationship belongs to
            # by checking the source node's document
            source_node = all_nodes.get(rel.source_id)
            if source_node and "source_chunk_id" in source_node.properties:
                # Find which document this chunk belongs to
                for doc_id, doc_group in document_groups.items():
                    if any(c["chunk_id"] == source_node.properties["source_chunk_id"] 
                           for c in doc_group["chunks"]):
                        if doc_id not in rel_by_doc:
                            rel_by_doc[doc_id] = []
                        rel_by_doc[doc_id].append(rel)
                        break
        
        # Compare relationship types across documents
        if len(rel_by_doc) > 1:
            rel_types_by_doc = {}
            for doc_id, rels in rel_by_doc.items():
                rel_types_by_doc[doc_id] = set(r.rel_type for r in rels)
            
            doc_list = list(rel_types_by_doc.keys())
            for i in range(len(doc_list)):
                for j in range(i + 1, len(doc_list)):
                    doc1 = doc_list[i]
                    doc2 = doc_list[j]
                    types1 = rel_types_by_doc[doc1]
                    types2 = rel_types_by_doc[doc2]
                    
                    # Find unique relationship types
                    only_in_doc1 = types1 - types2
                    only_in_doc2 = types2 - types1
                    
                    if only_in_doc1:
                        diff_msg = (
                            f"Document {doc1} has unique relationship types: "
                            f"{', '.join(only_in_doc1)}"
                        )
                        differences.append(diff_msg)
                    
                    if only_in_doc2:
                        diff_msg = (
                            f"Document {doc2} has unique relationship types: "
                            f"{', '.join(only_in_doc2)}"
                        )
                        differences.append(diff_msg)
        
        return differences
    
    def _execute_general_query(self, entity_names: List[str], depth: int) -> GraphResult:
        """
        Execute general entity query (fallback).
        
        Retrieves entity and its immediate neighborhood.
        
        Args:
            entity_names: List of entity names to query
            depth: Maximum traversal depth
        
        Returns:
            GraphResult with entity neighborhood
        """
        if not entity_names:
            return GraphResult()
        
        with self.driver.session() as session:
            cypher = f"""
            UNWIND $entity_names AS entity_name
            MATCH (e:Entity)
            WHERE e.name = entity_name OR e.canonical_name = entity_name
            
            // Get all relationships within depth
            OPTIONAL MATCH path = (e)-[*1..{depth}]-(related)
            
            RETURN e AS center_node,
                   nodes(path) AS path_nodes,
                   relationships(path) AS path_rels
            LIMIT 50
            """
            
            result = session.run(cypher, entity_names=entity_names)
            return self._process_graph_result(result)
    
    def _process_graph_result(self, result) -> GraphResult:
        """
        Process Neo4j query result into GraphResult.
        
        Extracts nodes, relationships, and associated chunks from query result.
        
        Args:
            result: Neo4j query result
        
        Returns:
            GraphResult with deduplicated nodes and relationships
        """
        nodes_dict: Dict[str, GraphNode] = {}
        relationships_dict: Dict[str, GraphRelationship] = {}
        chunk_ids = set()
        
        for record in result:
            # Process center node if present
            if "center_node" in record:
                center_node = record["center_node"]
                if center_node:
                    self._add_node_to_dict(center_node, nodes_dict)
            
            # Process individual nodes (e1, e2, etc.)
            for key in record.keys():
                if key in ["e", "e1", "e2", "wf", "conflicting", "dependent", "dependency"]:
                    node = record[key]
                    if node:
                        self._add_node_to_dict(node, nodes_dict)
            
            # Process path nodes
            if "path_nodes" in record and record["path_nodes"]:
                for node in record["path_nodes"]:
                    if node:
                        self._add_node_to_dict(node, nodes_dict)
            
            # Process path relationships
            if "path_rels" in record and record["path_rels"]:
                for rel in record["path_rels"]:
                    if rel:
                        self._add_relationship_to_dict(rel, relationships_dict)
            
            # Process individual relationship
            if "r" in record and record["r"]:
                self._add_relationship_to_dict(record["r"], relationships_dict)
        
        # Collect chunk IDs from entity source_chunk_id
        for node in nodes_dict.values():
            if "source_chunk_id" in node.properties:
                chunk_ids.add(node.properties["source_chunk_id"])
        
        # Retrieve associated chunks
        chunks = self._retrieve_chunks(list(chunk_ids))
        
        return GraphResult(
            nodes=list(nodes_dict.values()),
            relationships=list(relationships_dict.values()),
            chunks=chunks
        )
    
    def _add_node_to_dict(self, neo4j_node, nodes_dict: Dict[str, GraphNode]) -> None:
        """
        Add Neo4j node to nodes dictionary.
        
        Args:
            neo4j_node: Neo4j node object
            nodes_dict: Dictionary to add node to
        """
        # Get node ID (try different ID fields)
        node_id = None
        if hasattr(neo4j_node, "get"):
            node_id = neo4j_node.get("entity_id") or neo4j_node.get("chunk_id") or neo4j_node.get("doc_id") or neo4j_node.get("section_id")
        elif hasattr(neo4j_node, "id"):
            node_id = str(neo4j_node.id)
        
        if not node_id:
            return
        
        # Skip if already added
        if node_id in nodes_dict:
            return
        
        # Get node labels/type
        node_type = "Unknown"
        if hasattr(neo4j_node, "labels"):
            labels = list(neo4j_node.labels)
            if labels:
                # Prefer specific type over generic "Entity"
                node_type = labels[0] if len(labels) == 1 else next((l for l in labels if l != "Entity"), labels[0])
        
        # Get node properties
        properties = dict(neo4j_node) if hasattr(neo4j_node, "__iter__") else {}
        
        # Create GraphNode
        graph_node = GraphNode(
            node_id=node_id,
            node_type=node_type,
            properties=properties
        )
        
        nodes_dict[node_id] = graph_node
    
    def _add_relationship_to_dict(self, neo4j_rel, relationships_dict: Dict[str, GraphRelationship]) -> None:
        """
        Add Neo4j relationship to relationships dictionary.
        
        Args:
            neo4j_rel: Neo4j relationship object
            relationships_dict: Dictionary to add relationship to
        """
        # Get relationship ID
        rel_id = None
        if hasattr(neo4j_rel, "get"):
            rel_id = neo4j_rel.get("rel_id")
        if not rel_id and hasattr(neo4j_rel, "id"):
            rel_id = str(neo4j_rel.id)
        
        if not rel_id:
            return
        
        # Skip if already added
        if rel_id in relationships_dict:
            return
        
        # Get relationship type
        rel_type = neo4j_rel.type if hasattr(neo4j_rel, "type") else "RELATED"
        
        # Get source and target node IDs
        source_id = None
        target_id = None
        
        if hasattr(neo4j_rel, "start_node") and hasattr(neo4j_rel, "end_node"):
            start_node = neo4j_rel.start_node
            end_node = neo4j_rel.end_node
            
            if hasattr(start_node, "get"):
                source_id = start_node.get("entity_id") or start_node.get("chunk_id") or str(start_node.id)
            elif hasattr(start_node, "id"):
                source_id = str(start_node.id)
            
            if hasattr(end_node, "get"):
                target_id = end_node.get("entity_id") or end_node.get("chunk_id") or str(end_node.id)
            elif hasattr(end_node, "id"):
                target_id = str(end_node.id)
        
        if not source_id or not target_id:
            return
        
        # Get relationship properties
        properties = dict(neo4j_rel) if hasattr(neo4j_rel, "__iter__") else {}
        
        # Create GraphRelationship
        graph_rel = GraphRelationship(
            rel_id=rel_id,
            rel_type=rel_type,
            source_id=source_id,
            target_id=target_id,
            properties=properties
        )
        
        relationships_dict[rel_id] = graph_rel
    
    def _retrieve_chunks(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve chunk text from Neo4j.
        
        Args:
            chunk_ids: List of chunk IDs to retrieve
        
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not chunk_ids:
            return []
        
        with self.driver.session() as session:
            cypher = """
            UNWIND $chunk_ids AS chunk_id
            MATCH (c:Chunk {chunk_id: chunk_id})
            RETURN c.chunk_id AS chunk_id,
                   c.text AS text,
                   c.breadcrumbs AS breadcrumbs,
                   c.doc_id AS doc_id,
                   c.section AS section
            """
            
            result = session.run(cypher, chunk_ids=chunk_ids)
            
            chunks = []
            for record in result:
                chunk = {
                    "chunk_id": record["chunk_id"],
                    "text": record["text"],
                    "breadcrumbs": record["breadcrumbs"],
                    "doc_id": record["doc_id"],
                    "section": record["section"]
                }
                chunks.append(chunk)
            
            return chunks
    
    def _compute_impact_radius(self, entity_names: List[str], depth: int, session: Session) -> int:
        """
        Compute impact radius (number of affected systems) for dependency queries.
        
        The impact radius is the total number of unique systems that would be affected
        by changes to the queried entities, considering both forward and backward dependencies.
        
        Args:
            entity_names: List of entity names to compute impact for
            depth: Maximum traversal depth
            session: Neo4j session
        
        Returns:
            Number of unique affected systems
        """
        if not entity_names:
            return 0
        
        cypher = f"""
        UNWIND $entity_names AS entity_name
        MATCH (e)
        WHERE (e.name = entity_name OR e.canonical_name = entity_name)
          AND (e:System OR e:Entity)
        
        // Find all systems in forward dependency chain (what depends on this)
        OPTIONAL MATCH (forward_dependent)-[:DEPENDS_ON*1..{depth}]->(e)
        WHERE forward_dependent:System OR forward_dependent:Entity
        
        // Find all systems in backward dependency chain (what this depends on)
        OPTIONAL MATCH (e)-[:DEPENDS_ON*1..{depth}]->(backward_dependency)
        WHERE backward_dependency:System OR backward_dependency:Entity
        
        WITH e, 
             collect(DISTINCT forward_dependent) AS forward_deps,
             collect(DISTINCT backward_dependency) AS backward_deps
        
        // Combine and count unique systems
        UNWIND (forward_deps + backward_deps + [e]) AS affected_system
        WITH DISTINCT affected_system
        WHERE affected_system IS NOT NULL
        
        RETURN count(affected_system) AS impact_radius
        """
        
        result = session.run(cypher, entity_names=entity_names)
        
        # Handle both list and result object
        if isinstance(result, list):
            # Mock result (list)
            if result and "impact_radius" in result[0]:
                return result[0]["impact_radius"]
            return 0
        else:
            # Real Neo4j result object
            record = result.single()
            if record:
                return record["impact_radius"]
            return 0
    
    def _detect_circular_dependencies(self, entity_names: List[str], depth: int, session: Session) -> List[List[str]]:
        """
        Detect circular dependencies in the dependency graph.
        
        A circular dependency exists when entity A depends on entity B, and B depends on A
        (directly or through intermediate entities), creating a cycle.
        
        Args:
            entity_names: List of entity names to check for circular dependencies
            depth: Maximum traversal depth
            session: Neo4j session
        
        Returns:
            List of circular dependency chains, where each chain is a list of entity names
        """
        if not entity_names:
            return []
        
        circular_deps = []
        
        # For each entity, check if there's a path back to itself
        for entity_name in entity_names:
            cypher = f"""
            MATCH (e)
            WHERE (e.name = $entity_name OR e.canonical_name = $entity_name)
              AND (e:System OR e:Entity)
            
            // Find circular paths (entity depends on itself through other entities)
            MATCH cycle_path = (e)-[:DEPENDS_ON*1..{depth}]->(e)
            
            // Return the cycle with entity names
            RETURN [node IN nodes(cycle_path) | COALESCE(node.name, node.canonical_name, node.entity_id)] AS cycle_entities
            LIMIT 10
            """
            
            result = session.run(cypher, entity_name=entity_name)
            
            for record in result:
                cycle = record["cycle_entities"]
                if cycle and len(cycle) > 1:
                    # Remove duplicates while preserving order
                    unique_cycle = []
                    seen = set()
                    for entity in cycle:
                        if entity and entity not in seen:
                            seen.add(entity)
                            unique_cycle.append(entity)
                    
                    if len(unique_cycle) > 1 and unique_cycle not in circular_deps:
                        circular_deps.append(unique_cycle)
                        logger.warning(f"Circular dependency detected: {' -> '.join(unique_cycle)}")
        
        return circular_deps
