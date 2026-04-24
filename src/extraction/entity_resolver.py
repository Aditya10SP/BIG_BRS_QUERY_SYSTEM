"""Entity resolver for deduplicating entities across chunks and creating canonical nodes."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple
from collections import defaultdict

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

from src.extraction.entity_extractor import Entity
from src.utils.cache import EntityResolutionCache


logger = logging.getLogger(__name__)


@dataclass
class Relationship:
    """Represents a relationship between two entities in the knowledge graph."""
    rel_id: str
    rel_type: str  # SAME_AS, CONFLICTS_WITH, DEPENDS_ON, etc.
    source_entity_id: str
    target_entity_id: str
    properties: Dict[str, Any] = field(default_factory=dict)


class EntityResolver:
    """
    Deduplicates entities across chunks and creates canonical nodes.
    
    Uses fuzzy matching and DBSCAN clustering to identify duplicate entities,
    then creates SAME_AS relationships linking all mentions to canonical entities.
    """
    
    def __init__(self, similarity_threshold: float = 0.85, enable_cache: bool = True, cache_size: int = 5000):
        """
        Initialize with similarity threshold for matching.
        
        Args:
            similarity_threshold: Minimum similarity score for entity matching (0-1)
            enable_cache: Whether to enable entity resolution caching (default: True)
            cache_size: Maximum number of entity mappings to cache (default: 5000)
        """
        self.similarity_threshold = similarity_threshold
        self.enable_cache = enable_cache
        
        # Initialize cache if enabled
        if enable_cache:
            self.cache = EntityResolutionCache(max_size=cache_size)
            logger.info(f"Entity resolution cache enabled with size={cache_size}")
        else:
            self.cache = None
            logger.info("Entity resolution cache disabled")
        
        logger.info(f"Initialized EntityResolver with similarity_threshold={similarity_threshold}")
    
    def resolve(self, entities: List[Entity]) -> Tuple[List[Entity], List[Relationship]]:
        """
        Deduplicate entities and create SAME_AS relationships.
        
        Algorithm:
        1. Group entities by type (only match within same type)
        2. For each type group:
           a. Compute pairwise similarity between all entities
           b. Use DBSCAN clustering to group similar entities
           c. Select canonical entity for each cluster
           d. Create SAME_AS relationships from all cluster members to canonical
        
        Args:
            entities: List of Entity objects to resolve
            
        Returns:
            Tuple of (canonical_entities, same_as_relationships)
        """
        if not entities:
            logger.info("No entities to resolve")
            return [], []
        
        canonical_entities = []
        same_as_relationships = []
        
        # Group entities by type
        entities_by_type = self._group_by_type(entities)
        
        # Resolve each type group independently
        for entity_type, type_entities in entities_by_type.items():
            logger.info(f"Resolving {len(type_entities)} entities of type '{entity_type}'")
            
            # Resolve this type group
            type_canonical, type_relationships = self._resolve_type_group(
                type_entities, entity_type
            )
            
            canonical_entities.extend(type_canonical)
            same_as_relationships.extend(type_relationships)
        
        logger.info(
            f"Resolution complete: {len(entities)} entities → "
            f"{len(canonical_entities)} canonical entities, "
            f"{len(same_as_relationships)} SAME_AS relationships"
        )
        
        return canonical_entities, same_as_relationships
    
    def _group_by_type(self, entities: List[Entity]) -> Dict[str, List[Entity]]:
        """
        Group entities by entity_type.
        
        Args:
            entities: List of entities to group
            
        Returns:
            Dictionary mapping entity_type to list of entities
        """
        groups = defaultdict(list)
        
        for entity in entities:
            groups[entity.entity_type].append(entity)
        
        return dict(groups)
    
    def _resolve_type_group(
        self, entities: List[Entity], entity_type: str
    ) -> Tuple[List[Entity], List[Relationship]]:
        """
        Resolve entities within a single type group.
        
        Args:
            entities: List of entities of the same type
            entity_type: The entity type being resolved
            
        Returns:
            Tuple of (canonical_entities, same_as_relationships)
        """
        if len(entities) <= 1:
            # No duplicates possible with 0 or 1 entity
            return entities, []
        
        # Compute pairwise similarity matrix
        similarity_matrix = self._compute_similarity_matrix(entities)
        
        # Cluster similar entities using DBSCAN
        clusters = self._cluster_entities(similarity_matrix, entities)
        
        # Select canonical entity for each cluster and create relationships
        canonical_entities = []
        same_as_relationships = []
        
        for cluster_id, cluster_members in clusters.items():
            if len(cluster_members) == 1:
                # Single entity, no duplicates
                canonical_entities.append(cluster_members[0])
            else:
                # Multiple entities, select canonical and create SAME_AS relationships
                canonical = self._select_canonical(cluster_members)
                canonical_entities.append(canonical)
                
                # Create SAME_AS relationships from all members to canonical
                for member in cluster_members:
                    if member.entity_id != canonical.entity_id:
                        rel = self._create_same_as_relationship(
                            member, canonical, entity_type
                        )
                        same_as_relationships.append(rel)
                
                logger.debug(
                    f"Merged {len(cluster_members)} entities into canonical: "
                    f"{canonical.canonical_name} (ID: {canonical.entity_id})"
                )
        
        return canonical_entities, same_as_relationships
    
    def _compute_similarity_matrix(self, entities: List[Entity]) -> np.ndarray:
        """
        Compute pairwise similarity between all entities.
        
        Uses a combination of:
        - String similarity (Levenshtein-based)
        - Canonical name matching
        
        Args:
            entities: List of entities to compare
            
        Returns:
            NxN similarity matrix where N = len(entities)
        """
        n = len(entities)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                else:
                    sim = self._compute_entity_similarity(entities[i], entities[j])
                    similarity_matrix[i][j] = sim
                    similarity_matrix[j][i] = sim  # Symmetric
        
        return similarity_matrix
    
    def _compute_entity_similarity(self, entity1: Entity, entity2: Entity) -> float:
        """
        Compute similarity between two entities.
        
        Combines multiple similarity metrics:
        - Canonical name similarity (70% weight)
        - Original name similarity (30% weight)
        - Containment bonus
        - Acronym matching bonus
        
        Args:
            entity1: First entity
            entity2: Second entity
            
        Returns:
            Similarity score between 0 and 1
        """
        # Canonical name similarity (primary signal)
        canonical_sim = self._string_similarity(
            entity1.canonical_name.lower(),
            entity2.canonical_name.lower()
        )
        
        # Original name similarity (secondary signal)
        name_sim = self._string_similarity(
            entity1.name.lower(),
            entity2.name.lower()
        )
        
        # Check if one name contains the other (e.g., "NEFT" in "NEFT System")
        name1_lower = entity1.canonical_name.lower()
        name2_lower = entity2.canonical_name.lower()
        
        containment_bonus = 0.0
        if name1_lower in name2_lower or name2_lower in name1_lower:
            # Boost similarity if one name contains the other
            containment_bonus = 0.3
        
        # Check for acronym matching (e.g., "NEFT" matches "National Electronic Funds Transfer")
        acronym_bonus = 0.0
        if self._is_acronym_match(name1_lower, name2_lower):
            acronym_bonus = 0.5  # Strong signal for acronym matches
        
        # Weighted combination
        combined_sim = 0.7 * canonical_sim + 0.3 * name_sim + containment_bonus + acronym_bonus
        
        # Cap at 1.0
        return min(1.0, combined_sim)
    
    def _is_acronym_match(self, str1: str, str2: str) -> bool:
        """
        Check if one string is an acronym of the other.
        
        Examples:
        - "neft" matches "national electronic funds transfer"
        - "rtgs" matches "real time gross settlement"
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            True if one is an acronym of the other
        """
        # Check both directions
        return self._is_acronym_of(str1, str2) or self._is_acronym_of(str2, str1)
    
    def _is_acronym_of(self, acronym: str, full_text: str) -> bool:
        """
        Check if acronym matches the first letters of words in full_text.
        
        Args:
            acronym: Potential acronym (e.g., "neft")
            full_text: Full text (e.g., "national electronic funds transfer")
            
        Returns:
            True if acronym matches first letters of words
        """
        # Split full text into words
        words = full_text.split()
        
        # If acronym is longer than number of words, can't match
        if len(acronym) > len(words):
            return False
        
        # Check if acronym matches first letters of words
        first_letters = ''.join(word[0] for word in words if word)
        
        # Check if acronym matches the beginning of first_letters
        return first_letters.startswith(acronym) and len(acronym) >= 3
    
    def _string_similarity(self, str1: str, str2: str) -> float:
        """
        Compute string similarity using normalized Levenshtein distance.
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Similarity score between 0 and 1
        """
        if str1 == str2:
            return 1.0
        
        if not str1 or not str2:
            return 0.0
        
        # Compute Levenshtein distance
        distance = self._levenshtein_distance(str1, str2)
        
        # Normalize by max length
        max_len = max(len(str1), len(str2))
        similarity = 1.0 - (distance / max_len)
        
        return max(0.0, similarity)
    
    def _levenshtein_distance(self, str1: str, str2: str) -> int:
        """
        Compute Levenshtein distance between two strings.
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Edit distance
        """
        m, n = len(str1), len(str2)
        
        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i - 1] == str2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],      # Deletion
                        dp[i][j - 1],      # Insertion
                        dp[i - 1][j - 1]   # Substitution
                    )
        
        return dp[m][n]
    
    def _cluster_entities(
        self, similarity_matrix: np.ndarray, entities: List[Entity]
    ) -> Dict[int, List[Entity]]:
        """
        Cluster entities using DBSCAN based on similarity matrix.
        
        Args:
            similarity_matrix: NxN similarity matrix
            entities: List of entities corresponding to matrix rows/cols
            
        Returns:
            Dictionary mapping cluster_id to list of entities
        """
        # Convert similarity to distance for DBSCAN
        # Distance = 1 - similarity
        distance_matrix = 1.0 - similarity_matrix
        
        # Use DBSCAN with precomputed distance matrix
        # eps = 1 - similarity_threshold (distance threshold)
        # min_samples = 1 (allow single-entity clusters)
        eps = 1.0 - self.similarity_threshold
        
        clustering = DBSCAN(
            eps=eps,
            min_samples=1,
            metric='precomputed'
        )
        
        labels = clustering.fit_predict(distance_matrix)
        
        # Group entities by cluster label
        clusters = defaultdict(list)
        for idx, label in enumerate(labels):
            clusters[label].append(entities[idx])
        
        logger.debug(f"DBSCAN clustering: {len(entities)} entities → {len(clusters)} clusters")
        
        return dict(clusters)
    
    def _select_canonical(self, cluster_members: List[Entity]) -> Entity:
        """
        Select canonical entity from cluster members.
        
        Selection criteria (in order of priority):
        1. Most source chunk references (if we had multiple mentions)
        2. Longest canonical name (more complete)
        3. Longest context (more information)
        4. First entity (deterministic fallback)
        
        Args:
            cluster_members: List of entities in the cluster
            
        Returns:
            Selected canonical entity
        """
        if len(cluster_members) == 1:
            return cluster_members[0]
        
        # Score each entity
        scored_entities = []
        
        for entity in cluster_members:
            score = 0
            
            # Prefer longer canonical names (more complete)
            score += len(entity.canonical_name) * 10
            
            # Prefer longer context (more information)
            score += len(entity.context)
            
            # Prefer entities with more properties
            score += len(entity.properties) * 5
            
            scored_entities.append((score, entity))
        
        # Sort by score descending
        scored_entities.sort(key=lambda x: x[0], reverse=True)
        
        canonical = scored_entities[0][1]
        
        # Merge source chunk references from all members
        canonical = self._merge_entity_properties(canonical, cluster_members)
        
        return canonical
    
    def _merge_entity_properties(
        self, canonical: Entity, cluster_members: List[Entity]
    ) -> Entity:
        """
        Merge properties from all cluster members into canonical entity.
        
        Preserves all source chunk references and combines properties.
        
        Args:
            canonical: The canonical entity
            cluster_members: All entities in the cluster
            
        Returns:
            Canonical entity with merged properties
        """
        # Collect all source chunk IDs
        source_chunks = set()
        for member in cluster_members:
            source_chunks.add(member.source_chunk_id)
        
        # Update canonical entity properties
        canonical.properties["source_chunk_ids"] = list(source_chunks)
        canonical.properties["mention_count"] = len(cluster_members)
        
        # Collect all unique names (aliases)
        aliases = set()
        for member in cluster_members:
            if member.name != canonical.name:
                aliases.add(member.name)
        
        if aliases:
            canonical.properties["aliases"] = list(aliases)
        
        return canonical
    
    def _create_same_as_relationship(
        self, source: Entity, target: Entity, entity_type: str
    ) -> Relationship:
        """
        Create SAME_AS relationship from source entity to canonical target.
        
        Args:
            source: Source entity (mention)
            target: Target entity (canonical)
            entity_type: Entity type
            
        Returns:
            Relationship object
        """
        rel_id = f"same_as_{source.entity_id}_{target.entity_id}"
        
        relationship = Relationship(
            rel_id=rel_id,
            rel_type="SAME_AS",
            source_entity_id=source.entity_id,
            target_entity_id=target.entity_id,
            properties={
                "entity_type": entity_type,
                "source_name": source.name,
                "target_name": target.name,
                "canonical_name": target.canonical_name
            }
        )
        
        return relationship
