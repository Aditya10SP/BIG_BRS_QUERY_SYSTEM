"""Conflict detector for identifying contradictory information and creating CONFLICTS_WITH relationships."""

import json
import logging
import re
from typing import Any, Dict, List, Optional

import requests

from src.chunking.hierarchical_chunker import Chunk
from src.extraction.entity_extractor import Entity
from src.extraction.entity_resolver import Relationship


logger = logging.getLogger(__name__)


class ConflictDetector:
    """
    Identifies contradictory information and creates CONFLICTS_WITH relationships.
    
    Detects three types of conflicts:
    1. Property conflicts: Same entity with different property values
    2. Rule conflicts: Contradictory rules for same scenario
    3. Workflow conflicts: Different process flows for same operation
    
    Uses LLM-based semantic analysis to identify conflicts beyond exact contradictions.
    """
    
    # LLM prompt template for semantic conflict detection
    CONFLICT_DETECTION_PROMPT = """Compare these two statements about {entity_name}:

Statement 1 (from {doc1}): {text1}

Statement 2 (from {doc2}): {text2}

Do they conflict or contradict each other? Consider:
- Different values for the same property (e.g., different limits, different requirements)
- Contradictory rules or policies
- Incompatible process flows or workflows

Respond with JSON only, no other text:
{{
  "conflicts": true or false,
  "conflict_type": "property" or "rule" or "workflow",
  "explanation": "Brief explanation of the conflict"
}}"""
    
    def __init__(self, ollama_base_url: str = None, llm_model: str = None):
        """
        Initialize with LLM client for semantic conflict detection.
        
        Args:
            ollama_base_url: Base URL for Ollama API
            llm_model: Name of LLM model to use
        """
        self.ollama_base_url = ollama_base_url
        self.llm_model = llm_model
        logger.info(f"Initialized ConflictDetector with LLM model: {llm_model}")
    
    def detect(self, entities: List[Entity], chunks: List[Chunk]) -> List[Relationship]:
        """
        Detect conflicts and return CONFLICTS_WITH relationships.
        
        Algorithm:
        1. Group entities by canonical name (same entity across documents)
        2. For each entity group with multiple mentions:
           a. Compare all pairs of mentions for property conflicts
           b. Use LLM to detect semantic conflicts
           c. Create bidirectional CONFLICTS_WITH relationships
        
        Args:
            entities: List of Entity objects to check for conflicts
            chunks: List of Chunk objects for context retrieval
            
        Returns:
            List of Relationship objects with conflict metadata
        """
        if not entities:
            logger.info("No entities to check for conflicts")
            return []
        
        if not self.ollama_base_url or not self.llm_model:
            logger.warning("LLM configuration not provided. Conflict detection will be limited.")
            return []
        
        conflicts = []
        
        # Create chunk lookup for fast access
        chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
        
        # Group entities by canonical name
        entity_groups = self._group_by_canonical_name(entities)
        
        # Check each group for conflicts
        for canonical_name, group_entities in entity_groups.items():
            if len(group_entities) < 2:
                # Need at least 2 entities to have a conflict
                continue
            
            logger.info(f"Checking {len(group_entities)} mentions of '{canonical_name}' for conflicts")
            
            # Compare all pairs of entities
            for i in range(len(group_entities)):
                for j in range(i + 1, len(group_entities)):
                    entity1 = group_entities[i]
                    entity2 = group_entities[j]
                    
                    # Get chunks for context
                    chunk1 = chunk_map.get(entity1.source_chunk_id)
                    chunk2 = chunk_map.get(entity2.source_chunk_id)
                    
                    if not chunk1 or not chunk2:
                        logger.warning(f"Missing chunks for entities {entity1.entity_id}, {entity2.entity_id}")
                        continue
                    
                    # Check for property conflicts first (fast)
                    property_conflict = self._check_property_conflict(entity1, entity2)
                    
                    if property_conflict:
                        # Create conflict relationships
                        conflict_rels = self._create_conflict_relationships(
                            entity1, entity2, chunk1, chunk2, property_conflict
                        )
                        conflicts.extend(conflict_rels)
                        logger.info(f"Property conflict detected: {canonical_name}")
                        continue
                    
                    # Check for semantic conflicts using LLM (slower)
                    semantic_conflict = self._check_semantic_conflict(
                        entity1, entity2, chunk1, chunk2, canonical_name
                    )
                    
                    if semantic_conflict:
                        # Create conflict relationships
                        conflict_rels = self._create_conflict_relationships(
                            entity1, entity2, chunk1, chunk2, semantic_conflict
                        )
                        conflicts.extend(conflict_rels)
                        logger.info(f"Semantic conflict detected: {canonical_name}")
        
        logger.info(f"Conflict detection complete: {len(conflicts)} CONFLICTS_WITH relationships created")
        
        return conflicts
    
    def _group_by_canonical_name(self, entities: List[Entity]) -> Dict[str, List[Entity]]:
        """
        Group entities by canonical name (case-insensitive).
        
        Args:
            entities: List of entities to group
            
        Returns:
            Dictionary mapping canonical_name to list of entities
        """
        groups: Dict[str, List[Entity]] = {}
        
        for entity in entities:
            key = entity.canonical_name.lower()
            if key not in groups:
                groups[key] = []
            groups[key].append(entity)
        
        return groups
    
    def _check_property_conflict(
        self, entity1: Entity, entity2: Entity
    ) -> Optional[Dict[str, Any]]:
        """
        Check for property conflicts between two entities.
        
        Property conflicts occur when the same entity has different values
        for the same property (e.g., different limits, different requirements).
        
        Args:
            entity1: First entity
            entity2: Second entity
            
        Returns:
            Conflict metadata dict if conflict found, None otherwise
        """
        # Check if entities have conflicting properties
        conflicting_properties = []
        
        # Get all property keys from both entities
        all_keys = set(entity1.properties.keys()) | set(entity2.properties.keys())
        
        for key in all_keys:
            # Skip metadata properties
            if key in ["extraction_method", "spacy_label", "start_char", "end_char", 
                      "source_chunk_ids", "mention_count", "aliases"]:
                continue
            
            val1 = entity1.properties.get(key)
            val2 = entity2.properties.get(key)
            
            # If both have the property and values differ
            if val1 is not None and val2 is not None and val1 != val2:
                conflicting_properties.append({
                    "property": key,
                    "value1": val1,
                    "value2": val2
                })
        
        if conflicting_properties:
            return {
                "conflicts": True,
                "conflict_type": "property",
                "explanation": f"Different values for properties: {', '.join(p['property'] for p in conflicting_properties)}",
                "conflicting_properties": conflicting_properties
            }
        
        return None
    
    def _check_semantic_conflict(
        self, entity1: Entity, entity2: Entity, chunk1: Chunk, chunk2: Chunk, entity_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Check for semantic conflicts using LLM analysis.
        
        Args:
            entity1: First entity
            entity2: Second entity
            chunk1: Chunk containing entity1
            chunk2: Chunk containing entity2
            entity_name: Canonical entity name
            
        Returns:
            Conflict metadata dict if conflict found, None otherwise
        """
        # Get context for both entities
        context1 = entity1.context or chunk1.text[:500]
        context2 = entity2.context or chunk2.text[:500]
        
        # Get document identifiers
        doc1 = chunk1.doc_id or "Document 1"
        doc2 = chunk2.doc_id or "Document 2"
        
        # Create prompt
        prompt = self.CONFLICT_DETECTION_PROMPT.format(
            entity_name=entity_name,
            doc1=doc1,
            doc2=doc2,
            text1=context1,
            text2=context2
        )
        
        # Call LLM
        response = self._call_ollama(prompt)
        
        if not response:
            return None
        
        # Parse response
        conflict_data = self._parse_conflict_response(response)
        
        if conflict_data and conflict_data.get("conflicts"):
            return conflict_data
        
        return None
    
    def _call_ollama(self, prompt: str, max_retries: int = 2) -> Optional[str]:
        """
        Call Ollama API for LLM generation.
        
        Args:
            prompt: Prompt to send to LLM
            max_retries: Maximum number of retry attempts
            
        Returns:
            LLM response text or None if failed
        """
        url = f"{self.ollama_base_url}/api/generate"
        
        payload = {
            "model": self.llm_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for more deterministic detection
                "num_predict": 500  # Limit response length
            }
        }
        
        for attempt in range(max_retries + 1):
            try:
                response = requests.post(url, json=payload, timeout=120)  # 2 minute timeout for conflict detection
                response.raise_for_status()
                
                result = response.json()
                return result.get("response", "")
            
            except requests.exceptions.RequestException as e:
                logger.warning(f"Ollama API call failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                if attempt == max_retries:
                    logger.error(f"Ollama API call failed after {max_retries + 1} attempts")
                    return None
        
        return None
    
    def _parse_conflict_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Parse LLM response to extract conflict data.
        
        Args:
            response: LLM response text
            
        Returns:
            Conflict data dictionary or None if parsing failed
        """
        try:
            # Try to find JSON object in response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                conflict_data = json.loads(json_str)
                
                if isinstance(conflict_data, dict):
                    return conflict_data
            
            # If no JSON object found, try parsing entire response
            conflict_data = json.loads(response)
            if isinstance(conflict_data, dict):
                return conflict_data
            
            logger.warning("LLM response is not a JSON object")
            return None
        
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Response: {response[:500]}")
            return None
    
    def _create_conflict_relationships(
        self, entity1: Entity, entity2: Entity, chunk1: Chunk, chunk2: Chunk,
        conflict_data: Dict[str, Any]
    ) -> List[Relationship]:
        """
        Create bidirectional CONFLICTS_WITH relationships.
        
        Args:
            entity1: First entity
            entity2: Second entity
            chunk1: Chunk containing entity1
            chunk2: Chunk containing entity2
            conflict_data: Conflict metadata from detection
            
        Returns:
            List of two Relationship objects (bidirectional)
        """
        relationships = []
        
        # Extract conflict metadata
        conflict_type = conflict_data.get("conflict_type", "unknown")
        explanation = conflict_data.get("explanation", "Conflicting information detected")
        
        # Create metadata for relationships
        metadata = {
            "conflict_type": conflict_type,
            "explanation": explanation,
            "source_chunk_ids": [chunk1.chunk_id, chunk2.chunk_id],
            "doc_ids": [chunk1.doc_id, chunk2.doc_id],
            "entity1_context": entity1.context[:200] if entity1.context else "",
            "entity2_context": entity2.context[:200] if entity2.context else ""
        }
        
        # Add conflicting properties if available
        if "conflicting_properties" in conflict_data:
            metadata["conflicting_properties"] = conflict_data["conflicting_properties"]
        
        # Create forward relationship (entity1 -> entity2)
        rel_id_forward = f"conflict_{entity1.entity_id}_{entity2.entity_id}"
        rel_forward = Relationship(
            rel_id=rel_id_forward,
            rel_type="CONFLICTS_WITH",
            source_entity_id=entity1.entity_id,
            target_entity_id=entity2.entity_id,
            properties=metadata.copy()
        )
        relationships.append(rel_forward)
        
        # Create backward relationship (entity2 -> entity1) for bidirectionality
        rel_id_backward = f"conflict_{entity2.entity_id}_{entity1.entity_id}"
        rel_backward = Relationship(
            rel_id=rel_id_backward,
            rel_type="CONFLICTS_WITH",
            source_entity_id=entity2.entity_id,
            target_entity_id=entity1.entity_id,
            properties=metadata.copy()
        )
        relationships.append(rel_backward)
        
        return relationships
