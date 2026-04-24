"""Graph populator for creating Neo4j knowledge graph from entities and relationships."""

import logging
import json
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import ServiceUnavailable, Neo4jError

from src.extraction.entity_extractor import Entity
from src.extraction.entity_resolver import Relationship
from src.chunking.hierarchical_chunker import Chunk
from src.utils.batch_processor import Neo4jBatchProcessor
from src.utils.indexing import Neo4jIndexManager


logger = logging.getLogger(__name__)


class GraphPopulator:
    """
    Creates Neo4j knowledge graph from entities and relationships.
    
    Manages Neo4j connection and provides methods to:
    - Create schema (node types and indexes)
    - Populate graph with nodes and relationships
    - Batch operations for efficient bulk inserts
    """
    
    # Use optimized batch sizes from Neo4jBatchProcessor
    NODE_BATCH_SIZE = Neo4jBatchProcessor.NODE_BATCH_SIZE
    RELATIONSHIP_BATCH_SIZE = Neo4jBatchProcessor.RELATIONSHIP_BATCH_SIZE
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        """
        Initialize Neo4j connection.
        
        Args:
            neo4j_uri: Neo4j connection URI (e.g., "bolt://localhost:7687")
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
        
        Raises:
            ServiceUnavailable: If connection to Neo4j fails
        """
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.driver: Optional[Driver] = None
        
        logger.info(
            f"Initializing GraphPopulator with Neo4j at {neo4j_uri} "
            f"(node_batch_size={self.NODE_BATCH_SIZE}, "
            f"relationship_batch_size={self.RELATIONSHIP_BATCH_SIZE})"
        )
        
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
    
    def create_schema(self) -> None:
        """
        Create node types, indexes, and constraints in Neo4j.
        
        Creates:
        - Indexes on entity names and types for fast lookup
        - Uniqueness constraints on entity IDs
        - Indexes on chunk IDs and document IDs
        - Composite indexes for common query patterns
        - Full-text search indexes
        """
        with self.driver.session() as session:
            try:
                # Use Neo4jIndexManager to create optimized indexes
                Neo4jIndexManager.create_optimized_indexes(session)
                
                logger.info("Neo4j schema and indexes created successfully")
                
            except Neo4jError as e:
                logger.error(f"Failed to create schema: {e}")
                raise
    
    def populate(
        self,
        entities: List[Entity],
        relationships: List[Relationship],
        chunks: List[Chunk],
        documents: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Populate Neo4j graph with nodes and relationships.
        
        Creates:
        - Document nodes
        - Section nodes (extracted from chunks)
        - Entity nodes (System, PaymentMode, Workflow, Rule, Field)
        - Chunk nodes
        - Relationships: CONTAINS, HAS_CHUNK, MENTIONS, SAME_AS, CONFLICTS_WITH, etc.
        
        Args:
            entities: List of Entity objects to create as nodes
            relationships: List of Relationship objects to create as edges
            chunks: List of Chunk objects to create as nodes
            documents: Optional list of document metadata dictionaries
        
        Raises:
            Neo4jError: If graph population fails
        """
        logger.info(
            f"Populating graph: {len(entities)} entities, "
            f"{len(relationships)} relationships, {len(chunks)} chunks"
        )
        
        with self.driver.session() as session:
            try:
                # Create document nodes
                if documents:
                    self._create_document_nodes(session, documents)
                
                # Create entity nodes
                self._create_entity_nodes(session, entities)
                
                # Create chunk nodes
                self._create_chunk_nodes(session, chunks)
                
                # Create section nodes (from chunks)
                self._create_section_nodes(session, chunks)
                
                # Create relationships
                self._create_relationships(session, relationships)
                
                # Create MENTIONS relationships (chunks to entities)
                self._create_mentions_relationships(session, entities)
                
                # Create document structure relationships (CONTAINS, HAS_CHUNK)
                self._create_structure_relationships(session, chunks)
                
                logger.info("Graph population completed successfully")
                
            except Neo4jError as e:
                logger.error(f"Failed to populate graph: {e}")
                raise
    
    def _create_document_nodes(
        self, session: Session, documents: List[Dict[str, Any]]
    ) -> None:
        """
        Create Document nodes in batches.
        
        Args:
            session: Neo4j session
            documents: List of document metadata dictionaries
        """
        if not documents:
            return
        
        query = """
        UNWIND $batch AS doc
        MERGE (d:Document {doc_id: doc.doc_id})
        SET d.title = doc.title,
            d.file_path = doc.file_path,
            d.file_type = doc.file_type,
            d.metadata_json = doc.metadata_json
        """
        
        # Serialize metadata to JSON strings
        documents_with_json = [
            {
                **doc,
                "metadata_json": json.dumps(doc.get("metadata", {}))
            }
            for doc in documents
        ]
        
        self._execute_batch(session, query, documents_with_json, "documents")
    
    def _create_entity_nodes(self, session: Session, entities: List[Entity]) -> None:
        """
        Create Entity nodes in batches.
        
        Creates nodes with labels: Entity and specific type (System, PaymentMode, etc.)
        
        Args:
            session: Neo4j session
            entities: List of Entity objects
        """
        if not entities:
            return
        
        # Convert entities to dictionaries for batch processing
        entity_dicts = []
        for entity in entities:
            entity_dict = {
                "entity_id": entity.entity_id,
                "entity_type": entity.entity_type,
                "name": entity.name,
                "canonical_name": entity.canonical_name,
                "source_chunk_id": entity.source_chunk_id,
                "context": entity.context,
                "properties": entity.properties
            }
            entity_dicts.append(entity_dict)
        
        # Use dynamic label based on entity_type
        query = """
        UNWIND $batch AS ent
        CALL apoc.create.node(['Entity', ent.entity_type], {
            entity_id: ent.entity_id,
            name: ent.name,
            canonical_name: ent.canonical_name,
            entity_type: ent.entity_type,
            source_chunk_id: ent.source_chunk_id,
            context: ent.context,
            properties: ent.properties
        }) YIELD node
        RETURN count(node)
        """
        
        # Serialize properties to JSON strings
        for ent_dict in entity_dicts:
            ent_dict["properties_json"] = json.dumps(ent_dict.get("properties", {}))
        
        # Fallback query without APOC (simpler but less efficient)
        fallback_query = """
        UNWIND $batch AS ent
        MERGE (e:Entity {entity_id: ent.entity_id})
        SET e.name = ent.name,
            e.canonical_name = ent.canonical_name,
            e.entity_type = ent.entity_type,
            e.source_chunk_id = ent.source_chunk_id,
            e.context = ent.context,
            e.properties_json = ent.properties_json
        """
        
        try:
            self._execute_batch(session, query, entity_dicts, "entities")
        except Neo4jError as e:
            # If APOC is not available, use fallback
            logger.warning(f"APOC not available, using fallback query: {e}")
            self._execute_batch(session, fallback_query, entity_dicts, "entities")
    
    def _create_chunk_nodes(self, session: Session, chunks: List[Chunk]) -> None:
        """
        Create Chunk nodes in batches.
        
        Args:
            session: Neo4j session
            chunks: List of Chunk objects
        """
        if not chunks:
            return
        
        # Convert chunks to dictionaries
        chunk_dicts = []
        for chunk in chunks:
            chunk_dict = {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "text": chunk.text,
                "chunk_type": chunk.chunk_type,
                "parent_chunk_id": chunk.parent_chunk_id,
                "breadcrumbs": chunk.breadcrumbs,
                "section": chunk.section,
                "token_count": chunk.token_count,
                "metadata": chunk.metadata
            }
            chunk_dicts.append(chunk_dict)
        
        # Serialize metadata to JSON strings
        for chunk_dict in chunk_dicts:
            chunk_dict["metadata_json"] = json.dumps(chunk_dict.get("metadata", {}))
        
        query = """
        UNWIND $batch AS chunk
        MERGE (c:Chunk {chunk_id: chunk.chunk_id})
        SET c.doc_id = chunk.doc_id,
            c.text = chunk.text,
            c.chunk_type = chunk.chunk_type,
            c.parent_chunk_id = chunk.parent_chunk_id,
            c.breadcrumbs = chunk.breadcrumbs,
            c.section = chunk.section,
            c.token_count = chunk.token_count,
            c.metadata_json = chunk.metadata_json
        """
        
        self._execute_batch(session, query, chunk_dicts, "chunks")
    
    def _create_section_nodes(self, session: Session, chunks: List[Chunk]) -> None:
        """
        Create Section nodes from unique sections in chunks.
        
        Args:
            session: Neo4j session
            chunks: List of Chunk objects
        """
        if not chunks:
            return
        
        # Extract unique sections
        sections = {}
        for chunk in chunks:
            if chunk.section and chunk.section not in sections:
                # Create section ID from doc_id and section name
                section_id = f"{chunk.doc_id}_{chunk.section.replace(' ', '_')}"
                sections[section_id] = {
                    "section_id": section_id,
                    "doc_id": chunk.doc_id,
                    "heading": chunk.section,
                    "level": 1  # Default level, could be extracted from breadcrumbs
                }
        
        if not sections:
            return
        
        section_list = list(sections.values())
        
        query = """
        UNWIND $batch AS sec
        MERGE (s:Section {section_id: sec.section_id})
        SET s.doc_id = sec.doc_id,
            s.heading = sec.heading,
            s.level = sec.level
        """
        
        self._execute_batch(session, query, section_list, "sections")
    
    def _create_relationships(
        self, session: Session, relationships: List[Relationship]
    ) -> None:
        """
        Create relationship edges in batches.
        
        Handles: SAME_AS, CONFLICTS_WITH, DEPENDS_ON, INTEGRATES_WITH, etc.
        
        Args:
            session: Neo4j session
            relationships: List of Relationship objects
        """
        if not relationships:
            return
        
        # Convert relationships to dictionaries
        rel_dicts = []
        for rel in relationships:
            rel_dict = {
                "rel_id": rel.rel_id,
                "rel_type": rel.rel_type,
                "source_entity_id": rel.source_entity_id,
                "target_entity_id": rel.target_entity_id,
                "properties": rel.properties
            }
            rel_dicts.append(rel_dict)
        
        # Serialize properties to JSON strings
        for rel_dict in rel_dicts:
            rel_dict["properties_json"] = json.dumps(rel_dict.get("properties", {}))
        
        # Group by relationship type for efficient creation
        rels_by_type = {}
        for rel_dict in rel_dicts:
            rel_type = rel_dict["rel_type"]
            if rel_type not in rels_by_type:
                rels_by_type[rel_type] = []
            rels_by_type[rel_type].append(rel_dict)
        
        # Create relationships for each type
        for rel_type, rels in rels_by_type.items():
            query = f"""
            UNWIND $batch AS rel
            MATCH (source:Entity {{entity_id: rel.source_entity_id}})
            MATCH (target:Entity {{entity_id: rel.target_entity_id}})
            MERGE (source)-[r:{rel_type}]->(target)
            SET r.rel_id = rel.rel_id,
                r.properties_json = rel.properties_json
            """
            
            self._execute_batch(session, query, rels, f"{rel_type} relationships")
    
    def _create_mentions_relationships(
        self, session: Session, entities: List[Entity]
    ) -> None:
        """
        Create MENTIONS relationships from chunks to entities.
        
        Args:
            session: Neo4j session
            entities: List of Entity objects
        """
        if not entities:
            return
        
        # Create MENTIONS relationships
        mentions = []
        for entity in entities:
            mentions.append({
                "chunk_id": entity.source_chunk_id,
                "entity_id": entity.entity_id
            })
        
        query = """
        UNWIND $batch AS mention
        MATCH (c:Chunk {chunk_id: mention.chunk_id})
        MATCH (e:Entity {entity_id: mention.entity_id})
        MERGE (c)-[r:MENTIONS]->(e)
        """
        
        self._execute_batch(session, query, mentions, "MENTIONS relationships")
    
    def _create_structure_relationships(
        self, session: Session, chunks: List[Chunk]
    ) -> None:
        """
        Create document structure relationships.
        
        Creates:
        - Document CONTAINS Section
        - Section HAS_CHUNK Chunk
        - Parent Chunk CONTAINS Child Chunk
        
        Args:
            session: Neo4j session
            chunks: List of Chunk objects
        """
        if not chunks:
            return
        
        # Create Document CONTAINS Section relationships
        doc_section_rels = []
        seen_doc_sections = set()
        
        for chunk in chunks:
            if chunk.section:
                section_id = f"{chunk.doc_id}_{chunk.section.replace(' ', '_')}"
                key = (chunk.doc_id, section_id)
                if key not in seen_doc_sections:
                    doc_section_rels.append({
                        "doc_id": chunk.doc_id,
                        "section_id": section_id
                    })
                    seen_doc_sections.add(key)
        
        if doc_section_rels:
            query = """
            UNWIND $batch AS rel
            MATCH (d:Document {doc_id: rel.doc_id})
            MATCH (s:Section {section_id: rel.section_id})
            MERGE (d)-[r:CONTAINS]->(s)
            """
            self._execute_batch(session, query, doc_section_rels, "CONTAINS relationships")
        
        # Create Section HAS_CHUNK Chunk relationships
        section_chunk_rels = []
        for chunk in chunks:
            if chunk.section:
                section_id = f"{chunk.doc_id}_{chunk.section.replace(' ', '_')}"
                section_chunk_rels.append({
                    "section_id": section_id,
                    "chunk_id": chunk.chunk_id
                })
        
        if section_chunk_rels:
            query = """
            UNWIND $batch AS rel
            MATCH (s:Section {section_id: rel.section_id})
            MATCH (c:Chunk {chunk_id: rel.chunk_id})
            MERGE (s)-[r:HAS_CHUNK]->(c)
            """
            self._execute_batch(session, query, section_chunk_rels, "HAS_CHUNK relationships")
        
        # Create parent-child chunk relationships
        parent_child_rels = []
        for chunk in chunks:
            if chunk.parent_chunk_id:
                parent_child_rels.append({
                    "parent_id": chunk.parent_chunk_id,
                    "child_id": chunk.chunk_id
                })
        
        if parent_child_rels:
            query = """
            UNWIND $batch AS rel
            MATCH (parent:Chunk {chunk_id: rel.parent_id})
            MATCH (child:Chunk {chunk_id: rel.child_id})
            MERGE (parent)-[r:CONTAINS]->(child)
            """
            self._execute_batch(session, query, parent_child_rels, "parent-child CONTAINS relationships")
    
    def _execute_batch(
        self,
        session: Session,
        query: str,
        data: List[Dict[str, Any]],
        description: str,
        batch_size: Optional[int] = None
    ) -> None:
        """
        Execute query in batches for efficient bulk operations.
        
        Args:
            session: Neo4j session
            query: Cypher query with $batch parameter
            data: List of data dictionaries
            description: Description for logging
            batch_size: Override default batch size (optional)
        """
        if not data:
            return
        
        # Determine batch size based on operation type
        if batch_size is None:
            if 'relationship' in description.lower():
                batch_size = self.RELATIONSHIP_BATCH_SIZE
            else:
                batch_size = self.NODE_BATCH_SIZE
        
        total = len(data)
        batches = (total + batch_size - 1) // batch_size
        
        logger.info(f"Creating {total} {description} in {batches} batches (batch_size={batch_size})")
        
        for i in range(0, total, batch_size):
            batch = data[i:i + batch_size]
            try:
                session.run(query, batch=batch)
                logger.debug(f"Processed batch {i // batch_size + 1}/{batches} for {description}")
            except Neo4jError as e:
                logger.error(f"Failed to process batch for {description}: {e}")
                raise
        
        logger.info(f"Successfully created {total} {description}")
