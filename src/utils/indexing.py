"""Database indexing utilities for performance optimization."""

import logging
from typing import List, Dict, Any, Optional
from psycopg2 import sql
from neo4j import Session

logger = logging.getLogger(__name__)


class PostgreSQLIndexManager:
    """
    Manages PostgreSQL indexes for optimal query performance.
    
    Provides utilities for:
    - Creating indexes on frequently queried columns
    - Creating composite indexes for multi-column queries
    - Creating partial indexes for filtered queries
    - Analyzing index usage and recommendations
    """
    
    @staticmethod
    def create_optimized_indexes(cursor) -> None:
        """
        Create optimized indexes for the document store.
        
        Creates indexes on:
        - Primary keys (already created by schema)
        - Foreign keys for joins
        - Frequently filtered columns
        - Text search columns
        - Composite indexes for common query patterns
        
        Args:
            cursor: PostgreSQL cursor
        """
        logger.info("Creating optimized PostgreSQL indexes")
        
        # Indexes on documents table
        indexes = [
            # Basic indexes (already in schema, but ensure they exist)
            "CREATE INDEX IF NOT EXISTS idx_documents_title ON documents(title)",
            "CREATE INDEX IF NOT EXISTS idx_documents_file_type ON documents(file_type)",
            "CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at)",
            
            # JSONB indexes for metadata queries
            "CREATE INDEX IF NOT EXISTS idx_documents_metadata_gin ON documents USING GIN (metadata)",
            
            # Indexes on chunks table
            # Basic indexes (already in schema)
            "CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id)",
            "CREATE INDEX IF NOT EXISTS idx_chunks_parent ON chunks(parent_chunk_id)",
            "CREATE INDEX IF NOT EXISTS idx_chunks_section ON chunks(section)",
            
            # Additional indexes for common queries
            "CREATE INDEX IF NOT EXISTS idx_chunks_chunk_type ON chunks(chunk_type)",
            "CREATE INDEX IF NOT EXISTS idx_chunks_token_count ON chunks(token_count)",
            "CREATE INDEX IF NOT EXISTS idx_chunks_created_at ON chunks(created_at)",
            
            # Composite indexes for common query patterns
            "CREATE INDEX IF NOT EXISTS idx_chunks_doc_section ON chunks(doc_id, section)",
            "CREATE INDEX IF NOT EXISTS idx_chunks_doc_type ON chunks(doc_id, chunk_type)",
            
            # JSONB indexes for metadata queries
            "CREATE INDEX IF NOT EXISTS idx_chunks_metadata_gin ON chunks USING GIN (metadata)",
            
            # Full-text search index on chunk text
            "CREATE INDEX IF NOT EXISTS idx_chunks_text_fts ON chunks USING GIN (to_tsvector('english', text))",
            
            # Partial indexes for specific chunk types (more efficient for filtered queries)
            "CREATE INDEX IF NOT EXISTS idx_chunks_parent_type ON chunks(doc_id, parent_chunk_id) WHERE chunk_type = 'child'",
        ]
        
        for index_sql in indexes:
            try:
                cursor.execute(index_sql)
                logger.debug(f"Created index: {index_sql[:80]}...")
            except Exception as e:
                logger.warning(f"Failed to create index: {str(e)}")
        
        logger.info("PostgreSQL indexes created successfully")
    
    @staticmethod
    def analyze_index_usage(cursor) -> List[Dict[str, Any]]:
        """
        Analyze index usage statistics.
        
        Returns information about:
        - Index size
        - Number of scans
        - Tuples read/fetched
        - Unused indexes
        
        Args:
            cursor: PostgreSQL cursor
        
        Returns:
            List of index usage statistics
        """
        query = """
        SELECT
            schemaname,
            tablename,
            indexname,
            idx_scan as scans,
            idx_tup_read as tuples_read,
            idx_tup_fetch as tuples_fetched,
            pg_size_pretty(pg_relation_size(indexrelid)) as index_size
        FROM pg_stat_user_indexes
        WHERE schemaname = 'public'
        ORDER BY idx_scan ASC, tablename, indexname
        """
        
        try:
            cursor.execute(query)
            results = cursor.fetchall()
            
            # Convert to list of dictionaries
            columns = [desc[0] for desc in cursor.description]
            index_stats = [dict(zip(columns, row)) for row in results]
            
            logger.info(f"Retrieved usage statistics for {len(index_stats)} indexes")
            return index_stats
            
        except Exception as e:
            logger.error(f"Failed to analyze index usage: {str(e)}")
            return []
    
    @staticmethod
    def get_missing_indexes(cursor) -> List[str]:
        """
        Identify potential missing indexes based on query patterns.
        
        Analyzes sequential scans and suggests indexes.
        
        Args:
            cursor: PostgreSQL cursor
        
        Returns:
            List of index recommendations
        """
        query = """
        SELECT
            schemaname,
            tablename,
            seq_scan,
            seq_tup_read,
            idx_scan,
            seq_tup_read / seq_scan as avg_seq_tup_read
        FROM pg_stat_user_tables
        WHERE schemaname = 'public'
          AND seq_scan > 0
        ORDER BY seq_tup_read DESC
        """
        
        try:
            cursor.execute(query)
            results = cursor.fetchall()
            
            recommendations = []
            for row in results:
                schemaname, tablename, seq_scan, seq_tup_read, idx_scan, avg_seq_tup_read = row
                
                # If table has many sequential scans with high tuple reads, suggest indexing
                if seq_scan > 100 and avg_seq_tup_read > 1000:
                    recommendations.append(
                        f"Consider adding indexes to {tablename}: "
                        f"{seq_scan} sequential scans reading {seq_tup_read} tuples"
                    )
            
            logger.info(f"Generated {len(recommendations)} index recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to get missing indexes: {str(e)}")
            return []


class Neo4jIndexManager:
    """
    Manages Neo4j indexes and constraints for optimal graph query performance.
    
    Provides utilities for:
    - Creating node property indexes
    - Creating composite indexes
    - Creating full-text search indexes
    - Creating uniqueness constraints
    """
    
    @staticmethod
    def create_optimized_indexes(session: Session) -> None:
        """
        Create optimized indexes and constraints for the knowledge graph.
        
        Creates:
        - Uniqueness constraints (also create indexes)
        - Property indexes for fast lookups
        - Composite indexes for multi-property queries
        - Full-text search indexes
        
        Args:
            session: Neo4j session
        """
        logger.info("Creating optimized Neo4j indexes and constraints")
        
        # Uniqueness constraints (also create indexes)
        constraints = [
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
            "CREATE CONSTRAINT doc_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
            "CREATE CONSTRAINT section_id_unique IF NOT EXISTS FOR (s:Section) REQUIRE s.section_id IS UNIQUE",
        ]
        
        for constraint in constraints:
            try:
                session.run(constraint)
                logger.debug(f"Created constraint: {constraint[:80]}...")
            except Exception as e:
                logger.debug(f"Constraint already exists or failed: {str(e)}")
        
        # Property indexes for efficient lookups
        indexes = [
            # Entity indexes
            "CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_canonical_idx IF NOT EXISTS FOR (e:Entity) ON (e.canonical_name)",
            "CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)",
            
            # Chunk indexes
            "CREATE INDEX chunk_doc_idx IF NOT EXISTS FOR (c:Chunk) ON (c.doc_id)",
            "CREATE INDEX chunk_section_idx IF NOT EXISTS FOR (c:Chunk) ON (c.section)",
            "CREATE INDEX chunk_type_idx IF NOT EXISTS FOR (c:Chunk) ON (c.chunk_type)",
            
            # Document indexes
            "CREATE INDEX doc_title_idx IF NOT EXISTS FOR (d:Document) ON (d.title)",
            "CREATE INDEX doc_file_type_idx IF NOT EXISTS FOR (d:Document) ON (d.file_type)",
            
            # Section indexes
            "CREATE INDEX section_heading_idx IF NOT EXISTS FOR (s:Section) ON (s.heading)",
            "CREATE INDEX section_doc_idx IF NOT EXISTS FOR (s:Section) ON (s.doc_id)",
        ]
        
        for index in indexes:
            try:
                session.run(index)
                logger.debug(f"Created index: {index[:80]}...")
            except Exception as e:
                logger.debug(f"Index already exists or failed: {str(e)}")
        
        # Composite indexes for common query patterns
        composite_indexes = [
            # Entity type + name for filtered lookups
            "CREATE INDEX entity_type_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.entity_type, e.name)",
            
            # Chunk doc + section for document navigation
            "CREATE INDEX chunk_doc_section_idx IF NOT EXISTS FOR (c:Chunk) ON (c.doc_id, c.section)",
        ]
        
        for index in composite_indexes:
            try:
                session.run(index)
                logger.debug(f"Created composite index: {index[:80]}...")
            except Exception as e:
                logger.debug(f"Composite index already exists or failed: {str(e)}")
        
        # Full-text search indexes
        fulltext_indexes = [
            # Entity name full-text search
            """
            CREATE FULLTEXT INDEX entity_name_fulltext IF NOT EXISTS
            FOR (e:Entity)
            ON EACH [e.name, e.canonical_name]
            """,
            
            # Chunk text full-text search
            """
            CREATE FULLTEXT INDEX chunk_text_fulltext IF NOT EXISTS
            FOR (c:Chunk)
            ON EACH [c.text, c.breadcrumbs]
            """,
            
            # Document title full-text search
            """
            CREATE FULLTEXT INDEX document_title_fulltext IF NOT EXISTS
            FOR (d:Document)
            ON EACH [d.title]
            """,
        ]
        
        for index in fulltext_indexes:
            try:
                session.run(index)
                logger.debug(f"Created full-text index: {index[:80]}...")
            except Exception as e:
                logger.debug(f"Full-text index already exists or failed: {str(e)}")
        
        logger.info("Neo4j indexes and constraints created successfully")
    
    @staticmethod
    def analyze_index_usage(session: Session) -> List[Dict[str, Any]]:
        """
        Analyze Neo4j index usage and performance.
        
        Returns information about:
        - Index provider
        - Index state
        - Index population progress
        
        Args:
            session: Neo4j session
        
        Returns:
            List of index information
        """
        query = "SHOW INDEXES"
        
        try:
            result = session.run(query)
            indexes = []
            
            for record in result:
                index_info = {
                    "name": record.get("name"),
                    "state": record.get("state"),
                    "type": record.get("type"),
                    "entity_type": record.get("entityType"),
                    "properties": record.get("properties"),
                    "provider": record.get("provider"),
                }
                indexes.append(index_info)
            
            logger.info(f"Retrieved information for {len(indexes)} Neo4j indexes")
            return indexes
            
        except Exception as e:
            logger.error(f"Failed to analyze Neo4j indexes: {str(e)}")
            return []


class QdrantIndexManager:
    """
    Manages Qdrant payload indexes for efficient filtering.
    
    Provides utilities for:
    - Creating payload field indexes
    - Optimizing filter queries
    """
    
    @staticmethod
    def create_payload_indexes(client, collection_name: str) -> None:
        """
        Create payload indexes for efficient filtering in Qdrant.
        
        Creates indexes on:
        - doc_id for document filtering
        - section for section filtering
        - chunk_type for type filtering
        
        Args:
            client: Qdrant client
            collection_name: Name of the collection
        """
        logger.info(f"Creating Qdrant payload indexes for collection '{collection_name}'")
        
        # Payload fields to index
        payload_fields = [
            "doc_id",
            "section",
            "chunk_type",
            "chunk_id",
        ]
        
        for field in payload_fields:
            try:
                # Create payload index
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field,
                    field_schema="keyword"  # Use keyword for exact matching
                )
                logger.debug(f"Created payload index on field: {field}")
            except Exception as e:
                # Index might already exist
                logger.debug(f"Payload index on {field} already exists or failed: {str(e)}")
        
        logger.info(f"Qdrant payload indexes created for collection '{collection_name}'")
    
    @staticmethod
    def get_collection_info(client, collection_name: str) -> Dict[str, Any]:
        """
        Get collection information including index status.
        
        Args:
            client: Qdrant client
            collection_name: Name of the collection
        
        Returns:
            Collection information dictionary
        """
        try:
            collection_info = client.get_collection(collection_name=collection_name)
            
            info = {
                "vectors_count": collection_info.vectors_count,
                "points_count": collection_info.points_count,
                "segments_count": collection_info.segments_count,
                "status": collection_info.status,
                "optimizer_status": collection_info.optimizer_status,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
            }
            
            logger.info(f"Retrieved info for collection '{collection_name}'")
            return info
            
        except Exception as e:
            logger.error(f"Failed to get collection info: {str(e)}")
            return {}
