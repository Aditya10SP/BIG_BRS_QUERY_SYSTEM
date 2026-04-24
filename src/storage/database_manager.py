"""PostgreSQL database manager with connection pooling"""

import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Any
from datetime import datetime
import psycopg2
import psycopg2.extras
from psycopg2 import pool, sql
from psycopg2.extras import RealDictCursor

from src.utils.indexing import PostgreSQLIndexManager

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages PostgreSQL connections and operations for documents and chunks.
    
    Provides connection pooling and CRUD operations for the document store.
    """
    
    def __init__(self, connection_string: str, min_connections: int = 1, max_connections: int = 10):
        """
        Initialize database manager with connection pooling.
        
        Args:
            connection_string: PostgreSQL connection string
            min_connections: Minimum number of connections in pool
            max_connections: Maximum number of connections in pool
        """
        self.connection_string = connection_string
        self.pool = None
        self.min_connections = min_connections
        self.max_connections = max_connections
        
    def initialize(self) -> None:
        """
        Initialize connection pool and create schema if needed.
        
        Raises:
            psycopg2.Error: If connection or schema creation fails
        """
        try:
            # Create connection pool
            self.pool = psycopg2.pool.ThreadedConnectionPool(
                self.min_connections,
                self.max_connections,
                self.connection_string
            )
            logger.info(
                "Database connection pool initialized",
                extra={
                    "min_connections": self.min_connections,
                    "max_connections": self.max_connections
                }
            )
            
            # Create schema
            self.create_schema()
            
        except psycopg2.Error as e:
            logger.error(
                "Failed to initialize database",
                extra={
                    "error": str(e),
                    "connection_string": self._mask_password(self.connection_string)
                }
            )
            raise
    
    def close(self) -> None:
        """Close all connections in the pool."""
        if self.pool:
            self.pool.closeall()
            logger.info("Database connection pool closed")
    
    @contextmanager
    def get_connection(self):
        """
        Context manager for getting a connection from the pool.
        
        Yields:
            psycopg2 connection object
        """
        conn = None
        try:
            conn = self.pool.getconn()
            yield conn
        finally:
            if conn:
                self.pool.putconn(conn)
    
    def create_schema(self) -> None:
        """
        Create documents and chunks tables if they don't exist.
        Also creates optimized indexes for query performance.
        
        Schema:
            - documents: Stores document metadata
            - chunks: Stores text chunks with hierarchical relationships
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                # Create documents table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        doc_id VARCHAR(255) PRIMARY KEY,
                        title TEXT NOT NULL,
                        file_path TEXT,
                        file_type VARCHAR(10),
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                """)
                
                # Create chunks table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS chunks (
                        chunk_id VARCHAR(255) PRIMARY KEY,
                        doc_id VARCHAR(255) REFERENCES documents(doc_id) ON DELETE CASCADE,
                        text TEXT NOT NULL,
                        chunk_type VARCHAR(20),
                        parent_chunk_id VARCHAR(255),
                        breadcrumbs TEXT,
                        section TEXT,
                        token_count INT,
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                """)
                
                # Create optimized indexes using IndexManager
                PostgreSQLIndexManager.create_optimized_indexes(cur)
                
                conn.commit()
                logger.info("Database schema and indexes created successfully")
    
    def create_document(
        self,
        doc_id: str,
        title: str,
        file_path: Optional[str] = None,
        file_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Create a new document record.
        
        Args:
            doc_id: Unique document identifier
            title: Document title
            file_path: Path to source file
            file_type: File type (docx, pdf)
            metadata: Additional metadata as JSON
        
        Raises:
            psycopg2.Error: If insert fails
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO documents (doc_id, title, file_path, file_type, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (doc_id) DO UPDATE
                    SET title = EXCLUDED.title,
                        file_path = EXCLUDED.file_path,
                        file_type = EXCLUDED.file_type,
                        metadata = EXCLUDED.metadata
                    """,
                    (doc_id, title, file_path, file_type, psycopg2.extras.Json(metadata or {}))
                )
                conn.commit()
                logger.debug(f"Document created: {doc_id}")
    
    def create_chunk(
        self,
        chunk_id: str,
        doc_id: str,
        text: str,
        chunk_type: str,
        parent_chunk_id: Optional[str] = None,
        breadcrumbs: Optional[str] = None,
        section: Optional[str] = None,
        token_count: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Create a new chunk record.
        
        Args:
            chunk_id: Unique chunk identifier
            doc_id: Parent document ID
            text: Chunk text content
            chunk_type: Type of chunk ('parent' or 'child')
            parent_chunk_id: ID of parent chunk (for child chunks)
            breadcrumbs: Hierarchical context path
            section: Section name
            token_count: Number of tokens in chunk
            metadata: Additional metadata as JSON
        
        Raises:
            psycopg2.Error: If insert fails
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chunks (
                        chunk_id, doc_id, text, chunk_type, parent_chunk_id,
                        breadcrumbs, section, token_count, metadata
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (chunk_id) DO UPDATE
                    SET text = EXCLUDED.text,
                        chunk_type = EXCLUDED.chunk_type,
                        parent_chunk_id = EXCLUDED.parent_chunk_id,
                        breadcrumbs = EXCLUDED.breadcrumbs,
                        section = EXCLUDED.section,
                        token_count = EXCLUDED.token_count,
                        metadata = EXCLUDED.metadata
                    """,
                    (
                        chunk_id, doc_id, text, chunk_type, parent_chunk_id,
                        breadcrumbs, section, token_count,
                        psycopg2.extras.Json(metadata or {})
                    )
                )
                conn.commit()
                logger.debug(f"Chunk created: {chunk_id}")
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a chunk by its ID.
        
        Args:
            chunk_id: Chunk identifier
        
        Returns:
            Dictionary with chunk data or None if not found
        """
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT chunk_id, doc_id, text, chunk_type, parent_chunk_id,
                           breadcrumbs, section, token_count, metadata, created_at
                    FROM chunks
                    WHERE chunk_id = %s
                    """,
                    (chunk_id,)
                )
                result = cur.fetchone()
                return dict(result) if result else None
    
    def get_chunks_by_doc_id(self, doc_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all chunks for a document.
        
        Args:
            doc_id: Document identifier
        
        Returns:
            List of chunk dictionaries
        """
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT chunk_id, doc_id, text, chunk_type, parent_chunk_id,
                           breadcrumbs, section, token_count, metadata, created_at
                    FROM chunks
                    WHERE doc_id = %s
                    ORDER BY created_at
                    """,
                    (doc_id,)
                )
                results = cur.fetchall()
                return [dict(row) for row in results]
    
    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """
        Retrieve all chunks from the database.
        
        Returns:
            List of chunk dictionaries
        """
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT chunk_id, doc_id, text, chunk_type, parent_chunk_id,
                           breadcrumbs, section, token_count, metadata, created_at
                    FROM chunks
                    ORDER BY created_at
                    """
                )
                results = cur.fetchall()
                return [dict(row) for row in results]
    
    def get_chunks_by_section(self, doc_id: str, section: str) -> List[Dict[str, Any]]:
        """
        Retrieve chunks filtered by document and section.
        
        Args:
            doc_id: Document identifier
            section: Section name
        
        Returns:
            List of chunk dictionaries
        """
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT chunk_id, doc_id, text, chunk_type, parent_chunk_id,
                           breadcrumbs, section, token_count, metadata, created_at
                    FROM chunks
                    WHERE doc_id = %s AND section = %s
                    ORDER BY created_at
                    """,
                    (doc_id, section)
                )
                results = cur.fetchall()
                return [dict(row) for row in results]
    
    def update_chunk(
        self,
        chunk_id: str,
        text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update chunk text or metadata.
        
        Args:
            chunk_id: Chunk identifier
            text: New text content (optional)
            metadata: New metadata (optional)
        
        Returns:
            True if chunk was updated, False if not found
        """
        if text is None and metadata is None:
            return False
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                updates = []
                params = []
                
                if text is not None:
                    updates.append("text = %s")
                    params.append(text)
                
                if metadata is not None:
                    updates.append("metadata = %s")
                    params.append(psycopg2.extras.Json(metadata))
                
                params.append(chunk_id)
                
                query = f"""
                    UPDATE chunks
                    SET {', '.join(updates)}
                    WHERE chunk_id = %s
                """
                
                cur.execute(query, params)
                conn.commit()
                
                updated = cur.rowcount > 0
                if updated:
                    logger.debug(f"Chunk updated: {chunk_id}")
                return updated
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document and all its chunks (cascade).
        
        Args:
            doc_id: Document identifier
        
        Returns:
            True if document was deleted, False if not found
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM documents WHERE doc_id = %s",
                    (doc_id,)
                )
                conn.commit()
                
                deleted = cur.rowcount > 0
                if deleted:
                    logger.info(f"Document deleted: {doc_id}")
                return deleted
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document by its ID.
        
        Args:
            doc_id: Document identifier
        
        Returns:
            Dictionary with document data or None if not found
        """
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT doc_id, title, file_path, file_type, metadata, created_at
                    FROM documents
                    WHERE doc_id = %s
                    """,
                    (doc_id,)
                )
                result = cur.fetchone()
                return dict(result) if result else None
    
    @staticmethod
    def _mask_password(connection_string: str) -> str:
        """Mask password in connection string for logging."""
        if "@" in connection_string and ":" in connection_string:
            parts = connection_string.split("@")
            if len(parts) == 2:
                user_pass = parts[0].split("://")
                if len(user_pass) == 2:
                    protocol = user_pass[0]
                    credentials = user_pass[1].split(":")
                    if len(credentials) == 2:
                        return f"{protocol}://{credentials[0]}:****@{parts[1]}"
        return connection_string
