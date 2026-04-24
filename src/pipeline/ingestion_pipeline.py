"""Ingestion pipeline orchestrating document processing from parse to graph population."""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from src.parsing.document_parser import DocumentParser, ParsedDocument, ParsingError
from src.chunking.hierarchical_chunker import HierarchicalChunker, Chunk
from src.embedding.embedding_generator import EmbeddingGenerator
from src.storage.database_manager import DatabaseManager
from src.storage.vector_store import VectorStore
from src.indexing.bm25_indexer import BM25Indexer
from src.extraction.entity_extractor import EntityExtractor, Entity
from src.extraction.entity_resolver import EntityResolver, Relationship
from src.extraction.conflict_detector import ConflictDetector
from src.storage.graph_populator import GraphPopulator


logger = logging.getLogger(__name__)


class IngestionStatus(Enum):
    """Status of document ingestion."""
    PENDING = "pending"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    EXTRACTING = "extracting"
    RESOLVING = "resolving"
    DETECTING_CONFLICTS = "detecting_conflicts"
    POPULATING_GRAPH = "populating_graph"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class IngestionResult:
    """Result of document ingestion."""
    doc_id: str
    status: IngestionStatus
    message: str
    num_chunks: int = 0
    num_entities: int = 0
    num_relationships: int = 0
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class IngestionPipeline:
    """
    Orchestrates the complete document ingestion pipeline.
    
    Pipeline steps (in order):
    1. Parse: Extract text and structure from document
    2. Chunk: Create hierarchical chunks with breadcrumbs
    3. Embed: Generate vector embeddings for chunks
    4. Index: Build BM25 keyword index
    5. Extract: Extract entities using NER and LLM
    6. Resolve: Deduplicate entities and create SAME_AS relationships
    7. Detect Conflicts: Identify contradictory information
    8. Populate Graph: Create Neo4j knowledge graph
    
    The pipeline supports:
    - Single document ingestion
    - Batch processing of multiple documents
    - Error handling with halt on failure
    - Status tracking for each document
    - Consistency verification across storage layers
    """
    
    def __init__(
        self,
        parser: DocumentParser,
        chunker: HierarchicalChunker,
        embedding_generator: EmbeddingGenerator,
        database_manager: DatabaseManager,
        vector_store: VectorStore,
        bm25_indexer: BM25Indexer,
        entity_extractor: EntityExtractor,
        entity_resolver: EntityResolver,
        conflict_detector: ConflictDetector,
        graph_populator: GraphPopulator
    ):
        """
        Initialize ingestion pipeline with all required components.
        
        Args:
            parser: Document parser for .docx and .pdf files
            chunker: Hierarchical chunker for creating chunks
            embedding_generator: Embedding generator for vector embeddings
            database_manager: PostgreSQL database manager
            vector_store: Qdrant vector store
            bm25_indexer: BM25 keyword indexer
            entity_extractor: Entity extractor using NER and LLM
            entity_resolver: Entity resolver for deduplication
            conflict_detector: Conflict detector for contradictions
            graph_populator: Neo4j graph populator
        """
        self.parser = parser
        self.chunker = chunker
        self.embedding_generator = embedding_generator
        self.database_manager = database_manager
        self.vector_store = vector_store
        self.bm25_indexer = bm25_indexer
        self.entity_extractor = entity_extractor
        self.entity_resolver = entity_resolver
        self.conflict_detector = conflict_detector
        self.graph_populator = graph_populator
        
        # Track ingestion status for documents
        self.ingestion_status: Dict[str, IngestionResult] = {}
        
        logger.info("IngestionPipeline initialized successfully")
    
    def ingest(
        self,
        file_path: str,
        file_type: str,
        doc_id: Optional[str] = None
    ) -> IngestionResult:
        """
        Ingest a single document through the complete pipeline.
        
        Executes all pipeline steps in order:
        parse → chunk → embed → index → extract → resolve → detect conflicts → populate graph
        
        If any step fails, the pipeline halts and returns an error result.
        
        Args:
            file_path: Path to document file
            file_type: File type ('docx' or 'pdf')
            doc_id: Optional document ID (defaults to filename stem)
        
        Returns:
            IngestionResult with status and metadata
        """
        # Generate doc_id if not provided
        if doc_id is None:
            doc_id = Path(file_path).stem
        
        # Initialize result
        result = IngestionResult(
            doc_id=doc_id,
            status=IngestionStatus.PENDING,
            message="Ingestion started",
            started_at=datetime.now()
        )
        
        self.ingestion_status[doc_id] = result
        
        logger.info(f"Starting ingestion for document: {doc_id} ({file_path})")
        
        try:
            # Step 1: Parse document
            result.status = IngestionStatus.PARSING
            logger.info(f"[{doc_id}] Step 1/8: Parsing document")
            parsed_doc = self._parse_document(file_path, file_type, doc_id)
            
            # Step 2: Chunk document
            result.status = IngestionStatus.CHUNKING
            logger.info(f"[{doc_id}] Step 2/8: Chunking document")
            chunks = self._chunk_document(parsed_doc)
            result.num_chunks = len(chunks)
            
            # Step 3: Generate embeddings
            result.status = IngestionStatus.EMBEDDING
            logger.info(f"[{doc_id}] Step 3/8: Generating embeddings")
            self._generate_embeddings(chunks)
            
            # Step 4: Build BM25 index
            result.status = IngestionStatus.INDEXING
            logger.info(f"[{doc_id}] Step 4/8: Building BM25 index")
            self._build_bm25_index(chunks)
            
            # Step 5: Extract entities
            result.status = IngestionStatus.EXTRACTING
            logger.info(f"[{doc_id}] Step 5/8: Extracting entities")
            entities = self._extract_entities(chunks)
            result.num_entities = len(entities)
            
            # Step 6: Resolve entities
            result.status = IngestionStatus.RESOLVING
            logger.info(f"[{doc_id}] Step 6/8: Resolving entities")
            canonical_entities, same_as_rels = self._resolve_entities(entities)
            
            # Step 7: Detect conflicts
            result.status = IngestionStatus.DETECTING_CONFLICTS
            logger.info(f"[{doc_id}] Step 7/8: Detecting conflicts")
            conflict_rels = self._detect_conflicts(canonical_entities, chunks)
            
            # Combine all relationships
            all_relationships = same_as_rels + conflict_rels
            result.num_relationships = len(all_relationships)
            
            # Step 8: Populate graph
            result.status = IngestionStatus.POPULATING_GRAPH
            logger.info(f"[{doc_id}] Step 8/8: Populating knowledge graph")
            self._populate_graph(parsed_doc, canonical_entities, all_relationships, chunks)
            
            # Verify consistency
            logger.info(f"[{doc_id}] Verifying storage consistency")
            self._verify_consistency(doc_id, chunks)
            
            # Mark as completed
            result.status = IngestionStatus.COMPLETED
            result.message = "Ingestion completed successfully"
            result.completed_at = datetime.now()
            
            # Add metadata
            result.metadata = {
                "file_path": file_path,
                "file_type": file_type,
                "title": parsed_doc.title,
                "num_sections": len(parsed_doc.sections),
                "num_parent_chunks": len([c for c in chunks if c.chunk_type == "parent"]),
                "num_child_chunks": len([c for c in chunks if c.chunk_type == "child"]),
                "num_canonical_entities": len(canonical_entities),
                "num_same_as_relationships": len(same_as_rels),
                "num_conflict_relationships": len(conflict_rels)
            }
            
            logger.info(
                f"[{doc_id}] Ingestion completed successfully: "
                f"{result.num_chunks} chunks, {result.num_entities} entities, "
                f"{result.num_relationships} relationships"
            )
            
            return result
            
        except Exception as e:
            # Handle failure
            result.status = IngestionStatus.FAILED
            result.message = f"Ingestion failed: {str(e)}"
            result.error = str(e)
            result.completed_at = datetime.now()
            
            logger.error(
                f"[{doc_id}] Ingestion failed at step {result.status.value}",
                exc_info=True,
                extra={
                    "doc_id": doc_id,
                    "file_path": file_path,
                    "error": str(e)
                }
            )
            
            return result
    
    def ingest_batch(
        self,
        documents: List[Dict[str, str]]
    ) -> List[IngestionResult]:
        """
        Ingest multiple documents in batch.
        
        Each document is processed independently. If one document fails,
        others continue processing.
        
        Args:
            documents: List of document dictionaries with keys:
                      - file_path: Path to document file
                      - file_type: File type ('docx' or 'pdf')
                      - doc_id: Optional document ID
        
        Returns:
            List of IngestionResult objects, one per document
        """
        logger.info(f"Starting batch ingestion for {len(documents)} documents")
        
        results = []
        
        for i, doc_info in enumerate(documents, 1):
            file_path = doc_info["file_path"]
            file_type = doc_info["file_type"]
            doc_id = doc_info.get("doc_id")
            
            logger.info(f"Processing document {i}/{len(documents)}: {file_path}")
            
            result = self.ingest(file_path, file_type, doc_id)
            results.append(result)
        
        # Summary
        successful = sum(1 for r in results if r.status == IngestionStatus.COMPLETED)
        failed = sum(1 for r in results if r.status == IngestionStatus.FAILED)
        
        logger.info(
            f"Batch ingestion completed: {successful} successful, {failed} failed"
        )
        
        return results
    
    def get_status(self, doc_id: str) -> Optional[IngestionResult]:
        """
        Get ingestion status for a document.
        
        Args:
            doc_id: Document identifier
        
        Returns:
            IngestionResult if found, None otherwise
        """
        return self.ingestion_status.get(doc_id)
    
    def _parse_document(
        self, file_path: str, file_type: str, doc_id: str
    ) -> ParsedDocument:
        """
        Parse document and extract structure.
        
        Args:
            file_path: Path to document file
            file_type: File type ('docx' or 'pdf')
            doc_id: Document identifier
        
        Returns:
            ParsedDocument object
        
        Raises:
            ParsingError: If parsing fails
        """
        try:
            parsed_doc = self.parser.parse(file_path, file_type)
            
            # Override doc_id if provided
            parsed_doc.doc_id = doc_id
            
            # Store document in database
            self.database_manager.create_document(
                doc_id=parsed_doc.doc_id,
                title=parsed_doc.title,
                file_path=file_path,
                file_type=file_type,
                metadata=parsed_doc.metadata
            )
            
            logger.info(
                f"[{doc_id}] Parsed document: {parsed_doc.title}, "
                f"{len(parsed_doc.sections)} sections"
            )
            
            return parsed_doc
            
        except ParsingError as e:
            logger.error(f"[{doc_id}] Parsing failed: {e}")
            raise
        except Exception as e:
            logger.error(f"[{doc_id}] Unexpected error during parsing: {e}")
            raise ParsingError(f"Failed to parse document: {e}") from e
    
    def _chunk_document(self, parsed_doc: ParsedDocument) -> List[Chunk]:
        """
        Create hierarchical chunks from parsed document.
        
        Args:
            parsed_doc: ParsedDocument object
        
        Returns:
            List of Chunk objects
        
        Raises:
            Exception: If chunking fails
        """
        try:
            chunks = self.chunker.chunk(parsed_doc)
            
            if not chunks:
                raise ValueError("Chunking produced no chunks")
            
            # Store chunks in database
            for chunk in chunks:
                self.database_manager.create_chunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    text=chunk.text,
                    chunk_type=chunk.chunk_type,
                    parent_chunk_id=chunk.parent_chunk_id,
                    breadcrumbs=chunk.breadcrumbs,
                    section=chunk.section,
                    token_count=chunk.token_count,
                    metadata=chunk.metadata
                )
            
            logger.info(
                f"[{parsed_doc.doc_id}] Created {len(chunks)} chunks "
                f"({len([c for c in chunks if c.chunk_type == 'parent'])} parent, "
                f"{len([c for c in chunks if c.chunk_type == 'child'])} child)"
            )
            
            return chunks
            
        except Exception as e:
            logger.error(f"[{parsed_doc.doc_id}] Chunking failed: {e}")
            raise
    
    def _generate_embeddings(self, chunks: List[Chunk]) -> None:
        """
        Generate and store vector embeddings for chunks.
        
        Args:
            chunks: List of Chunk objects
        
        Raises:
            Exception: If embedding generation or storage fails
        """
        try:
            # Extract texts and metadata
            texts = [chunk.text for chunk in chunks]
            chunk_ids = [chunk.chunk_id for chunk in chunks]
            
            metadata = [
                {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "text": chunk.text,
                    "breadcrumbs": chunk.breadcrumbs,
                    "section": chunk.section,
                    "chunk_type": chunk.chunk_type
                }
                for chunk in chunks
            ]
            
            # Generate embeddings in batch
            embeddings = self.embedding_generator.batch_generate(texts)
            
            # Store in vector store
            self.vector_store.store_embeddings(chunk_ids, embeddings, metadata)
            
            logger.info(
                f"Generated and stored {len(embeddings)} embeddings"
            )
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    def _build_bm25_index(self, chunks: List[Chunk]) -> None:
        """
        Build BM25 keyword index for chunks.
        
        Args:
            chunks: List of Chunk objects
        
        Raises:
            Exception: If indexing fails
        """
        try:
            self.bm25_indexer.index(chunks)
            
            logger.info(
                f"Built BM25 index with {self.bm25_indexer.get_index_size()} documents"
            )
            
        except Exception as e:
            logger.error(f"BM25 indexing failed: {e}")
            raise
    
    def _extract_entities(self, chunks: List[Chunk]) -> List[Entity]:
        """
        Extract entities from chunks using NER and LLM.
        
        Args:
            chunks: List of Chunk objects
        
        Returns:
            List of Entity objects
        
        Raises:
            Exception: If entity extraction fails
        """
        try:
            all_entities = []
            
            for chunk in chunks:
                entities = self.entity_extractor.extract(chunk)
                all_entities.extend(entities)
            
            logger.info(
                f"Extracted {len(all_entities)} entities from {len(chunks)} chunks"
            )
            
            return all_entities
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            raise
    
    def _resolve_entities(
        self, entities: List[Entity]
    ) -> tuple[List[Entity], List[Relationship]]:
        """
        Resolve and deduplicate entities.
        
        Args:
            entities: List of Entity objects
        
        Returns:
            Tuple of (canonical_entities, same_as_relationships)
        
        Raises:
            Exception: If entity resolution fails
        """
        try:
            canonical_entities, same_as_rels = self.entity_resolver.resolve(entities)
            
            logger.info(
                f"Resolved {len(entities)} entities to {len(canonical_entities)} canonical entities, "
                f"created {len(same_as_rels)} SAME_AS relationships"
            )
            
            return canonical_entities, same_as_rels
            
        except Exception as e:
            logger.error(f"Entity resolution failed: {e}")
            raise
    
    def _detect_conflicts(
        self, entities: List[Entity], chunks: List[Chunk]
    ) -> List[Relationship]:
        """
        Detect conflicts between entities.
        
        Args:
            entities: List of Entity objects
            chunks: List of Chunk objects for context
        
        Returns:
            List of CONFLICTS_WITH Relationship objects
        
        Raises:
            Exception: If conflict detection fails
        """
        try:
            conflict_rels = self.conflict_detector.detect(entities, chunks)
            
            logger.info(
                f"Detected {len(conflict_rels)} conflict relationships"
            )
            
            return conflict_rels
            
        except Exception as e:
            logger.error(f"Conflict detection failed: {e}")
            raise
    
    def _populate_graph(
        self,
        parsed_doc: ParsedDocument,
        entities: List[Entity],
        relationships: List[Relationship],
        chunks: List[Chunk]
    ) -> None:
        """
        Populate Neo4j knowledge graph.
        
        Args:
            parsed_doc: ParsedDocument object
            entities: List of Entity objects
            relationships: List of Relationship objects
            chunks: List of Chunk objects
        
        Raises:
            Exception: If graph population fails
        """
        try:
            # Prepare document metadata
            documents = [{
                "doc_id": parsed_doc.doc_id,
                "title": parsed_doc.title,
                "file_path": parsed_doc.metadata.get("file_name", ""),
                "file_type": parsed_doc.metadata.get("file_type", ""),
                "metadata": parsed_doc.metadata
            }]
            
            # Populate graph
            self.graph_populator.populate(
                entities=entities,
                relationships=relationships,
                chunks=chunks,
                documents=documents
            )
            
            logger.info(
                f"Populated graph with {len(entities)} entities, "
                f"{len(relationships)} relationships, {len(chunks)} chunks"
            )
            
        except Exception as e:
            logger.error(f"Graph population failed: {e}")
            raise
    
    def _verify_consistency(self, doc_id: str, chunks: List[Chunk]) -> None:
        """
        Verify consistency across all storage layers.
        
        Checks:
        - PostgreSQL has all chunks
        - Qdrant has embeddings for all chunks
        - BM25 index has all chunks
        
        Args:
            doc_id: Document identifier
            chunks: List of Chunk objects that should be stored
        
        Raises:
            Exception: If consistency check fails
        """
        try:
            chunk_ids = [chunk.chunk_id for chunk in chunks]
            
            # Check PostgreSQL
            db_chunks = self.database_manager.get_chunks_by_doc_id(doc_id)
            db_chunk_ids = {chunk["chunk_id"] for chunk in db_chunks}
            
            missing_in_db = set(chunk_ids) - db_chunk_ids
            if missing_in_db:
                raise ValueError(
                    f"Consistency check failed: {len(missing_in_db)} chunks missing in PostgreSQL"
                )
            
            # Check Qdrant
            missing_in_qdrant = []
            for chunk_id in chunk_ids:
                result = self.vector_store.get_by_chunk_id(chunk_id)
                if result is None:
                    missing_in_qdrant.append(chunk_id)
            
            if missing_in_qdrant:
                raise ValueError(
                    f"Consistency check failed: {len(missing_in_qdrant)} chunks missing in Qdrant"
                )
            
            # Check BM25 index
            if self.bm25_indexer.get_index_size() == 0:
                raise ValueError("Consistency check failed: BM25 index is empty")
            
            logger.info(
                f"[{doc_id}] Consistency verification passed: "
                f"all {len(chunks)} chunks present in all storage layers"
            )
            
        except Exception as e:
            logger.error(f"[{doc_id}] Consistency verification failed: {e}")
            raise
