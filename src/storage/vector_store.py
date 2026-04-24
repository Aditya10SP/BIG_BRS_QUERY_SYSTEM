"""Vector storage using Qdrant for semantic similarity search"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import uuid
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from src.utils.batch_processor import QdrantBatchProcessor
from src.utils.indexing import QdrantIndexManager

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Wrapper for Qdrant vector database providing embedding storage and similarity search.
    
    This class manages a Qdrant collection for storing document chunk embeddings
    and performing cosine similarity search. The collection uses 384-dimensional
    vectors (from all-MiniLM-L6-v2 model) with cosine distance metric.
    
    Attributes:
        client: Qdrant client instance
        collection_name: Name of the Qdrant collection
        vector_size: Dimension of embedding vectors (384)
    """
    
    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_name: str = "banking_docs",
        vector_size: int = 384,
    ):
        """
        Initialize Qdrant client and create collection if it doesn't exist.
        
        Args:
            url: Qdrant server URL
            collection_name: Name of the collection to use
            vector_size: Dimension of embedding vectors (default: 384)
        
        Raises:
            Exception: If connection to Qdrant fails
        """
        self.collection_name = collection_name
        self.vector_size = vector_size
        
        logger.info(f"Initializing VectorStore with Qdrant at {url}")
        
        try:
            self.client = QdrantClient(url=url)
            logger.info("Connected to Qdrant successfully")
            
            # Create collection if it doesn't exist
            self._create_collection_if_not_exists()
            
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {str(e)}")
            raise
    
    def _create_collection_if_not_exists(self) -> None:
        """
        Create the collection with appropriate configuration if it doesn't exist.
        
        The collection is configured with:
        - Vector size: 384 dimensions
        - Distance metric: Cosine similarity
        - Payload schema for efficient filtering
        - Payload indexes for common filter fields
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name in collection_names:
                logger.info(f"Collection '{self.collection_name}' already exists")
                # Create payload indexes even if collection exists
                QdrantIndexManager.create_payload_indexes(self.client, self.collection_name)
                return
            
            # Create collection with cosine distance
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )
            
            logger.info(
                f"Created collection '{self.collection_name}' with "
                f"vector size {self.vector_size} and cosine distance"
            )
            
            # Create payload indexes for efficient filtering
            QdrantIndexManager.create_payload_indexes(self.client, self.collection_name)
            
        except Exception as e:
            logger.error(f"Failed to create collection: {str(e)}")
            raise
    
    @staticmethod
    def _chunk_id_to_uuid(chunk_id: str) -> str:
        """
        Convert a chunk_id string to a deterministic UUID string.
        
        This ensures the same chunk_id always maps to the same UUID,
        allowing for consistent retrieval and updates.
        
        Args:
            chunk_id: String chunk identifier
            
        Returns:
            UUID string representation
        """
        # Use UUID5 with a namespace to create deterministic UUIDs
        namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')  # DNS namespace
        return str(uuid.uuid5(namespace, chunk_id))
    
    def store_embeddings(
        self,
        chunk_ids: List[str],
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]],
    ) -> None:
        """
        Store embeddings with metadata using optimized batch upsert.
        
        This method performs batch upsert of embeddings to Qdrant. Each embedding
        is associated with a chunk_id and metadata including doc_id, text,
        breadcrumbs, section, and chunk_type.
        
        Uses optimized batch size from QdrantBatchProcessor for better performance.
        
        Args:
            chunk_ids: List of unique chunk identifiers
            embeddings: Numpy array of embeddings (shape: num_chunks x vector_size)
            metadata: List of metadata dictionaries, one per chunk
        
        Raises:
            ValueError: If input lengths don't match or embeddings have wrong shape
            Exception: If upsert operation fails
        """
        # Validate inputs
        if len(chunk_ids) != len(metadata):
            raise ValueError(
                f"Length mismatch: {len(chunk_ids)} chunk_ids vs {len(metadata)} metadata"
            )
        
        if embeddings.shape[0] != len(chunk_ids):
            raise ValueError(
                f"Length mismatch: {embeddings.shape[0]} embeddings vs {len(chunk_ids)} chunk_ids"
            )
        
        if embeddings.shape[1] != self.vector_size:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.vector_size}, got {embeddings.shape[1]}"
            )
        
        try:
            # Get optimal batch size for upsert operations
            batch_size = QdrantBatchProcessor.get_optimal_batch_size('upsert')
            total = len(chunk_ids)
            num_batches = (total + batch_size - 1) // batch_size
            
            logger.info(
                f"Storing {total} embeddings in {num_batches} batches "
                f"(batch_size={batch_size})"
            )
            
            # Process in batches
            for batch_idx in range(0, total, batch_size):
                batch_end = min(batch_idx + batch_size, total)
                batch_chunk_ids = chunk_ids[batch_idx:batch_end]
                batch_embeddings = embeddings[batch_idx:batch_end]
                batch_metadata = metadata[batch_idx:batch_end]
                
                # Create points for this batch
                points = []
                for i, chunk_id in enumerate(batch_chunk_ids):
                    # Ensure metadata has required fields
                    payload = {
                        "chunk_id": chunk_id,
                        "doc_id": batch_metadata[i].get("doc_id", ""),
                        "text": batch_metadata[i].get("text", ""),
                        "breadcrumbs": batch_metadata[i].get("breadcrumbs", ""),
                        "section": batch_metadata[i].get("section", ""),
                        "chunk_type": batch_metadata[i].get("chunk_type", ""),
                    }
                    
                    # Add any additional metadata fields
                    for key, value in batch_metadata[i].items():
                        if key not in payload:
                            payload[key] = value
                    
                    # Convert chunk_id to UUID for Qdrant
                    point_id = self._chunk_id_to_uuid(chunk_id)
                    
                    point = PointStruct(
                        id=point_id,
                        vector=batch_embeddings[i].tolist(),
                        payload=payload,
                    )
                    points.append(point)
                
                # Batch upsert
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                )
                
                logger.debug(
                    f"Stored batch {batch_idx // batch_size + 1}/{num_batches} "
                    f"({len(points)} embeddings)"
                )
            
            logger.info(f"Successfully stored {total} embeddings in collection '{self.collection_name}'")
            
        except Exception as e:
            logger.error(f"Failed to store embeddings: {str(e)}")
            raise
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar embeddings using cosine similarity.
        
        This method performs similarity search in the vector store and returns
        the top-k most similar chunks with their similarity scores and metadata.
        
        Args:
            query_embedding: Query embedding vector (shape: vector_size,)
            top_k: Maximum number of results to return (default: 10)
            score_threshold: Minimum similarity score threshold (optional)
            filter_conditions: Optional filters for doc_id, section, etc.
        
        Returns:
            List of tuples (chunk_id, similarity_score, metadata)
            Results are sorted by similarity score in descending order
        
        Raises:
            ValueError: If query_embedding has wrong shape
            Exception: If search operation fails
        """
        # Validate query embedding
        if query_embedding.shape[0] != self.vector_size:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {self.vector_size}, "
                f"got {query_embedding.shape[0]}"
            )
        
        try:
            # Build filter if provided
            query_filter = None
            if filter_conditions:
                conditions = []
                for key, value in filter_conditions.items():
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value),
                        )
                    )
                if conditions:
                    query_filter = Filter(must=conditions)
            
            # Perform search
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=query_filter,
            )
            
            # Format results
            results = []
            for hit in search_results:
                chunk_id = hit.payload.get("chunk_id", str(hit.id))
                score = hit.score
                metadata = hit.payload
                results.append((chunk_id, score, metadata))
            
            logger.info(
                f"Search returned {len(results)} results "
                f"(top_k={top_k}, threshold={score_threshold})"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search embeddings: {str(e)}")
            raise
    
    def get_by_chunk_id(self, chunk_id: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Retrieve a specific embedding by chunk_id.
        
        Args:
            chunk_id: Unique chunk identifier
        
        Returns:
            Tuple of (embedding, metadata) if found, None otherwise
        """
        try:
            # Convert chunk_id to UUID
            point_id = self._chunk_id_to_uuid(chunk_id)
            
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id],
                with_vectors=True,
            )
            
            if not points:
                logger.debug(f"Chunk {chunk_id} not found in vector store")
                return None
            
            point = points[0]
            embedding = np.array(point.vector)
            metadata = point.payload
            
            return (embedding, metadata)
            
        except Exception as e:
            logger.error(f"Failed to retrieve chunk {chunk_id}: {str(e)}")
            raise
    
    def delete_by_chunk_ids(self, chunk_ids: List[str]) -> None:
        """
        Delete embeddings by chunk IDs.
        
        Args:
            chunk_ids: List of chunk IDs to delete
        """
        try:
            # Convert chunk_ids to UUIDs
            point_ids = [self._chunk_id_to_uuid(cid) for cid in chunk_ids]
            
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=point_ids,
            )
            logger.info(f"Deleted {len(chunk_ids)} embeddings from vector store")
            
        except Exception as e:
            logger.error(f"Failed to delete embeddings: {str(e)}")
            raise
    
    def delete_by_doc_id(self, doc_id: str) -> None:
        """
        Delete all embeddings for a specific document.
        
        Args:
            doc_id: Document ID
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="doc_id",
                            match=MatchValue(value=doc_id),
                        )
                    ]
                ),
            )
            logger.info(f"Deleted all embeddings for document {doc_id}")
            
        except Exception as e:
            logger.error(f"Failed to delete embeddings for doc {doc_id}: {str(e)}")
            raise
    
    def count(self) -> int:
        """
        Get the total number of embeddings in the collection.
        
        Returns:
            Total count of embeddings
        """
        try:
            # Use scroll with limit=0 to get count without fetching points
            result = self.client.scroll(
                collection_name=self.collection_name,
                limit=1,
                with_payload=False,
                with_vectors=False,
            )
            # Get collection info using count method
            collection_info = self.client.count(
                collection_name=self.collection_name
            )
            return collection_info.count
            
        except Exception as e:
            logger.error(f"Failed to get collection count: {str(e)}")
            raise
    
    def close(self) -> None:
        """Close the Qdrant client connection."""
        try:
            self.client.close()
            logger.info("Closed Qdrant client connection")
        except Exception as e:
            logger.warning(f"Error closing Qdrant client: {str(e)}")
