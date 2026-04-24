"""Batch processing utilities for performance optimization."""

import logging
from typing import List, Callable, Any, TypeVar, Generic, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class BatchProcessor(Generic[T, R]):
    """
    Generic batch processor for efficient bulk operations.
    
    Provides utilities for:
    - Batching large lists into smaller chunks
    - Parallel batch processing with thread pools
    - Progress tracking and error handling
    - Adaptive batch sizing based on performance
    
    Attributes:
        batch_size: Default batch size for operations
        max_workers: Maximum number of parallel workers
    """
    
    def __init__(self, batch_size: int = 100, max_workers: int = 4):
        """
        Initialize batch processor.
        
        Args:
            batch_size: Default batch size (default: 100)
            max_workers: Maximum parallel workers (default: 4)
        """
        self.batch_size = batch_size
        self.max_workers = max_workers
        logger.info(f"Initialized BatchProcessor with batch_size={batch_size}, max_workers={max_workers}")
    
    def process_batches(
        self,
        items: List[T],
        process_fn: Callable[[List[T]], R],
        batch_size: Optional[int] = None,
        parallel: bool = False,
        description: str = "items"
    ) -> List[R]:
        """
        Process items in batches using the provided function.
        
        Args:
            items: List of items to process
            process_fn: Function that processes a batch of items
            batch_size: Batch size (uses default if None)
            parallel: Whether to process batches in parallel
            description: Description for logging
        
        Returns:
            List of results from each batch
        
        Raises:
            Exception: If batch processing fails
        """
        if not items:
            logger.debug(f"No {description} to process")
            return []
        
        batch_size = batch_size or self.batch_size
        total = len(items)
        num_batches = (total + batch_size - 1) // batch_size
        
        logger.info(
            f"Processing {total} {description} in {num_batches} batches "
            f"(batch_size={batch_size}, parallel={parallel})"
        )
        
        # Create batches
        batches = [
            items[i:i + batch_size]
            for i in range(0, total, batch_size)
        ]
        
        if parallel:
            return self._process_parallel(batches, process_fn, description)
        else:
            return self._process_sequential(batches, process_fn, description)
    
    def _process_sequential(
        self,
        batches: List[List[T]],
        process_fn: Callable[[List[T]], R],
        description: str
    ) -> List[R]:
        """
        Process batches sequentially.
        
        Args:
            batches: List of batches to process
            process_fn: Function to process each batch
            description: Description for logging
        
        Returns:
            List of results
        """
        results = []
        start_time = time.time()
        
        for i, batch in enumerate(batches):
            try:
                batch_start = time.time()
                result = process_fn(batch)
                batch_time = time.time() - batch_start
                
                results.append(result)
                
                logger.debug(
                    f"Processed batch {i + 1}/{len(batches)} for {description} "
                    f"({len(batch)} items in {batch_time:.2f}s)"
                )
                
            except Exception as e:
                logger.error(
                    f"Failed to process batch {i + 1}/{len(batches)} for {description}: {str(e)}"
                )
                raise
        
        total_time = time.time() - start_time
        logger.info(
            f"Completed processing {len(batches)} batches for {description} "
            f"in {total_time:.2f}s (avg {total_time / len(batches):.2f}s per batch)"
        )
        
        return results
    
    def _process_parallel(
        self,
        batches: List[List[T]],
        process_fn: Callable[[List[T]], R],
        description: str
    ) -> List[R]:
        """
        Process batches in parallel using thread pool.
        
        Args:
            batches: List of batches to process
            process_fn: Function to process each batch
            description: Description for logging
        
        Returns:
            List of results in original order
        """
        results = [None] * len(batches)
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all batches
            future_to_index = {
                executor.submit(process_fn, batch): i
                for i, batch in enumerate(batches)
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                
                try:
                    result = future.result()
                    results[index] = result
                    completed += 1
                    
                    logger.debug(
                        f"Completed batch {completed}/{len(batches)} for {description}"
                    )
                    
                except Exception as e:
                    logger.error(
                        f"Failed to process batch {index + 1}/{len(batches)} for {description}: {str(e)}"
                    )
                    raise
        
        total_time = time.time() - start_time
        logger.info(
            f"Completed parallel processing of {len(batches)} batches for {description} "
            f"in {total_time:.2f}s (avg {total_time / len(batches):.2f}s per batch)"
        )
        
        return results
    
    def adaptive_batch_size(
        self,
        items: List[T],
        process_fn: Callable[[List[T]], R],
        min_batch_size: int = 10,
        max_batch_size: int = 1000,
        target_time: float = 1.0,
        description: str = "items"
    ) -> int:
        """
        Determine optimal batch size based on processing time.
        
        Tests different batch sizes and selects one that achieves
        target processing time per batch.
        
        Args:
            items: Sample items to test
            process_fn: Function to process batches
            min_batch_size: Minimum batch size to test
            max_batch_size: Maximum batch size to test
            target_time: Target processing time per batch in seconds
            description: Description for logging
        
        Returns:
            Optimal batch size
        """
        if len(items) < min_batch_size:
            logger.debug(f"Not enough {description} to determine optimal batch size")
            return min_batch_size
        
        logger.info(f"Determining optimal batch size for {description}")
        
        # Test with minimum batch size
        test_batch = items[:min_batch_size]
        start_time = time.time()
        
        try:
            process_fn(test_batch)
            elapsed = time.time() - start_time
            
            # Calculate optimal batch size based on target time
            if elapsed > 0:
                optimal_size = int(min_batch_size * (target_time / elapsed))
                optimal_size = max(min_batch_size, min(optimal_size, max_batch_size))
            else:
                optimal_size = max_batch_size
            
            logger.info(
                f"Optimal batch size for {description}: {optimal_size} "
                f"(test batch of {min_batch_size} took {elapsed:.3f}s)"
            )
            
            return optimal_size
            
        except Exception as e:
            logger.warning(
                f"Failed to determine optimal batch size for {description}: {str(e)}. "
                f"Using default: {self.batch_size}"
            )
            return self.batch_size


class EmbeddingBatchProcessor:
    """
    Specialized batch processor for embedding generation.
    
    Optimizes batch sizes based on:
    - GPU memory availability
    - Text length distribution
    - Model inference time
    """
    
    # Optimal batch sizes for different scenarios
    SMALL_TEXT_BATCH_SIZE = 64  # For texts < 100 tokens
    MEDIUM_TEXT_BATCH_SIZE = 32  # For texts 100-300 tokens
    LARGE_TEXT_BATCH_SIZE = 16  # For texts > 300 tokens
    
    @staticmethod
    def get_optimal_batch_size(texts: List[str], avg_token_count: Optional[int] = None) -> int:
        """
        Determine optimal batch size based on text characteristics.
        
        Args:
            texts: List of texts to embed
            avg_token_count: Average token count (computed if not provided)
        
        Returns:
            Optimal batch size
        """
        if not texts:
            return EmbeddingBatchProcessor.MEDIUM_TEXT_BATCH_SIZE
        
        # Estimate average token count if not provided
        if avg_token_count is None:
            # Rough estimate: 1 token ≈ 4 characters
            avg_length = sum(len(text) for text in texts) / len(texts)
            avg_token_count = int(avg_length / 4)
        
        # Select batch size based on text length
        if avg_token_count < 100:
            batch_size = EmbeddingBatchProcessor.SMALL_TEXT_BATCH_SIZE
        elif avg_token_count < 300:
            batch_size = EmbeddingBatchProcessor.MEDIUM_TEXT_BATCH_SIZE
        else:
            batch_size = EmbeddingBatchProcessor.LARGE_TEXT_BATCH_SIZE
        
        logger.debug(
            f"Selected batch size {batch_size} for texts with "
            f"avg token count {avg_token_count}"
        )
        
        return batch_size


class Neo4jBatchProcessor:
    """
    Specialized batch processor for Neo4j operations.
    
    Optimizes batch sizes for:
    - Node creation
    - Relationship creation
    - Property updates
    """
    
    # Optimal batch sizes for different operations
    NODE_BATCH_SIZE = 100
    RELATIONSHIP_BATCH_SIZE = 100
    PROPERTY_UPDATE_BATCH_SIZE = 200
    
    @staticmethod
    def get_optimal_batch_size(operation_type: str) -> int:
        """
        Get optimal batch size for Neo4j operation type.
        
        Args:
            operation_type: Type of operation ('node', 'relationship', 'property')
        
        Returns:
            Optimal batch size
        """
        if operation_type == 'node':
            return Neo4jBatchProcessor.NODE_BATCH_SIZE
        elif operation_type == 'relationship':
            return Neo4jBatchProcessor.RELATIONSHIP_BATCH_SIZE
        elif operation_type == 'property':
            return Neo4jBatchProcessor.PROPERTY_UPDATE_BATCH_SIZE
        else:
            logger.warning(f"Unknown operation type: {operation_type}, using default")
            return 100


class QdrantBatchProcessor:
    """
    Specialized batch processor for Qdrant operations.
    
    Optimizes batch sizes for:
    - Vector upserts
    - Batch searches
    - Deletions
    """
    
    # Optimal batch sizes for different operations
    UPSERT_BATCH_SIZE = 100
    SEARCH_BATCH_SIZE = 50
    DELETE_BATCH_SIZE = 200
    
    @staticmethod
    def get_optimal_batch_size(operation_type: str) -> int:
        """
        Get optimal batch size for Qdrant operation type.
        
        Args:
            operation_type: Type of operation ('upsert', 'search', 'delete')
        
        Returns:
            Optimal batch size
        """
        if operation_type == 'upsert':
            return QdrantBatchProcessor.UPSERT_BATCH_SIZE
        elif operation_type == 'search':
            return QdrantBatchProcessor.SEARCH_BATCH_SIZE
        elif operation_type == 'delete':
            return QdrantBatchProcessor.DELETE_BATCH_SIZE
        else:
            logger.warning(f"Unknown operation type: {operation_type}, using default")
            return 100
