"""Cross-encoder reranking for retrieved results using transformer models."""

import logging
from dataclasses import dataclass
from typing import List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.retrieval.vector_retriever import RetrievedChunk
from src.retrieval.result_fusion import FusedResults
from src.utils.cache import CrossEncoderScoreCache

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Reranks retrieved results by query relevance using transformer cross-encoder model.
    
    This class uses a pre-trained cross-encoder model to score (query, chunk_text) pairs
    for relevance. Cross-encoders jointly encode the query and document text, providing
    more accurate relevance scores than bi-encoders used in initial retrieval.
    
    The reranker:
    1. Takes query and list of retrieved chunks
    2. Creates (query, chunk_text) pairs
    3. Scores each pair using the cross-encoder model
    4. Reorders chunks by cross-encoder score
    5. Returns top-k chunks with updated scores
    
    Attributes:
        model_name: Name of the cross-encoder model
        tokenizer: Tokenizer for the model
        model: Cross-encoder model for scoring
        device: Device for model inference (cuda/cpu)
        max_length: Maximum sequence length for tokenization
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length: int = 512,
        enable_cache: bool = True,
        cache_size: int = 5000
    ):
        """
        Initialize with cross-encoder model.
        
        Args:
            model_name: HuggingFace model name for cross-encoder (default: ms-marco-MiniLM-L-6-v2)
            max_length: Maximum sequence length for tokenization (default: 512)
            enable_cache: Whether to enable score caching (default: True)
            cache_size: Maximum number of scores to cache (default: 5000)
        
        Raises:
            Exception: If model loading fails
        """
        self.model_name = model_name
        self.max_length = max_length
        self.enable_cache = enable_cache
        
        # Initialize cache if enabled
        if enable_cache:
            self.cache = CrossEncoderScoreCache(max_size=cache_size)
            logger.info(f"Cross-encoder score cache enabled with size={cache_size}")
        else:
            self.cache = None
            logger.info("Cross-encoder score cache disabled")
        
        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            logger.info(f"Loading cross-encoder model: {model_name}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Move model to device
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            logger.info(
                f"CrossEncoderReranker initialized successfully",
                extra={
                    "model_name": model_name,
                    "device": str(self.device),
                    "max_length": max_length
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to load cross-encoder model {model_name}: {str(e)}")
            raise
    
    def rerank(
        self,
        query: str,
        results: FusedResults,
        top_k: int = 5
    ) -> List[RetrievedChunk]:
        """
        Score and rerank results by relevance using cross-encoder.
        
        This method:
        1. Creates (query, chunk_text) pairs for all chunks
        2. Tokenizes pairs and runs through cross-encoder model
        3. Extracts relevance scores from model output
        4. Sorts chunks by cross-encoder score (descending)
        5. Returns top-k chunks with updated scores
        6. Preserves all original metadata
        
        Args:
            query: Original search query
            results: FusedResults containing chunks to rerank
            top_k: Number of top results to return (default: 5)
        
        Returns:
            List of top-k RetrievedChunk objects sorted by cross-encoder score
            All original metadata is preserved
        
        Raises:
            ValueError: If query is empty or no chunks provided
            Exception: If model inference fails
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        if not results.chunks:
            logger.warning("No chunks provided for reranking")
            return []
        
        logger.info(
            f"Reranking {len(results.chunks)} chunks for query: '{query}' (top_k={top_k})"
        )
        
        try:
            # Step 1: Create (query, chunk_text) pairs
            pairs = [(query, chunk.text) for chunk in results.chunks]
            
            # Step 2: Score pairs using cross-encoder
            scores = self._score_pairs(pairs)
            
            # Step 3: Create (chunk, score) tuples
            chunk_scores = list(zip(results.chunks, scores))
            
            # Step 4: Sort by cross-encoder score (descending)
            chunk_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Step 5: Take top-k and update scores
            top_chunks = []
            for i, (chunk, score) in enumerate(chunk_scores[:top_k]):
                # Create new chunk with updated score
                reranked_chunk = RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    breadcrumbs=chunk.breadcrumbs,
                    doc_id=chunk.doc_id,
                    section=chunk.section,
                    score=score,  # Updated with cross-encoder score
                    retrieval_source=chunk.retrieval_source,
                    vector_score=chunk.vector_score,
                    bm25_score=chunk.bm25_score
                )
                top_chunks.append(reranked_chunk)
            
            logger.info(
                f"Reranking complete: returned {len(top_chunks)} chunks",
                extra={
                    "query": query,
                    "input_chunks": len(results.chunks),
                    "output_chunks": len(top_chunks),
                    "top_score": top_chunks[0].score if top_chunks else 0.0
                }
            )
            
            return top_chunks
            
        except Exception as e:
            logger.error(
                f"Failed to rerank results: {str(e)}",
                extra={"query": query, "chunks_count": len(results.chunks)}
            )
            raise
    
    def _score_pairs(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Score (query, text) pairs using cross-encoder model.
        
        This method processes pairs in batches for efficiency and handles
        tokenization, model inference, and score extraction. Uses cache for
        previously scored pairs.
        
        Args:
            pairs: List of (query, text) tuples to score
        
        Returns:
            List of relevance scores (higher = more relevant)
            Scores are normalized to 0-1 range using sigmoid
        """
        if not pairs:
            return []
        
        logger.debug(f"Scoring {len(pairs)} query-text pairs")
        
        # Check cache for each pair
        scores = []
        pairs_to_compute = []
        pair_indices = []
        
        if self.cache:
            for i, (query, text) in enumerate(pairs):
                cached_score = self.cache.get(query, text)
                if cached_score is not None:
                    scores.append((i, cached_score))
                else:
                    pairs_to_compute.append((query, text))
                    pair_indices.append(i)
        else:
            pairs_to_compute = pairs
            pair_indices = list(range(len(pairs)))
        
        try:
            # Compute scores for uncached pairs
            if pairs_to_compute:
                # Tokenize all pairs
                inputs = self.tokenizer(
                    pairs_to_compute,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                
                # Move inputs to device
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                
                # Run inference
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    
                    # Apply sigmoid to normalize scores to 0-1 range
                    computed_scores = torch.sigmoid(logits.squeeze()).cpu().numpy()
                    
                    # Handle single result case (sigmoid returns scalar)
                    if computed_scores.ndim == 0:
                        computed_scores = [float(computed_scores)]
                    else:
                        computed_scores = computed_scores.tolist()
                
                # Store in cache
                if self.cache:
                    for (query, text), score in zip(pairs_to_compute, computed_scores):
                        self.cache.put(query, text, score)
                
                logger.debug(f"Computed {len(computed_scores)} new scores")
            else:
                computed_scores = []
            
            # Combine cached and newly computed scores in correct order
            if scores:
                result = [0.0] * len(pairs)
                
                # Place cached scores
                for idx, score in scores:
                    result[idx] = score
                
                # Place newly computed scores
                for i, idx in enumerate(pair_indices):
                    result[idx] = computed_scores[i]
                
                logger.debug(
                    f"Used {len(scores)} cached scores, computed {len(computed_scores)} new scores"
                )
                logger.debug(
                    f"Score stats: min={min(result):.3f}, max={max(result):.3f}, "
                    f"mean={sum(result)/len(result):.3f}"
                )
                
                return result
            else:
                logger.debug(
                    f"Scoring complete: min={min(computed_scores):.3f}, max={max(computed_scores):.3f}, "
                    f"mean={sum(computed_scores)/len(computed_scores):.3f}"
                )
                return computed_scores
            
        except Exception as e:
            logger.error(f"Failed to score pairs: {str(e)}")
            # Return zero scores as fallback
            return [0.0] * len(pairs)