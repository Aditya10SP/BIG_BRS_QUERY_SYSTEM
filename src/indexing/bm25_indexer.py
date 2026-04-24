"""BM25 keyword indexer for exact term matching and acronym search."""

import logging
import re
from typing import List, Tuple, Optional
from rank_bm25 import BM25Okapi

from src.chunking.hierarchical_chunker import Chunk

logger = logging.getLogger(__name__)


class BM25Indexer:
    """
    BM25-based keyword search index for exact term matching.
    
    This class provides keyword-based search functionality using the BM25
    ranking algorithm. It's particularly useful for finding exact matches
    of acronyms and technical terms that may not be captured well by
    semantic similarity search.
    
    The tokenization process preserves acronyms (e.g., "NEFT", "RTGS")
    while normalizing other text to lowercase and removing punctuation.
    
    Attributes:
        k1: Term frequency saturation parameter (default: 1.5)
        b: Length normalization parameter (default: 0.75)
        bm25: BM25Okapi index instance
        chunk_ids: List of chunk IDs corresponding to indexed documents
        corpus_tokens: Tokenized corpus for the BM25 index
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 indexer with parameters.
        
        Args:
            k1: Term frequency saturation parameter (default: 1.5)
                Higher values give more weight to term frequency
            b: Length normalization parameter (default: 0.75)
                Controls how much document length affects scoring (0-1)
        """
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.chunk_ids = []
        self.corpus_tokens = []
        
        logger.info(f"Initialized BM25Indexer with k1={k1}, b={b}")
    
    def index(self, chunks: List[Chunk]) -> None:
        """
        Build BM25 index from chunks.
        
        This method tokenizes all chunks and builds the BM25 index.
        Both chunk text and breadcrumbs are indexed to enable context matching.
        
        Args:
            chunks: List of Chunk objects to index
        
        Raises:
            ValueError: If chunks list is empty
        """
        if not chunks:
            raise ValueError("Cannot index empty chunks list")
        
        logger.info(f"Indexing {len(chunks)} chunks with BM25")
        
        # Reset index state
        self.chunk_ids = []
        self.corpus_tokens = []
        
        # Tokenize each chunk
        for chunk in chunks:
            # Combine text and breadcrumbs for richer context matching
            combined_text = f"{chunk.text} {chunk.breadcrumbs}"
            tokens = self._tokenize(combined_text)
            
            self.corpus_tokens.append(tokens)
            self.chunk_ids.append(chunk.chunk_id)
        
        # Build BM25 index
        self.bm25 = BM25Okapi(
            self.corpus_tokens,
            k1=self.k1,
            b=self.b
        )
        
        logger.info(
            f"BM25 index built successfully with {len(self.chunk_ids)} documents"
        )
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: Optional[float] = None
    ) -> List[Tuple[str, float]]:
        """
        Search index and return chunk IDs with BM25 scores.
        
        Args:
            query: Search query string
            top_k: Maximum number of results to return (default: 10)
            score_threshold: Minimum BM25 score threshold (optional)
        
        Returns:
            List of (chunk_id, bm25_score) tuples, sorted by score descending
        
        Raises:
            ValueError: If index has not been built (call index() first)
            ValueError: If query is empty
        """
        if self.bm25 is None:
            raise ValueError("Index not built. Call index() before searching.")
        
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        if not query_tokens:
            logger.warning("Query tokenization resulted in empty token list")
            return []
        
        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(query_tokens)
        
        # Create list of (chunk_id, score) tuples
        results = list(zip(self.chunk_ids, scores))
        
        # Apply score threshold if provided
        if score_threshold is not None:
            results = [(cid, score) for cid, score in results if score >= score_threshold]
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Limit to top_k
        results = results[:top_k]
        
        logger.debug(
            f"BM25 search for '{query}' returned {len(results)} results "
            f"(top_k={top_k}, threshold={score_threshold})"
        )
        
        return results
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text while preserving acronyms.
        
        Tokenization strategy:
        1. Preserve acronyms (all-caps words like "NEFT", "RTGS")
        2. Convert other text to lowercase
        3. Remove punctuation (except in acronyms)
        4. Split on whitespace
        
        Args:
            text: Text to tokenize
        
        Returns:
            List of tokens
        """
        if not text:
            return []
        
        # Split text into words
        words = text.split()
        
        tokens = []
        for word in words:
            # Check if word is an acronym (all uppercase letters, 2+ chars)
            # Allow numbers in acronyms (e.g., "ISO20022")
            if self._is_acronym(word):
                # Preserve acronym as-is, but remove surrounding punctuation
                clean_word = re.sub(r'^[^\w]+|[^\w]+$', '', word)
                if clean_word:
                    tokens.append(clean_word)
            else:
                # Lowercase and remove punctuation for non-acronyms
                clean_word = re.sub(r'[^\w\s]', '', word.lower())
                if clean_word:
                    tokens.append(clean_word)
        
        return tokens
    
    def _is_acronym(self, word: str) -> bool:
        """
        Check if a word is an acronym.
        
        An acronym is defined as:
        - At least 2 characters long
        - All uppercase letters (may include numbers)
        - May have surrounding punctuation
        
        Args:
            word: Word to check
        
        Returns:
            True if word is an acronym, False otherwise
        """
        # Remove surrounding punctuation
        clean_word = re.sub(r'^[^\w]+|[^\w]+$', '', word)
        
        if len(clean_word) < 2:
            return False
        
        # Check if all alphabetic characters are uppercase
        # Allow numbers (e.g., "ISO20022")
        alphabetic_chars = [c for c in clean_word if c.isalpha()]
        
        if not alphabetic_chars:
            return False
        
        return all(c.isupper() for c in alphabetic_chars)
    
    def get_index_size(self) -> int:
        """
        Get the number of documents in the index.
        
        Returns:
            Number of indexed documents
        """
        return len(self.chunk_ids)
    
    def clear(self) -> None:
        """Clear the index and reset state."""
        self.bm25 = None
        self.chunk_ids = []
        self.corpus_tokens = []
        logger.info("BM25 index cleared")
