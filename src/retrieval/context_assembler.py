"""Context assembly for LLM generation with citations and token management."""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any
import tiktoken

from src.retrieval.vector_retriever import RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """
    Represents a source citation for context information.
    
    Attributes:
        citation_id: Unique citation identifier (e.g., "doc1:section2")
        doc_id: Parent document identifier
        section: Section name
        chunk_id: Source chunk identifier
        breadcrumbs: Hierarchical context path
    """
    citation_id: str
    doc_id: str
    section: str
    chunk_id: str
    breadcrumbs: str


@dataclass
class AssembledContext:
    """
    Represents assembled context for LLM generation.
    
    Attributes:
        context_text: Formatted context text with citations
        citations: Dictionary mapping citation_id to Citation objects
        token_count: Total token count of the context
    """
    context_text: str
    citations: Dict[str, Citation] = field(default_factory=dict)
    token_count: int = 0


class ContextAssembler:
    """
    Assembles structured context for LLM generation with citations and token management.
    
    This class creates formatted context by:
    1. Formatting graph facts as structured statements
    2. Including text chunks with breadcrumb context
    3. Generating citations in [doc_id:section] format
    4. Managing token limits with intelligent truncation
    5. Preserving citations even after truncation
    
    The assembled context follows this structure:
    - Query statement
    - Knowledge Graph Facts (with citations)
    - Relevant Document Excerpts (with citations and breadcrumbs)
    
    Attributes:
        max_tokens: Maximum token limit for assembled context (default: 4096)
        tokenizer: tiktoken encoder for token counting
    """
    
    def __init__(self, max_tokens: int = 4096):
        """
        Initialize ContextAssembler with token limit.
        
        Args:
            max_tokens: Maximum token limit for assembled context (default: 4096)
        
        Raises:
            Exception: If tiktoken encoder loading fails
        """
        self.max_tokens = max_tokens
        
        # Initialize tiktoken encoder (using cl100k_base for GPT-4 compatibility)
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            logger.info(
                "ContextAssembler initialized",
                extra={
                    "max_tokens": max_tokens,
                    "tokenizer": "cl100k_base"
                }
            )
        except Exception as e:
            logger.warning(f"Failed to load tiktoken encoder: {e}. Falling back to approximate counting.")
            self.tokenizer = None
    
    def assemble(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        graph_facts: List[str]
    ) -> AssembledContext:
        """
        Create structured context with citations and token management.
        
        This method:
        1. Creates citations for all chunks and graph facts
        2. Formats graph facts with citations
        3. Formats text chunks with breadcrumbs and citations
        4. Manages token limits by truncating lower-ranked results
        5. Preserves citations for all included content
        
        Args:
            query: Original search query
            chunks: List of RetrievedChunk objects (should be pre-ranked)
            graph_facts: List of formatted graph fact strings
        
        Returns:
            AssembledContext with formatted text, citations, and token count
        
        Raises:
            ValueError: If query is empty
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        logger.info(
            f"Assembling context for query: '{query}' "
            f"({len(chunks)} chunks, {len(graph_facts)} facts)"
        )
        
        # Initialize context components
        citations = {}
        context_parts = []
        
        # Add query header
        query_header = f"Query: {query}\n\n"
        context_parts.append(query_header)
        
        # Step 1: Process graph facts with citations
        if graph_facts:
            facts_section = self._format_graph_facts(graph_facts, chunks, citations)
            context_parts.append(facts_section)
        
        # Step 2: Process text chunks with citations
        if chunks:
            chunks_section = self._format_text_chunks(chunks, citations)
            context_parts.append(chunks_section)
        
        # Step 3: Combine all parts and check token limit
        full_context = "".join(context_parts)
        token_count = self._count_tokens(full_context)
        
        # Step 4: Truncate if necessary while preserving citations
        if token_count > self.max_tokens:
            logger.info(
                f"Context exceeds token limit ({token_count} > {self.max_tokens}), truncating"
            )
            full_context, citations, token_count = self._truncate_context(
                query_header, graph_facts, chunks, citations
            )
        
        logger.info(
            f"Context assembly complete: {token_count} tokens, {len(citations)} citations"
        )
        
        return AssembledContext(
            context_text=full_context,
            citations=citations,
            token_count=token_count
        )
    
    def _format_graph_facts(
        self,
        graph_facts: List[str],
        chunks: List[RetrievedChunk],
        citations: Dict[str, Citation]
    ) -> str:
        """
        Format graph facts with citations.
        
        For each graph fact, attempts to find the source chunk and create a citation.
        If no specific source is found, uses the first available chunk as citation source.
        
        Args:
            graph_facts: List of formatted graph fact strings
            chunks: List of chunks for citation source lookup
            citations: Dictionary to populate with citations
        
        Returns:
            Formatted graph facts section with citations
        """
        if not graph_facts:
            return ""
        
        facts_lines = ["Knowledge Graph Facts:\n"]
        
        for i, fact in enumerate(graph_facts, 1):
            # Try to find source chunk for this fact (simplified approach)
            # In a more sophisticated implementation, this could track fact-to-chunk mappings
            source_chunk = chunks[0] if chunks else None
            
            if source_chunk:
                citation_id = f"{source_chunk.doc_id}:{source_chunk.section}"
                
                # Create citation if not already exists
                if citation_id not in citations:
                    citations[citation_id] = Citation(
                        citation_id=citation_id,
                        doc_id=source_chunk.doc_id,
                        section=source_chunk.section,
                        chunk_id=source_chunk.chunk_id,
                        breadcrumbs=source_chunk.breadcrumbs
                    )
                
                facts_lines.append(f"{i}. {fact} [{citation_id}]\n")
            else:
                facts_lines.append(f"{i}. {fact}\n")
        
        facts_lines.append("\n")
        return "".join(facts_lines)
    
    def _format_text_chunks(
        self,
        chunks: List[RetrievedChunk],
        citations: Dict[str, Citation]
    ) -> str:
        """
        Format text chunks with breadcrumbs and citations.
        
        Each chunk is formatted with:
        - Citation header [doc_id:section]
        - Breadcrumbs for context hierarchy
        - Full chunk text
        
        Args:
            chunks: List of RetrievedChunk objects
            citations: Dictionary to populate with citations
        
        Returns:
            Formatted text chunks section
        """
        if not chunks:
            return ""
        
        chunks_lines = ["Relevant Document Excerpts:\n\n"]
        
        for chunk in chunks:
            citation_id = f"{chunk.doc_id}:{chunk.section}"
            
            # Create citation
            citations[citation_id] = Citation(
                citation_id=citation_id,
                doc_id=chunk.doc_id,
                section=chunk.section,
                chunk_id=chunk.chunk_id,
                breadcrumbs=chunk.breadcrumbs
            )
            
            # Format chunk with citation and breadcrumbs
            chunks_lines.append(f"[{citation_id}] ({chunk.breadcrumbs})\n")
            chunks_lines.append(f"{chunk.text}\n\n")
        
        return "".join(chunks_lines)
    
    def _truncate_context(
        self,
        query_header: str,
        graph_facts: List[str],
        chunks: List[RetrievedChunk],
        original_citations: Dict[str, Citation]
    ) -> tuple[str, Dict[str, Citation], int]:
        """
        Truncate context to fit token limit while preserving citations.
        
        Truncation strategy:
        1. Always keep query header
        2. Keep all graph facts (they're usually short)
        3. Truncate text chunks from lowest-ranked first
        4. Ensure remaining content has valid citations
        
        Args:
            query_header: Query header text
            graph_facts: List of graph facts
            chunks: List of chunks (pre-sorted by relevance)
            original_citations: Original citations dictionary
        
        Returns:
            Tuple of (truncated_context, updated_citations, token_count)
        """
        # Start with query header (always included)
        context_parts = [query_header]
        citations = {}
        
        # Add graph facts section (usually small, so include all)
        if graph_facts:
            facts_section = self._format_graph_facts(graph_facts, chunks, citations)
            context_parts.append(facts_section)
        
        # Add text chunks one by one until token limit
        chunks_header = "Relevant Document Excerpts:\n\n"
        context_parts.append(chunks_header)
        
        for chunk in chunks:
            # Format this chunk
            citation_id = f"{chunk.doc_id}:{chunk.section}"
            chunk_text = f"[{citation_id}] ({chunk.breadcrumbs})\n{chunk.text}\n\n"
            
            # Check if adding this chunk would exceed limit
            test_context = "".join(context_parts) + chunk_text
            test_token_count = self._count_tokens(test_context)
            
            if test_token_count <= self.max_tokens:
                # Add chunk and citation
                context_parts.append(chunk_text)
                citations[citation_id] = Citation(
                    citation_id=citation_id,
                    doc_id=chunk.doc_id,
                    section=chunk.section,
                    chunk_id=chunk.chunk_id,
                    breadcrumbs=chunk.breadcrumbs
                )
            else:
                # Stop adding chunks
                logger.debug(f"Stopped at chunk {chunk.chunk_id} to stay within token limit")
                break
        
        final_context = "".join(context_parts)
        final_token_count = self._count_tokens(final_context)
        
        logger.info(
            f"Context truncated: {len(chunks)} -> {len([c for c in citations.values() if c.chunk_id])} chunks, "
            f"{final_token_count} tokens"
        )
        
        return final_context, citations, final_token_count
    
    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Token count
        """
        if not text:
            return 0
        
        if self.tokenizer is not None:
            try:
                return len(self.tokenizer.encode(text))
            except Exception as e:
                logger.warning(f"Token counting failed: {e}. Using approximate count.")
        
        # Fallback: approximate token count (roughly 1 token per 4 characters)
        return len(text) // 4