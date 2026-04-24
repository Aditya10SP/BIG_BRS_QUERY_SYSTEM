"""Hierarchical chunker for creating parent and child chunks with breadcrumbs."""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import spacy
import tiktoken

from src.parsing.document_parser import ParsedDocument, Section


logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a chunk of document text with hierarchical context."""
    chunk_id: str
    doc_id: str
    text: str
    chunk_type: str  # 'parent' or 'child'
    parent_chunk_id: Optional[str]
    breadcrumbs: str  # "Doc Title > Section > Subsection"
    section: str
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class HierarchicalChunker:
    """
    Splits documents into parent and child chunks with contextual breadcrumbs.
    
    Parent chunks represent major sections (up to parent_size tokens).
    Child chunks are split from parents (max child_size tokens) while preserving
    sentence boundaries for semantic coherence.
    """
    
    def __init__(self, parent_size: int = 2048, child_size: int = 512, overlap: int = 50):
        """
        Initialize chunker with size limits.
        
        Args:
            parent_size: Max tokens for parent chunks (section-level)
            child_size: Max tokens for child chunks
            overlap: Number of tokens to overlap between adjacent child chunks
        """
        self.parent_size = parent_size
        self.child_size = child_size
        self.overlap = overlap
        
        # Initialize tiktoken encoder (using cl100k_base for GPT-4 compatibility)
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"Failed to load tiktoken encoder: {e}. Falling back to approximate counting.")
            self.tokenizer = None
        
        # Initialize spaCy for sentence segmentation
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "textcat"])
        except OSError:
            logger.warning("spaCy model 'en_core_web_sm' not found. Falling back to simple sentence splitting.")
            self.nlp = None
    
    def chunk(self, document: ParsedDocument) -> List[Chunk]:
        """
        Create hierarchical chunks with breadcrumbs.
        
        Args:
            document: ParsedDocument with sections
            
        Returns:
            List of Chunk objects (both parent and child)
        """
        chunks = []
        
        for section in document.sections:
            # Create breadcrumbs for this section
            breadcrumbs = self._create_breadcrumbs(document.title, section)
            
            # Create parent chunk for the section
            parent_chunk = self._create_parent_chunk(
                document.doc_id,
                section,
                breadcrumbs
            )
            
            # Only add parent chunk if it has content
            if parent_chunk.text.strip():
                chunks.append(parent_chunk)
                
                # Create child chunks from parent
                child_chunks = self._create_child_chunks(
                    document.doc_id,
                    parent_chunk,
                    section,
                    breadcrumbs
                )
                chunks.extend(child_chunks)
        
        return chunks
    
    def _create_breadcrumbs(self, doc_title: str, section: Section) -> str:
        """
        Create breadcrumb path for context hierarchy.
        
        Args:
            doc_title: Document title
            section: Section object
            
        Returns:
            Breadcrumb string like "Doc Title > Section > Subsection"
        """
        parts = [doc_title]
        
        # Add section heading
        if section.heading:
            parts.append(section.heading)
        
        return " > ".join(parts)
    
    def _create_parent_chunk(
        self,
        doc_id: str,
        section: Section,
        breadcrumbs: str
    ) -> Chunk:
        """
        Create a parent chunk from a section.
        
        Parent chunks represent major sections and may be up to parent_size tokens.
        If a section exceeds parent_size, it's truncated (child chunks will cover full content).
        
        Args:
            doc_id: Document ID
            section: Section to chunk
            breadcrumbs: Breadcrumb path
            
        Returns:
            Parent Chunk object
        """
        text = section.text.strip()
        token_count = self._count_tokens(text)
        
        # If section exceeds parent size, truncate at sentence boundary
        if token_count > self.parent_size:
            text = self._truncate_to_token_limit(text, self.parent_size)
            token_count = self._count_tokens(text)
        
        chunk_id = f"{doc_id}_{section.section_id}_parent"
        
        return Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            text=text,
            chunk_type="parent",
            parent_chunk_id=None,
            breadcrumbs=breadcrumbs,
            section=section.heading,
            token_count=token_count,
            metadata={
                "section_id": section.section_id,
                "section_level": section.level,
                "page_numbers": section.page_numbers
            }
        )
    
    def _create_child_chunks(
        self,
        doc_id: str,
        parent_chunk: Chunk,
        section: Section,
        breadcrumbs: str
    ) -> List[Chunk]:
        """
        Create child chunks from parent chunk text.
        
        Child chunks are split at sentence boundaries to maintain semantic coherence.
        Adjacent chunks overlap by self.overlap tokens for context continuity.
        
        Args:
            doc_id: Document ID
            parent_chunk: Parent chunk to split
            section: Original section
            breadcrumbs: Breadcrumb path
            
        Returns:
            List of child Chunk objects
        """
        text = section.text.strip()
        
        # Split text into sentences
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return []
        
        child_chunks = []
        child_counter = 0
        
        current_chunk_sentences = []
        current_token_count = 0
        
        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)
            
            # If single sentence exceeds child_size, split it forcefully
            if sentence_tokens > self.child_size:
                # Save current chunk if it has content
                if current_chunk_sentences:
                    child_chunks.append(
                        self._build_child_chunk(
                            doc_id,
                            parent_chunk,
                            section,
                            breadcrumbs,
                            child_counter,
                            current_chunk_sentences
                        )
                    )
                    child_counter += 1
                    current_chunk_sentences = []
                    current_token_count = 0
                
                # Split long sentence into smaller pieces
                split_chunks = self._split_long_sentence(sentence, self.child_size)
                for split_text in split_chunks:
                    child_chunks.append(
                        self._build_child_chunk(
                            doc_id,
                            parent_chunk,
                            section,
                            breadcrumbs,
                            child_counter,
                            [split_text]
                        )
                    )
                    child_counter += 1
                continue
            
            # Check if adding this sentence would exceed child_size
            if current_token_count + sentence_tokens > self.child_size and current_chunk_sentences:
                # Save current chunk
                child_chunks.append(
                    self._build_child_chunk(
                        doc_id,
                        parent_chunk,
                        section,
                        breadcrumbs,
                        child_counter,
                        current_chunk_sentences
                    )
                )
                child_counter += 1
                
                # Start new chunk with overlap
                # Keep last few sentences for overlap
                overlap_sentences = self._get_overlap_sentences(
                    current_chunk_sentences,
                    self.overlap
                )
                current_chunk_sentences = overlap_sentences
                current_token_count = sum(self._count_tokens(s) for s in overlap_sentences)
            
            # Add sentence to current chunk
            current_chunk_sentences.append(sentence)
            current_token_count += sentence_tokens
        
        # Add final chunk if it has content
        if current_chunk_sentences:
            child_chunks.append(
                self._build_child_chunk(
                    doc_id,
                    parent_chunk,
                    section,
                    breadcrumbs,
                    child_counter,
                    current_chunk_sentences
                )
            )
        
        return child_chunks
    
    def _build_child_chunk(
        self,
        doc_id: str,
        parent_chunk: Chunk,
        section: Section,
        breadcrumbs: str,
        child_index: int,
        sentences: List[str]
    ) -> Chunk:
        """Build a child chunk from sentences."""
        text = " ".join(sentences).strip()
        token_count = self._count_tokens(text)
        chunk_id = f"{doc_id}_{section.section_id}_child_{child_index}"
        
        return Chunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            text=text,
            chunk_type="child",
            parent_chunk_id=parent_chunk.chunk_id,
            breadcrumbs=breadcrumbs,
            section=section.heading,
            token_count=token_count,
            metadata={
                "section_id": section.section_id,
                "section_level": section.level,
                "page_numbers": section.page_numbers,
                "child_index": child_index
            }
        )
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using spaCy or fallback to simple splitting.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        if self.nlp is not None:
            # Use spaCy for accurate sentence segmentation
            doc = self.nlp(text)
            return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        else:
            # Fallback to simple regex-based splitting
            # Split on sentence-ending punctuation followed by whitespace
            sentences = re.split(r'(?<=[.!?])\s+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def _split_long_sentence(self, sentence: str, max_tokens: int) -> List[str]:
        """
        Split a sentence that exceeds max_tokens into smaller pieces.
        
        Tries to split at punctuation or whitespace boundaries.
        
        Args:
            sentence: Long sentence to split
            max_tokens: Maximum tokens per piece
            
        Returns:
            List of text pieces
        """
        # Try splitting at commas, semicolons, or other punctuation
        parts = re.split(r'([,;:])', sentence)
        
        chunks = []
        current_chunk = ""
        
        for part in parts:
            test_chunk = current_chunk + part
            if self._count_tokens(test_chunk) <= max_tokens:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = part
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # If still too long, split by words
        final_chunks = []
        for chunk in chunks:
            if self._count_tokens(chunk) <= max_tokens:
                final_chunks.append(chunk)
            else:
                # Split by words
                words = chunk.split()
                current_word_chunk = []
                for word in words:
                    test_chunk = " ".join(current_word_chunk + [word])
                    if self._count_tokens(test_chunk) <= max_tokens:
                        current_word_chunk.append(word)
                    else:
                        if current_word_chunk:
                            final_chunks.append(" ".join(current_word_chunk))
                        current_word_chunk = [word]
                
                if current_word_chunk:
                    final_chunks.append(" ".join(current_word_chunk))
        
        return [c for c in final_chunks if c]
    
    def _get_overlap_sentences(self, sentences: List[str], overlap_tokens: int) -> List[str]:
        """
        Get the last few sentences that fit within overlap_tokens.
        
        Args:
            sentences: List of sentences
            overlap_tokens: Target overlap in tokens
            
        Returns:
            List of sentences for overlap
        """
        if not sentences:
            return []
        
        overlap_sentences = []
        token_count = 0
        
        # Work backwards from the end
        for sentence in reversed(sentences):
            sentence_tokens = self._count_tokens(sentence)
            if token_count + sentence_tokens <= overlap_tokens:
                overlap_sentences.insert(0, sentence)
                token_count += sentence_tokens
            else:
                break
        
        return overlap_sentences
    
    def _truncate_to_token_limit(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to max_tokens while preserving sentence boundaries.
        
        Args:
            text: Text to truncate
            max_tokens: Maximum token count
            
        Returns:
            Truncated text
        """
        sentences = self._split_into_sentences(text)
        
        result_sentences = []
        token_count = 0
        
        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)
            if token_count + sentence_tokens <= max_tokens:
                result_sentences.append(sentence)
                token_count += sentence_tokens
            else:
                break
        
        return " ".join(result_sentences)
    
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
