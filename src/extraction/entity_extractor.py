"""Entity extractor for extracting domain entities from document chunks."""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests
import spacy
from spacy.tokens import Doc

from src.chunking.hierarchical_chunker import Chunk
from src.utils.errors import (
    LLMError,
    ErrorContext,
    retry_with_backoff,
    RetryConfig,
    log_error_with_context
)


logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Represents an extracted entity with type, name, and context."""
    entity_id: str
    entity_type: str  # System, PaymentMode, Workflow, Rule, Field
    name: str
    canonical_name: str  # Normalized form
    source_chunk_id: str
    context: str  # Surrounding text
    properties: Dict[str, Any] = field(default_factory=dict)


class EntityExtractor:
    """
    Extracts domain entities from chunks using NER and LLM.
    
    Uses a two-stage extraction approach:
    1. Stage 1 - spaCy NER: Extract standard entities (ORG, PRODUCT, MONEY, DATE)
    2. Stage 2 - LLM Extraction: Extract domain-specific entities using prompted LLM
    """
    
    # Entity types for banking domain
    ENTITY_TYPES = {
        "System",
        "PaymentMode",
        "Workflow",
        "Rule",
        "Field"
    }
    
    # spaCy entity types to extract
    SPACY_ENTITY_TYPES = {"ORG", "PRODUCT", "MONEY", "DATE", "GPE", "PERSON"}
    
    # LLM prompt template for entity extraction
    LLM_PROMPT_TEMPLATE = """You are extracting entities from banking documentation.

Entity types:
- System: Banking systems or applications (e.g., "NEFT", "Core Banking", "Payment Gateway")
- PaymentMode: Payment methods (e.g., "RTGS", "IMPS", "UPI", "NEFT")
- Workflow: Business processes (e.g., "Payment Authorization Flow", "KYC Process")
- Rule: Business rules or policies (e.g., "Transaction Limit Rule", "Validation Rule")
- Field: Data fields or attributes (e.g., "Account Number", "IFSC Code", "Transaction ID")

Text: {chunk_text}

Extract all entities as JSON array. For each entity, provide:
- type: One of System, PaymentMode, Workflow, Rule, Field
- name: The entity name as it appears in text
- context: A brief surrounding context (1-2 sentences)

Return ONLY a valid JSON array, no other text:
[{{"type": "System", "name": "NEFT", "context": "..."}}]"""
    
    def __init__(self, spacy_model: str = "en_core_web_sm", ollama_base_url: str = None, llm_model: str = None):
        """
        Initialize with spaCy model and LLM client.
        
        Args:
            spacy_model: Name of spaCy model to load
            ollama_base_url: Base URL for Ollama API
            llm_model: Name of LLM model to use
        """
        self.spacy_model_name = spacy_model
        self.ollama_base_url = ollama_base_url
        self.llm_model = llm_model
        
        # Initialize spaCy NER
        try:
            self.nlp = spacy.load(spacy_model)
            logger.info(f"Loaded spaCy model: {spacy_model}")
        except OSError:
            logger.warning(f"spaCy model '{spacy_model}' not found. Stage 1 extraction will be skipped.")
            self.nlp = None
    
    def extract(self, chunk: Chunk) -> List[Entity]:
        """
        Extract entities from chunk text using two-stage extraction.
        
        Args:
            chunk: Chunk object with text to extract entities from
            
        Returns:
            List of Entity objects with type, text, and context
        """
        entities = []
        
        # Stage 1: spaCy NER extraction
        spacy_entities = self._extract_with_spacy(chunk)
        entities.extend(spacy_entities)
        
        # Stage 2: LLM-based extraction for domain-specific entities
        llm_entities = self._extract_with_llm(chunk)
        entities.extend(llm_entities)
        
        # Deduplicate entities by name (case-insensitive)
        entities = self._deduplicate_entities(entities)
        
        logger.info(f"Extracted {len(entities)} entities from chunk {chunk.chunk_id}")
        
        return entities
    
    def _extract_with_spacy(self, chunk: Chunk) -> List[Entity]:
        """
        Extract entities using spaCy NER (Stage 1).
        
        Args:
            chunk: Chunk to extract entities from
            
        Returns:
            List of Entity objects from spaCy NER
        """
        if self.nlp is None:
            return []
        
        entities = []
        
        try:
            doc = self.nlp(chunk.text)
            
            for ent in doc.ents:
                # Only extract relevant entity types
                if ent.label_ not in self.SPACY_ENTITY_TYPES:
                    continue
                
                # Map spaCy entity type to domain entity type
                entity_type = self._map_spacy_to_domain_type(ent.label_)
                
                # Get context (sentence containing the entity)
                context = self._get_entity_context(doc, ent)
                
                # Normalize entity name
                canonical_name = self._normalize_entity_name(ent.text)
                
                # Create entity ID
                entity_id = self._generate_entity_id(chunk.chunk_id, canonical_name)
                
                entity = Entity(
                    entity_id=entity_id,
                    entity_type=entity_type,
                    name=ent.text,
                    canonical_name=canonical_name,
                    source_chunk_id=chunk.chunk_id,
                    context=context,
                    properties={
                        "spacy_label": ent.label_,
                        "start_char": ent.start_char,
                        "end_char": ent.end_char
                    }
                )
                entities.append(entity)
        
        except Exception as e:
            logger.error(f"spaCy extraction failed for chunk {chunk.chunk_id}: {e}")
        
        return entities
    
    def _extract_with_llm(self, chunk: Chunk) -> List[Entity]:
        """
        Extract domain-specific entities using LLM (Stage 2).
        
        Args:
            chunk: Chunk to extract entities from
            
        Returns:
            List of Entity objects from LLM extraction
        """
        if not self.ollama_base_url or not self.llm_model:
            logger.debug("LLM configuration not provided. Skipping LLM extraction.")
            return []
        
        entities = []
        
        try:
            # Create prompt
            prompt = self.LLM_PROMPT_TEMPLATE.format(chunk_text=chunk.text)
            
            # Call Ollama API
            response = self._call_ollama(prompt)
            
            if not response:
                return []
            
            # Parse JSON response
            extracted_entities = self._parse_llm_response(response)
            
            # Convert to Entity objects
            for ent_data in extracted_entities:
                entity_type = ent_data.get("type", "System")
                name = ent_data.get("name", "")
                context = ent_data.get("context", "")
                
                # Validate entity type
                if entity_type not in self.ENTITY_TYPES:
                    logger.warning(f"Invalid entity type '{entity_type}', defaulting to 'System'")
                    entity_type = "System"
                
                if not name:
                    continue
                
                # Normalize entity name
                canonical_name = self._normalize_entity_name(name)
                
                # Create entity ID
                entity_id = self._generate_entity_id(chunk.chunk_id, canonical_name)
                
                entity = Entity(
                    entity_id=entity_id,
                    entity_type=entity_type,
                    name=name,
                    canonical_name=canonical_name,
                    source_chunk_id=chunk.chunk_id,
                    context=context or chunk.text[:200],  # Use chunk text if no context
                    properties={
                        "extraction_method": "llm"
                    }
                )
                entities.append(entity)
        
        except Exception as e:
            logger.error(f"LLM extraction failed for chunk {chunk.chunk_id}: {e}")
        
        return entities
    
    def _call_ollama(self, prompt: str, max_retries: int = 2) -> Optional[str]:
        """
        Call Ollama API for LLM generation with retry logic.
        
        Args:
            prompt: Prompt to send to LLM
            max_retries: Maximum number of retry attempts
            
        Returns:
            LLM response text or None if failed
        """
        url = f"{self.ollama_base_url}/api/generate"
        
        payload = {
            "model": self.llm_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for more deterministic extraction
                "num_predict": 1000  # Limit response length
            }
        }
        
        # Configure retry with exponential backoff
        retry_config = RetryConfig(
            max_attempts=max_retries + 1,
            initial_delay=1.0,
            max_delay=10.0,
            exponential_base=2.0,
            retryable_exceptions=(
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.RequestException
            )
        )
        
        @retry_with_backoff(retry_config)
        def _make_request():
            response = requests.post(url, json=payload, timeout=120)  # 2 minute timeout for entity extraction
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        
        try:
            return _make_request()
        except Exception as e:
            log_error_with_context(
                e,
                component="EntityExtractor",
                operation="_call_ollama",
                url=url,
                model=self.llm_model,
                max_retries=max_retries
            )
            return None
    
    def _parse_llm_response(self, response: str) -> List[Dict[str, str]]:
        """
        Parse LLM response to extract entity data.
        
        Args:
            response: LLM response text
            
        Returns:
            List of entity dictionaries
        """
        try:
            # Try to find JSON array in response
            # Look for content between [ and ]
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                entities = json.loads(json_str)
                
                if isinstance(entities, list):
                    return entities
            
            # If no JSON array found, try parsing entire response
            entities = json.loads(response)
            if isinstance(entities, list):
                return entities
            
            logger.warning("LLM response is not a JSON array")
            return []
        
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Response: {response[:500]}")
            return []
    
    def _map_spacy_to_domain_type(self, spacy_label: str) -> str:
        """
        Map spaCy entity label to domain entity type.
        
        Args:
            spacy_label: spaCy entity label (ORG, PRODUCT, etc.)
            
        Returns:
            Domain entity type
        """
        mapping = {
            "ORG": "System",
            "PRODUCT": "System",
            "MONEY": "Field",
            "DATE": "Field",
            "GPE": "System",
            "PERSON": "Field"
        }
        
        return mapping.get(spacy_label, "System")
    
    def _get_entity_context(self, doc: Doc, ent) -> str:
        """
        Get context sentence containing the entity.
        
        Args:
            doc: spaCy Doc object
            ent: spaCy entity
            
        Returns:
            Context sentence
        """
        # Find the sentence containing this entity
        for sent in doc.sents:
            if ent.start >= sent.start and ent.end <= sent.end:
                return sent.text.strip()
        
        # Fallback: return entity with surrounding words
        start_idx = max(0, ent.start - 10)
        end_idx = min(len(doc), ent.end + 10)
        context_tokens = doc[start_idx:end_idx]
        return " ".join([token.text for token in context_tokens])
    
    def _normalize_entity_name(self, name: str) -> str:
        """
        Normalize entity name to canonical form.
        
        Normalization rules:
        - Convert to uppercase for acronyms (all caps or mixed case short names)
        - Remove extra whitespace
        - Remove common suffixes like "System", "Module", "Service"
        - Standardize punctuation
        
        Args:
            name: Original entity name
            
        Returns:
            Normalized canonical name
        """
        # Remove extra whitespace
        name = " ".join(name.split())
        
        # Check if it's an acronym (short and mostly uppercase)
        if len(name) <= 6 and name.isupper():
            return name.upper()
        
        # Check if it's a mixed-case acronym like "NEFT System"
        words = name.split()
        if words and len(words[0]) <= 6 and words[0].isupper():
            # Keep the acronym, remove common suffixes
            canonical = words[0].upper()
            return canonical
        
        # Remove common suffixes
        suffixes_to_remove = [
            " System", " Module", " Service", " Application",
            " Process", " Workflow", " Rule", " Field"
        ]
        
        canonical = name
        for suffix in suffixes_to_remove:
            if canonical.endswith(suffix):
                canonical = canonical[:-len(suffix)]
                break
        
        # Title case for multi-word names
        if " " in canonical:
            canonical = canonical.title()
        
        return canonical.strip()
    
    def _generate_entity_id(self, chunk_id: str, canonical_name: str) -> str:
        """
        Generate unique entity ID.
        
        Args:
            chunk_id: Source chunk ID
            canonical_name: Canonical entity name
            
        Returns:
            Entity ID
        """
        # Create a simple hash-like ID from canonical name
        name_hash = abs(hash(canonical_name.lower())) % 10000
        return f"ent_{name_hash}_{chunk_id}"
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """
        Deduplicate entities by canonical name (case-insensitive).
        
        If multiple entities have the same canonical name, keep the one with
        the most complete information (longest context).
        
        Args:
            entities: List of entities to deduplicate
            
        Returns:
            Deduplicated list of entities
        """
        if not entities:
            return []
        
        # Group by canonical name (case-insensitive)
        entity_groups: Dict[str, List[Entity]] = {}
        
        for entity in entities:
            key = entity.canonical_name.lower()
            if key not in entity_groups:
                entity_groups[key] = []
            entity_groups[key].append(entity)
        
        # Keep the best entity from each group
        deduplicated = []
        for group in entity_groups.values():
            if len(group) == 1:
                deduplicated.append(group[0])
            else:
                # Choose entity with longest context
                best_entity = max(group, key=lambda e: len(e.context))
                deduplicated.append(best_entity)
        
        return deduplicated
