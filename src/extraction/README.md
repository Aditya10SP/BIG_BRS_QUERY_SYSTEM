# Entity Extraction Module

This module provides entity extraction capabilities for the Graph RAG Layer system, extracting domain-specific entities from banking document chunks.

## Overview

The `EntityExtractor` class implements a two-stage extraction approach:

1. **Stage 1 - spaCy NER**: Extracts standard entities (ORG, PRODUCT, MONEY, DATE) using spaCy's Named Entity Recognition
2. **Stage 2 - LLM Extraction**: Extracts domain-specific banking entities using a prompted LLM

## Entity Types

The system extracts the following banking domain entity types:

- **System**: Banking systems or applications (e.g., "NEFT", "Core Banking")
- **PaymentMode**: Payment methods (e.g., "RTGS", "IMPS", "UPI")
- **Workflow**: Business processes (e.g., "Payment Authorization Flow")
- **Rule**: Business rules or policies (e.g., "Transaction Limit Rule")
- **Field**: Data fields or attributes (e.g., "Account Number", "IFSC Code")

## Features

### Two-Stage Extraction

- **spaCy NER**: Fast, accurate extraction of standard entity types
- **LLM Extraction**: Flexible extraction of domain-specific entities with context understanding

### Entity Normalization

Entities are normalized to canonical forms for consistency:

- Acronyms are uppercased: "neft" → "NEFT"
- Common suffixes are removed: "NEFT System" → "NEFT"
- Multi-word names are title-cased: "core banking" → "Core Banking"

### Deduplication

Entities are automatically deduplicated within each chunk based on canonical names, keeping the entity with the most complete context.

### Context Preservation

Each extracted entity includes:
- The entity name as it appears in text
- Canonical normalized name
- Surrounding context (sentence or nearby text)
- Source chunk reference
- Additional properties (extraction method, spaCy label, etc.)

## Usage

### Basic Usage (spaCy only)

```python
from src.extraction.entity_extractor import EntityExtractor
from src.chunking.hierarchical_chunker import Chunk

# Initialize extractor
extractor = EntityExtractor(spacy_model="en_core_web_sm")

# Create a chunk
chunk = Chunk(
    chunk_id="doc1_s1_child_0",
    doc_id="doc1",
    text="The NEFT System processes payments through Core Banking.",
    chunk_type="child",
    parent_chunk_id="doc1_s1_parent",
    breadcrumbs="Document > Section",
    section="Section",
    token_count=10,
    metadata={}
)

# Extract entities
entities = extractor.extract(chunk)

# Process entities
for entity in entities:
    print(f"{entity.entity_type}: {entity.name} -> {entity.canonical_name}")
```

### With LLM Enhancement

```python
# Initialize with LLM configuration
extractor = EntityExtractor(
    spacy_model="en_core_web_sm",
    ollama_base_url="http://localhost:11434",
    llm_model="llama2"
)

# Extract entities (uses both spaCy and LLM)
entities = extractor.extract(chunk)
```

### Entity Normalization

```python
# Normalize entity names
canonical = extractor._normalize_entity_name("NEFT System")
# Returns: "NEFT"

canonical = extractor._normalize_entity_name("Core Banking Platform")
# Returns: "Core Banking Platform"
```

## Entity Data Structure

```python
@dataclass
class Entity:
    entity_id: str              # Unique identifier
    entity_type: str            # System, PaymentMode, Workflow, Rule, Field
    name: str                   # Original name from text
    canonical_name: str         # Normalized canonical form
    source_chunk_id: str        # Source chunk reference
    context: str                # Surrounding text context
    properties: Dict[str, Any]  # Additional metadata
```

## Configuration

### spaCy Model

The extractor uses spaCy for NER. Install the model:

```bash
python -m spacy download en_core_web_sm
```

If the model is not available, Stage 1 extraction is skipped gracefully.

### LLM Configuration

For Stage 2 extraction, configure Ollama:

- `ollama_base_url`: Base URL for Ollama API (e.g., "http://localhost:11434")
- `llm_model`: Model name (e.g., "llama2", "mistral")

If LLM is not configured, Stage 2 extraction is skipped.

## Implementation Details

### spaCy Entity Mapping

spaCy entity types are mapped to domain types:

- ORG, PRODUCT, GPE → System
- MONEY, DATE, PERSON → Field

### LLM Prompt Template

The LLM is prompted with:
- Entity type definitions and examples
- Chunk text
- JSON output format specification

The prompt uses low temperature (0.1) for deterministic extraction.

### Error Handling

- Missing spaCy model: Logs warning, skips Stage 1
- LLM API failure: Logs error, retries up to 2 times, then skips Stage 2
- Invalid JSON response: Logs warning, returns empty list
- Empty chunk: Returns empty list

## Testing

Run unit tests:
```bash
pytest tests/test_entity_extractor.py -v
```

Run integration tests:
```bash
pytest tests/test_entity_extractor_integration.py -v
```

Run example:
```bash
python examples/entity_extractor_example.py
```

## Requirements

- spacy >= 3.0.0
- requests >= 2.28.0
- en_core_web_sm (spaCy model)

## Design References

- **Requirements**: 6.1, 6.2, 6.3, 6.4, 6.5
- **Design Document**: Section 5 - Entity Extractor
- **Task**: 8.1 - Create EntityExtractor class

## Future Enhancements

- Support for custom entity types
- Configurable normalization rules
- Entity linking across documents
- Confidence scoring for extracted entities
- Support for additional NER models (Hugging Face, etc.)


## EntityResolver

The `EntityResolver` class deduplicates entities across chunks and creates canonical nodes with SAME_AS relationships for the knowledge graph.

### Features

- **Fuzzy String Matching**: Uses Levenshtein distance for similarity computation
- **Acronym Matching**: Automatically matches acronyms to full names (e.g., "NEFT" ↔ "National Electronic Funds Transfer")
- **DBSCAN Clustering**: Groups similar entities using density-based clustering
- **Configurable Threshold**: Adjustable similarity threshold for matching (default: 0.85)
- **Source Preservation**: Maintains all source chunk references in canonical entities
- **Type-Based Grouping**: Only matches entities within the same type

### Algorithm

1. Group entities by type (System, PaymentMode, etc.)
2. For each type group:
   - Compute pairwise similarity between all entities
   - Use DBSCAN clustering to group similar entities
   - Select canonical entity for each cluster (longest name, most context)
   - Create SAME_AS relationships from all cluster members to canonical

### Usage

```python
from src.extraction import EntityResolver, Entity

# Initialize resolver with similarity threshold
resolver = EntityResolver(similarity_threshold=0.85)

# Resolve entities
canonical_entities, same_as_relationships = resolver.resolve(entities)

print(f"Resolved {len(entities)} entities into {len(canonical_entities)} canonical entities")
print(f"Created {len(same_as_relationships)} SAME_AS relationships")

# Access canonical entity properties
for entity in canonical_entities:
    if "source_chunk_ids" in entity.properties:
        print(f"{entity.canonical_name}: {entity.properties['source_chunk_ids']}")
    if "aliases" in entity.properties:
        print(f"  Aliases: {entity.properties['aliases']}")
```

### Similarity Computation

The resolver computes entity similarity using multiple signals:

- **Canonical name similarity** (70% weight): Levenshtein-based string similarity
- **Original name similarity** (30% weight): Similarity of original entity names
- **Containment bonus** (+0.3): If one name contains the other (e.g., "NEFT" in "NEFT System")
- **Acronym bonus** (+0.5): If one name is an acronym of the other (e.g., "NEFT" ↔ "National Electronic Funds Transfer")

### Canonical Entity Selection

When multiple entities are merged, the canonical entity is selected based on:

1. Longest canonical name (more complete)
2. Longest context (more information)
3. Most properties (more metadata)

The canonical entity's properties are enriched with:
- `source_chunk_ids`: List of all source chunks
- `mention_count`: Number of merged entities
- `aliases`: List of alternative names

### Relationship Data Structure

```python
@dataclass
class Relationship:
    rel_id: str                 # Unique identifier
    rel_type: str               # SAME_AS, CONFLICTS_WITH, DEPENDS_ON, etc.
    source_entity_id: str       # Source entity ID
    target_entity_id: str       # Target entity ID (canonical)
    properties: Dict[str, Any]  # Relationship metadata
```

### Threshold Selection

- **0.85 (default)**: Conservative matching, requires close similarity
- **0.7-0.8**: Moderate matching, catches most duplicates
- **0.6**: Aggressive matching, includes acronym matches

### Testing

Run unit tests:
```bash
pytest tests/test_entity_resolver.py -v
```

Run example:
```bash
python examples/entity_resolver_example.py
```

### Requirements

- scikit-learn >= 1.0.0 (for DBSCAN clustering)
- numpy >= 1.20.0

### Design References

- **Requirements**: 7.1, 7.2, 7.3, 7.4, 7.5
- **Design Document**: Section 6 - Entity Resolver
- **Task**: 9.1 - Create EntityResolver class


## ConflictDetector

The `ConflictDetector` class identifies contradictory information across documents and creates bidirectional CONFLICTS_WITH relationships for the knowledge graph.

### Features

- **Property Conflict Detection**: Fast detection of entities with different property values
- **Semantic Conflict Detection**: LLM-based analysis for contradictory rules and policies
- **Bidirectional Relationships**: Creates symmetric CONFLICTS_WITH edges
- **Complete Metadata**: Includes conflict type, explanation, and source references
- **Three Conflict Types**: Property, Rule, and Workflow conflicts

### Conflict Types

1. **Property Conflicts**: Same entity with different property values
   - Example: "NEFT limit: 2 lakhs" vs "NEFT limit: 5 lakhs"

2. **Rule Conflicts**: Contradictory rules for the same scenario
   - Example: "Approve if amount < 1L" vs "Reject if amount < 1L"

3. **Workflow Conflicts**: Different process flows for the same operation
   - Example: Different approval workflows for the same transaction type

### Algorithm

1. Group entities by canonical name (case-insensitive)
2. For each entity group with multiple mentions:
   - Compare all pairs of entities
   - Check for property conflicts (fast, no LLM)
   - If no property conflict, check for semantic conflicts (LLM-based)
   - Create bidirectional CONFLICTS_WITH relationships with metadata

### Usage

```python
from src.extraction import ConflictDetector, Entity
from src.chunking.hierarchical_chunker import Chunk

# Initialize detector with LLM configuration
detector = ConflictDetector(
    ollama_base_url="http://localhost:11434",
    llm_model="llama2"
)

# Create entities with potential conflicts
entities = [
    Entity(
        entity_id="ent1",
        entity_type="PaymentMode",
        name="NEFT",
        canonical_name="NEFT",
        source_chunk_id="chunk1",
        context="NEFT limit is 2 lakhs",
        properties={"limit": "2 lakhs"}
    ),
    Entity(
        entity_id="ent2",
        entity_type="PaymentMode",
        name="NEFT",
        canonical_name="NEFT",
        source_chunk_id="chunk2",
        context="NEFT limit is 5 lakhs",
        properties={"limit": "5 lakhs"}
    )
]

# Detect conflicts
conflicts = detector.detect(entities, chunks)

# Process conflicts
for conflict in conflicts:
    print(f"Conflict: {conflict.source_entity_id} ↔ {conflict.target_entity_id}")
    print(f"Type: {conflict.properties['conflict_type']}")
    print(f"Explanation: {conflict.properties['explanation']}")
```

### Property Conflict Detection

Property conflicts are detected by comparing entity properties directly:

```python
# Check for property conflicts (fast, no LLM)
property_conflict = detector._check_property_conflict(entity1, entity2)

if property_conflict:
    print(f"Conflict type: {property_conflict['conflict_type']}")
    print(f"Conflicting properties: {property_conflict['conflicting_properties']}")
```

Metadata properties are automatically ignored:
- `extraction_method`, `spacy_label`, `start_char`, `end_char`
- `source_chunk_ids`, `mention_count`, `aliases`

### Semantic Conflict Detection

Semantic conflicts are detected using LLM analysis:

```python
# LLM analyzes entity contexts for contradictions
semantic_conflict = detector._check_semantic_conflict(
    entity1, entity2, chunk1, chunk2, "NEFT"
)

if semantic_conflict:
    print(f"Semantic conflict detected: {semantic_conflict['explanation']}")
```

The LLM prompt compares entity contexts and identifies:
- Different values for the same property
- Contradictory rules or policies
- Incompatible process flows or workflows

### Conflict Metadata

Each CONFLICTS_WITH relationship includes complete metadata:

```python
{
    "conflict_type": "property" | "rule" | "workflow",
    "explanation": "Brief explanation of the conflict",
    "source_chunk_ids": ["chunk1", "chunk2"],
    "doc_ids": ["doc1", "doc2"],
    "entity1_context": "Context from first entity",
    "entity2_context": "Context from second entity",
    "conflicting_properties": [  # For property conflicts
        {
            "property": "limit",
            "value1": "2 lakhs",
            "value2": "5 lakhs"
        }
    ]
}
```

### Bidirectional Relationships

For each conflict, two relationships are created:

```python
# Forward relationship: entity1 → entity2
Relationship(
    rel_id="conflict_ent1_ent2",
    rel_type="CONFLICTS_WITH",
    source_entity_id="ent1",
    target_entity_id="ent2",
    properties={...}
)

# Backward relationship: entity2 → entity1
Relationship(
    rel_id="conflict_ent2_ent1",
    rel_type="CONFLICTS_WITH",
    source_entity_id="ent2",
    target_entity_id="ent1",
    properties={...}
)
```

### Configuration

#### LLM Configuration

The detector requires LLM configuration for semantic conflict detection:

- `ollama_base_url`: Base URL for Ollama API (e.g., "http://localhost:11434")
- `llm_model`: Model name (e.g., "llama2", "mistral")

If LLM is not configured, only property conflicts are detected.

#### LLM Parameters

- **Temperature**: 0.1 (low for deterministic detection)
- **Max Tokens**: 500 (limit response length)
- **Retries**: 2 attempts on failure

### Error Handling

- **Missing LLM config**: Logs warning, skips semantic detection
- **Missing chunks**: Logs warning, skips that entity pair
- **LLM API failure**: Logs error, retries, then skips semantic detection
- **Invalid JSON response**: Logs warning, treats as no conflict

### Testing

Run unit tests:
```bash
pytest tests/test_conflict_detector.py -v
```

Run example:
```bash
python examples/conflict_detector_example.py
```

### Requirements

- requests >= 2.28.0 (for LLM API calls)

### Design References

- **Requirements**: 8.1, 8.2, 8.3, 8.4, 8.5
- **Design Document**: Section 7 - Conflict Detector
- **Task**: 10.1 - Create ConflictDetector class

### Future Enhancements

- Confidence scoring for conflicts
- Conflict severity ranking
- Temporal conflict detection (version conflicts)
- Custom conflict detection rules
- Conflict resolution suggestions
