# Retrieval Module

This module provides retrieval components for the Graph RAG Layer system.

## Components

### GraphRetriever

The `GraphRetriever` class implements graph-based retrieval using Neo4j knowledge graph traversal. It supports multiple query patterns for relational and structural queries.

#### Features

- **Entity Extraction**: Automatically extracts entity mentions from queries
- **Pattern Detection**: Classifies queries into 5 patterns (dependency, integration, workflow, conflict, comparison)
- **Depth Limiting**: Configurable maximum traversal depth (default: 3 hops)
- **Multiple Query Patterns**: Specialized Cypher queries for each pattern type
- **Chunk Retrieval**: Fetches associated text chunks via MENTIONS relationships

#### Query Patterns

1. **Dependency Queries**: Forward/backward dependency traversal
   - Keywords: depend, depends, dependency, require, impact, affect
   - Example: "What systems depend on NEFT?"

2. **Integration Queries**: System connection paths
   - Keywords: integrate, integration, connect, interface, communicate
   - Example: "How does NEFT integrate with Core Banking?"

3. **Workflow Queries**: Process chain traversal
   - Keywords: workflow, process, flow, step, procedure, sequence
   - Example: "Show the payment authorization workflow"

4. **Conflict Queries**: Contradiction detection
   - Keywords: conflict, contradiction, inconsistent, disagree
   - Example: "What conflicts exist for NEFT?"

5. **Comparison Queries**: Entity comparison
   - Keywords: compare, comparison, difference, versus, between
   - Example: "Compare NEFT and RTGS"

#### Usage

```python
from src.retrieval import GraphRetriever

# Initialize retriever
retriever = GraphRetriever(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password",
    max_depth=3
)

# Retrieve graph
result = retriever.retrieve(
    query="What systems depend on NEFT?",
    max_depth=2  # Optional override
)

# Access results
print(f"Nodes: {len(result.nodes)}")
print(f"Relationships: {len(result.relationships)}")
print(f"Chunks: {len(result.chunks)}")

for node in result.nodes:
    print(f"Node: {node.node_type} - {node.properties.get('name')}")

for rel in result.relationships:
    print(f"Relationship: {rel.source_id} -{rel.rel_type}-> {rel.target_id}")

# Close connection
retriever.close()
```

#### Data Classes

**GraphNode**: Represents a graph node with:
- `node_id`: Unique node identifier
- `node_type`: Type of node (System, PaymentMode, Workflow, etc.)
- `properties`: Dictionary of node properties

**GraphRelationship**: Represents a graph relationship with:
- `rel_id`: Unique relationship identifier
- `rel_type`: Type of relationship (DEPENDS_ON, INTEGRATES_WITH, etc.)
- `source_id`: Source node identifier
- `target_id`: Target node identifier
- `properties`: Dictionary of relationship properties

**GraphResult**: Contains retrieval results with:
- `nodes`: List of GraphNode objects
- `relationships`: List of GraphRelationship objects
- `chunks`: List of associated chunk dictionaries

#### Requirements Validated

This component validates requirements:
- 12.1: Entity extraction from queries
- 12.2: Cypher query generation for traversal patterns
- 12.3: Depth limiting (max 3 hops)
- 12.4: Subgraph retrieval with nodes and relationships
- 12.5: Full node properties and relationship types
- 12.6: Support for dependency, integration, workflow, conflict patterns

### VectorRetriever

The `VectorRetriever` class implements hybrid retrieval combining:
- **Vector similarity search**: Semantic matching using embeddings
- **BM25 keyword search**: Exact term matching for acronyms and technical terms
- **Reciprocal Rank Fusion (RRF)**: Score combination for unified ranking

#### Features

- **Parallel Search**: Executes vector and BM25 searches concurrently for efficiency
- **Similarity Threshold**: Filters vector results by minimum similarity score (default: 0.7)
- **Top-K Limiting**: Returns at most top-k results (default: 10)
- **Full Metadata**: Retrieves complete chunk text and metadata from document store
- **Score Tracking**: Preserves both original scores (vector, BM25) and combined RRF score

#### Usage

```python
from src.retrieval import VectorRetriever
from src.storage import VectorStore, DatabaseManager
from src.indexing import BM25Indexer
from src.embedding import EmbeddingGenerator

# Initialize components
vector_store = VectorStore(url="http://localhost:6333")
bm25_index = BM25Indexer()
embedding_gen = EmbeddingGenerator()
doc_store = DatabaseManager(connection_string="postgresql://...")

# Create retriever
retriever = VectorRetriever(
    vector_store=vector_store,
    bm25_index=bm25_index,
    embedding_generator=embedding_gen,
    doc_store=doc_store,
    similarity_threshold=0.7
)

# Retrieve chunks
results = retriever.retrieve(
    query="What is NEFT?",
    top_k=10
)

# Access results
for chunk in results:
    print(f"Chunk: {chunk.chunk_id}")
    print(f"Text: {chunk.text}")
    print(f"Score: {chunk.score}")
    print(f"Source: {chunk.retrieval_source}")
    print(f"Breadcrumbs: {chunk.breadcrumbs}")
```

#### Reciprocal Rank Fusion (RRF)

RRF combines rankings from multiple sources using the formula:

```
score(chunk) = sum(1 / (k + rank_i))
```

where:
- `k` is a constant (default: 60)
- `rank_i` is the rank of the chunk in result list i (0-indexed)

This approach:
- Doesn't require score normalization
- Handles different score scales naturally
- Gives higher weight to top-ranked results
- Combines evidence from multiple retrieval methods

#### Data Classes

**RetrievedChunk**: Represents a retrieved chunk with:
- `chunk_id`: Unique identifier
- `text`: Full text content
- `breadcrumbs`: Hierarchical context path
- `doc_id`: Parent document ID
- `section`: Section name
- `score`: Combined RRF score
- `retrieval_source`: Source indicator ('vector', 'bm25', or 'vector+bm25')
- `vector_score`: Original vector similarity score
- `bm25_score`: Original BM25 score

## Design Decisions

### Why Parallel Search?

Running vector and BM25 searches in parallel reduces latency by ~50% compared to sequential execution. The searches are independent and can be executed concurrently.

### Why RRF over Score Normalization?

RRF is preferred because:
1. No need to normalize scores from different scales
2. Robust to outliers and score distribution differences
3. Simple and effective in practice
4. Well-studied in information retrieval literature

### Why Threshold Filtering After Search?

Applying the similarity threshold after vector search (rather than during) allows:
1. Better RRF fusion with more candidates
2. Flexibility to adjust threshold without re-searching
3. Consistent behavior across different query types

### ResultFusion

The `ResultFusion` class merges and deduplicates results from vector and graph retrieval for HYBRID mode queries.

#### Features

- **Chunk Deduplication**: Removes duplicate chunks appearing in both vector and graph results
- **Graph Fact Extraction**: Converts graph relationships to human-readable facts
- **Score Combination**: Combines vector and graph scores using weighted average (default: 0.6 vector + 0.4 graph)
- **Metadata Preservation**: Maintains all chunk metadata from both sources
- **Sorted Results**: Returns chunks sorted by combined score in descending order

#### Usage

```python
from src.retrieval import ResultFusion, VectorRetriever, GraphRetriever

# Initialize retrievers
vector_retriever = VectorRetriever(...)
graph_retriever = GraphRetriever(...)

# Initialize fusion with default weights
fusion = ResultFusion(vector_weight=0.6, graph_weight=0.4)

# Retrieve from both sources
vector_results = vector_retriever.retrieve(query="What is NEFT?")
graph_results = graph_retriever.retrieve(query="What is NEFT?")

# Fuse results
fused = fusion.fuse(vector_results, graph_results)

# Access fused results
print(f"Chunks: {len(fused.chunks)}")
print(f"Graph facts: {len(fused.graph_facts)}")

# Display graph facts
for fact in fused.graph_facts:
    print(f"  - {fact}")

# Display chunks sorted by combined score
for chunk in fused.chunks:
    combined_score = fused.combined_score[chunk.chunk_id]
    print(f"Chunk: {chunk.chunk_id} (score: {combined_score:.3f})")
    print(f"  Text: {chunk.text[:100]}...")
    print(f"  Source: {chunk.retrieval_source}")
```

#### Graph Fact Formatting

Graph relationships are converted to readable facts:

- **Basic relationships**: `"System A DEPENDS_ON System B"`
- **Workflow steps**: `"Workflow X NEXT_STEP Workflow Y"`
- **Conflicts**: `"Entity E1 CONFLICTS_WITH Entity E2 (reason: different limits)"`
- **Rules**: `"Rule R APPLIES_TO System S"`

#### Score Combination

Combined scores are computed using weighted average:

```
combined_score = (vector_weight × vector_score) + (graph_weight × graph_centrality)
```

Default weights:
- Vector weight: 0.6 (semantic relevance)
- Graph weight: 0.4 (structural importance)

For chunks appearing in both sources, both scores contribute to the final ranking.

#### Data Classes

**FusedResults**: Contains fusion results with:
- `chunks`: List of deduplicated RetrievedChunk objects
- `graph_facts`: List of formatted graph facts as strings
- `combined_score`: Dictionary mapping chunk_id to combined score

#### Deduplication Strategy

When a chunk appears in both vector and graph results:
1. The chunk appears only once in the fused results
2. The vector score is preserved from the vector retrieval
3. Graph centrality is added based on graph presence
4. Combined score reflects both sources

#### Custom Weights

You can customize the fusion weights based on your use case:

```python
# Emphasize vector similarity (semantic relevance)
fusion_semantic = ResultFusion(vector_weight=0.8, graph_weight=0.2)

# Emphasize graph structure (relationship importance)
fusion_structural = ResultFusion(vector_weight=0.4, graph_weight=0.6)

# Equal weighting
fusion_balanced = ResultFusion(vector_weight=0.5, graph_weight=0.5)
```

## Testing

See test files for unit tests and integration tests:
- `tests/test_vector_retriever.py`: Vector retrieval tests
- `tests/test_graph_retriever.py`: Graph retrieval tests
- `tests/test_result_fusion.py`: Result fusion tests

## Requirements Validated

### VectorRetriever
- 11.1: Query embedding generation
- 11.2: Vector store similarity search
- 11.3: Similarity threshold filtering (0.7)
- 11.4: Top-k limiting (10 results)
- 11.5: Full text and metadata retrieval

### GraphRetriever
- 12.1: Entity extraction from queries
- 12.2: Cypher query generation for traversal patterns
- 12.3: Depth limiting (max 3 hops)
- 12.4: Subgraph retrieval with nodes and relationships
- 12.5: Full node properties and relationship types
- 12.6: Support for dependency, integration, workflow, conflict patterns

### ResultFusion
- 13.1: Execute both vector and graph retrieval in HYBRID mode
- 13.2: Deduplicate chunks by chunk_id
- 13.3: Merge graph facts with text chunks
- 13.4: Preserve both vector similarity and graph centrality scores
- 13.5: Combine scores using weighted average (0.6 vector + 0.4 graph)
