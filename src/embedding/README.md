# Embedding Generation Module

This module provides vector embedding generation for semantic search using sentence-transformers.

## Overview

The `EmbeddingGenerator` class generates dense vector embeddings for text chunks using the `all-MiniLM-L6-v2` model from sentence-transformers. These embeddings enable semantic similarity search in the Graph RAG Layer system.

## Features

- **384-dimensional embeddings**: Compact yet effective representation
- **L2 normalization**: All embeddings are normalized for cosine similarity
- **Batch processing**: Efficient generation of multiple embeddings
- **Consistent results**: Same text always produces identical embeddings
- **Error handling**: Comprehensive validation and error messages

## Usage

### Basic Usage

```python
from src.embedding.embedding_generator import EmbeddingGenerator

# Initialize the generator
generator = EmbeddingGenerator()

# Generate a single embedding
text = "NEFT is a payment system"
embedding = generator.generate(text)
print(f"Shape: {embedding.shape}")  # (384,)
print(f"Norm: {np.linalg.norm(embedding)}")  # 1.0
```

### Batch Processing

```python
# Generate embeddings for multiple texts efficiently
texts = [
    "NEFT is used for retail payments",
    "RTGS handles high-value transactions",
    "IMPS enables instant transfers"
]

embeddings = generator.batch_generate(texts)
print(f"Shape: {embeddings.shape}")  # (3, 384)
```

### Semantic Similarity

```python
# Compute cosine similarity between texts
text1 = "NEFT payment system"
text2 = "Electronic fund transfer"

emb1 = generator.generate(text1)
emb2 = generator.generate(text2)

# Cosine similarity (dot product of normalized vectors)
similarity = np.dot(emb1, emb2)
print(f"Similarity: {similarity:.4f}")
```

### Custom Model

```python
# Use a different sentence-transformers model
generator = EmbeddingGenerator(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
```

## API Reference

### EmbeddingGenerator

#### `__init__(model_name: str = "sentence-transformers/all-MiniLM-L6-v2")`

Initialize the embedding generator.

**Parameters:**
- `model_name` (str): Name of the sentence-transformers model

**Raises:**
- `Exception`: If the model cannot be loaded

#### `generate(text: str) -> np.ndarray`

Generate a normalized embedding for a single text.

**Parameters:**
- `text` (str): Input text to embed

**Returns:**
- `np.ndarray`: L2-normalized embedding vector of shape (384,)

**Raises:**
- `ValueError`: If text is empty or None

#### `batch_generate(texts: List[str]) -> np.ndarray`

Generate normalized embeddings for multiple texts efficiently.

**Parameters:**
- `texts` (List[str]): List of input texts to embed

**Returns:**
- `np.ndarray`: L2-normalized embedding matrix of shape (num_texts, 384)

**Raises:**
- `ValueError`: If texts list is empty or contains empty strings

#### `get_embedding_dimension() -> int`

Get the dimension of the embedding vectors.

**Returns:**
- `int`: Embedding dimension (384 for all-MiniLM-L6-v2)

## Model Details

### all-MiniLM-L6-v2

- **Dimension**: 384
- **Max sequence length**: 256 tokens
- **Performance**: Good balance of speed and quality
- **Use case**: General-purpose semantic similarity

The model automatically truncates texts longer than 256 tokens. For longer documents, consider chunking the text first.

## L2 Normalization

All embeddings are L2-normalized, meaning their Euclidean norm is 1.0. This enables efficient cosine similarity computation using dot products:

```python
# For normalized vectors, cosine similarity = dot product
similarity = np.dot(embedding1, embedding2)
```

## Performance Considerations

### Batch Processing

For better performance, use `batch_generate()` instead of multiple `generate()` calls:

```python
# Good: Batch processing
embeddings = generator.batch_generate(texts)

# Less efficient: Individual calls
embeddings = [generator.generate(text) for text in texts]
```

### Caching

Consider caching embeddings for frequently accessed chunks to avoid recomputation:

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_cached_embedding(text: str) -> np.ndarray:
    return generator.generate(text)
```

## Integration with Storage

### Qdrant Vector Store

```python
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

# Generate embeddings
embeddings = generator.batch_generate(chunk_texts)

# Store in Qdrant
points = [
    PointStruct(
        id=chunk_id,
        vector=embedding.tolist(),
        payload={"text": text, "doc_id": doc_id}
    )
    for chunk_id, embedding, text in zip(chunk_ids, embeddings, chunk_texts)
]

client.upsert(collection_name="banking_docs", points=points)
```

### Similarity Search

```python
# Generate query embedding
query = "What is NEFT?"
query_embedding = generator.generate(query)

# Search Qdrant
results = client.search(
    collection_name="banking_docs",
    query_vector=query_embedding.tolist(),
    limit=10
)
```

## Testing

The module includes comprehensive unit and integration tests:

```bash
# Run unit tests
pytest tests/test_embedding_generator.py -v

# Run integration tests
pytest tests/test_embedding_generator_integration.py -v

# Run all tests
pytest tests/test_embedding_generator*.py -v
```

## Examples

See `examples/embedding_example.py` for a complete demonstration:

```bash
PYTHONPATH=. python examples/embedding_example.py
```

## Requirements

- Python 3.10+
- sentence-transformers==2.2.2
- numpy==1.26.2
- torch==2.1.1

## Design Document Reference

This implementation follows the design specifications in:
- **Design Document**: `.kiro/specs/graph-rag-layer/design.md` (Section 3: Embedding Generator)
- **Requirements**: Requirements 3.1 (Vector Embedding and Indexing)
- **Task**: Task 5.1 (Create EmbeddingGenerator class)

## Future Enhancements

- Support for multilingual models
- Embedding caching layer
- GPU acceleration support
- Custom fine-tuned models for banking domain
- Embedding dimension reduction options
