# Query Pipeline Components

This module contains components for the query pipeline, including query routing and retrieval.

## Components

### QueryRouter

Routes queries to appropriate retrieval mode using LLM-based classification.

**Features:**
- Classifies queries into VECTOR, GRAPH, or HYBRID modes
- Uses LLM to analyze query intent
- Provides confidence scoring
- Defaults to HYBRID mode for low confidence (<0.7)
- Handles LLM failures gracefully

**Query Modes:**

1. **VECTOR Mode**: Factual/definitional questions about single concepts
   - Examples: "What is NEFT?", "Define transaction limit", "Explain RTGS process"

2. **GRAPH Mode**: Relational/structural/comparison questions
   - Examples: "What systems depend on NEFT?", "Compare RTGS and IMPS", "Show payment workflow"

3. **HYBRID Mode**: Complex questions requiring both relationships and full text
   - Examples: "How does NEFT integrate with Core Banking and what are the limits?"

**Usage:**

```python
from src.query.query_router import QueryRouter, QueryMode

# Initialize router
router = QueryRouter(
    ollama_base_url="http://localhost:11434",
    llm_model="llama2",
    confidence_threshold=0.7
)

# Route a query
mode, confidence = router.route("What systems depend on NEFT?")

if mode == QueryMode.VECTOR:
    # Use vector retrieval
    pass
elif mode == QueryMode.GRAPH:
    # Use graph retrieval
    pass
else:  # HYBRID
    # Use both retrievals
    pass
```

**Configuration:**

- `ollama_base_url`: Base URL for Ollama API
- `llm_model`: Name of LLM model to use
- `confidence_threshold`: Minimum confidence to accept classification (default: 0.7)

**Error Handling:**

The router handles various failure scenarios gracefully:
- Empty queries → HYBRID mode
- LLM API failures → HYBRID mode
- Invalid mode responses → HYBRID mode
- Low confidence → HYBRID mode
- JSON parsing errors → HYBRID mode

All failures default to HYBRID mode to ensure queries can still be processed.

## Testing

Run tests with:

```bash
python -m pytest tests/test_query_router.py -v
```

## Example

See `examples/query_router_example.py` for a complete example with interactive mode.

```bash
python examples/query_router_example.py
```
