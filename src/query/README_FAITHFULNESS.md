# Faithfulness Validator

## Overview

The `FaithfulnessValidator` validates LLM-generated responses to ensure they are grounded in the provided source context. It detects hallucinations and unsupported claims by checking each factual statement against the context used for generation.

## Purpose

In RAG systems, it's critical that LLM responses are faithful to the retrieved context. The validator:

1. **Extracts claims** from generated responses
2. **Checks entailment** for each claim against source context
3. **Computes faithfulness score** (proportion of supported claims)
4. **Generates warnings** for low scores or unsupported claims
5. **Identifies specific unsupported claims** for review

## Architecture

```
Generated Response → Claim Extraction → Entailment Checking → Faithfulness Score
                                              ↓
                                        Source Context
```

### Validation Process

1. **Claim Extraction**: Parse response into individual factual claims using LLM
2. **Entailment Checking**: For each claim, verify if context supports it
3. **Score Computation**: `faithfulness_score = supported_claims / total_claims`
4. **Warning Generation**: Flag responses with score < threshold (default: 0.8)

## Usage

### Basic Usage

```python
from src.query.faithfulness_validator import FaithfulnessValidator
from src.query.llm_generator import GeneratedResponse
from src.retrieval.context_assembler import AssembledContext

# Initialize validator
validator = FaithfulnessValidator(
    base_url="http://localhost:11434",
    model="llama2",
    faithfulness_threshold=0.8
)

# Validate response
result = validator.validate(response, context)

# Check results
print(f"Faithfulness Score: {result.faithfulness_score}")
print(f"Supported Claims: {result.supported_claims}/{result.total_claims}")

if result.warnings:
    print("Warnings:")
    for warning in result.warnings:
        print(f"  - {warning}")

if result.unsupported_claims:
    print("Unsupported Claims:")
    for claim in result.unsupported_claims:
        print(f"  - {claim}")
```

### Integration with Query Pipeline

```python
# After LLM generation
response = llm_generator.generate(query, context)

# Validate faithfulness
validation = validator.validate(response, context)

# Include validation in API response
return {
    "answer": response.answer,
    "citations": response.citations_used,
    "faithfulness_score": validation.faithfulness_score,
    "warnings": validation.warnings if validation.faithfulness_score < 0.8 else []
}
```

## Configuration

### Parameters

- **base_url**: Ollama API base URL (e.g., `"http://localhost:11434"`)
- **model**: LLM model for entailment checking (e.g., `"llama2"`, `"mistral"`)
- **faithfulness_threshold**: Score threshold for warnings (default: `0.8`)

### Environment Variables

```bash
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=llama2
```

## Validation Result

### ValidationResult Fields

```python
@dataclass
class ValidationResult:
    faithfulness_score: float        # 0.0 to 1.0
    total_claims: int                # Number of claims extracted
    supported_claims: int            # Number of supported claims
    unsupported_claims: List[str]    # List of unsupported claim texts
    warnings: List[str]              # Warning messages
```

### Interpreting Scores

| Score Range | Interpretation | Recommended Action |
|-------------|----------------|-------------------|
| 0.8 - 1.0   | High faithfulness | Proceed normally |
| 0.5 - 0.8   | Moderate faithfulness | Review unsupported claims |
| 0.0 - 0.5   | Low faithfulness | Regenerate response |

## Implementation Details

### Claim Extraction

The validator uses two approaches:

1. **LLM-based extraction** (primary): Uses LLM to parse response into discrete claims
2. **Sentence splitting** (fallback): Simple sentence-based splitting if LLM fails

```python
# LLM prompt for claim extraction
CLAIM_EXTRACTION_PROMPT = """Extract individual factual claims from the following answer.

Answer: {answer}

Return the claims as a JSON array of strings:
["claim 1", "claim 2", "claim 3"]
"""
```

### Entailment Checking

For each claim, the validator checks if the context supports it:

```python
# LLM prompt for entailment checking
ENTAILMENT_PROMPT = """Does the following context support this claim?

Context: {context_excerpt}

Claim: {claim}

Respond with JSON:
{
  "supported": true/false,
  "confidence": 0.0-1.0,
  "explanation": "..."
}
"""
```

### Optimizations

- **Context truncation**: Long contexts are truncated to 2000 chars for efficiency
- **Retry logic**: API calls retry up to 2 times on failure
- **Conservative defaults**: Assumes claims are unsupported if validation fails

## Error Handling

### Graceful Degradation

- **LLM failure**: Falls back to sentence-based claim extraction
- **API timeout**: Returns conservative result (assumes unsupported)
- **Invalid JSON**: Falls back to sentence splitting or returns False

### Logging

The validator logs:
- Initialization parameters
- Validation progress (claims extracted, entailment checks)
- Warnings for low scores or failures
- API call failures and retries

## Testing

### Unit Tests

Run unit tests:
```bash
pytest tests/test_faithfulness_validator.py -v
```

### Test Coverage

- Initialization validation
- Claim extraction (LLM and fallback)
- Entailment checking
- Score computation
- Warning generation
- Error handling and retries

## Requirements Validation

This implementation satisfies:

- **Requirement 17.1**: Validates each claim against source context
- **Requirement 17.2**: Uses entailment checking to verify claims
- **Requirement 17.3**: Flags unsupported claims with warnings
- **Requirement 17.4**: Computes faithfulness score (0-1)
- **Requirement 17.5**: Generates warnings for scores < 0.8
- **Requirement 17.6**: Identifies specific unsupported claims

## Example

See `examples/faithfulness_validator_example.py` for complete usage examples.

## Dependencies

- `requests`: HTTP client for Ollama API
- `json`: JSON parsing for LLM responses
- `re`: Regular expressions for claim extraction
- `logging`: Structured logging

## Performance Considerations

### Latency

- Claim extraction: ~1-2 seconds per response
- Entailment checking: ~0.5-1 second per claim
- Total validation time: ~2-5 seconds for typical responses (3-5 claims)

### Optimization Tips

1. **Batch validation**: Validate multiple responses in parallel
2. **Cache results**: Cache validation results for identical responses
3. **Adjust threshold**: Lower threshold for faster validation
4. **Limit claims**: Extract fewer claims for faster validation

## Limitations

1. **LLM dependency**: Requires Ollama service to be running
2. **Latency**: Adds 2-5 seconds to query pipeline
3. **False negatives**: May miss subtle hallucinations
4. **Context truncation**: Long contexts are truncated, may miss relevant info

## Future Enhancements

1. **Caching**: Cache entailment results for common claim-context pairs
2. **Batch processing**: Check multiple claims in single LLM call
3. **Fine-tuned models**: Use specialized entailment models
4. **Confidence thresholds**: Adjust based on claim confidence scores
5. **Semantic similarity**: Use embeddings for faster approximate checking
