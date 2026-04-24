"""Example usage of FaithfulnessValidator for validating LLM response faithfulness."""

import os
import sys
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.query.faithfulness_validator import FaithfulnessValidator, ValidationResult
from src.query.llm_generator import GeneratedResponse
from src.retrieval.context_assembler import AssembledContext, Citation


def main():
    """Demonstrate FaithfulnessValidator usage."""
    
    print("=" * 80)
    print("FaithfulnessValidator Example")
    print("=" * 80)
    print()
    
    # Initialize validator
    print("1. Initializing FaithfulnessValidator...")
    validator = FaithfulnessValidator(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        model=os.getenv("LLM_MODEL", "llama2"),
        faithfulness_threshold=0.8
    )
    print(f"   ✓ Validator initialized with model: {validator.model}")
    print(f"   ✓ Faithfulness threshold: {validator.faithfulness_threshold}")
    print()
    
    # Example 1: High faithfulness response
    print("2. Example 1: Validating high-faithfulness response")
    print("-" * 80)
    
    # Create context
    context1 = AssembledContext(
        context_text="""Query: What is NEFT?

Knowledge Graph Facts:
1. NEFT INTEGRATES_WITH Core Banking System [doc1:section2]

Relevant Document Excerpts:

[doc1:section2] (Banking Systems > Payment Modes > NEFT)
NEFT (National Electronic Funds Transfer) is a nationwide payment system facilitating 
one-to-one funds transfer. It operates in hourly batches and is available 24x7. 
NEFT has a transaction limit of Rs. 2 lakhs per transaction.
""",
        citations={
            "doc1:section2": Citation(
                citation_id="doc1:section2",
                doc_id="doc1",
                section="section2",
                chunk_id="chunk1",
                breadcrumbs="Banking Systems > Payment Modes > NEFT"
            )
        },
        token_count=150
    )
    
    # Create response that is well-grounded
    response1 = GeneratedResponse(
        answer="""NEFT (National Electronic Funds Transfer) is a nationwide payment system 
that facilitates one-to-one funds transfer [doc1:section2]. It operates in hourly batches 
and is available 24x7 [doc1:section2]. NEFT has a transaction limit of Rs. 2 lakhs per 
transaction [doc1:section2].""",
        citations_used=["doc1:section2"],
        model="llama2",
        timestamp=datetime.now()
    )
    
    print(f"   Response: {response1.answer[:100]}...")
    print(f"   Citations used: {response1.citations_used}")
    print()
    
    # Note: This example requires Ollama to be running
    print("   Note: Validation requires Ollama to be running.")
    print("   Skipping actual validation in this example.")
    print("   In production, you would call:")
    print("   result1 = validator.validate(response1, context1)")
    print()
    
    # Example 2: Low faithfulness response
    print("3. Example 2: Response with unsupported claims")
    print("-" * 80)
    
    # Create response with unsupported claims
    response2 = GeneratedResponse(
        answer="""NEFT is a nationwide payment system [doc1:section2]. 
NEFT has no transaction fees [doc1:section2]. 
NEFT is the fastest payment method in India [doc1:section2].""",
        citations_used=["doc1:section2"],
        model="llama2",
        timestamp=datetime.now()
    )
    
    print(f"   Response: {response2.answer}")
    print()
    print("   Expected validation result:")
    print("   - Claim 1: 'NEFT is a nationwide payment system' → SUPPORTED")
    print("   - Claim 2: 'NEFT has no transaction fees' → UNSUPPORTED (not in context)")
    print("   - Claim 3: 'NEFT is the fastest payment method' → UNSUPPORTED (not in context)")
    print("   - Faithfulness score: ~0.33 (1/3 claims supported)")
    print("   - Warnings: Low faithfulness score, unsupported claims detected")
    print()
    
    # Example 3: Insufficient information response
    print("4. Example 3: Insufficient information response")
    print("-" * 80)
    
    response3 = GeneratedResponse(
        answer="Insufficient information in documents to answer this question.",
        citations_used=[],
        model="llama2",
        timestamp=datetime.now()
    )
    
    print(f"   Response: {response3.answer}")
    print()
    print("   Expected validation result:")
    print("   - No factual claims extracted")
    print("   - Faithfulness score: 1.0 (vacuously true)")
    print("   - Warnings: No claims extracted from response")
    print()
    
    # Example 4: Interpreting validation results
    print("5. Interpreting Validation Results")
    print("-" * 80)
    print()
    print("   ValidationResult fields:")
    print("   - faithfulness_score: Float between 0.0 and 1.0")
    print("     * 1.0 = All claims supported by context")
    print("     * 0.0 = No claims supported by context")
    print()
    print("   - total_claims: Number of claims extracted from response")
    print("   - supported_claims: Number of claims supported by context")
    print("   - unsupported_claims: List of specific unsupported claim texts")
    print("   - warnings: List of warning messages")
    print()
    print("   Recommended actions based on score:")
    print("   - Score >= 0.8: Response is trustworthy, proceed normally")
    print("   - Score 0.5-0.8: Review unsupported claims, consider regenerating")
    print("   - Score < 0.5: High risk of hallucination, regenerate response")
    print()
    
    # Example 5: Integration with query pipeline
    print("6. Integration with Query Pipeline")
    print("-" * 80)
    print()
    print("   Typical usage in query pipeline:")
    print()
    print("   # After LLM generation")
    print("   response = llm_generator.generate(query, context)")
    print()
    print("   # Validate faithfulness")
    print("   validation = validator.validate(response, context)")
    print()
    print("   # Check score and warnings")
    print("   if validation.faithfulness_score < 0.8:")
    print("       # Log warnings")
    print("       for warning in validation.warnings:")
    print("           logger.warning(warning)")
    print()
    print("       # Include warnings in API response")
    print("       return {")
    print("           'answer': response.answer,")
    print("           'citations': response.citations_used,")
    print("           'faithfulness_score': validation.faithfulness_score,")
    print("           'warnings': validation.warnings")
    print("       }")
    print()
    
    print("=" * 80)
    print("Example complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
