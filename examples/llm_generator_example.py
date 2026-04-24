#!/usr/bin/env python3
"""
Example usage of LLMGenerator for grounded response generation.

This example demonstrates:
1. Creating an LLMGenerator with Ollama configuration
2. Assembling context with citations
3. Generating grounded responses with citation enforcement
4. Handling insufficient context scenarios
"""

import os
import sys
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.query.llm_generator import LLMGenerator, GeneratedResponse
from src.retrieval.context_assembler import AssembledContext, Citation


def create_sample_context() -> AssembledContext:
    """Create sample context for demonstration."""
    
    # Sample citations
    citations = {
        "doc1:section2": Citation(
            citation_id="doc1:section2",
            doc_id="doc1",
            section="section2",
            chunk_id="chunk_123",
            breadcrumbs="NEFT Specification > Transaction Processing"
        ),
        "doc2:section1": Citation(
            citation_id="doc2:section1",
            doc_id="doc2",
            section="section1",
            chunk_id="chunk_456",
            breadcrumbs="Core Banking > Payment Systems"
        )
    }
    
    # Sample context text with citations
    context_text = """Query: What is NEFT and how does it work?

Knowledge Graph Facts:
1. NEFT DEPENDS_ON Core Banking System [doc1:section2]
2. NEFT INTEGRATES_WITH RTGS System [doc2:section1]

Relevant Document Excerpts:

[doc1:section2] (NEFT Specification > Transaction Processing)
NEFT (National Electronic Funds Transfer) is a nation-wide payment system facilitating one-to-one funds transfer. It operates in hourly batches and is available 24x7 throughout the year including holidays. The system can handle transactions from Rs. 1 to Rs. 10 lakhs for individuals.

[doc2:section1] (Core Banking > Payment Systems)
The Core Banking System integrates with NEFT to process electronic fund transfers. All NEFT transactions are validated against account balances and regulatory limits before processing. The system maintains transaction logs for audit and reconciliation purposes.
"""
    
    return AssembledContext(
        context_text=context_text,
        citations=citations,
        token_count=150  # Approximate token count
    )


def main():
    """Demonstrate LLMGenerator usage."""
    
    print("=== LLM Generator Example ===\n")
    
    # Configuration (normally from environment variables)
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    llm_model = os.getenv("LLM_MODEL", "llama2")
    
    print(f"Ollama URL: {ollama_base_url}")
    print(f"LLM Model: {llm_model}\n")
    
    try:
        # Initialize LLM generator
        print("1. Initializing LLMGenerator...")
        generator = LLMGenerator(
            base_url=ollama_base_url,
            model=llm_model
        )
        print("✓ LLMGenerator initialized successfully\n")
        
        # Create sample context
        print("2. Creating sample context...")
        context = create_sample_context()
        print(f"✓ Context created: {context.token_count} tokens, {len(context.citations)} citations\n")
        
        # Generate response for a query
        print("3. Generating response...")
        query = "What is NEFT and how does it work?"
        
        response = generator.generate(query, context)
        
        print("✓ Response generated successfully\n")
        
        # Display results
        print("=== Generated Response ===")
        print(f"Query: {query}")
        print(f"Model: {response.model}")
        print(f"Timestamp: {response.timestamp}")
        print(f"Citations Used: {response.citations_used}")
        print(f"\nAnswer:\n{response.answer}")
        
        # Test insufficient context scenario
        print("\n" + "="*50)
        print("4. Testing insufficient context scenario...")
        
        # Create minimal context
        minimal_context = AssembledContext(
            context_text="Query: What is the capital of Mars?\n\nNo relevant information available.",
            citations={},
            token_count=10
        )
        
        insufficient_query = "What is the capital of Mars?"
        insufficient_response = generator.generate(insufficient_query, minimal_context)
        
        print(f"Query: {insufficient_query}")
        print(f"Answer: {insufficient_response.answer}")
        print(f"Citations: {insufficient_response.citations_used}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    print("\n=== Example completed successfully ===")
    return 0


if __name__ == "__main__":
    exit(main())