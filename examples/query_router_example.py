"""Example usage of QueryRouter for query classification"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.query.query_router import QueryRouter, QueryMode
from config.system_config import SystemConfig


def main():
    """Demonstrate QueryRouter functionality"""
    
    print("=" * 80)
    print("Query Router Example")
    print("=" * 80)
    print()
    
    # Load configuration
    try:
        config = SystemConfig.from_env()
        print(f"✓ Configuration loaded")
        print(f"  LLM Model: {config.llm_model}")
        print(f"  Ollama URL: {config.ollama_base_url}")
        print()
    except ValueError as e:
        print(f"✗ Configuration error: {e}")
        print("\nPlease set the following environment variables:")
        print("  - OLLAMA_BASE_URL (e.g., http://localhost:11434)")
        print("  - LLM_MODEL (e.g., llama2)")
        return
    
    # Initialize QueryRouter
    router = QueryRouter(
        ollama_base_url=config.ollama_base_url,
        llm_model=config.llm_model,
        confidence_threshold=0.7
    )
    print("✓ QueryRouter initialized")
    print()
    
    # Example queries for each mode
    example_queries = [
        # VECTOR mode examples (factual/definitional)
        "What is NEFT?",
        "Define transaction limit",
        "Explain RTGS process",
        
        # GRAPH mode examples (relational/structural)
        "What systems depend on NEFT?",
        "Compare RTGS and IMPS",
        "Show payment workflow",
        
        # HYBRID mode examples (complex)
        "How does NEFT integrate with Core Banking and what are the limits?",
        "What are the conflicts between payment rules across documents?",
    ]
    
    print("Classifying example queries:")
    print("-" * 80)
    
    for query in example_queries:
        print(f"\nQuery: {query}")
        
        try:
            mode, confidence = router.route(query)
            
            # Format output
            mode_str = mode.value.upper()
            confidence_pct = confidence * 100
            
            # Color coding (if terminal supports it)
            if mode == QueryMode.VECTOR:
                mode_color = "\033[94m"  # Blue
            elif mode == QueryMode.GRAPH:
                mode_color = "\033[92m"  # Green
            else:
                mode_color = "\033[93m"  # Yellow
            reset_color = "\033[0m"
            
            print(f"  Mode: {mode_color}{mode_str}{reset_color}")
            print(f"  Confidence: {confidence_pct:.1f}%")
            
            if confidence < 0.7:
                print(f"  ⚠ Low confidence - defaulted to HYBRID")
        
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print()
    print("-" * 80)
    print()
    
    # Interactive mode
    print("Interactive Mode (type 'quit' to exit):")
    print()
    
    while True:
        try:
            query = input("Enter query: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            mode, confidence = router.route(query)
            
            mode_str = mode.value.upper()
            confidence_pct = confidence * 100
            
            print(f"  → Mode: {mode_str} (confidence: {confidence_pct:.1f}%)")
            
            if confidence < 0.7:
                print(f"  ⚠ Low confidence - defaulted to HYBRID")
            
            print()
        
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"  ✗ Error: {e}")
            print()


if __name__ == "__main__":
    main()
