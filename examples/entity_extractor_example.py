"""Example usage of EntityExtractor for extracting entities from banking documents."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extraction.entity_extractor import EntityExtractor
from src.chunking.hierarchical_chunker import Chunk


def main():
    """Demonstrate EntityExtractor usage."""
    
    print("=" * 80)
    print("Entity Extractor Example")
    print("=" * 80)
    
    # Create sample banking document chunk
    sample_chunk = Chunk(
        chunk_id="banking_doc_s1_child_0",
        doc_id="banking_doc",
        text="""
        The National Electronic Funds Transfer (NEFT) System is a nationwide payment system 
        facilitating one-to-one funds transfer. NEFT operates in hourly batches and settles 
        transactions through the Reserve Bank of India (RBI). The system integrates with 
        Core Banking Solutions to process transactions. Each transaction requires an 
        Account Number, IFSC Code, and beneficiary details. The Transaction Limit Rule 
        specifies that individual transactions cannot exceed Rs. 2,00,000 without additional 
        authorization. The Payment Authorization Workflow involves validation, risk assessment, 
        and final approval stages. RTGS (Real Time Gross Settlement) is another payment mode 
        for high-value transactions.
        """,
        chunk_type="child",
        parent_chunk_id="banking_doc_s1_parent",
        breadcrumbs="Banking Systems > Payment Systems > NEFT",
        section="NEFT",
        token_count=120,
        metadata={}
    )
    
    print("\nSample Chunk:")
    print("-" * 80)
    print(f"Chunk ID: {sample_chunk.chunk_id}")
    print(f"Section: {sample_chunk.section}")
    print(f"Breadcrumbs: {sample_chunk.breadcrumbs}")
    print(f"\nText:\n{sample_chunk.text.strip()}")
    
    # Initialize EntityExtractor
    print("\n" + "=" * 80)
    print("Initializing EntityExtractor...")
    print("=" * 80)
    
    # Get LLM configuration from environment (optional)
    ollama_base_url = os.getenv("OLLAMA_BASE_URL")
    llm_model = os.getenv("LLM_MODEL")
    
    if ollama_base_url and llm_model:
        print(f"LLM Configuration: {llm_model} at {ollama_base_url}")
        extractor = EntityExtractor(
            spacy_model="en_core_web_sm",
            ollama_base_url=ollama_base_url,
            llm_model=llm_model
        )
    else:
        print("LLM not configured. Using spaCy NER only.")
        extractor = EntityExtractor(spacy_model="en_core_web_sm")
    
    # Extract entities
    print("\n" + "=" * 80)
    print("Extracting Entities...")
    print("=" * 80)
    
    entities = extractor.extract(sample_chunk)
    
    print(f"\nExtracted {len(entities)} entities:")
    print("-" * 80)
    
    if entities:
        # Group entities by type
        entities_by_type = {}
        for entity in entities:
            if entity.entity_type not in entities_by_type:
                entities_by_type[entity.entity_type] = []
            entities_by_type[entity.entity_type].append(entity)
        
        # Display entities grouped by type
        for entity_type, type_entities in sorted(entities_by_type.items()):
            print(f"\n{entity_type} ({len(type_entities)}):")
            for entity in type_entities:
                print(f"  - {entity.name}")
                print(f"    Canonical: {entity.canonical_name}")
                print(f"    Context: {entity.context[:100]}...")
                if entity.properties:
                    print(f"    Properties: {entity.properties}")
                print()
    else:
        print("\nNo entities extracted.")
        print("Note: spaCy model 'en_core_web_sm' may not be installed.")
        print("Install it with: python -m spacy download en_core_web_sm")
    
    # Demonstrate entity normalization
    print("\n" + "=" * 80)
    print("Entity Normalization Examples")
    print("=" * 80)
    
    test_names = [
        "NEFT",
        "NEFT System",
        "National Electronic Funds Transfer",
        "Core Banking Platform",
        "Transaction Limit Rule",
        "RTGS",
        "Payment Authorization Workflow"
    ]
    
    print("\nOriginal Name -> Canonical Name:")
    print("-" * 80)
    for name in test_names:
        canonical = extractor._normalize_entity_name(name)
        print(f"{name:40} -> {canonical}")
    
    print("\n" + "=" * 80)
    print("Example Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
