"""Example usage of ConflictDetector for identifying contradictory information."""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.chunking.hierarchical_chunker import Chunk
from src.extraction.entity_extractor import Entity
from src.extraction.conflict_detector import ConflictDetector


def main():
    """Demonstrate ConflictDetector usage."""
    
    print("=" * 80)
    print("ConflictDetector Example")
    print("=" * 80)
    
    # Initialize ConflictDetector with LLM configuration
    print("\n1. Initializing ConflictDetector...")
    detector = ConflictDetector(
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        llm_model=os.getenv("LLM_MODEL", "llama2")
    )
    print(f"   ✓ Initialized with model: {detector.llm_model}")
    
    # Create sample chunks representing different documents
    print("\n2. Creating sample chunks from different documents...")
    chunks = [
        Chunk(
            chunk_id="chunk_doc1_001",
            doc_id="FSD_NEFT_v1.0",
            text="NEFT (National Electronic Funds Transfer) has a transaction limit of 2 lakhs per transaction. "
                 "This limit applies to all retail customers. NEFT operates in hourly batches.",
            chunk_type="child",
            parent_chunk_id="parent_001",
            breadcrumbs="NEFT Specification > Transaction Limits > Retail Limits",
            section="Transaction Limits",
            token_count=45,
            metadata={"version": "1.0", "date": "2023-01-15"}
        ),
        Chunk(
            chunk_id="chunk_doc2_001",
            doc_id="FSD_NEFT_v2.0",
            text="NEFT (National Electronic Funds Transfer) transaction limit has been increased to 5 lakhs per transaction "
                 "for retail customers. This change is effective from April 2023. NEFT operates in hourly batches.",
            chunk_type="child",
            parent_chunk_id="parent_002",
            breadcrumbs="NEFT Specification > Transaction Limits > Updated Limits",
            section="Transaction Limits",
            token_count=48,
            metadata={"version": "2.0", "date": "2023-04-01"}
        ),
        Chunk(
            chunk_id="chunk_doc3_001",
            doc_id="FSD_Payment_Rules",
            text="Transaction approval rule: All transactions below 1 lakh should be automatically approved "
                 "without manual intervention. This applies to NEFT, RTGS, and IMPS.",
            chunk_type="child",
            parent_chunk_id="parent_003",
            breadcrumbs="Payment Rules > Approval Rules > Auto-Approval",
            section="Approval Rules",
            token_count=38,
            metadata={"version": "1.0"}
        ),
        Chunk(
            chunk_id="chunk_doc4_001",
            doc_id="FSD_Risk_Management",
            text="Risk management policy: All transactions below 1 lakh must undergo manual review "
                 "for fraud detection. This is mandatory for all payment modes.",
            chunk_type="child",
            parent_chunk_id="parent_004",
            breadcrumbs="Risk Management > Fraud Detection > Review Policy",
            section="Fraud Detection",
            token_count=35,
            metadata={"version": "1.0"}
        )
    ]
    print(f"   ✓ Created {len(chunks)} chunks from different documents")
    
    # Create sample entities with potential conflicts
    print("\n3. Creating sample entities...")
    entities = [
        # NEFT entities with different transaction limits (property conflict)
        Entity(
            entity_id="ent_neft_001",
            entity_type="PaymentMode",
            name="NEFT",
            canonical_name="NEFT",
            source_chunk_id="chunk_doc1_001",
            context="NEFT has a transaction limit of 2 lakhs per transaction.",
            properties={
                "transaction_limit": "2 lakhs",
                "customer_type": "retail",
                "operation_mode": "hourly batches"
            }
        ),
        Entity(
            entity_id="ent_neft_002",
            entity_type="PaymentMode",
            name="NEFT",
            canonical_name="NEFT",
            source_chunk_id="chunk_doc2_001",
            context="NEFT transaction limit has been increased to 5 lakhs per transaction.",
            properties={
                "transaction_limit": "5 lakhs",
                "customer_type": "retail",
                "operation_mode": "hourly batches"
            }
        ),
        # Rule entities with contradictory policies (semantic conflict)
        Entity(
            entity_id="ent_rule_001",
            entity_type="Rule",
            name="Transaction Approval Rule",
            canonical_name="Transaction Approval Rule",
            source_chunk_id="chunk_doc3_001",
            context="All transactions below 1 lakh should be automatically approved without manual intervention.",
            properties={
                "threshold": "1 lakh",
                "action": "auto-approve"
            }
        ),
        Entity(
            entity_id="ent_rule_002",
            entity_type="Rule",
            name="Transaction Approval Rule",
            canonical_name="Transaction Approval Rule",
            source_chunk_id="chunk_doc4_001",
            context="All transactions below 1 lakh must undergo manual review for fraud detection.",
            properties={
                "threshold": "1 lakh",
                "action": "manual-review"
            }
        )
    ]
    print(f"   ✓ Created {len(entities)} entities")
    
    # Detect conflicts
    print("\n4. Detecting conflicts...")
    print("   This will:")
    print("   - Check for property conflicts (different values for same property)")
    print("   - Use LLM to detect semantic conflicts (contradictory rules/policies)")
    print("   - Create bidirectional CONFLICTS_WITH relationships")
    print()
    
    conflicts = detector.detect(entities, chunks)
    
    # Display results
    print(f"\n5. Conflict Detection Results:")
    print(f"   Found {len(conflicts)} conflict relationships")
    print()
    
    if conflicts:
        # Group conflicts by entity pairs (bidirectional relationships)
        conflict_pairs = {}
        for conflict in conflicts:
            # Create a sorted tuple to identify unique pairs
            pair_key = tuple(sorted([conflict.source_entity_id, conflict.target_entity_id]))
            if pair_key not in conflict_pairs:
                conflict_pairs[pair_key] = conflict
        
        print(f"   Unique conflict pairs: {len(conflict_pairs)}")
        print()
        
        for idx, (pair_key, conflict) in enumerate(conflict_pairs.items(), 1):
            print(f"   Conflict #{idx}:")
            print(f"   ├─ Type: {conflict.properties.get('conflict_type', 'unknown').upper()}")
            print(f"   ├─ Between: {conflict.source_entity_id} ↔ {conflict.target_entity_id}")
            print(f"   ├─ Explanation: {conflict.properties.get('explanation', 'N/A')}")
            print(f"   ├─ Source Documents:")
            for doc_id in conflict.properties.get('doc_ids', []):
                print(f"   │  • {doc_id}")
            print(f"   ├─ Source Chunks:")
            for chunk_id in conflict.properties.get('source_chunk_ids', []):
                print(f"   │  • {chunk_id}")
            
            # Show conflicting properties if available
            if 'conflicting_properties' in conflict.properties:
                print(f"   └─ Conflicting Properties:")
                for prop in conflict.properties['conflicting_properties']:
                    print(f"      • {prop['property']}: '{prop['value1']}' vs '{prop['value2']}'")
            else:
                print(f"   └─ Contexts:")
                print(f"      • Entity 1: {conflict.properties.get('entity1_context', 'N/A')[:80]}...")
                print(f"      • Entity 2: {conflict.properties.get('entity2_context', 'N/A')[:80]}...")
            print()
    else:
        print("   No conflicts detected.")
    
    # Example: Property conflict detection without LLM
    print("\n6. Property Conflict Detection (without LLM):")
    print("   Property conflicts are detected by comparing entity properties directly.")
    print("   This is fast and doesn't require LLM calls.")
    print()
    
    entity1 = entities[0]  # NEFT with 2 lakhs limit
    entity2 = entities[1]  # NEFT with 5 lakhs limit
    
    property_conflict = detector._check_property_conflict(entity1, entity2)
    
    if property_conflict:
        print(f"   ✓ Property conflict detected!")
        print(f"   ├─ Conflict Type: {property_conflict['conflict_type']}")
        print(f"   ├─ Explanation: {property_conflict['explanation']}")
        print(f"   └─ Conflicting Properties:")
        for prop in property_conflict['conflicting_properties']:
            print(f"      • {prop['property']}: '{prop['value1']}' vs '{prop['value2']}'")
    
    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
