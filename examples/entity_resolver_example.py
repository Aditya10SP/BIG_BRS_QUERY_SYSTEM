"""Example demonstrating EntityResolver functionality."""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.extraction.entity_extractor import Entity
from src.extraction.entity_resolver import EntityResolver


def main():
    """Demonstrate entity resolution with duplicate entities."""
    
    print("=" * 80)
    print("Entity Resolution Example")
    print("=" * 80)
    
    # Create sample entities with duplicates
    entities = [
        Entity(
            entity_id="ent_1",
            entity_type="System",
            name="NEFT",
            canonical_name="NEFT",
            source_chunk_id="doc1_chunk_1",
            context="NEFT is a payment system used for electronic fund transfers."
        ),
        Entity(
            entity_id="ent_2",
            entity_type="System",
            name="NEFT System",
            canonical_name="NEFT",
            source_chunk_id="doc1_chunk_5",
            context="The NEFT System processes payments in batches."
        ),
        Entity(
            entity_id="ent_3",
            entity_type="System",
            name="National Electronic Funds Transfer",
            canonical_name="National Electronic Funds Transfer",
            source_chunk_id="doc2_chunk_3",
            context="National Electronic Funds Transfer (NEFT) is operated by RBI."
        ),
        Entity(
            entity_id="ent_4",
            entity_type="PaymentMode",
            name="RTGS",
            canonical_name="RTGS",
            source_chunk_id="doc1_chunk_2",
            context="RTGS is used for large value transactions."
        ),
        Entity(
            entity_id="ent_5",
            entity_type="PaymentMode",
            name="RTGS System",
            canonical_name="RTGS",
            source_chunk_id="doc2_chunk_1",
            context="The RTGS System provides real-time gross settlement."
        ),
        Entity(
            entity_id="ent_6",
            entity_type="System",
            name="Core Banking",
            canonical_name="Core Banking",
            source_chunk_id="doc1_chunk_8",
            context="Core Banking platform integrates with payment systems."
        )
    ]
    
    print(f"\nInput: {len(entities)} entities")
    print("-" * 80)
    for entity in entities:
        print(f"  [{entity.entity_type}] {entity.name} (ID: {entity.entity_id})")
    
    # Create resolver with default threshold (0.85)
    print("\n\nResolving with threshold=0.85 (strict matching)...")
    resolver_strict = EntityResolver(similarity_threshold=0.85)
    canonical_strict, relationships_strict = resolver_strict.resolve(entities)
    
    print(f"\nOutput: {len(canonical_strict)} canonical entities, {len(relationships_strict)} SAME_AS relationships")
    print("-" * 80)
    for entity in canonical_strict:
        print(f"  [{entity.entity_type}] {entity.canonical_name} (ID: {entity.entity_id})")
        if "source_chunk_ids" in entity.properties:
            print(f"    Source chunks: {entity.properties['source_chunk_ids']}")
        if "aliases" in entity.properties:
            print(f"    Aliases: {entity.properties['aliases']}")
    
    print("\nSAME_AS Relationships:")
    print("-" * 80)
    for rel in relationships_strict:
        print(f"  {rel.source_entity_id} → {rel.target_entity_id}")
        print(f"    {rel.properties['source_name']} → {rel.properties['target_name']}")
    
    # Create resolver with lower threshold (0.6) for more aggressive merging
    print("\n\n" + "=" * 80)
    print("Resolving with threshold=0.6 (loose matching)...")
    resolver_loose = EntityResolver(similarity_threshold=0.6)
    canonical_loose, relationships_loose = resolver_loose.resolve(entities)
    
    print(f"\nOutput: {len(canonical_loose)} canonical entities, {len(relationships_loose)} SAME_AS relationships")
    print("-" * 80)
    for entity in canonical_loose:
        print(f"  [{entity.entity_type}] {entity.canonical_name} (ID: {entity.entity_id})")
        if "source_chunk_ids" in entity.properties:
            print(f"    Source chunks: {entity.properties['source_chunk_ids']}")
        if "aliases" in entity.properties:
            print(f"    Aliases: {entity.properties['aliases']}")
        if "mention_count" in entity.properties:
            print(f"    Mentions: {entity.properties['mention_count']}")
    
    print("\nSAME_AS Relationships:")
    print("-" * 80)
    for rel in relationships_loose:
        print(f"  {rel.source_entity_id} → {rel.target_entity_id}")
        print(f"    {rel.properties['source_name']} → {rel.properties['target_name']}")
    
    print("\n" + "=" * 80)
    print("Key Observations:")
    print("=" * 80)
    print("1. Entities are grouped by type (System vs PaymentMode)")
    print("2. Similar entities within each type are merged into canonical entities")
    print("3. SAME_AS relationships link all mentions to the canonical entity")
    print("4. Source chunk references are preserved in canonical entity properties")
    print("5. Lower threshold (0.6) merges more aggressively, including acronym matches")
    print("6. Higher threshold (0.85) is more conservative, requiring closer matches")
    print("=" * 80)


if __name__ == "__main__":
    main()
