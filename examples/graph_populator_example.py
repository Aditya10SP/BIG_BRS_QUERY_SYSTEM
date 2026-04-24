"""Example usage of GraphPopulator for creating Neo4j knowledge graph."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage.graph_populator import GraphPopulator
from src.extraction.entity_extractor import Entity
from src.extraction.entity_resolver import Relationship
from src.chunking.hierarchical_chunker import Chunk


def main():
    """Demonstrate GraphPopulator usage."""
    
    # Initialize GraphPopulator with Neo4j connection
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
    
    print(f"Connecting to Neo4j at {neo4j_uri}...")
    
    try:
        populator = GraphPopulator(
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password
        )
        
        print("✓ Connected to Neo4j successfully")
        
        # Step 1: Create schema (indexes and constraints)
        print("\nCreating Neo4j schema...")
        populator.create_schema()
        print("✓ Schema created successfully")
        
        # Step 2: Prepare sample data
        print("\nPreparing sample data...")
        
        # Sample documents
        documents = [
            {
                "doc_id": "doc_001",
                "title": "NEFT Payment System Specification",
                "file_path": "/docs/neft_spec.pdf",
                "file_type": "pdf",
                "metadata": {"version": "1.0", "author": "Banking Team"}
            },
            {
                "doc_id": "doc_002",
                "title": "RTGS Integration Guide",
                "file_path": "/docs/rtgs_guide.pdf",
                "file_type": "pdf",
                "metadata": {"version": "2.1", "author": "Integration Team"}
            }
        ]
        
        # Sample chunks
        chunks = [
            Chunk(
                chunk_id="chunk_001",
                doc_id="doc_001",
                text="NEFT (National Electronic Funds Transfer) is a nationwide payment system facilitating one-to-one funds transfer.",
                chunk_type="parent",
                parent_chunk_id=None,
                breadcrumbs="NEFT Payment System Specification > Introduction",
                section="Introduction",
                token_count=25,
                metadata={"page": 1}
            ),
            Chunk(
                chunk_id="chunk_002",
                doc_id="doc_001",
                text="NEFT operates in hourly batches. Transactions are settled in batches during banking hours.",
                chunk_type="child",
                parent_chunk_id="chunk_001",
                breadcrumbs="NEFT Payment System Specification > Introduction > Operation",
                section="Introduction",
                token_count=15,
                metadata={"page": 1}
            ),
            Chunk(
                chunk_id="chunk_003",
                doc_id="doc_002",
                text="RTGS (Real Time Gross Settlement) is used for high-value transactions. It provides real-time settlement.",
                chunk_type="parent",
                parent_chunk_id=None,
                breadcrumbs="RTGS Integration Guide > Overview",
                section="Overview",
                token_count=20,
                metadata={"page": 1}
            ),
            Chunk(
                chunk_id="chunk_004",
                doc_id="doc_002",
                text="RTGS integrates with NEFT for seamless payment processing across different transaction types.",
                chunk_type="child",
                parent_chunk_id="chunk_003",
                breadcrumbs="RTGS Integration Guide > Overview > Integration",
                section="Overview",
                token_count=15,
                metadata={"page": 2}
            )
        ]
        
        # Sample entities
        entities = [
            Entity(
                entity_id="ent_001",
                entity_type="System",
                name="NEFT",
                canonical_name="NEFT",
                source_chunk_id="chunk_001",
                context="NEFT (National Electronic Funds Transfer) is a nationwide payment system",
                properties={"full_name": "National Electronic Funds Transfer"}
            ),
            Entity(
                entity_id="ent_002",
                entity_type="System",
                name="RTGS",
                canonical_name="RTGS",
                source_chunk_id="chunk_003",
                context="RTGS (Real Time Gross Settlement) is used for high-value transactions",
                properties={"full_name": "Real Time Gross Settlement"}
            ),
            Entity(
                entity_id="ent_003",
                entity_type="PaymentMode",
                name="Batch Processing",
                canonical_name="Batch Processing",
                source_chunk_id="chunk_002",
                context="NEFT operates in hourly batches",
                properties={"type": "batch"}
            ),
            Entity(
                entity_id="ent_004",
                entity_type="PaymentMode",
                name="Real-time Settlement",
                canonical_name="Real-time Settlement",
                source_chunk_id="chunk_003",
                context="RTGS provides real-time settlement",
                properties={"type": "realtime"}
            )
        ]
        
        # Sample relationships
        relationships = [
            Relationship(
                rel_id="rel_001",
                rel_type="INTEGRATES_WITH",
                source_entity_id="ent_002",  # RTGS
                target_entity_id="ent_001",  # NEFT
                properties={
                    "description": "RTGS integrates with NEFT for seamless payment processing",
                    "source_chunk": "chunk_004"
                }
            ),
            Relationship(
                rel_id="rel_002",
                rel_type="USES",
                source_entity_id="ent_001",  # NEFT
                target_entity_id="ent_003",  # Batch Processing
                properties={
                    "description": "NEFT uses batch processing mode",
                    "source_chunk": "chunk_002"
                }
            ),
            Relationship(
                rel_id="rel_003",
                rel_type="USES",
                source_entity_id="ent_002",  # RTGS
                target_entity_id="ent_004",  # Real-time Settlement
                properties={
                    "description": "RTGS uses real-time settlement mode",
                    "source_chunk": "chunk_003"
                }
            )
        ]
        
        print(f"  - {len(documents)} documents")
        print(f"  - {len(chunks)} chunks")
        print(f"  - {len(entities)} entities")
        print(f"  - {len(relationships)} relationships")
        
        # Step 3: Populate the graph
        print("\nPopulating Neo4j graph...")
        populator.populate(
            entities=entities,
            relationships=relationships,
            chunks=chunks,
            documents=documents
        )
        print("✓ Graph populated successfully")
        
        # Step 4: Summary
        print("\n" + "="*60)
        print("Graph Population Summary")
        print("="*60)
        print(f"Documents created:     {len(documents)}")
        print(f"Chunks created:        {len(chunks)}")
        print(f"Entities created:      {len(entities)}")
        print(f"Relationships created: {len(relationships)}")
        print("\nNode types created:")
        print("  - Document nodes")
        print("  - Section nodes (extracted from chunks)")
        print("  - Entity nodes (System, PaymentMode)")
        print("  - Chunk nodes")
        print("\nRelationship types created:")
        print("  - CONTAINS (Document → Section, Section → Chunk, Parent → Child)")
        print("  - HAS_CHUNK (Section → Chunk)")
        print("  - MENTIONS (Chunk → Entity)")
        print("  - INTEGRATES_WITH (System → System)")
        print("  - USES (System → PaymentMode)")
        print("\nYou can now query the graph using Cypher queries in Neo4j Browser")
        print(f"Neo4j Browser: http://localhost:7474")
        print("\nExample Cypher queries:")
        print("  1. Find all systems:")
        print("     MATCH (e:Entity) WHERE e.entity_type = 'System' RETURN e")
        print("\n  2. Find NEFT integrations:")
        print("     MATCH (e1:Entity {canonical_name: 'NEFT'})-[r]-(e2:Entity)")
        print("     RETURN e1, r, e2")
        print("\n  3. Find document structure:")
        print("     MATCH (d:Document)-[:CONTAINS]->(s:Section)-[:HAS_CHUNK]->(c:Chunk)")
        print("     RETURN d.title, s.heading, count(c) as chunk_count")
        
        # Close connection
        populator.close()
        print("\n✓ Connection closed")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
