# Storage Layer

This module provides the storage layer for the Graph RAG system, managing PostgreSQL database operations for documents and chunks, Qdrant vector storage, and Neo4j knowledge graph.

## Components

- **DatabaseManager**: PostgreSQL document and chunk storage
- **VectorStore**: Qdrant vector embeddings storage
- **GraphPopulator**: Neo4j knowledge graph creation

---

## GraphPopulator

The `GraphPopulator` class creates and populates a Neo4j knowledge graph from extracted entities, relationships, and document chunks.

### Features

- **Neo4j Connection Management**: Handles driver initialization and connection pooling
- **Schema Creation**: Creates indexes and constraints for efficient graph queries
- **Batch Operations**: Efficient bulk inserts with configurable batch size (default: 100)
- **Node Types**: Document, Section, Entity (System, PaymentMode, Workflow, Rule, Field), Chunk
- **Relationship Types**: CONTAINS, HAS_CHUNK, MENTIONS, SAME_AS, CONFLICTS_WITH, DEPENDS_ON, INTEGRATES_WITH, USES, etc.
- **Referential Integrity**: Ensures all relationships have valid source and target nodes

### Neo4j Graph Schema

#### Node Types
```cypher
// Document nodes
CREATE (d:Document {doc_id, title, file_path, file_type, metadata})

// Section nodes
CREATE (s:Section {section_id, doc_id, heading, level})

// Entity nodes (with type-specific labels)
CREATE (e:Entity:System {entity_id, name, canonical_name, entity_type, context, properties})
CREATE (e:Entity:PaymentMode {entity_id, name, canonical_name, entity_type, context, properties})
CREATE (e:Entity:Workflow {entity_id, name, canonical_name, entity_type, context, properties})
CREATE (e:Entity:Rule {entity_id, name, canonical_name, entity_type, context, properties})
CREATE (e:Entity:Field {entity_id, name, canonical_name, entity_type, context, properties})

// Chunk nodes
CREATE (c:Chunk {chunk_id, doc_id, text, chunk_type, breadcrumbs, section, token_count, metadata})
```

#### Relationship Types
```cypher
// Document structure
(Document)-[:CONTAINS]->(Section)
(Section)-[:HAS_CHUNK]->(Chunk)
(ParentChunk)-[:CONTAINS]->(ChildChunk)

// Entity relationships
(Chunk)-[:MENTIONS]->(Entity)
(Entity)-[:SAME_AS]->(CanonicalEntity)
(Entity)-[:CONFLICTS_WITH]->(Entity)
(System)-[:DEPENDS_ON]->(System)
(System)-[:INTEGRATES_WITH]->(System)
(System)-[:USES]->(PaymentMode)
(Workflow)-[:NEXT_STEP]->(Workflow)
(Rule)-[:APPLIES_TO]->(System)
(Field)-[:DEFINED_IN]->(System)
```

#### Indexes and Constraints
```cypher
// Uniqueness constraints
CREATE CONSTRAINT entity_id_unique FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE
CREATE CONSTRAINT chunk_id_unique FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE
CREATE CONSTRAINT doc_id_unique FOR (d:Document) REQUIRE d.doc_id IS UNIQUE
CREATE CONSTRAINT section_id_unique FOR (s:Section) REQUIRE s.section_id IS UNIQUE

// Indexes for efficient lookups
CREATE INDEX entity_name_idx FOR (e:Entity) ON (e.name)
CREATE INDEX entity_canonical_idx FOR (e:Entity) ON (e.canonical_name)
CREATE INDEX entity_type_idx FOR (e:Entity) ON (e.entity_type)
CREATE INDEX chunk_doc_idx FOR (c:Chunk) ON (c.doc_id)
CREATE INDEX doc_title_idx FOR (d:Document) ON (d.title)
```

### Usage Example

```python
from src.storage.graph_populator import GraphPopulator
from src.extraction.entity_extractor import Entity
from src.extraction.entity_resolver import Relationship
from src.chunking.hierarchical_chunker import Chunk

# Initialize GraphPopulator
populator = GraphPopulator(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password"
)

# Create schema (indexes and constraints)
populator.create_schema()

# Prepare data
documents = [{
    "doc_id": "doc_001",
    "title": "NEFT Payment System",
    "file_path": "/docs/neft.pdf",
    "file_type": "pdf",
    "metadata": {"version": "1.0"}
}]

entities = [
    Entity(
        entity_id="ent_001",
        entity_type="System",
        name="NEFT",
        canonical_name="NEFT",
        source_chunk_id="chunk_001",
        context="NEFT is a payment system",
        properties={"full_name": "National Electronic Funds Transfer"}
    )
]

relationships = [
    Relationship(
        rel_id="rel_001",
        rel_type="INTEGRATES_WITH",
        source_entity_id="ent_001",
        target_entity_id="ent_002",
        properties={"description": "Integration details"}
    )
]

chunks = [
    Chunk(
        chunk_id="chunk_001",
        doc_id="doc_001",
        text="NEFT is a nationwide payment system...",
        chunk_type="parent",
        parent_chunk_id=None,
        breadcrumbs="NEFT Payment System > Introduction",
        section="Introduction",
        token_count=25,
        metadata={"page": 1}
    )
]

# Populate graph
populator.populate(
    entities=entities,
    relationships=relationships,
    chunks=chunks,
    documents=documents
)

# Close connection
populator.close()
```

### Example Cypher Queries

```cypher
-- Find all systems
MATCH (e:Entity) WHERE e.entity_type = 'System' RETURN e

-- Find entity relationships
MATCH (e1:Entity {canonical_name: 'NEFT'})-[r]-(e2:Entity)
RETURN e1, r, e2

-- Find document structure
MATCH (d:Document)-[:CONTAINS]->(s:Section)-[:HAS_CHUNK]->(c:Chunk)
RETURN d.title, s.heading, count(c) as chunk_count

-- Find entity mentions in chunks
MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
WHERE e.canonical_name = 'NEFT'
RETURN c.text, c.breadcrumbs

-- Find system dependencies
MATCH (s1:System)-[:DEPENDS_ON*1..3]->(s2:System)
WHERE s1.canonical_name = 'NEFT'
RETURN s1, s2

-- Find conflicts
MATCH (e1:Entity)-[r:CONFLICTS_WITH]-(e2:Entity)
RETURN e1.name, e2.name, r.properties
```

### Testing

```bash
# Run unit tests
pytest tests/test_graph_populator.py -v

# Run with Neo4j (requires Neo4j running)
docker-compose up -d neo4j
pytest tests/test_graph_populator.py -v
```

### Requirements

- Neo4j 4.4+ or 5.x
- neo4j Python driver
- Connection URI format: `bolt://host:port` or `neo4j://host:port`

---

## DatabaseManager

The `DatabaseManager` class provides connection pooling and CRUD operations for the PostgreSQL document store.

### Features

- **Connection Pooling**: Efficient connection management with configurable pool size
- **Schema Management**: Automatic creation of documents and chunks tables
- **CRUD Operations**: Create, read, update, and delete operations for documents and chunks
- **Filtering**: Query chunks by document ID or section
- **Cascade Deletion**: Deleting a document automatically removes all associated chunks
- **JSON Metadata**: Support for storing complex metadata as JSONB

### Database Schema

#### Documents Table
```sql
CREATE TABLE documents (
    doc_id VARCHAR(255) PRIMARY KEY,
    title TEXT NOT NULL,
    file_path TEXT,
    file_type VARCHAR(10),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);
```

#### Chunks Table
```sql
CREATE TABLE chunks (
    chunk_id VARCHAR(255) PRIMARY KEY,
    doc_id VARCHAR(255) REFERENCES documents(doc_id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    chunk_type VARCHAR(20),
    parent_chunk_id VARCHAR(255),
    breadcrumbs TEXT,
    section TEXT,
    token_count INT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### Usage Example

```python
from src.storage.database_manager import DatabaseManager

# Initialize database manager
db_manager = DatabaseManager(
    connection_string="postgresql://user:pass@localhost:5432/graph_rag",
    min_connections=1,
    max_connections=10
)
db_manager.initialize()

# Create a document
db_manager.create_document(
    doc_id="doc1",
    title="Banking FSD",
    file_path="/docs/banking_fsd.pdf",
    file_type="pdf",
    metadata={"version": "1.0", "author": "Bank Team"}
)

# Create a chunk
db_manager.create_chunk(
    chunk_id="chunk1",
    doc_id="doc1",
    text="NEFT is a nationwide payment system...",
    chunk_type="child",
    parent_chunk_id="parent1",
    breadcrumbs="Banking FSD > Payment Systems > NEFT",
    section="Payment Systems",
    token_count=128,
    metadata={"page": 5}
)

# Query chunks
chunks = db_manager.get_chunks_by_doc_id("doc1")
section_chunks = db_manager.get_chunks_by_section("doc1", "Payment Systems")

# Update chunk
db_manager.update_chunk("chunk1", text="Updated text...")

# Delete document (cascades to chunks)
db_manager.delete_document("doc1")

# Close connections
db_manager.close()
```

### Testing

To run tests, ensure PostgreSQL is running:

```bash
# Start Docker services
make start

# Run unit tests (requires test database)
pytest tests/test_database_manager.py -v

# Run integration tests
pytest tests/test_database_manager_integration.py -v
```

### Requirements

- PostgreSQL 12+
- psycopg2-binary
- Connection string format: `postgresql://user:password@host:port/database`
