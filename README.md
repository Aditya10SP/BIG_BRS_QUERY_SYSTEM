# Graph RAG Layer for Banking Documents

A hybrid retrieval system that combines vector-based semantic search with knowledge graph traversal to answer complex queries about banking Functional Specification Documents (FSDs).

## Features

- **Hybrid Retrieval**: Combines vector search (Qdrant), keyword search (BM25), and graph traversal (Neo4j)
- **Entity Extraction**: Automatic extraction of banking entities and relationships
- **Conflict Detection**: Identifies contradictions across documents
- **Grounded Generation**: LLM responses with source citations
- **Faithfulness Validation**: Verifies responses are grounded in source documents

## Architecture
graph TB
    subgraph "Ingestion Pipeline (Offline)"
        A[Document Input] --> B[Parser]
        B --> C[Hierarchical Chunker]
        C --> D[Embedding Generator]
        C --> E[BM25 Indexer]
        C --> F[Entity Extractor]
        F --> G[Entity Resolver]
        G --> H[Conflict Detector]
        H --> I[Graph Populator]
    end

    subgraph "Storage Layer"
        D --> J[(Qdrant)]
        E --> K[(BM25 Index)]
        C --> L[(PostgreSQL)]
        I --> M[(Neo4j)]
    end

    subgraph "Query Pipeline (Runtime)"
        N[User Query] --> O[Query Router]
        O --> P[Vector Retriever]
        O --> Q[Graph Retriever]
        P --> R[Result Fusion]
        Q --> R
        R --> S[Cross-Encoder Reranker]
        S --> T[Context Assembler]
        T --> U[LLM Generator]
        U --> V[Faithfulness Validator]
        V --> W[Response with Citations]
    end

    J --> P
    K --> P
    L --> P
    M --> Q


The system uses a two-pipeline approach:

1. **Ingestion Pipeline (Offline)**: Processes documents through parsing, chunking, embedding, entity extraction, and graph population
2. **Query Pipeline (Runtime)**: Routes queries to appropriate retrieval modes, fuses results, and generates grounded responses

### Storage Layers

- **Qdrant**: Vector embeddings for semantic similarity search
- **BM25 Index**: Keyword-based search for acronyms and exact terms
- **PostgreSQL**: Full text storage with metadata
- **Neo4j**: Knowledge graph for entity relationships

## Prerequisites

- Python 3.10+
- Docker and Docker Compose
- Ollama (for LLM inference)

## Quick Start

### 1. Clone and Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Copy environment variables
cp .env.example .env
# Edit .env with your configuration
```

### 2. Start Storage Services

```bash
# Start Qdrant, Neo4j, and PostgreSQL
docker-compose up -d

# Verify services are running
docker-compose ps
```

Services will be available at:
- Qdrant: http://localhost:6333
- Neo4j Browser: http://localhost:7474 (user: neo4j, password: graphrag123)
- PostgreSQL: localhost:5432 (user: postgres, password: postgres, db: graph_rag)

### 3. Run the Application

```bash
# Start the FastAPI server
python src/main.py
```

The API will be available at http://localhost:8000

- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

## Configuration

Configuration is managed through environment variables. See `.env.example` for all available options.

### Required Configuration

- `OLLAMA_BASE_URL`: Ollama API endpoint
- `LLM_MODEL`: LLM model name

### Optional Configuration

All other settings have sensible defaults. Key configurations:

- `CHILD_CHUNK_SIZE`: Maximum tokens per chunk (default: 512)
- `SIMILARITY_THRESHOLD`: Minimum similarity for retrieval (default: 0.7)
- `MAX_GRAPH_DEPTH`: Maximum graph traversal depth (default: 3)
- `FAITHFULNESS_THRESHOLD`: Minimum faithfulness score (default: 0.8)

## Project Structure

```
.
├── src/                    # Source code
│   ├── main.py            # FastAPI application
│   ├── parsing/           # Document parsing
│   │   └── document_parser.py
│   ├── chunking/          # Hierarchical chunking
│   │   └── hierarchical_chunker.py
│   ├── storage/           # Storage layer
│   │   └── database_manager.py
│   └── utils/             # Utility modules
│       └── logging.py     # Structured logging
├── config/                # Configuration
│   └── system_config.py   # SystemConfig dataclass
├── tests/                 # Test suite
├── examples/              # Usage examples
│   └── database_example.py
├── docker-compose.yml     # Docker services
├── init-db.sql           # PostgreSQL schema
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run property-based tests only
pytest -m property
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

## API Endpoints

### Query Endpoint (Coming Soon)

```bash
POST /query
{
  "query_text": "What systems depend on NEFT?",
  "mode": "HYBRID",  # Optional: VECTOR, GRAPH, or HYBRID
  "top_k": 10,       # Optional
  "max_depth": 3     # Optional
}
```

### Ingestion Endpoint (Coming Soon)

```bash
POST /ingest
# Upload document file (.docx or .pdf)

GET /ingest/{job_id}
# Check ingestion status
```

## Troubleshooting

### Docker Services Not Starting

```bash
# Check logs
docker-compose logs

# Restart services
docker-compose restart

# Clean restart
docker-compose down -v
docker-compose up -d
```

### Connection Errors

Verify services are accessible:

```bash
# Test Qdrant
curl http://localhost:6333/collections

# Test PostgreSQL
psql -h localhost -U postgres -d graph_rag -c "SELECT 1;"

# Test Neo4j (requires neo4j-client)
cypher-shell -a bolt://localhost:7687 -u neo4j -p graphrag123 "RETURN 1;"
```

## License

MIT License

## Contributing

Contributions are welcome! Please read the design document in `.kiro/specs/graph-rag-layer/design.md` for architecture details.
