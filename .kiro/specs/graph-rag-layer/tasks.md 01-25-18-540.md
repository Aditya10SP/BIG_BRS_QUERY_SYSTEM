# Implementation Plan: Graph RAG Layer for Banking Documents

## Overview

This implementation plan breaks down the Graph RAG Layer system into discrete coding tasks following a phased approach:
1. **Phase 1**: Vector RAG pipeline (foundation)
2. **Phase 2**: Graph layer (intelligence on top)
3. **Phase 3**: Query routing and fusion

Each task builds incrementally, with testing integrated throughout to validate correctness early.

## Tasks

- [x] 1. Set up project structure and core infrastructure
  - Create Python project with FastAPI framework
  - Set up directory structure: `src/`, `tests/`, `config/`
  - Configure environment variables and SystemConfig dataclass
  - Set up logging infrastructure with structured logging
  - Create Docker Compose for local development (Qdrant, Neo4j, PostgreSQL)
  - _Requirements: 25.1, 25.2, 25.3, 25.4, 26.1_

- [ ] 2. Implement document parsing
  - [x] 2.1 Create DocumentParser class with .docx and .pdf support
    - Implement parse() method using python-docx and pdfplumber
    - Extract document structure: title, sections, headings, hierarchy
    - Create ParsedDocument and Section dataclasses
    - _Requirements: 1.1, 1.2, 1.4, 1.5_
  
  - [ ]* 2.2 Write property test for document parsing
    - **Property 1: Document Parsing Preserves Structure**
    - **Validates: Requirements 1.1, 1.2, 1.4, 1.5**
  
  - [ ]* 2.3 Write property test for parsing error handling
    - **Property 2: Parsing Errors Are Descriptive**
    - **Validates: Requirements 1.3**

- [ ] 3. Implement hierarchical chunking
  - [x] 3.1 Create HierarchicalChunker class
    - Implement chunk() method with parent/child chunk generation
    - Add breadcrumb generation for context hierarchy
    - Use tiktoken for token counting
    - Preserve sentence boundaries using NLTK or spaCy
    - Create Chunk dataclass
    - _Requirements: 2.1, 2.2, 2.3, 2.6_
  
  - [ ]* 3.2 Write property test for token limits
    - **Property 3: Child Chunks Respect Token Limits**
    - **Validates: Requirements 2.4**
  
  - [ ]* 3.3 Write property test for sentence boundaries
    - **Property 4: Chunks Preserve Sentence Boundaries**
    - **Validates: Requirements 2.6**
  
  - [ ]* 3.4 Write property test for breadcrumbs
    - **Property 5: Child Chunks Have Breadcrumbs**
    - **Validates: Requirements 2.3**

- [ ] 4. Set up storage layer - PostgreSQL
  - [x] 4.1 Create PostgreSQL schema and connection manager
    - Define documents and chunks tables
    - Implement DatabaseManager class with connection pooling
    - Add create, read, update operations for chunks
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_
  
  - [ ]* 4.2 Write property test for document store round-trip
    - **Property 9: Document Store Round-trip**
    - **Validates: Requirements 5.1, 5.3, 5.5**
  
  - [ ]* 4.3 Write property test for filtering
    - **Property 10: Document Store Filtering Works**
    - **Validates: Requirements 5.4**

- [ ] 5. Implement embedding generation and vector storage
  - [x] 5.1 Create EmbeddingGenerator class
    - Initialize sentence-transformers model (all-MiniLM-L6-v2)
    - Implement generate() and batch_generate() methods
    - Add L2 normalization for embeddings
    - _Requirements: 3.1_
  
  - [x] 5.2 Create VectorStore wrapper for Qdrant
    - Initialize Qdrant client and create collection
    - Implement store_embeddings() with batch upsert
    - Implement search() with cosine similarity
    - _Requirements: 3.2, 3.3, 3.4, 3.5_
  
  - [ ]* 5.3 Write property test for embedding storage round-trip
    - **Property 6: Embedding Storage Round-trip**
    - **Validates: Requirements 3.2, 3.4**
  
  - [ ]* 5.4 Write property test for similarity search ranking
    - **Property 7: Vector Similarity Search Returns Ranked Results**
    - **Validates: Requirements 3.5**

- [ ] 6. Implement BM25 keyword indexing
  - [x] 6.1 Create BM25Indexer class
    - Initialize rank_bm25 index
    - Implement index() method with tokenization
    - Implement search() method returning chunk IDs and scores
    - Preserve acronyms during tokenization
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_
  
  - [ ]* 6.2 Write property test for BM25 keyword matching
    - **Property 8: BM25 Index Returns Keyword Matches**
    - **Validates: Requirements 4.2, 4.3, 4.5**

- [x] 7. Checkpoint - Ensure vector RAG pipeline works end-to-end
  - Test ingesting a sample document through parse → chunk → embed → store
  - Test querying with vector + BM25 hybrid search
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 8. Implement entity extraction
  - [x] 8.1 Create EntityExtractor class
    - Initialize spaCy NER model (en_core_web_sm)
    - Implement extract() method with two-stage extraction (spaCy + LLM)
    - Create LLM prompt template for domain entity extraction
    - Implement entity normalization logic
    - Create Entity dataclass
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_
  
  - [ ]* 8.2 Write property test for entity field completeness
    - **Property 11: Entity Extraction Captures Required Fields**
    - **Validates: Requirements 6.4**
  
  - [ ]* 8.3 Write property test for entity normalization consistency
    - **Property 12: Entity Normalization Is Consistent**
    - **Validates: Requirements 6.5**

- [ ] 9. Implement entity resolution and deduplication
  - [x] 9.1 Create EntityResolver class
    - Implement resolve() method with fuzzy matching
    - Use DBSCAN clustering for entity grouping
    - Implement canonical entity selection logic
    - Create SAME_AS relationships
    - Create Relationship dataclass
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_
  
  - [ ]* 9.2 Write property test for SAME_AS relationship creation
    - **Property 13: Entity Resolution Creates SAME_AS Relationships**
    - **Validates: Requirements 7.2, 7.3**
  
  - [ ]* 9.3 Write property test for source reference preservation
    - **Property 14: Entity Merging Preserves Source References**
    - **Validates: Requirements 7.4**

- [ ] 10. Implement conflict detection
  - [x] 10.1 Create ConflictDetector class
    - Implement detect() method with property conflict detection
    - Create LLM prompt template for semantic conflict detection
    - Implement conflict metadata creation
    - Create bidirectional CONFLICTS_WITH relationships
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_
  
  - [ ]* 10.2 Write property test for bidirectional conflict edges
    - **Property 15: Conflict Detection Creates Bidirectional Edges**
    - **Validates: Requirements 8.1, 8.4**
  
  - [ ]* 10.3 Write property test for conflict metadata completeness
    - **Property 16: Conflict Metadata Is Complete**
    - **Validates: Requirements 8.3, 20.3**

- [ ] 11. Set up Neo4j knowledge graph
  - [x] 11.1 Create GraphPopulator class
    - Initialize Neo4j driver and connection
    - Implement create_schema() with node types and indexes
    - Implement populate() method with batch operations
    - Create nodes for Documents, Sections, Entities, Chunks
    - Create relationships: CONTAINS, HAS_CHUNK, DEPENDS_ON, etc.
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6_
  
  - [ ]* 11.2 Write property test for graph population round-trip
    - **Property 17: Graph Population Round-trip**
    - **Validates: Requirements 9.1**
  
  - [ ]* 11.3 Write property test for referential integrity
    - **Property 18: Graph Referential Integrity**
    - **Validates: Requirements 9.6**

- [ ] 12. Implement ingestion pipeline orchestration
  - [x] 12.1 Create IngestionPipeline class
    - Implement ingest() method orchestrating all steps
    - Add error handling and logging for each step
    - Implement batch processing support
    - Add status tracking for documents
    - Implement consistency verification across storage layers
    - _Requirements: 23.1, 23.2, 23.3, 23.4, 23.5_
  
  - [ ]* 12.2 Write property test for pipeline step ordering
    - **Property 44: Ingestion Pipeline Executes In Order**
    - **Validates: Requirements 23.1**
  
  - [ ]* 12.3 Write property test for failure handling
    - **Property 45: Ingestion Failure Halts Processing**
    - **Validates: Requirements 23.2**

- [ ] 13. Checkpoint - Ensure ingestion pipeline works end-to-end
  - Test ingesting multiple documents with entity extraction and graph population
  - Verify all storage layers are populated correctly
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 14. Implement query routing
  - [x] 14.1 Create QueryRouter class
    - Implement route() method with LLM-based classification
    - Create classification prompt template
    - Implement confidence scoring
    - Add default to HYBRID for low confidence
    - Create QueryMode enum
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6_
  
  - [ ]* 14.2 Write property test for valid mode classification
    - **Property 19: Query Router Produces Valid Mode**
    - **Validates: Requirements 10.1**
  
  - [ ]* 14.3 Write property test for low confidence default
    - **Property 20: Low Confidence Defaults to HYBRID**
    - **Validates: Requirements 10.6**
  
  - [ ]* 14.4 Write unit tests for query mode examples
    - Test factual queries → VECTOR mode
    - Test relational queries → GRAPH mode
    - Test complex queries → HYBRID mode
    - _Requirements: 10.2, 10.3, 10.4_

- [ ] 15. Implement vector retrieval
  - [x] 15.1 Create VectorRetriever class
    - Implement retrieve() method with parallel vector + BM25 search
    - Implement Reciprocal Rank Fusion (RRF) for score combination
    - Add similarity threshold filtering (0.7)
    - Add top-k limiting (10 results)
    - Create RetrievedChunk dataclass
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_
  
  - [ ]* 15.2 Write property test for similarity threshold
    - **Property 21: Vector Retrieval Respects Similarity Threshold**
    - **Validates: Requirements 11.3**
  
  - [ ]* 15.3 Write property test for top-k limit
    - **Property 22: Vector Retrieval Respects Top-K Limit**
    - **Validates: Requirements 11.4**

- [ ] 16. Implement graph retrieval
  - [x] 16.1 Create GraphRetriever class
    - Implement retrieve() method with entity extraction from query
    - Implement Cypher query generation for 5 patterns:
      - Dependency queries (forward/backward)
      - Integration queries
      - Workflow queries
      - Conflict queries
      - Comparison queries
    - Add depth limiting (max 3 hops)
    - Create GraphResult, GraphNode, GraphRelationship dataclasses
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6_
  
  - [ ]* 16.2 Write property test for depth limiting
    - **Property 23: Graph Traversal Respects Depth Limit**
    - **Validates: Requirements 12.3**
  
  - [ ]* 16.3 Write unit tests for traversal patterns
    - Test dependency pattern
    - Test integration pattern
    - Test workflow pattern
    - Test conflict pattern
    - Test comparison pattern
    - _Requirements: 12.6_

- [x] 17. Implement result fusion
  - [x] 17.1 Create ResultFusion class
    - Implement fuse() method with chunk deduplication
    - Implement graph fact extraction and formatting
    - Implement score combination (0.6 vector + 0.4 graph)
    - Create FusedResults dataclass
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5_
  
  - [ ]* 17.2 Write property test for hybrid mode execution
    - **Property 24: Hybrid Mode Executes Both Retrievals**
    - **Validates: Requirements 13.1**
  
  - [ ]* 17.3 Write property test for deduplication
    - **Property 25: Result Fusion Deduplicates Chunks**
    - **Validates: Requirements 13.2**
  
  - [ ]* 17.4 Write property test for score preservation
    - **Property 26: Fusion Preserves Both Score Types**
    - **Validates: Requirements 13.4**

- [x] 18. Implement cross-encoder reranking
  - [x] 18.1 Create CrossEncoderReranker class
    - Initialize cross-encoder model (ms-marco-MiniLM-L-6-v2)
    - Implement rerank() method with batch scoring
    - Add top-k selection (5 results)
    - Preserve metadata during reranking
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5_
  
  - [ ]* 18.2 Write property test for top-k preservation
    - **Property 27: Reranking Preserves Top-K**
    - **Validates: Requirements 14.4**
  
  - [ ]* 18.3 Write property test for metadata preservation
    - **Property 28: Reranking Preserves Metadata**
    - **Validates: Requirements 14.5**

- [x] 19. Implement context assembly
  - [x] 19.1 Create ContextAssembler class
    - Implement assemble() method with context formatting
    - Add graph facts formatting
    - Add citation generation ([doc_id:section])
    - Add token counting and truncation logic
    - Create AssembledContext and Citation dataclasses
    - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.5, 15.6_
  
  - [ ]* 19.2 Write property test for token limit
    - **Property 29: Context Assembly Respects Token Limit**
    - **Validates: Requirements 15.5**
  
  - [ ]* 19.3 Write property test for citation inclusion
    - **Property 30: Context Includes Citations**
    - **Validates: Requirements 15.4**
  
  - [ ]* 19.4 Write property test for truncation citation preservation
    - **Property 31: Context Truncation Preserves Citations**
    - **Validates: Requirements 15.6**

- [x] 20. Implement LLM generation
  - [x] 20.1 Create LLMGenerator class
    - Initialize Ollama client with configured base URL and model
    - Implement generate() method with system prompt
    - Add citation enforcement in prompt
    - Add insufficient context handling
    - Create GeneratedResponse dataclass
    - _Requirements: 16.1, 16.2, 16.3, 16.4, 16.5, 16.6_
  
  - [ ]* 20.2 Write property test for citation inclusion
    - **Property 32: LLM Response Includes Citations**
    - **Validates: Requirements 16.3, 16.4**
  
  - [ ]* 20.3 Write property test for insufficient context refusal
    - **Property 33: Insufficient Context Triggers Refusal**
    - **Validates: Requirements 16.6**

- [ ] 21. Implement faithfulness validation
  - [x] 21.1 Create FaithfulnessValidator class
    - Implement validate() method with claim extraction
    - Implement entailment checking using LLM
    - Compute faithfulness score
    - Add warning generation for low scores
    - Create ValidationResult dataclass
    - _Requirements: 17.1, 17.2, 17.3, 17.4, 17.5, 17.6_
  
  - [ ]* 21.2 Write property test for faithfulness score computation
    - **Property 34: Faithfulness Validation Computes Score**
    - **Validates: Requirements 17.4**
  
  - [ ]* 21.3 Write property test for low score warnings
    - **Property 35: Low Faithfulness Triggers Warning**
    - **Validates: Requirements 17.5**

- [ ] 22. Implement query pipeline orchestration
  - [x] 22.1 Create QueryPipeline class
    - Implement query() method orchestrating all steps
    - Add parallel execution for HYBRID mode retrievals
    - Add error handling and logging for each step
    - Add query execution metrics logging
    - _Requirements: 24.1, 24.2, 24.3, 24.5_
  
  - [ ]* 22.2 Write property test for pipeline step ordering
    - **Property 46: Query Pipeline Executes In Order**
    - **Validates: Requirements 24.1**
  
  - [ ]* 22.3 Write property test for failure handling
    - **Property 47: Query Failure Returns Error Response**
    - **Validates: Requirements 24.3**

- [x] 23. Checkpoint - Ensure query pipeline works end-to-end
  - Test all three query modes (VECTOR, GRAPH, HYBRID)
  - Test with sample banking queries
  - Verify citations and faithfulness validation
  - Ensure all tests pass, ask the user if questions arise.

- [x] 24. Implement advanced query patterns
  - [x] 24.1 Add dependency query support
    - Implement forward and backward dependency traversal
    - Add impact radius computation
    - Add circular dependency detection
    - _Requirements: 18.1, 18.2, 18.3, 18.4, 18.5_
  
  - [ ]* 24.2 Write property test for bidirectional traversal
    - **Property 36: Dependency Traversal Is Bidirectional**
    - **Validates: Requirements 18.2**
  
  - [ ]* 24.3 Write property test for complete chains
    - **Property 37: Dependency Chains Are Complete**
    - **Validates: Requirements 18.3**

- [x] 25. Implement cross-document comparison
  - [x] 25.1 Add comparison query support
    - Implement multi-document retrieval
    - Add document grouping logic
    - Add common entity identification using SAME_AS
    - Add difference highlighting
    - _Requirements: 19.1, 19.2, 19.3, 19.4, 19.5_
  
  - [ ]* 25.2 Write property test for document grouping
    - **Property 38: Comparison Groups By Document**
    - **Validates: Requirements 19.2**
  
  - [ ]* 25.3 Write property test for common entity identification
    - **Property 39: Comparison Identifies Common Entities**
    - **Validates: Requirements 19.3**

- [x] 26. Implement conflict and process queries
  - [x] 26.1 Add conflict query support
    - Implement CONFLICTS_WITH relationship retrieval
    - Add conflict categorization
    - Add severity ranking
    - _Requirements: 20.1, 20.2, 20.3, 20.4, 20.5_
  
  - [x] 26.2 Add process chain traversal
    - Implement NEXT_STEP relationship traversal
    - Add branching workflow support
    - Add gap detection for incomplete chains
    - _Requirements: 21.1, 21.2, 21.3, 21.4, 21.5_
  
  - [x] 26.3 Add risk rule analysis
    - Implement Rule node retrieval
    - Add APPLIES_TO traversal
    - Add rule overlap detection
    - Add rule ranking by specificity
    - _Requirements: 22.1, 22.2, 22.3, 22.4, 22.5_
  
  - [ ]* 26.4 Write property test for conflict result completeness
    - **Property 40: Conflict Results Include Both Sides**
    - **Validates: Requirements 20.3**
  
  - [ ]* 26.5 Write property test for complete process chains
    - **Property 41: Process Chains Are Complete**
    - **Validates: Requirements 21.2, 21.3**
  
  - [ ]* 26.6 Write property test for incomplete process indication
    - **Property 42: Incomplete Processes Indicate Gaps**
    - **Validates: Requirements 21.5**
  
  - [ ]* 26.7 Write property test for rule result completeness
    - **Property 43: Rule Results Include Complete Information**
    - **Validates: Requirements 22.3**

- [x] 27. Implement REST API endpoints
  - [x] 27.1 Create FastAPI application and query endpoint
    - Implement POST /query endpoint
    - Add request validation with Pydantic models
    - Add optional parameters: mode, top_k, max_depth
    - Add response formatting with required fields
    - Add error handling (400, 500)
    - _Requirements: 27.1, 27.2, 27.3, 27.4, 27.5, 27.6_
  
  - [x] 27.2 Create ingestion API endpoints
    - Implement POST /ingest endpoint with file upload
    - Add file format validation
    - Add job ID generation and tracking
    - Implement GET /ingest/{job_id} status endpoint
    - Add success/failure status responses
    - _Requirements: 28.1, 28.2, 28.3, 28.4, 28.5, 28.6_
  
  - [ ]* 27.3 Write property test for API response fields
    - **Property 50: API Responses Have Required Fields**
    - **Validates: Requirements 27.4**
  
  - [ ]* 27.4 Write property test for malformed request handling
    - **Property 51: Malformed Requests Return 400**
    - **Validates: Requirements 27.5**
  
  - [ ]* 27.5 Write property test for ingestion job ID
    - **Property 52: Ingestion Returns Job ID**
    - **Validates: Requirements 28.3**
  
  - [ ]* 27.6 Write property test for ingestion status tracking
    - **Property 53: Ingestion Status Is Trackable**
    - **Validates: Requirements 28.5, 28.6**

- [-] 28. Implement error handling and logging
  - [x] 28.1 Add comprehensive error handling
    - Implement error classes: ParsingError, StorageError, RetrievalError, LLMError, ValidationError
    - Add retry logic with exponential backoff
    - Add graceful degradation (fallback modes)
    - _Requirements: 26.2, 26.5_
  
  - [ ]* 28.2 Write property test for error logging completeness
    - **Property 49: Error Logging Is Complete**
    - **Validates: Requirements 26.1, 26.3**
  
  - [ ]* 28.3 Write property test for configuration validation
    - **Property 48: Configuration Validation At Startup**
    - **Validates: Requirements 25.3, 25.4**

- [x] 29. Add performance optimizations
  - [x] 29.1 Implement caching layer
    - Add embedding cache (LRU)
    - Add entity resolution cache
    - Add Cypher query result cache
    - Add cross-encoder score cache
  
  - [x] 29.2 Add batch processing optimizations
    - Optimize embedding generation batching
    - Optimize Neo4j batch writes
    - Optimize Qdrant batch upserts
  
  - [x] 29.3 Add database indexing
    - Create PostgreSQL indexes
    - Create Neo4j indexes
    - Create Qdrant payload indexes

- [x] 30. Create deployment configuration
  - [x] 30.1 Create Docker containers
    - Create Dockerfile for API service
    - Create Dockerfile for ingestion worker
    - Create docker-compose.yml with all services
  
  - [x] 30.2 Add monitoring and observability
    - Add Prometheus metrics endpoints
    - Add structured logging configuration
    - Add health check endpoints

- [x] 31. Final checkpoint - End-to-end system validation
  - Test complete ingestion pipeline with multiple documents
  - Test all query modes and advanced patterns
  - Test API endpoints with various scenarios
  - Verify all property tests pass
  - Verify error handling and logging
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional property-based tests that can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at major milestones
- Property tests validate universal correctness properties with 100+ iterations
- Unit tests validate specific examples, edge cases, and integration points
- The implementation follows a phased approach: Vector RAG → Graph Layer → Query Routing
