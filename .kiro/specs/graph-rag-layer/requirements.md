# Requirements Document: Graph RAG Layer for Banking Documents

## Introduction

The Graph RAG Layer system enhances traditional vector-based Retrieval-Augmented Generation (RAG) with knowledge graph capabilities to enable intelligent querying of banking Functional Specification Documents (FSDs). The system combines semantic search, keyword matching, and graph-based relationship traversal to answer complex queries about banking systems, dependencies, conflicts, and processes.

## Glossary

- **System**: The Graph RAG Layer application
- **RAG**: Retrieval-Augmented Generation - technique combining information retrieval with LLM generation
- **Vector_Store**: Qdrant database storing document chunk embeddings
- **Knowledge_Graph**: Neo4j graph database storing entities and relationships
- **BM25_Index**: Keyword-based search index for acronyms and exact matches
- **Document_Store**: PostgreSQL database storing full text chunks with metadata
- **Ingestion_Pipeline**: Offline process that parses, chunks, and indexes documents
- **Query_Pipeline**: Runtime process that handles user queries and generates responses
- **Entity**: A named concept extracted from documents (System, PaymentMode, Workflow, Rule, Field)
- **Chunk**: A segment of document text with hierarchical context (breadcrumbs)
- **Query_Router**: Component that determines which retrieval mode to use
- **Cross_Encoder**: Reranking model that scores relevance of retrieved results
- **Faithfulness_Validator**: Component that verifies LLM responses are grounded in retrieved context
- **FSD**: Functional Specification Document - banking system documentation

## Requirements

### Requirement 1: Document Ingestion and Parsing

**User Story:** As a system administrator, I want to ingest banking documents in various formats, so that the system can process and index their content for querying.

#### Acceptance Criteria

1. WHEN a .docx file is provided, THE Ingestion_Pipeline SHALL parse it and extract text content with structural metadata
2. WHEN a .pdf file is provided, THE Ingestion_Pipeline SHALL parse it and extract text content with structural metadata
3. WHEN parsing fails, THE Ingestion_Pipeline SHALL return a descriptive error message indicating the failure reason
4. WHEN a document is parsed, THE System SHALL preserve document structure including sections, headings, and hierarchy
5. THE Ingestion_Pipeline SHALL extract metadata including document title, sections, and page numbers

### Requirement 2: Hierarchical Chunking

**User Story:** As a developer, I want documents chunked hierarchically with context preservation, so that retrieved chunks maintain their relationship to parent content.

#### Acceptance Criteria

1. WHEN a document is chunked, THE Ingestion_Pipeline SHALL create parent chunks representing major sections
2. WHEN a document is chunked, THE Ingestion_Pipeline SHALL create child chunks from parent chunks with size limits
3. WHEN a child chunk is created, THE System SHALL attach breadcrumb metadata linking it to its parent and document
4. THE System SHALL ensure child chunks do not exceed 512 tokens in length
5. THE System SHALL ensure parent chunks capture section-level context
6. WHEN chunks overlap boundaries, THE System SHALL maintain semantic coherence by preserving sentence boundaries

### Requirement 3: Vector Embedding and Indexing

**User Story:** As a developer, I want document chunks embedded and indexed in vector storage, so that semantic similarity search can retrieve relevant content.

#### Acceptance Criteria

1. WHEN a chunk is created, THE System SHALL generate a vector embedding using the configured embedding model
2. WHEN an embedding is generated, THE System SHALL store it in the Vector_Store with chunk metadata
3. THE System SHALL index embeddings to enable efficient similarity search
4. WHEN storing embeddings, THE System SHALL associate them with chunk IDs for retrieval
5. THE Vector_Store SHALL support cosine similarity search across all indexed embeddings

### Requirement 4: BM25 Keyword Indexing

**User Story:** As a user, I want keyword-based search for acronyms and exact terms, so that I can find specific banking terminology regardless of semantic similarity.

#### Acceptance Criteria

1. WHEN a chunk is created, THE System SHALL index it in the BM25_Index with term frequencies
2. THE BM25_Index SHALL support exact keyword matching for acronyms and technical terms
3. WHEN a query contains acronyms, THE System SHALL retrieve matching chunks from the BM25_Index
4. THE System SHALL maintain inverse document frequency statistics for BM25 scoring
5. THE BM25_Index SHALL return results ranked by BM25 relevance score

### Requirement 5: Full Text Storage

**User Story:** As a developer, I want full text chunks stored with metadata, so that the system can retrieve complete context for LLM generation.

#### Acceptance Criteria

1. WHEN a chunk is created, THE System SHALL store the full text in the Document_Store
2. WHEN storing chunks, THE System SHALL include metadata: document_id, section, breadcrumbs, chunk_type
3. THE Document_Store SHALL support retrieval by chunk_id
4. THE Document_Store SHALL support filtering by document_id and section
5. WHEN retrieving chunks, THE System SHALL return full text with all associated metadata

### Requirement 6: Entity Extraction

**User Story:** As a developer, I want entities extracted from documents using NER and LLM, so that the knowledge graph can represent key concepts and their relationships.

#### Acceptance Criteria

1. WHEN a chunk is processed, THE System SHALL extract entities using spaCy NER for standard entity types
2. WHEN spaCy NER is insufficient, THE System SHALL use LLM-based extraction for domain-specific entities
3. THE System SHALL extract entity types: System, PaymentMode, Workflow, Rule, Field, Document, Section
4. WHEN an entity is extracted, THE System SHALL capture entity text, type, and source chunk reference
5. THE System SHALL normalize entity names to canonical forms (e.g., "NEFT" and "National Electronic Funds Transfer" → "NEFT")

### Requirement 7: Entity Resolution and Deduplication

**User Story:** As a developer, I want entities deduplicated across documents, so that the knowledge graph represents each concept once with all its mentions linked.

#### Acceptance Criteria

1. WHEN entities are extracted, THE System SHALL identify duplicate entities across chunks using fuzzy matching
2. WHEN duplicates are found, THE System SHALL merge them into a single canonical entity node
3. THE System SHALL create SAME_AS relationships between entity mentions and canonical nodes
4. WHEN merging entities, THE System SHALL preserve all source chunk references
5. THE System SHALL use similarity threshold of 0.85 for entity matching

### Requirement 8: Conflict Detection

**User Story:** As a user, I want automatic detection of conflicting information, so that I can identify inconsistencies across banking documents.

#### Acceptance Criteria

1. WHEN entities with the same name have contradictory properties, THE System SHALL create CONFLICTS_WITH relationships
2. WHEN rules or workflows contradict each other, THE System SHALL detect and flag the conflict
3. THE System SHALL store conflict metadata including conflict type and source chunks
4. WHEN a conflict is detected, THE System SHALL link both conflicting entities with bidirectional CONFLICTS_WITH edges
5. THE System SHALL use LLM-based analysis to identify semantic conflicts beyond exact contradictions

### Requirement 9: Knowledge Graph Population

**User Story:** As a developer, I want extracted entities and relationships stored in Neo4j, so that graph queries can traverse document knowledge.

#### Acceptance Criteria

1. WHEN entities are extracted, THE System SHALL create corresponding nodes in the Knowledge_Graph
2. THE System SHALL create node types: Document, Section, System, PaymentMode, Workflow, Rule, Field, Chunk
3. THE System SHALL create relationship types: CONTAINS, HAS_CHUNK, DEPENDS_ON, INTEGRATES_WITH, NEXT_STEP, APPLIES_TO, DEFINED_IN, CONFLICTS_WITH, USES, MENTIONS, SAME_AS
4. WHEN a relationship is identified, THE System SHALL create a directed edge with relationship metadata
5. THE Knowledge_Graph SHALL support Cypher queries for relationship traversal
6. WHEN populating the graph, THE System SHALL ensure referential integrity between nodes and edges

### Requirement 10: Query Intent Classification

**User Story:** As a user, I want my queries automatically classified by intent, so that the system routes them to the appropriate retrieval mode.

#### Acceptance Criteria

1. WHEN a query is received, THE Query_Router SHALL classify it into one of three modes: VECTOR, GRAPH, or HYBRID
2. WHEN a query asks factual/definitional questions about single concepts, THE Query_Router SHALL classify it as VECTOR mode
3. WHEN a query asks structural/relational/comparison questions, THE Query_Router SHALL classify it as GRAPH mode
4. WHEN a query requires both relationship context and full text, THE Query_Router SHALL classify it as HYBRID mode
5. THE Query_Router SHALL use LLM-based classification with confidence scoring
6. WHEN classification confidence is below 0.7, THE Query_Router SHALL default to HYBRID mode

### Requirement 11: Vector-Based Retrieval

**User Story:** As a user, I want semantic search across document chunks, so that I can find relevant information based on meaning rather than exact keywords.

#### Acceptance Criteria

1. WHEN a VECTOR mode query is processed, THE System SHALL generate a query embedding
2. WHEN a query embedding is generated, THE System SHALL search the Vector_Store for top-k similar chunks
3. THE System SHALL retrieve chunks with cosine similarity above 0.7 threshold
4. THE System SHALL return up to 10 chunks ranked by similarity score
5. WHEN retrieving chunks, THE System SHALL include full text and metadata from the Document_Store

### Requirement 12: Graph-Based Retrieval

**User Story:** As a user, I want to query relationships and dependencies in the knowledge graph, so that I can understand how banking systems and processes connect.

#### Acceptance Criteria

1. WHEN a GRAPH mode query is processed, THE System SHALL extract entity mentions from the query
2. WHEN entities are extracted, THE System SHALL construct a Cypher query to traverse relevant relationships
3. THE System SHALL execute graph traversal with configurable depth limits (default: 3 hops)
4. THE System SHALL return subgraphs containing relevant nodes and relationships
5. WHEN retrieving graph results, THE System SHALL include node properties and relationship types
6. THE System SHALL support traversal patterns: dependencies, integrations, workflows, conflicts

### Requirement 13: Hybrid Retrieval and Fusion

**User Story:** As a user, I want complex queries answered using both vector search and graph traversal, so that I get comprehensive results combining text and relationships.

#### Acceptance Criteria

1. WHEN a HYBRID mode query is processed, THE System SHALL execute vector retrieval and graph retrieval in parallel
2. WHEN both retrievals complete, THE System SHALL fuse results by deduplicating overlapping chunks
3. THE System SHALL merge graph facts with corresponding text chunks based on chunk references
4. THE System SHALL preserve both semantic relevance scores and graph relationship context
5. WHEN fusing results, THE System SHALL maintain ranking that balances vector similarity and graph centrality

### Requirement 14: Cross-Encoder Reranking

**User Story:** As a developer, I want retrieved results reranked by relevance, so that the most pertinent information appears first for LLM context.

#### Acceptance Criteria

1. WHEN retrieval results are obtained, THE Cross_Encoder SHALL score each result against the original query
2. THE Cross_Encoder SHALL use a transformer-based reranking model
3. THE System SHALL reorder results by cross-encoder scores in descending order
4. THE System SHALL retain top 5 results after reranking for LLM context
5. WHEN reranking, THE System SHALL preserve result metadata including source and retrieval mode

### Requirement 15: Context Assembly

**User Story:** As a developer, I want retrieved information assembled into structured context, so that the LLM receives both graph facts and supporting text.

#### Acceptance Criteria

1. WHEN results are reranked, THE System SHALL assemble context combining graph facts and text chunks
2. THE System SHALL format graph facts as structured statements (e.g., "System A DEPENDS_ON System B")
3. THE System SHALL include chunk text with breadcrumb context for grounding
4. THE System SHALL attach source citations to each piece of context (document, section, chunk_id)
5. THE System SHALL ensure total context does not exceed 4096 tokens
6. WHEN context exceeds token limit, THE System SHALL truncate lower-ranked results while preserving citations

### Requirement 16: Grounded LLM Response Generation

**User Story:** As a user, I want answers generated from retrieved context with citations, so that I can verify the information source and trust the response.

#### Acceptance Criteria

1. WHEN context is assembled, THE System SHALL send it to the LLM with the user query
2. THE System SHALL instruct the LLM to answer only using provided context
3. THE System SHALL instruct the LLM to include citations in the format [doc_id:section]
4. WHEN the LLM generates a response, THE System SHALL return it with inline citations
5. THE System SHALL use the configured LLM_MODEL from environment variables
6. WHEN the LLM cannot answer from context, THE System SHALL return "Insufficient information in documents" rather than hallucinating

### Requirement 17: Faithfulness Validation

**User Story:** As a developer, I want LLM responses validated for faithfulness to source context, so that hallucinations and unsupported claims are detected.

#### Acceptance Criteria

1. WHEN an LLM response is generated, THE Faithfulness_Validator SHALL check each claim against source context
2. THE Faithfulness_Validator SHALL use entailment checking to verify claims are supported
3. WHEN a claim is unsupported, THE System SHALL flag it with a warning
4. THE System SHALL compute a faithfulness score (0-1) indicating response grounding quality
5. WHEN faithfulness score is below 0.8, THE System SHALL include a warning with the response
6. THE Faithfulness_Validator SHALL identify specific unsupported claims in the response

### Requirement 18: Dependency and Impact Queries

**User Story:** As a user, I want to query system dependencies and impact chains, so that I can understand how changes propagate through banking systems.

#### Acceptance Criteria

1. WHEN a dependency query is received, THE System SHALL traverse DEPENDS_ON relationships in the Knowledge_Graph
2. THE System SHALL support forward traversal (what depends on X) and backward traversal (what X depends on)
3. THE System SHALL return dependency chains with all intermediate nodes
4. THE System SHALL include impact radius (number of affected systems) in results
5. WHEN traversing dependencies, THE System SHALL detect and handle circular dependencies

### Requirement 19: Cross-Document Comparison

**User Story:** As a user, I want to compare information across multiple documents, so that I can identify differences and commonalities in banking system specifications.

#### Acceptance Criteria

1. WHEN a comparison query is received, THE System SHALL retrieve relevant chunks from multiple documents
2. THE System SHALL group results by document for side-by-side comparison
3. THE System SHALL identify common entities mentioned across documents using SAME_AS relationships
4. THE System SHALL highlight differences in entity properties or relationships across documents
5. THE System SHALL present comparison results in structured format showing per-document information

### Requirement 20: Conflict Detection Queries

**User Story:** As a user, I want to query for conflicts and inconsistencies, so that I can identify and resolve contradictions in banking documentation.

#### Acceptance Criteria

1. WHEN a conflict query is received, THE System SHALL retrieve all CONFLICTS_WITH relationships from the Knowledge_Graph
2. THE System SHALL return conflicting entity pairs with conflict metadata
3. THE System SHALL include source chunks for both sides of each conflict
4. THE System SHALL categorize conflicts by type (property conflict, rule conflict, workflow conflict)
5. THE System SHALL rank conflicts by severity based on entity importance and conflict scope

### Requirement 21: Process Chain Traversal

**User Story:** As a user, I want to traverse workflow and process chains, so that I can understand end-to-end banking processes across systems.

#### Acceptance Criteria

1. WHEN a process query is received, THE System SHALL traverse NEXT_STEP relationships in the Knowledge_Graph
2. THE System SHALL construct complete process chains from start to end nodes
3. THE System SHALL include all intermediate steps with step metadata
4. THE System SHALL support branching workflows with conditional paths
5. WHEN a process chain is incomplete, THE System SHALL indicate missing steps or gaps

### Requirement 22: Risk Rule Analysis

**User Story:** As a user, I want to analyze risk rules and their applicability, so that I can understand which rules apply to specific banking scenarios.

#### Acceptance Criteria

1. WHEN a risk rule query is received, THE System SHALL retrieve Rule nodes from the Knowledge_Graph
2. THE System SHALL traverse APPLIES_TO relationships to find applicable entities
3. THE System SHALL return rules with their conditions, actions, and scope
4. THE System SHALL identify overlapping or conflicting rules for the same scenario
5. THE System SHALL rank rules by specificity and priority when multiple rules apply

### Requirement 23: Ingestion Pipeline Orchestration

**User Story:** As a system administrator, I want the ingestion pipeline to execute all processing steps in correct order, so that documents are fully indexed and ready for querying.

#### Acceptance Criteria

1. WHEN ingestion is triggered, THE Ingestion_Pipeline SHALL execute steps in order: parse, chunk, embed, index, extract, resolve, detect conflicts, populate graph
2. WHEN a step fails, THE Ingestion_Pipeline SHALL log the error and halt processing for that document
3. THE Ingestion_Pipeline SHALL support batch processing of multiple documents
4. THE Ingestion_Pipeline SHALL track processing status for each document
5. WHEN ingestion completes, THE System SHALL verify all storage layers are consistent

### Requirement 24: Query Pipeline Orchestration

**User Story:** As a developer, I want the query pipeline to execute all processing steps efficiently, so that users receive accurate responses with minimal latency.

#### Acceptance Criteria

1. WHEN a query is received, THE Query_Pipeline SHALL execute steps in order: classify, route, retrieve, fuse, rerank, assemble, generate, validate
2. THE Query_Pipeline SHALL execute parallel retrievals concurrently when in HYBRID mode
3. WHEN a step fails, THE Query_Pipeline SHALL return an error response with failure details
4. THE Query_Pipeline SHALL complete queries within 10 seconds for 95th percentile
5. THE Query_Pipeline SHALL log query execution metrics including latency per step

### Requirement 25: Configuration Management

**User Story:** As a system administrator, I want system configuration managed through environment variables, so that deployment settings can be changed without code modifications.

#### Acceptance Criteria

1. THE System SHALL read LLM configuration from OLLAMA_BASE_URL and LLM_MODEL environment variables
2. THE System SHALL support configuration of: embedding model, chunk sizes, retrieval limits, similarity thresholds
3. WHEN a required configuration is missing, THE System SHALL fail startup with a descriptive error
4. THE System SHALL validate configuration values at startup
5. THE System SHALL support configuration overrides for testing and development environments

### Requirement 26: Error Handling and Logging

**User Story:** As a developer, I want comprehensive error handling and logging, so that I can diagnose and resolve issues in production.

#### Acceptance Criteria

1. WHEN an error occurs, THE System SHALL log it with severity level, timestamp, and context
2. THE System SHALL distinguish between recoverable errors (warnings) and fatal errors
3. THE System SHALL include stack traces for unexpected exceptions
4. THE System SHALL log query execution traces for debugging
5. WHEN external services fail (Qdrant, Neo4j, PostgreSQL), THE System SHALL log connection errors with retry information

### Requirement 27: API Interface

**User Story:** As a client application, I want a REST API to submit queries and receive responses, so that I can integrate the Graph RAG Layer into banking applications.

#### Acceptance Criteria

1. THE System SHALL expose a POST /query endpoint accepting JSON query requests
2. WHEN a query request is received, THE System SHALL validate required fields: query_text
3. THE System SHALL support optional parameters: mode (VECTOR/GRAPH/HYBRID), top_k, max_depth
4. THE System SHALL return JSON responses with: answer, citations, faithfulness_score, retrieval_mode
5. WHEN a request is malformed, THE System SHALL return HTTP 400 with error details
6. THE System SHALL return HTTP 500 for internal errors with error messages

### Requirement 28: Ingestion API Interface

**User Story:** As a system administrator, I want an API to trigger document ingestion, so that I can add new documents to the system programmatically.

#### Acceptance Criteria

1. THE System SHALL expose a POST /ingest endpoint accepting document uploads
2. WHEN a document is uploaded, THE System SHALL validate file format (.docx or .pdf)
3. THE System SHALL return an ingestion job ID for tracking
4. THE System SHALL expose a GET /ingest/{job_id} endpoint for status checking
5. WHEN ingestion completes, THE System SHALL return success status with document_id
6. WHEN ingestion fails, THE System SHALL return error status with failure reason
