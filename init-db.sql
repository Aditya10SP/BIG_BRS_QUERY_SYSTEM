-- Initialize Graph RAG PostgreSQL Database
-- This script is automatically run when the PostgreSQL container starts

-- Create documents table
CREATE TABLE IF NOT EXISTS documents (
    doc_id VARCHAR(255) PRIMARY KEY,
    title TEXT NOT NULL,
    file_path TEXT,
    file_type VARCHAR(10),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create chunks table with foreign key to documents
CREATE TABLE IF NOT EXISTS chunks (
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

-- Create indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_parent ON chunks(parent_chunk_id);
CREATE INDEX IF NOT EXISTS idx_chunks_section ON chunks(section);

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
