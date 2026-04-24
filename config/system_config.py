"""System configuration management using environment variables"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SystemConfig:
    """System configuration loaded from environment variables"""
    
    # LLM Configuration
    ollama_base_url: str
    llm_model: str
    
    # Embedding Configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # Chunking Configuration
    parent_chunk_size: int = 2048
    child_chunk_size: int = 512
    chunk_overlap: int = 50
    
    # Retrieval Configuration
    vector_top_k: int = 10
    bm25_top_k: int = 10
    rerank_top_k: int = 5
    similarity_threshold: float = 0.7
    
    # Graph Configuration
    max_graph_depth: int = 3
    entity_similarity_threshold: float = 0.85
    
    # Context Configuration
    max_context_tokens: int = 4096
    faithfulness_threshold: float = 0.8
    
    # Storage Configuration
    qdrant_url: str = "http://localhost:6333"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    postgres_connection_string: str = ""
    
    @classmethod
    def from_env(cls) -> "SystemConfig":
        """
        Load configuration from environment variables.
        
        Raises:
            ValueError: If required configuration is missing
        """
        # Required configurations
        ollama_base_url = os.getenv("OLLAMA_BASE_URL")
        if not ollama_base_url:
            raise ValueError("OLLAMA_BASE_URL environment variable is required")
        
        llm_model = os.getenv("LLM_MODEL")
        if not llm_model:
            raise ValueError("LLM_MODEL environment variable is required")
        
        # Optional configurations with defaults
        config = cls(
            ollama_base_url=ollama_base_url,
            llm_model=llm_model,
            embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            embedding_dimension=int(os.getenv("EMBEDDING_DIMENSION", "384")),
            parent_chunk_size=int(os.getenv("PARENT_CHUNK_SIZE", "2048")),
            child_chunk_size=int(os.getenv("CHILD_CHUNK_SIZE", "512")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
            vector_top_k=int(os.getenv("VECTOR_TOP_K", "10")),
            bm25_top_k=int(os.getenv("BM25_TOP_K", "10")),
            rerank_top_k=int(os.getenv("RERANK_TOP_K", "5")),
            similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.7")),
            max_graph_depth=int(os.getenv("MAX_GRAPH_DEPTH", "3")),
            entity_similarity_threshold=float(os.getenv("ENTITY_SIMILARITY_THRESHOLD", "0.85")),
            max_context_tokens=int(os.getenv("MAX_CONTEXT_TOKENS", "4096")),
            faithfulness_threshold=float(os.getenv("FAITHFULNESS_THRESHOLD", "0.8")),
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD", ""),
            postgres_connection_string=os.getenv(
                "POSTGRES_CONNECTION_STRING",
                "postgresql://postgres:postgres@localhost:5432/graph_rag"
            ),
        )
        
        # Validate configuration
        config.validate()
        
        return config
    
    def validate(self) -> None:
        """
        Validate configuration values.
        
        Raises:
            ValueError: If configuration values are invalid
        """
        if self.parent_chunk_size <= 0:
            raise ValueError("parent_chunk_size must be positive")
        
        if self.child_chunk_size <= 0:
            raise ValueError("child_chunk_size must be positive")
        
        if self.child_chunk_size > self.parent_chunk_size:
            raise ValueError("child_chunk_size cannot exceed parent_chunk_size")
        
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        
        if not 0 <= self.similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")
        
        if not 0 <= self.entity_similarity_threshold <= 1:
            raise ValueError("entity_similarity_threshold must be between 0 and 1")
        
        if not 0 <= self.faithfulness_threshold <= 1:
            raise ValueError("faithfulness_threshold must be between 0 and 1")
        
        if self.max_graph_depth <= 0:
            raise ValueError("max_graph_depth must be positive")
        
        if self.vector_top_k <= 0:
            raise ValueError("vector_top_k must be positive")
        
        if self.rerank_top_k <= 0:
            raise ValueError("rerank_top_k must be positive")
        
        if self.max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be positive")
