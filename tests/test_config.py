"""Tests for system configuration"""

import os
import pytest
from config.system_config import SystemConfig


class TestSystemConfig:
    """Test SystemConfig dataclass"""
    
    def test_from_env_with_required_vars(self, monkeypatch):
        """Test loading configuration from environment variables"""
        # Set required environment variables
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
        monkeypatch.setenv("LLM_MODEL", "llama2")
        
        config = SystemConfig.from_env()
        
        assert config.ollama_base_url == "http://localhost:11434"
        assert config.llm_model == "llama2"
        assert config.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.child_chunk_size == 512
        assert config.parent_chunk_size == 2048
    
    def test_from_env_missing_ollama_url(self, monkeypatch):
        """Test that missing OLLAMA_BASE_URL raises ValueError"""
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.setenv("LLM_MODEL", "llama2")
        
        with pytest.raises(ValueError, match="OLLAMA_BASE_URL"):
            SystemConfig.from_env()
    
    def test_from_env_missing_llm_model(self, monkeypatch):
        """Test that missing LLM_MODEL raises ValueError"""
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
        monkeypatch.delenv("LLM_MODEL", raising=False)
        
        with pytest.raises(ValueError, match="LLM_MODEL"):
            SystemConfig.from_env()
    
    def test_validate_invalid_chunk_sizes(self):
        """Test validation catches invalid chunk sizes"""
        config = SystemConfig(
            ollama_base_url="http://localhost:11434",
            llm_model="llama2",
            child_chunk_size=2048,
            parent_chunk_size=512  # Invalid: child > parent
        )
        
        with pytest.raises(ValueError, match="child_chunk_size cannot exceed parent_chunk_size"):
            config.validate()
    
    def test_validate_invalid_similarity_threshold(self):
        """Test validation catches invalid similarity threshold"""
        config = SystemConfig(
            ollama_base_url="http://localhost:11434",
            llm_model="llama2",
            similarity_threshold=1.5  # Invalid: > 1
        )
        
        with pytest.raises(ValueError, match="similarity_threshold must be between 0 and 1"):
            config.validate()
    
    def test_validate_negative_chunk_overlap(self):
        """Test validation catches negative chunk overlap"""
        config = SystemConfig(
            ollama_base_url="http://localhost:11434",
            llm_model="llama2",
            chunk_overlap=-10  # Invalid: negative
        )
        
        with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
            config.validate()
    
    def test_validate_zero_max_graph_depth(self):
        """Test validation catches zero max_graph_depth"""
        config = SystemConfig(
            ollama_base_url="http://localhost:11434",
            llm_model="llama2",
            max_graph_depth=0  # Invalid: must be positive
        )
        
        with pytest.raises(ValueError, match="max_graph_depth must be positive"):
            config.validate()
    
    def test_custom_env_values(self, monkeypatch):
        """Test loading custom configuration values from environment"""
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
        monkeypatch.setenv("LLM_MODEL", "llama2")
        monkeypatch.setenv("CHILD_CHUNK_SIZE", "256")
        monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.8")
        monkeypatch.setenv("MAX_GRAPH_DEPTH", "5")
        
        config = SystemConfig.from_env()
        
        assert config.child_chunk_size == 256
        assert config.similarity_threshold == 0.8
        assert config.max_graph_depth == 5
