"""Integration tests for EmbeddingGenerator with real model"""

import pytest
import numpy as np
from src.embedding.embedding_generator import EmbeddingGenerator


class TestEmbeddingGeneratorIntegration:
    """Integration tests for EmbeddingGenerator with real sentence-transformers model"""
    
    @pytest.fixture(scope="class")
    def generator(self):
        """Create a single EmbeddingGenerator instance for all tests in this class"""
        return EmbeddingGenerator()
    
    def test_banking_domain_embeddings(self, generator):
        """Test embeddings for banking domain texts"""
        texts = [
            "NEFT is the National Electronic Funds Transfer system",
            "RTGS is used for high-value transactions",
            "IMPS enables instant money transfer",
            "Core Banking System manages customer accounts"
        ]
        
        embeddings = generator.batch_generate(texts)
        
        # All embeddings should be normalized
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)
        
        # NEFT and RTGS should be more similar to each other than to Core Banking
        neft_rtgs_sim = np.dot(embeddings[0], embeddings[1])
        neft_cbs_sim = np.dot(embeddings[0], embeddings[3])
        
        assert neft_rtgs_sim > neft_cbs_sim
    
    def test_semantic_similarity_payment_systems(self, generator):
        """Test semantic similarity between payment system descriptions"""
        text1 = "NEFT is a payment system for electronic fund transfers"
        text2 = "Electronic fund transfer system called NEFT"
        text3 = "Weather forecast for tomorrow"
        
        emb1 = generator.generate(text1)
        emb2 = generator.generate(text2)
        emb3 = generator.generate(text3)
        
        # Similar texts should have high similarity
        sim_12 = np.dot(emb1, emb2)
        assert sim_12 > 0.7, f"Similar texts should have high similarity, got {sim_12}"
        
        # Dissimilar texts should have lower similarity
        sim_13 = np.dot(emb1, emb3)
        assert sim_13 < sim_12, "Dissimilar texts should have lower similarity"
    
    def test_acronym_expansion_similarity(self, generator):
        """Test that acronyms and their expansions have similar embeddings"""
        acronym = "NEFT transaction"
        expansion = "National Electronic Funds Transfer transaction"
        
        emb_acronym = generator.generate(acronym)
        emb_expansion = generator.generate(expansion)
        
        similarity = np.dot(emb_acronym, emb_expansion)
        
        # Should have reasonable similarity (> 0.5)
        assert similarity > 0.5, f"Acronym and expansion should be similar, got {similarity}"
    
    def test_batch_processing_efficiency(self, generator):
        """Test that batch processing handles multiple texts efficiently"""
        # Create a batch of 50 texts
        texts = [f"Banking document chunk number {i} about payment systems" for i in range(50)]
        
        embeddings = generator.batch_generate(texts)
        
        assert embeddings.shape == (50, 384)
        
        # All should be normalized
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)
    
    def test_hierarchical_chunk_embeddings(self, generator):
        """Test embeddings for hierarchical chunks with breadcrumbs"""
        parent_chunk = "Section 3: Payment Systems. This section describes various payment systems including NEFT, RTGS, and IMPS."
        child_chunk = "NEFT (National Electronic Funds Transfer) is used for retail payments up to 2 lakhs."
        
        parent_emb = generator.generate(parent_chunk)
        child_emb = generator.generate(child_chunk)
        
        # Parent and child should have some similarity
        similarity = np.dot(parent_emb, child_emb)
        assert similarity > 0.3, "Parent and child chunks should have some similarity"
    
    def test_document_metadata_in_text(self, generator):
        """Test embeddings with document metadata included"""
        text_with_metadata = "Document: Banking FSD > Section 2.1 > NEFT System. Content: NEFT handles retail payments."
        text_without_metadata = "NEFT handles retail payments."
        
        emb_with = generator.generate(text_with_metadata)
        emb_without = generator.generate(text_without_metadata)
        
        # Should still have high similarity despite metadata
        similarity = np.dot(emb_with, emb_without)
        assert similarity > 0.6, "Metadata should not drastically change embedding"
    
    def test_technical_banking_terms(self, generator):
        """Test embeddings for technical banking terminology"""
        texts = [
            "IFSC code is used for bank identification",
            "MICR code is printed on cheques",
            "SWIFT code is for international transfers",
            "Account number uniquely identifies customer account"
        ]
        
        embeddings = generator.batch_generate(texts)
        
        # All should be normalized
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)
        
        # IFSC and MICR (both bank codes) should be more similar than IFSC and Account number
        ifsc_micr_sim = np.dot(embeddings[0], embeddings[1])
        ifsc_account_sim = np.dot(embeddings[0], embeddings[3])
        
        assert ifsc_micr_sim > ifsc_account_sim
    
    def test_numerical_values_in_text(self, generator):
        """Test handling of numerical values in banking text"""
        texts = [
            "Transaction limit is 2 lakhs",
            "Transaction limit is 5 lakhs",
            "Transaction limit is 10 lakhs"
        ]
        
        embeddings = generator.batch_generate(texts)
        
        # All should be normalized
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)
        
        # All three should be very similar (same concept, different values)
        sim_01 = np.dot(embeddings[0], embeddings[1])
        sim_12 = np.dot(embeddings[1], embeddings[2])
        
        assert sim_01 > 0.8, "Similar texts with different numbers should be similar"
        assert sim_12 > 0.8, "Similar texts with different numbers should be similar"
    
    def test_negation_handling(self, generator):
        """Test that negation affects embeddings"""
        positive = "NEFT is available for retail payments"
        negative = "NEFT is not available for retail payments"
        
        emb_pos = generator.generate(positive)
        emb_neg = generator.generate(negative)
        
        similarity = np.dot(emb_pos, emb_neg)
        
        # Should still be somewhat similar (same topic) but not identical
        assert 0.5 < similarity < 0.95, f"Negation should affect similarity, got {similarity}"
    
    def test_query_vs_document_embeddings(self, generator):
        """Test embeddings for queries vs document chunks"""
        query = "What is NEFT?"
        document = "NEFT (National Electronic Funds Transfer) is a nation-wide payment system facilitating one-to-one funds transfer."
        
        query_emb = generator.generate(query)
        doc_emb = generator.generate(document)
        
        similarity = np.dot(query_emb, doc_emb)
        
        # Query and relevant document should have reasonable similarity
        assert similarity > 0.4, f"Query and relevant document should be similar, got {similarity}"
    
    def test_multilingual_text_handling(self, generator):
        """Test handling of text with mixed languages (English + Hindi/regional)"""
        text = "NEFT system processes ₹2 lakh transactions"
        
        embedding = generator.generate(text)
        
        assert embedding.shape == (384,)
        norm = np.linalg.norm(embedding)
        assert np.isclose(norm, 1.0, atol=1e-6)
    
    def test_embedding_stability_across_calls(self, generator):
        """Test that embeddings are stable across multiple calls"""
        text = "Stable embedding test for banking system"
        
        embeddings = [generator.generate(text) for _ in range(5)]
        
        # All embeddings should be identical
        for i in range(1, 5):
            assert np.allclose(embeddings[0], embeddings[i], atol=1e-6)
    
    def test_real_world_chunk_sizes(self, generator):
        """Test with realistic chunk sizes (up to 512 tokens)"""
        # Create a realistic banking document chunk
        chunk = """
        Section 3.2: NEFT Payment System
        
        The National Electronic Funds Transfer (NEFT) system is a nation-wide payment system 
        facilitating one-to-one funds transfer. Under this Scheme, individuals, firms and 
        corporates can electronically transfer funds from any bank branch to any individual, 
        firm or corporate having an account with any other bank branch in the country 
        participating in the Scheme.
        
        Key Features:
        - Available 24x7 throughout the year
        - Settled in batches
        - Transaction limit: Up to ₹2 lakhs for retail
        - Charges: Nominal charges apply
        - Settlement: Half-hourly batches
        
        Integration Points:
        - Core Banking System (CBS)
        - Payment Gateway
        - Customer Portal
        """ * 3  # Repeat to make it longer
        
        embedding = generator.generate(chunk)
        
        assert embedding.shape == (384,)
        norm = np.linalg.norm(embedding)
        assert np.isclose(norm, 1.0, atol=1e-6)
