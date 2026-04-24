"""Example usage of EmbeddingGenerator for banking documents"""

import numpy as np
from src.embedding.embedding_generator import EmbeddingGenerator


def main():
    """Demonstrate EmbeddingGenerator usage"""
    
    print("=" * 80)
    print("EmbeddingGenerator Example - Banking Document Embeddings")
    print("=" * 80)
    
    # Initialize the generator
    print("\n1. Initializing EmbeddingGenerator...")
    generator = EmbeddingGenerator()
    print(f"   Model: {generator.model_name}")
    print(f"   Embedding dimension: {generator.embedding_dimension}")
    
    # Generate single embedding
    print("\n2. Generating single embedding...")
    text = "NEFT is the National Electronic Funds Transfer system used for retail payments."
    embedding = generator.generate(text)
    print(f"   Text: {text}")
    print(f"   Embedding shape: {embedding.shape}")
    print(f"   L2 norm: {np.linalg.norm(embedding):.6f}")
    print(f"   First 5 values: {embedding[:5]}")
    
    # Generate batch embeddings
    print("\n3. Generating batch embeddings...")
    texts = [
        "NEFT is used for retail electronic fund transfers",
        "RTGS handles high-value real-time gross settlement",
        "IMPS enables instant interbank mobile payment service",
        "Core Banking System manages customer accounts and transactions"
    ]
    
    embeddings = generator.batch_generate(texts)
    print(f"   Number of texts: {len(texts)}")
    print(f"   Embeddings shape: {embeddings.shape}")
    print(f"   All L2 normalized: {np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)}")
    
    # Compute similarity matrix
    print("\n4. Computing similarity matrix...")
    print("   Cosine similarities between texts:")
    print("   " + " " * 10 + "NEFT    RTGS    IMPS    CBS")
    
    for i, text in enumerate(texts):
        similarities = [np.dot(embeddings[i], embeddings[j]) for j in range(len(texts))]
        label = ["NEFT", "RTGS", "IMPS", "CBS"][i]
        print(f"   {label:8s} {similarities[0]:7.4f} {similarities[1]:7.4f} {similarities[2]:7.4f} {similarities[3]:7.4f}")
    
    # Demonstrate semantic search
    print("\n5. Semantic search example...")
    query = "What payment system is used for instant transfers?"
    query_embedding = generator.generate(query)
    
    print(f"   Query: {query}")
    print("   Similarities with documents:")
    
    for i, text in enumerate(texts):
        similarity = np.dot(query_embedding, embeddings[i])
        print(f"   [{i}] {similarity:.4f} - {text[:60]}...")
    
    # Find most similar document
    similarities = [np.dot(query_embedding, embeddings[i]) for i in range(len(texts))]
    most_similar_idx = np.argmax(similarities)
    print(f"\n   Most relevant document: [{most_similar_idx}] {texts[most_similar_idx]}")
    
    # Demonstrate consistency
    print("\n6. Testing embedding consistency...")
    test_text = "Testing consistency of embeddings"
    emb1 = generator.generate(test_text)
    emb2 = generator.generate(test_text)
    
    print(f"   Text: {test_text}")
    print(f"   Embeddings are identical: {np.allclose(emb1, emb2)}")
    print(f"   Max difference: {np.max(np.abs(emb1 - emb2)):.10f}")
    
    # Demonstrate with hierarchical chunks
    print("\n7. Hierarchical chunk example...")
    parent_chunk = "Section 3: Payment Systems. This section covers NEFT, RTGS, and IMPS payment systems."
    child_chunk = "NEFT (National Electronic Funds Transfer) processes retail payments up to ₹2 lakhs."
    
    parent_emb = generator.generate(parent_chunk)
    child_emb = generator.generate(child_chunk)
    
    similarity = np.dot(parent_emb, child_emb)
    print(f"   Parent chunk: {parent_chunk[:60]}...")
    print(f"   Child chunk: {child_chunk[:60]}...")
    print(f"   Similarity: {similarity:.4f}")
    
    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
