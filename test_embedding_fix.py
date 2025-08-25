#!/usr/bin/env python3
"""Test embedding dimension fix."""

import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer

def test_embedding_fix():
    """Test embedding with correct model."""
    
    print("🔍 Testing embedding dimension fix...")
    
    # Use the same model as ingestion
    model = SentenceTransformer("intfloat/e5-large-v2")
    print(f"✅ Model loaded: {model}")
    
    # Test embedding
    test_query = "reservoir temperature geothermometer"
    embedding = model.encode([test_query])
    print(f"✅ Embedding shape: {embedding.shape}")
    print(f"   Dimensions: {embedding.shape[1]}")
    
    # Connect to Chroma
    text_dir = Path("knowledge/text_emb")
    client = chromadb.PersistentClient(path=str(text_dir))
    
    try:
        collection = client.get_collection(name="text_emb")
        print(f"✅ Connected to collection with {collection.count()} items")
        
        # Try search with correct embedding
        print("\n🧠 Testing search with correct embedding...")
        results = collection.query(
            query_embeddings=[embedding[0].tolist()],  # Convert to list
            n_results=5,
            include=['documents', 'metadatas', 'distances']
        )
        
        if results['documents'] and results['documents'][0]:
            print(f"✅ SUCCESS! Found {len(results['documents'][0])} results")
            
            for i, (doc, meta, dist) in enumerate(zip(
                results['documents'][0][:3],
                results['metadatas'][0][:3], 
                results['distances'][0][:3]
            )):
                print(f"\n{i+1}. Distance: {dist:.3f}")
                print(f"   File: {meta.get('filename', 'unknown')}")
                print(f"   Page: {meta.get('page', 0)}")
                print(f"   Text: {doc[:200]}...")
                
                # Check if it contains temperature data
                if any(term in doc.lower() for term in ['geotherm', 'temperature', '°c', 'reservoir']):
                    print(f"   🌡️ CONTAINS TEMPERATURE DATA!")
        else:
            print("❌ Still no results")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_embedding_fix()
