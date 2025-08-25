#!/usr/bin/env python3
"""Test RAG directly without API."""

import os
import sys
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Add src to path
sys.path.insert(0, 'src')

def test_direct_rag():
    """Test RAG components directly."""
    
    print("🔍 Testing RAG components directly...")
    
    # Test 1: Direct Chroma access
    print("\n1. 📚 Testing direct Chroma access...")
    text_dir = Path("knowledge/text_emb")
    client = chromadb.PersistentClient(path=str(text_dir))
    collection = client.get_collection(name="text_emb")
    print(f"   ✅ Collection has {collection.count()} items")
    
    # Test 2: Model loading
    print("\n2. 🧠 Testing model loading...")
    model = SentenceTransformer("intfloat/e5-large-v2")
    print(f"   ✅ Model loaded with {model.get_sentence_embedding_dimension()} dimensions")
    
    # Test 3: Embedding generation
    print("\n3. 🔢 Testing embedding generation...")
    query = "geothermometer reservoir temperature"
    embedding = model.encode([query])
    print(f"   ✅ Generated embedding shape: {embedding.shape}")
    
    # Test 4: Direct Chroma query
    print("\n4. 🔍 Testing direct Chroma query...")
    try:
        results = collection.query(
            query_embeddings=[embedding[0].tolist()],
            n_results=5,
            include=['documents', 'metadatas', 'distances']
        )
        
        if results['documents'] and results['documents'][0]:
            print(f"   ✅ Found {len(results['documents'][0])} results!")
            
            # Check for Semurup data
            semurup_results = []
            for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
                if 'semurup' in doc.lower() or 'semurup' in str(meta).lower():
                    semurup_results.append((doc, meta, dist))
                    
            print(f"   📍 Found {len(semurup_results)} Semurup-related results")
            
            for i, (doc, meta, dist) in enumerate(semurup_results[:2]):
                print(f"\n   {i+1}. Distance: {dist:.3f}")
                print(f"      File: {meta.get('filename', 'unknown')}")
                print(f"      Text: {doc[:200]}...")
                
            # Check for geothermometer data
            geothermo_results = []
            for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
                if 'geotherm' in doc.lower():
                    geothermo_results.append((doc, meta, dist))
                    
            print(f"   🌡️ Found {len(geothermo_results)} geothermometer results")
            
            for i, (doc, meta, dist) in enumerate(geothermo_results[:2]):
                print(f"\n   {i+1}. Distance: {dist:.3f}")
                print(f"      File: {meta.get('filename', 'unknown')}")
                print(f"      Text: {doc[:200]}...")
                
        else:
            print("   ❌ No results from direct query")
            
    except Exception as e:
        print(f"   ❌ Direct query error: {e}")
        
    # Test 5: RAG class initialization
    print("\n5. 🏗️ Testing RAG class...")
    try:
        from config import CHROMA_TEXT_DIR_OBJ
        from tools.rag import TextStore
        
        # Initialize TextStore
        text_store = TextStore(chroma_dir=CHROMA_TEXT_DIR_OBJ)
        print(f"   ✅ TextStore initialized")
        print(f"   📊 Collection count: {text_store.count()}")
        
        # Test search
        results = text_store.search_similar("geothermometer temperature", top_k=3)
        print(f"   🔍 Search results: {len(results)} chunks")
        
        for i, chunk in enumerate(results[:2]):
            print(f"\n   {i+1}. Score: {chunk.get('score', 'N/A')}")
            print(f"      Text: {chunk.get('text', '')[:200]}...")
            
    except Exception as e:
        print(f"   ❌ RAG class error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_direct_rag()
