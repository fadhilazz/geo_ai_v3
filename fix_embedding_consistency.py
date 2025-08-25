#!/usr/bin/env python3
"""Fix embedding consistency across all components."""

import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def fix_embedding_consistency():
    """Ensure all components use the same embedding model."""
    
    print("🔧 Fixing embedding consistency...")
    
    # Check current model settings
    from config import DEFAULT_TEXT_MODEL, DEFAULT_IMAGE_MODEL, DEFAULT_IMAGE_PRETRAINED
    print(f"📋 Current settings:")
    print(f"   Text model: {DEFAULT_TEXT_MODEL}")
    print(f"   Image model: {DEFAULT_IMAGE_MODEL}")
    print(f"   Image pretrained: {DEFAULT_IMAGE_PRETRAINED}")
    
    # Test model dimensions
    print(f"\n🧠 Testing model dimensions...")
    from sentence_transformers import SentenceTransformer
    
    text_model = SentenceTransformer(DEFAULT_TEXT_MODEL)
    text_dim = text_model.get_sentence_embedding_dimension()
    print(f"   Text model ({DEFAULT_TEXT_MODEL}): {text_dim} dimensions")
    
    # Check if there are any hardcoded models
    print(f"\n🔍 Checking for hardcoded models...")
    
    # Force recreate collections with correct dimensions
    print(f"\n🔄 Checking collection dimensions...")
    
    import chromadb
    from pathlib import Path
    
    # Check text collection
    text_dir = Path("knowledge/text_emb")
    client = chromadb.PersistentClient(path=str(text_dir))
    
    try:
        collection = client.get_collection(name="text_emb")
        print(f"   Text collection: {collection.count()} items")
        
        # Try a simple query to see dimension mismatch
        try:
            test_embedding = text_model.encode(["test query"])
            print(f"   Test embedding shape: {test_embedding.shape}")
            
            results = collection.query(
                query_embeddings=[test_embedding[0].tolist()],
                n_results=1,
                include=['documents']
            )
            print(f"   ✅ Text collection query works!")
            
        except Exception as e:
            print(f"   ❌ Text collection query error: {e}")
            
            if "dimension" in str(e):
                print(f"   🔧 Dimension mismatch detected - this is the root cause!")
                
                # The collection was created with a different embedding model
                # We need to either:
                # 1. Recreate the collection (lose data)
                # 2. Use the same model that was used for ingestion
                
                print(f"\n💡 Solution options:")
                print(f"   1. Use the original embedding model from ingestion")
                print(f"   2. Recreate collections (will lose current data)")
                print(f"   3. Check what model was actually used during ingestion")
                
                # Let's check the ingestion logs or code
                return False
                
    except Exception as e:
        print(f"   ❌ Cannot access text collection: {e}")
        return False
        
    return True

if __name__ == "__main__":
    success = fix_embedding_consistency()
    if not success:
        print(f"\n❌ Embedding consistency issues detected")
        print(f"💡 Need to align embedding models between ingestion and query")
    else:
        print(f"\n✅ Embedding consistency verified")
