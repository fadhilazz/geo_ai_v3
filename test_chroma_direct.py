#!/usr/bin/env python3
"""Test direct Chroma access to debug search issues."""

import chromadb
from pathlib import Path

def test_chroma_direct():
    """Test direct access to Chroma collections."""
    
    # Connect to text collection
    print("🔍 Testing direct Chroma access...")
    
    text_dir = Path("knowledge/text_emb")
    client = chromadb.PersistentClient(path=str(text_dir))
    
    try:
        collection = client.get_collection(name="text_emb")
        print(f"✅ Connected to text collection with {collection.count()} items")
        
        # Get a few sample documents
        print("\n📚 Sample documents:")
        results = collection.get(limit=3, include=['documents', 'metadatas'])
        
        for i, (doc, meta) in enumerate(zip(results['documents'], results['metadatas'])):
            print(f"\n{i+1}. Doc ID: {meta.get('doc_id', 'unknown')}")
            print(f"   File: {meta.get('filename', 'unknown')}")
            print(f"   Page: {meta.get('page', 0)}")
            print(f"   Field: {meta.get('field', 'none')}")
            print(f"   Text: {doc[:200]}...")
            
        # Try simple query without embedding
        print("\n🔍 Testing query with 'geothermometer':")
        
        # Search by metadata only (no embedding)
        try:
            meta_results = collection.get(
                where={"filename": {"$ne": ""}},  # Get all with filename
                limit=10,
                include=['documents', 'metadatas']
            )
            
            geothermo_docs = []
            for doc, meta in zip(meta_results['documents'], meta_results['metadatas']):
                if 'geothermo' in doc.lower():
                    geothermo_docs.append((doc, meta))
                    
            print(f"   Found {len(geothermo_docs)} documents containing 'geothermo'")
            
            for i, (doc, meta) in enumerate(geothermo_docs[:2]):
                print(f"\n   {i+1}. {meta.get('filename', 'unknown')} (page {meta.get('page', 0)})")
                print(f"      Text: {doc[:300]}...")
                
        except Exception as e:
            print(f"   ❌ Metadata search error: {e}")
            
        # Try embedding search
        print(f"\n🧠 Testing embedding search:")
        try:
            # Simple query
            search_results = collection.query(
                query_texts=["geothermometer temperature"],
                n_results=3,
                include=['documents', 'metadatas', 'distances']
            )
            
            if search_results['documents'] and search_results['documents'][0]:
                print(f"   ✅ Found {len(search_results['documents'][0])} results")
                for i, (doc, meta, dist) in enumerate(zip(
                    search_results['documents'][0],
                    search_results['metadatas'][0], 
                    search_results['distances'][0]
                )):
                    print(f"\n   {i+1}. Distance: {dist:.3f}")
                    print(f"      File: {meta.get('filename', 'unknown')} (page {meta.get('page', 0)})")
                    print(f"      Text: {doc[:200]}...")
            else:
                print("   ❌ No embedding search results")
                
        except Exception as e:
            print(f"   ❌ Embedding search error: {e}")
            
    except Exception as e:
        print(f"❌ Error connecting to collection: {e}")

if __name__ == "__main__":
    test_chroma_direct()
