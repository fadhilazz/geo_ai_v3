#!/usr/bin/env python3
"""Final comprehensive RAG debugging."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def comprehensive_rag_debug():
    """Debug RAG system comprehensively."""
    
    print("🔍 COMPREHENSIVE RAG DEBUGGING...")
    
    # Test 1: Direct Chroma access
    print("\n1. 📚 Testing direct Chroma access...")
    import chromadb
    from pathlib import Path
    from sentence_transformers import SentenceTransformer
    
    text_dir = Path("knowledge/text_emb")
    client = chromadb.PersistentClient(path=str(text_dir))
    collection = client.get_collection(name="text_emb")
    print(f"   Collection: {collection.count()} items")
    
    # Test 2: Model consistency
    print("\n2. 🧠 Testing model consistency...")
    model = SentenceTransformer("intfloat/e5-large-v2")
    test_query = "geothermometer temperature Semurup"
    embedding = model.encode([test_query])
    print(f"   Query embedding shape: {embedding.shape}")
    
    # Test 3: Direct search
    print("\n3. 🔍 Testing direct search...")
    try:
        results = collection.query(
            query_embeddings=[embedding[0].tolist()],
            n_results=10,
            include=['documents', 'metadatas', 'distances']
        )
        
        if results['documents'] and results['documents'][0]:
            print(f"   ✅ Found {len(results['documents'][0])} results")
            
            # Look for Semurup and geothermometer data
            semurup_count = 0
            geothermo_count = 0
            
            for i, (doc, meta, dist) in enumerate(zip(
                results['documents'][0], 
                results['metadatas'][0], 
                results['distances'][0]
            )):
                if 'semurup' in doc.lower() or 'semurup' in str(meta).lower():
                    semurup_count += 1
                    print(f"\n   🎯 SEMURUP MATCH #{semurup_count} (distance: {dist:.3f})")
                    print(f"      File: {meta.get('filename', 'unknown')}")
                    print(f"      Page: {meta.get('page', 0)}")
                    print(f"      Field: {meta.get('field', 'none')}")
                    print(f"      Text: {doc[:200]}...")
                    
                if 'geotherm' in doc.lower():
                    geothermo_count += 1
                    if geothermo_count <= 2:  # Show first 2
                        print(f"\n   🌡️ GEOTHERMO MATCH #{geothermo_count} (distance: {dist:.3f})")
                        print(f"      File: {meta.get('filename', 'unknown')}")
                        print(f"      Text: {doc[:200]}...")
                        
            print(f"\n   📊 Summary: {semurup_count} Semurup matches, {geothermo_count} geothermo matches")
            
        else:
            print("   ❌ No results from direct search")
            
    except Exception as e:
        print(f"   ❌ Direct search error: {e}")
        
    # Test 4: RAG class search
    print("\n4. 🏗️ Testing RAG classes...")
    try:
        from tools.rag_text import get_text_rag
        
        text_rag = get_text_rag()
        print(f"   TextRAG initialized")
        
        # Test without filters
        chunks_no_filter = text_rag.search("geothermometer temperature Semurup", where=None, top_k=5)
        print(f"   Search without filters: {len(chunks_no_filter)} chunks")
        
        for i, chunk in enumerate(chunks_no_filter[:3]):
            print(f"   {i+1}. {chunk.filename} (page {chunk.page}, score: {chunk.score:.3f})")
            print(f"      Text: {chunk.text[:150]}...")
            
        # Test with simple field filter
        simple_filter = {"field": "Semurup"}
        chunks_with_filter = text_rag.search("geothermometer temperature", where=simple_filter, top_k=5)
        print(f"   Search with field filter: {len(chunks_with_filter)} chunks")
        
        # Test with fallback
        chunks_fallback = text_rag.search_with_fallback("geothermometer temperature Semurup", where=simple_filter, top_k=5)
        print(f"   Search with fallback: {len(chunks_fallback)} chunks")
        
    except Exception as e:
        print(f"   ❌ RAG class error: {e}")
        import traceback
        traceback.print_exc()
        
    # Test 5: Check metadata fields
    print("\n5. 📋 Checking metadata fields...")
    try:
        # Get some sample documents to see metadata structure
        sample_results = collection.get(limit=10, include=['documents', 'metadatas'])
        
        field_values = set()
        filenames = set()
        
        for meta in sample_results['metadatas']:
            if 'field' in meta and meta['field']:
                field_values.add(meta['field'])
            if 'filename' in meta:
                filenames.add(meta['filename'])
                
        print(f"   Field values found: {sorted(field_values)}")
        print(f"   Sample filenames: {sorted(list(filenames))[:5]}")
        
        # Look specifically for Semurup files
        semurup_files = [f for f in filenames if 'semurup' in f.lower()]
        print(f"   Semurup files: {semurup_files}")
        
    except Exception as e:
        print(f"   ❌ Metadata check error: {e}")

if __name__ == "__main__":
    comprehensive_rag_debug()
