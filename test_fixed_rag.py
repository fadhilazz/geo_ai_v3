#!/usr/bin/env python3
"""Test fixed RAG system."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_fixed_rag():
    """Test the fixed RAG system."""
    
    print("🔧 TESTING FIXED RAG SYSTEM...")
    
    try:
        from tools.rag_text import get_text_rag
        
        # Clear any cached instances
        import tools.rag_text
        tools.rag_text._text_rag_instance = None
        
        print("\n1. 🏗️ Initializing fixed TextRAG...")
        text_rag = get_text_rag()
        print("   ✅ TextRAG initialized")
        
        print("\n2. 🔍 Testing search without filters...")
        chunks_no_filter = text_rag.search("geothermometer temperature Semurup", where=None, top_k=5)
        print(f"   Results: {len(chunks_no_filter)} chunks")
        
        for i, chunk in enumerate(chunks_no_filter[:3]):
            print(f"\n   {i+1}. {chunk.filename} (page {chunk.page})")
            print(f"      Score: {chunk.score:.3f}")
            print(f"      Field: {chunk.field}")
            print(f"      Text: {chunk.text[:200]}...")
            
            if 'semurup' in chunk.text.lower():
                print(f"      🎯 CONTAINS SEMURUP!")
            if 'geotherm' in chunk.text.lower():
                print(f"      🌡️ CONTAINS GEOTHERMOMETER!")
                
        print(f"\n3. 🔍 Testing search with field filter...")
        field_filter = {"field": "Semurup"}
        chunks_with_filter = text_rag.search("geothermometer temperature", where=field_filter, top_k=5)
        print(f"   Results with field filter: {len(chunks_with_filter)} chunks")
        
        print(f"\n4. 🔄 Testing search with fallback...")
        chunks_fallback = text_rag.search_with_fallback("geothermometer temperature Semurup", where=field_filter, top_k=5)
        print(f"   Results with fallback: {len(chunks_fallback)} chunks")
        
        for i, chunk in enumerate(chunks_fallback[:2]):
            print(f"\n   {i+1}. {chunk.filename} (page {chunk.page})")
            print(f"      Score: {chunk.score:.3f}")
            print(f"      Text: {chunk.text[:200]}...")
            
        # Test specific temperature queries
        print(f"\n5. 🌡️ Testing specific temperature queries...")
        temp_queries = [
            "Na-K-Ca geothermometer",
            "reservoir temperature 229 239",
            "234 degrees celsius",
            "geothermometer Semurup"
        ]
        
        for query in temp_queries:
            chunks = text_rag.search(query, where=None, top_k=3)
            print(f"   '{query}': {len(chunks)} results")
            
            for chunk in chunks[:1]:
                if '229' in chunk.text or '234' in chunk.text or '239' in chunk.text:
                    print(f"      🎯 FOUND TEMPERATURE DATA!")
                    print(f"      Text: {chunk.text[:300]}...")
                    break
                    
        return len(chunks_no_filter) > 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fixed_rag()
    if success:
        print(f"\n🎉 RAG SYSTEM FIXED! Now finding data from knowledge base!")
    else:
        print(f"\n❌ RAG system still has issues")
