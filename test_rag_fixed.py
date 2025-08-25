#!/usr/bin/env python3
"""Test RAG with fixed imports."""

import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_rag_components():
    """Test RAG components with fixed imports."""
    
    print("🔍 Testing RAG components with fixed imports...")
    
    try:
        # Test 1: Import RAG modules
        print("\n1. 📚 Testing imports...")
        from tools.rag_text import get_text_rag
        from tools.rag_image import get_image_rag
        from tools.qm import get_question_matrix
        print("   ✅ All imports successful!")
        
        # Test 2: Initialize components
        print("\n2. 🏗️ Testing initialization...")
        text_rag = get_text_rag()
        image_rag = get_image_rag()
        qm = get_question_matrix()
        print("   ✅ All components initialized!")
        
        # Test 3: Test search
        print("\n3. 🔍 Testing search...")
        question = "What is the reservoir temperature in Semurup based on geothermometer analysis?"
        
        # Get intent and filters
        intent, confidence = qm.infer_intent(question)
        print(f"   Intent: {intent} (confidence: {confidence:.3f})")
        
        filters = qm.filters_for_intent(intent, "Semurup") if intent else {"field": "Semurup"}
        print(f"   Filters: {filters}")
        
        # Test text search
        text_chunks = text_rag.search_with_fallback(question, filters, top_k=3)
        print(f"   Text chunks found: {len(text_chunks)}")
        
        for i, chunk in enumerate(text_chunks[:2]):
            print(f"   {i+1}. {chunk.filename} (page {chunk.page})")
            print(f"      Score: {chunk.score:.3f}")
            print(f"      Text: {chunk.text[:150]}...")
            
        # Test image search  
        image_chunks = image_rag.search_with_fallback(question, filters, top_k=2)
        print(f"   Image chunks found: {len(image_chunks)}")
        
        for i, chunk in enumerate(image_chunks[:1]):
            print(f"   {i+1}. {chunk.filename} (page {chunk.page})")
            print(f"      Caption: {chunk.caption[:100]}...")
            
        # Test without filters
        print(f"\n4. 🔍 Testing search without filters...")
        text_chunks_no_filter = text_rag.search(question, where=None, top_k=3)
        print(f"   Text chunks without filters: {len(text_chunks_no_filter)}")
        
        for i, chunk in enumerate(text_chunks_no_filter[:2]):
            print(f"   {i+1}. {chunk.filename} (page {chunk.page})")
            if 'geotherm' in chunk.text.lower():
                print(f"      🌡️ CONTAINS GEOTHERMOMETER DATA!")
            print(f"      Text: {chunk.text[:150]}...")
            
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_rag_components()
    if success:
        print("\n🎉 RAG components working correctly!")
    else:
        print("\n❌ RAG components still have issues")
