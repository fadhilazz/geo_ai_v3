#!/usr/bin/env python3
"""Debug filter generation for geothermometer question."""

import sys
import os
sys.path.append('src')

from tools.qm import get_question_matrix
from tools.field_detect import get_field_detector

def debug_filters():
    """Debug filter generation process."""
    
    print("🔍 Debug filter generation...")
    
    # Test question
    question = "What is the reservoir temperature in Semurup based on geothermometer analysis?"
    field = "Semurup"
    
    try:
        # Load Question Matrix
        print("\n📋 Loading Question Matrix...")
        qm = get_question_matrix()
        print(f"✅ Loaded QM with {len(qm.rows)} patterns")
        
        # Test intent inference
        print(f"\n🧠 Testing intent inference for: '{question}'")
        intent, confidence = qm.infer_intent(question)
        print(f"   Intent: {intent}")
        print(f"   Confidence: {confidence}")
        
        # Test filter generation
        print(f"\n🔧 Testing filter generation...")
        filters = {}
        if intent:
            filters = qm.filters_for_intent(intent, field)
        elif field:
            filters = {"field": field}
            
        print(f"   Generated filters: {filters}")
        
        # Test field detection
        print(f"\n📍 Testing field detection...")
        field_detector = get_field_detector()
        detected_field = field_detector.detect_field(question)
        print(f"   Detected field: {detected_field}")
        
        # Test direct search without filters
        print(f"\n🔍 Testing direct search without filters...")
        from tools.rag_text import get_text_rag
        text_rag = get_text_rag()
        
        # Search without filters
        chunks_no_filter = text_rag.search(question, where=None, top_k=3)
        print(f"   Results without filters: {len(chunks_no_filter)} chunks")
        
        for i, chunk in enumerate(chunks_no_filter[:2]):
            print(f"   {i+1}. {chunk.filename} (page {chunk.page})")
            print(f"      Text: {chunk.text[:150]}...")
            
        # Search with filters
        if filters:
            print(f"\n🔧 Testing search with filters: {filters}")
            chunks_with_filter = text_rag.search_with_fallback(question, where=filters, top_k=3)
            print(f"   Results with filters: {len(chunks_with_filter)} chunks")
            
            for i, chunk in enumerate(chunks_with_filter[:2]):
                print(f"   {i+1}. {chunk.filename} (page {chunk.page})")
                print(f"      Text: {chunk.text[:150]}...")
        else:
            print("   No filters to test")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_filters()
