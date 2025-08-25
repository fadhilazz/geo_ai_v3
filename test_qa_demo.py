#!/usr/bin/env python3
"""Demo script for testing the QA engine functionality."""

import os
import sys
import asyncio
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.tools.qm import get_question_matrix
from src.tools.field_detect import detect_field_from_question
from src.tools.rag_text import get_text_rag
from src.tools.rag_image import get_image_rag


async def test_components():
    """Test individual components of the QA engine."""
    print("=" * 60)
    print("GEOTHERMAL QA ENGINE - COMPONENT TEST")
    print("=" * 60)
    
    # Test 1: Question Matrix Loading
    print("\n1. Testing Question Matrix Loading...")
    try:
        qm = get_question_matrix()
        print(f"   ✓ Loaded {len(qm.rows)} question patterns")
        print(f"   ✓ Found {len(qm.by_intent)} intent categories")
        
        # Test intent inference
        test_question = "Where is the caprock and how thick is it?"
        intent, confidence = qm.infer_intent(test_question)
        print(f"   ✓ Intent inference: '{intent}' (confidence: {confidence:.3f})")
        
    except Exception as e:
        print(f"   ✗ Question Matrix test failed: {e}")
        
    # Test 2: Field Detection
    print("\n2. Testing Field Detection...")
    try:
        field, score = detect_field_from_question("What is the caprock thickness in Semurup?")
        print(f"   ✓ Field detection: '{field}' (score: {score})")
        
        field2, score2 = detect_field_from_question("How does MT show subsurface structure?")
        print(f"   ✓ General question: '{field2}' (score: {score2})")
        
    except Exception as e:
        print(f"   ✗ Field detection test failed: {e}")
        
    # Test 3: Text RAG
    print("\n3. Testing Text RAG...")
    try:
        text_rag = get_text_rag()
        chunks = text_rag.search("caprock thickness", top_k=3)
        print(f"   ✓ Text search returned {len(chunks)} chunks")
        
        if chunks:
            print(f"   ✓ Sample chunk: {chunks[0].text[:100]}...")
            print(f"   ✓ Sample metadata: doc_id={chunks[0].doc_id}, page={chunks[0].page}")
            
    except Exception as e:
        print(f"   ✗ Text RAG test failed: {e}")
        
    # Test 4: Image RAG
    print("\n4. Testing Image RAG...")
    try:
        image_rag = get_image_rag()
        figures = image_rag.search("geological structure", top_k=3)
        print(f"   ✓ Image search returned {len(figures)} figures")
        
        if figures:
            print(f"   ✓ Sample figure: {figures[0].caption[:100]}...")
            print(f"   ✓ Sample metadata: doc_id={figures[0].doc_id}, page={figures[0].page}")
            
    except Exception as e:
        print(f"   ✗ Image RAG test failed: {e}")
        
    # Test 5: Integration Test
    print("\n5. Testing Integration (without LLM)...")
    try:
        question = "Where is the caprock in Semurup?"
        
        # Field detection
        field, _ = detect_field_from_question(question)
        print(f"   ✓ Detected field: {field}")
        
        # Intent inference
        qm = get_question_matrix()
        intent, confidence = qm.infer_intent(question)
        print(f"   ✓ Detected intent: {intent} (confidence: {confidence:.3f})")
        
        # Build filters
        filters = qm.filters_for_intent(intent, field) if intent else {"field": field} if field else {}
        print(f"   ✓ Generated filters: {filters}")
        
        # Search text and images
        text_rag = get_text_rag()
        image_rag = get_image_rag()
        
        text_results = text_rag.search_with_fallback(question, filters, top_k=3)
        image_results = image_rag.search_with_fallback(question, filters, top_k=2)
        
        print(f"   ✓ Found {len(text_results)} text chunks and {len(image_results)} figures")
        
        if text_results or image_results:
            print("   ✓ Integration test successful - ready for LLM!")
        else:
            print("   ⚠ No evidence found - may need more data or different query")
            
    except Exception as e:
        print(f"   ✗ Integration test failed: {e}")
        
    print("\n" + "=" * 60)
    print("COMPONENT TESTS COMPLETE")
    print("=" * 60)
    
    # Instructions for full API test
    print("\nTo test the full QA engine with LLM:")
    print("1. Set your OpenAI API key: $env:OPENAI_API_KEY='your-key-here'")
    print("2. Start the API server: uvicorn src.api:app --reload")
    print("3. Test with curl:")
    print('   curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \\')
    print('        -d "{\\"question\\":\\"Where is the caprock in Semurup?\\"}"')
    print("\nOr run the smoke tests:")
    print("   python tests/smoke_qm.py --api-key your-key-here")


if __name__ == "__main__":
    asyncio.run(test_components())
