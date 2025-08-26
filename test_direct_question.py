#!/usr/bin/env python3
"""Test direct user questions with integrated twin system."""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.app_graph import QAWorkflow

def test_direct_questions():
    """Test various direct user questions."""
    
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        return
    
    # Initialize QA workflow
    qa_workflow = QAWorkflow(api_key=api_key)
    
    # Test questions
    test_questions = [
        "Dimana area caprock berdasarkan nilai resistivitas yang rendah?",
        "Bagaimana karakteristik reservoir di field Semurup?",
        "Berikan ringkasan lengkap field Semurup",
        "Apa suhu reservoir di Semurup?",
        "Dimana lokasi caprock dengan resistivitas < 5 ohm-m?",
        "Berapa luas area reservoir di Semurup?"
    ]
    
    print("🌋 TESTING DIRECT USER QUESTIONS WITH INTEGRATED TWIN SYSTEM")
    print("=" * 70)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. PERTANYAAN: {question}")
        print("-" * 50)
        
        try:
            # Run QA workflow
            result = qa_workflow.run({
                "question": question,
                "field": None,  # Let it auto-detect
                "intent": None,
                "intent_confidence": 0.0,
                "filters": {},
                "text_ctx": [],
                "image_ctx": [],
                "numeric_ctx": None,
                "answer": "",
                "citations": [],
                "confidence": "",
                "error": None
            })
            
            # Check if twin data was used
            numeric_ctx = result.get("numeric_ctx")
            if numeric_ctx:
                print(f"✅ TWIN DATA USED:")
                print(f"   Type: {numeric_ctx.get('type')}")
                print(f"   Field: {numeric_ctx.get('field')}")
                print(f"   Intent: {numeric_ctx.get('intent')}")
                if numeric_ctx.get('type') == 'twin_query':
                    print(f"   Execution time: {numeric_ctx.get('execution_time_ms', 0):.1f} ms")
            else:
                print("ℹ️  No twin data used (RAG only)")
            
            # Show answer
            answer = result.get("answer", "")
            if answer:
                print(f"📝 JAWABAN: {answer[:200]}...")
            else:
                print("❌ No answer generated")
            
            # Show citations
            citations = result.get("citations", [])
            if citations:
                print(f"📚 Citations: {len(citations)} sources")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print()

if __name__ == "__main__":
    test_direct_questions()
