#!/usr/bin/env python3
"""Simple test for free-form QA functionality."""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_basic_functionality():
    """Test basic free-form functionality."""
    print("🧪 Testing basic free-form functionality...")
    
    try:
        # Test imports
        from tools.qm import load_qm
        from config import get_qa_paths
        
        print("✅ Imports successful")
        
        # Load Question Matrix
        qm_path = get_qa_paths()['question_matrix']
        qm_rows = load_qm(str(qm_path))
        
        print(f"✅ Loaded Question Matrix: {len(qm_rows)} rows")
        
        # Test semantic matching
        from tools.qm import match_rows, get_union_strategy
        
        test_questions = [
            "Where is the caprock and how thick is it?",
            "Best reservoir rock type and depth?",
            "What's going on structurally?"
        ]
        
        for question in test_questions:
            print(f"\n📝 Testing: '{question}'")
            
            matches = match_rows(question, k=3)
            if matches:
                print(f"   Found {len(matches)} matches")
                strategy = get_union_strategy(matches)
                print(f"   Intent: {strategy['intent_tag']}")
                print(f"   Confidence: {strategy['confidence']:.3f}")
                print(f"   Requires twin: {strategy['requires_twin']}")
                
                if strategy['confidence'] < 0.35:
                    print("   ⚠️  Would trigger clarifier")
                else:
                    print("   ✅ Would proceed with answer")
            else:
                print("   ❌ No matches found")
        
        print("\n✅ Basic functionality test completed")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def test_field_detection():
    """Test field detection."""
    print("\n🧪 Testing field detection...")
    
    try:
        from tools.field_detect import detect_field, get_available_fields
        from tools.rag_text import get_text_rag
        
        # Get available fields
        text_rag = get_text_rag()
        available_fields = get_available_fields(str(text_rag.chroma_dir))
        
        test_cases = [
            ("Where is the caprock in Semurup?", "Semurup"),
            ("Explain typical caprock lithologies", None),  # General
        ]
        
        for question, expected in test_cases:
            print(f"\n📝 Question: '{question}'")
            detected, confidence = detect_field(question, available_fields)
            print(f"   Detected: {detected} (confidence: {confidence:.1f}%)")
            
            if expected:
                if detected == expected:
                    print("   ✅ Correct field detected")
                else:
                    print("   ❌ Field detection failed")
            else:
                if not detected or confidence < 85:
                    print("   ✅ Correctly identified as general")
                else:
                    print("   ⚠️  Unexpected field detected")
        
        print("\n✅ Field detection test completed")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run tests."""
    print("🚀 SIMPLE FREE-FORM QA TEST")
    print("=" * 40)
    
    test_basic_functionality()
    test_field_detection()
    
    print("\n🎉 Tests completed!")

if __name__ == "__main__":
    main()
