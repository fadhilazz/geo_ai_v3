#!/usr/bin/env python3
"""Test free-form QA functionality."""

import sys
import json
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from src.tools.qm import match_rows, get_union_strategy, load_qm
    from src.config import get_qa_paths
except ImportError:
    # Try alternative import path
    from tools.qm import match_rows, get_union_strategy, load_qm
    from config import get_qa_paths

def test_semantic_matching():
    """Test semantic matching functionality."""
    print("🧪 Testing semantic matching...")
    
    # Load Question Matrix
    qm_path = get_qa_paths()['question_matrix']
    qm_rows = load_qm(str(qm_path))
    
    # Test cases
    test_cases = [
        "Where is the caprock and how thick is it?",
        "Best reservoir rock type and depth?",
        "Recommend targets near permeable corridors",
        "What does piper plot say about mixing?",
        "What's going on structurally?"
    ]
    
    for question in test_cases:
        print(f"\n📝 Question: '{question}'")
        
        # Get semantic matches
        matches = match_rows(question, k=3)
        
        if matches:
            print(f"   Top matches:")
            for i, (row, score) in enumerate(matches[:3]):
                print(f"   {i+1}. {row['intent_tag']} (score: {score:.3f})")
                print(f"      Template: {row['user_question']}")
            
            # Test union strategy
            strategy = get_union_strategy(matches)
            print(f"   Union strategy:")
            print(f"   - Intent: {strategy['intent_tag']}")
            print(f"   - Confidence: {strategy['confidence']:.3f}")
            print(f"   - Requires twin: {strategy['requires_twin']}")
            print(f"   - Filters: {strategy['filters']}")
        else:
            print("   ❌ No matches found")
    
    print("\n✅ Semantic matching test completed")

def test_clarifier_threshold():
    """Test clarifier threshold logic."""
    print("\n🧪 Testing clarifier threshold...")
    
    # Load Question Matrix
    qm_path = get_qa_paths()['question_matrix']
    qm_rows = load_qm(str(qm_path))
    
    # Test cases with expected confidence levels
    test_cases = [
        ("Where is the caprock and how thick is it?", "Should be high confidence"),
        ("Best reservoir rock type and depth?", "Should be high confidence"),
        ("What's going on structurally?", "Should be low confidence - triggers clarifier"),
        ("Tell me about geothermal", "Should be low confidence - triggers clarifier"),
        ("Random unrelated question about cooking", "Should be very low confidence")
    ]
    
    for question, expected in test_cases:
        print(f"\n📝 Question: '{question}'")
        print(f"   Expected: {expected}")
        
        matches = match_rows(question, k=3)
        if matches:
            strategy = get_union_strategy(matches)
            confidence = strategy['confidence']
            
            print(f"   Actual confidence: {confidence:.3f}")
            
            if confidence < 0.35:
                print("   ✅ Would trigger clarifier (τ_low = 0.35)")
            else:
                print("   ✅ Would proceed with answer")
        else:
            print("   ❌ No matches found")
    
    print("\n✅ Clarifier threshold test completed")

def test_union_strategy():
    """Test union strategy functionality."""
    print("\n🧪 Testing union strategy...")
    
    # Load Question Matrix
    qm_path = get_qa_paths()['question_matrix']
    qm_rows = load_qm(str(qm_path))
    
    # Test cases that should blend concepts
    test_cases = [
        "Caprock continuity and its effect on targets",
        "Reservoir temperature and flow direction",
        "Geology and geochemistry of the system"
    ]
    
    for question in test_cases:
        print(f"\n📝 Question: '{question}'")
        
        matches = match_rows(question, k=5)
        if matches:
            strategy = get_union_strategy(matches)
            
            print(f"   Union strategy:")
            print(f"   - Intent: {strategy['intent_tag']}")
            print(f"   - Confidence: {strategy['confidence']:.3f}")
            print(f"   - Requires twin: {strategy['requires_twin']}")
            print(f"   - Filters: {strategy['filters']}")
            
            # Check if multiple intents were considered
            intent_tags = [row['intent_tag'] for row, _ in matches]
            unique_intents = set(intent_tags)
            if len(unique_intents) > 1:
                print(f"   ✅ Multiple intents considered: {unique_intents}")
            else:
                print(f"   ℹ️  Single intent: {unique_intents}")
        else:
            print("   ❌ No matches found")
    
    print("\n✅ Union strategy test completed")

def test_field_detection():
    """Test field detection functionality."""
    print("\n🧪 Testing field detection...")
    
    try:
        from src.tools.field_detect import detect_field, get_available_fields
        from src.tools.rag_text import get_text_rag
    except ImportError:
        from tools.field_detect import detect_field, get_available_fields
        from tools.rag_text import get_text_rag
    
    # Get available fields
    text_rag = get_text_rag()
    available_fields = get_available_fields(str(text_rag.chroma_dir))
    
    # Test cases
    test_cases = [
        ("Where is the caprock in Semurup?", "Semurup"),
        ("Best reservoir rock type and depth in Semurup?", "Semurup"),
        ("Explain typical caprock lithologies in Indonesian volcanic settings", None),  # General
        ("What's the temperature in Semurup field?", "Semurup"),
        ("General geothermal exploration methods", None)  # General
    ]
    
    for question, expected_field in test_cases:
        print(f"\n📝 Question: '{question}'")
        print(f"   Expected field: {expected_field}")
        
        detected_field, confidence = detect_field(question, available_fields)
        
        print(f"   Detected field: {detected_field}")
        print(f"   Confidence: {confidence:.1f}%")
        
        if expected_field:
            if detected_field == expected_field:
                print("   ✅ Correct field detected")
            else:
                print("   ❌ Field detection failed")
        else:
            if not detected_field or confidence < 85:
                print("   ✅ Correctly identified as general question")
            else:
                print("   ⚠️  Unexpected field detected for general question")
    
    print("\n✅ Field detection test completed")

def test_api_integration():
    """Test API integration with free-form questions."""
    print("\n🧪 Testing API integration...")
    
    import requests
    
    # Test cases
    test_cases = [
        {
            "question": "Where is the caprock and how thick is it?",
            "expected": "Should return answer with citations"
        },
        {
            "question": "Best reservoir rock type and depth?",
            "expected": "Should route to reservoir intent"
        },
        {
            "question": "Recommend targets near permeable corridors",
            "expected": "Should route to Wells_Targeting intent"
        },
        {
            "question": "What does piper plot say about mixing?",
            "expected": "Should route to Geochemistry intent"
        },
        {
            "question": "What's going on structurally?",
            "expected": "Should trigger clarifier"
        }
    ]
    
    for test_case in test_cases:
        question = test_case["question"]
        expected = test_case["expected"]
        
        print(f"\n📝 Question: '{question}'")
        print(f"   Expected: {expected}")
        
        try:
            # Make API request
            response = requests.post(
                "http://127.0.0.1:8000/ask",
                json={"question": question},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                print(f"   Status: ✅ Success")
                print(f"   Intent: {data.get('intent', 'Unknown')}")
                print(f"   Confidence: {data.get('confidence', 'Unknown')}")
                print(f"   Needs clarification: {data.get('needs_clarification', False)}")
                print(f"   Text chunks: {data.get('text_chunks_found', 0)}")
                print(f"   Figures: {data.get('figures_found', 0)}")
                
                if data.get('plan_debug'):
                    plan = data['plan_debug']
                    print(f"   Plan debug:")
                    print(f"     - Top matches: {plan.get('top_matches', [])}")
                    print(f"     - Field detected: {plan.get('field_detected', 'Unknown')}")
                
                if data.get('needs_clarification'):
                    print(f"   Clarification options: {data.get('clarification_options', [])}")
                
            else:
                print(f"   Status: ❌ Error {response.status_code}")
                print(f"   Response: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("   Status: ❌ API server not running")
        except Exception as e:
            print(f"   Status: ❌ Error: {e}")
    
    print("\n✅ API integration test completed")

def main():
    """Run all tests."""
    print("🚀 FREE-FORM QA TEST SUITE")
    print("=" * 50)
    
    # Run tests
    test_semantic_matching()
    test_clarifier_threshold()
    test_union_strategy()
    test_field_detection()
    
    # API test (only if server is running)
    print("\n" + "=" * 50)
    print("🌐 API INTEGRATION TESTS")
    print("Note: Start the API server with 'python -m uvicorn src.api:app --host 127.0.0.1 --port 8000' to run these tests")
    
    try:
        import requests
        response = requests.get("http://127.0.0.1:8000/health", timeout=5)
        if response.status_code == 200:
            test_api_integration()
        else:
            print("❌ API server not responding correctly")
    except:
        print("❌ API server not running - skipping API tests")
    
    print("\n🎉 All tests completed!")

if __name__ == "__main__":
    main()
