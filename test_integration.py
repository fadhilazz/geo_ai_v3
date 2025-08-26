#!/usr/bin/env python3
"""Test twin integration without OpenAI API."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.twin import twin_summary, twin_query, clarify_needed, should_use_twin_summary
from src.tools.qm import load_qm, infer_intent
from src.config import get_qa_paths

def test_twin_integration():
    """Test twin integration with Question Matrix."""
    
    print("🌋 TESTING TWIN INTEGRATION WITH QUESTION MATRIX")
    print("=" * 60)
    
    # Load Question Matrix
    qm_path = get_qa_paths()['question_matrix']
    qm_rows = load_qm(str(qm_path))
    
    # Test questions
    test_questions = [
        "Where is the caprock based on low resistivity values?",
        "What is the reservoir rock type and characteristics?",
        "What is the reservoir extent and connectivity?",
        "Where is the caprock with resistivity < 5 ohm-m?",
        "What is the flow direction in the system?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. PERTANYAAN: {question}")
        print("-" * 50)
        
        try:
            # Step 1: Infer intent from Question Matrix
            intent, confidence = infer_intent(question, qm_rows)
            print(f"   Intent: {intent} (confidence: {confidence:.3f})")
            
            # Step 2: Check if twin is required
            intent_info = None
            for row in qm_rows:
                if row.get('intent_tag') == intent:
                    intent_info = row
                    break
            
            if intent_info:
                requires_twin = intent_info.get('requires_twin', False)
                print(f"   Requires Twin: {requires_twin}")
                
                if requires_twin:
                    # Step 3: Check if we should use summary or live query
                    use_summary = should_use_twin_summary(question, intent)
                    print(f"   Use Summary: {use_summary}")
                    
                    if use_summary:
                        # Step 4a: Get twin summary
                        twin_data = twin_summary("Semurup")
                        if "error" not in twin_data:
                            print(f"   ✅ Twin Summary loaded successfully")
                            print(f"   Caprock points: {twin_data.get('caprock', {}).get('points', 0)}")
                            print(f"   Reservoir extent: {twin_data.get('reservoir', {}).get('extent_km2', 0):.2f} km²")
                        else:
                            print(f"   ❌ Twin Summary error: {twin_data['error']}")
                    else:
                        # Step 4b: Execute live twin query
                        params = {}
                        if "resistivity" in question.lower() and "<" in question:
                            import re
                            match = re.search(r'< (\d+(?:\.\d+)?)', question)
                            if match:
                                params["res_threshold"] = float(match.group(1))
                        
                        twin_data = twin_query("Semurup", intent, question, params)
                        if "error" not in twin_data:
                            print(f"   ✅ Twin Query executed successfully")
                            print(f"   Execution time: {twin_data['execution_time_ms']:.1f} ms")
                            print(f"   Points found: {twin_data['metrics'].get('points', 0)}")
                        else:
                            print(f"   ❌ Twin Query error: {twin_data['error']}")
                    
                    # Step 5: Check if clarification is needed
                    clarification = clarify_needed(question, intent)
                    if clarification:
                        print(f"   ⚠️  Clarification needed: {clarification}")
                    else:
                        print(f"   ✅ No clarification needed")
                else:
                    print(f"   ℹ️  Twin not required for this intent")
            else:
                print(f"   ❌ Intent info not found in Question Matrix")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        print()

if __name__ == "__main__":
    test_twin_integration()
