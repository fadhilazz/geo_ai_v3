#!/usr/bin/env python3
"""Test open-domain fallback functionality."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.app_graph_simple import process_question

def test_open_domain_fallback():
    """Test if open-domain fallback works for questions not in knowledge base."""
    print("=== Testing Open-Domain Fallback ===")
    
    questions = [
        "Apakah aluvial deposit bisa menjadi caprock?",
        "Bagaimana proses alterasi hidrotermal pada batuan volkanik?",
        "Apa perbedaan antara geothermal system dan hydrothermal system?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{i}. Question: {question}")
        result = process_question(question)
        
        print(f"   Intent: {result['framework_debug']['qm_mapping']['intent']}")
        print(f"   Evidence Available: {result['framework_debug']['evidence_sources']['text_chunks'] > 0}")
        print(f"   Strategy: {result['framework_debug']['strategy_validation']}")
        print(f"   Answer Length: {len(result['answer'])} characters")
        print(f"   Answer Preview: {result['answer'][:200]}...")

if __name__ == "__main__":
    test_open_domain_fallback()
