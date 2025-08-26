#!/usr/bin/env python3
"""Final validation test for fine-tuning improvements."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.app_graph_simple import process_question

def test_temperature_consistency():
    """Test temperature data extraction consistency."""
    print("=== Testing Temperature Data Extraction Consistency ===")
    
    questions = [
        "Urutkan manifestasi dengan temperature tertinggi ke terendah",
        "Manifestasi mana yang memiliki temperature tertinggi?",
        "Berapa temperature manifestasi di Semurup?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{i}. Question: {question}")
        result = process_question(question)
        
        print(f"   Intent: {result['framework_debug']['qm_mapping']['intent']}")
        print(f"   Temperature Extracted: {result['framework_debug']['routing_strategy']['temperature_data_extracted']}")
        
        if result['temperature_data']:
            temp_data = result['temperature_data']
            print(f"   Total manifestations: {temp_data['total_manifestations']}")
            print(f"   Highest: {temp_data['highest_temperature']}°C")
            print(f"   Lowest: {temp_data['lowest_temperature']}°C")
            print(f"   Top 3: {[f'{d['location']}: {d['temperature']}°C' for d in temp_data['temperature_data'][:3]]}")
        else:
            print("   No temperature data extracted")

def test_digital_twin_routing():
    """Test Digital Twin routing for specific questions."""
    print("\n=== Testing Digital Twin Routing ===")
    
    questions = [
        "Berapa kedalaman low resistivity <10 ohmm.m?",
        "Berapa estimasi luas caprock?",
        "Dimana area dengan kontras densitas tinggi?",
        "Jelaskan struktur geologi di Semurup"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{i}. Question: {question}")
        result = process_question(question)
        
        print(f"   Intent: {result['framework_debug']['qm_mapping']['intent']}")
        print(f"   Requires Twin: {result['framework_debug']['qm_mapping']['requires_twin']}")
        print(f"   Use Twin: {result['framework_debug']['langgraph_decision']['use_twin']}")
        print(f"   Twin Summary Used: {result['framework_debug']['routing_strategy']['twin_summary_used']}")
        print(f"   Twin Live Query Used: {result['framework_debug']['routing_strategy']['twin_live_query_used']}")

def test_structure_focus():
    """Test if AI focuses only on geological structures."""
    print("\n=== Testing Structure Focus ===")
    
    questions = [
        "Jelaskan struktur geologi di Semurup",
        "Bagaimana sesar dan fault di Semurup?",
        "Dimana batas struktur geologi?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{i}. Question: {question}")
        result = process_question(question)
        
        answer = result['answer']
        print(f"   Answer length: {len(answer)} characters")
        print(f"   Contains 'litologi': {'litologi' in answer.lower()}")
        print(f"   Contains 'batuan': {'batuan' in answer.lower()}")
        print(f"   Contains 'struktur': {'struktur' in answer.lower()}")
        print(f"   Contains 'sesar': {'sesar' in answer.lower()}")
        print(f"   Contains 'fault': {'fault' in answer.lower()}")

def main():
    """Run all validation tests."""
    print("🔍 FINAL VALIDATION TEST FOR FINE-TUNING IMPROVEMENTS")
    print("=" * 60)
    
    test_temperature_consistency()
    test_digital_twin_routing()
    test_structure_focus()
    
    print("\n" + "=" * 60)
    print("✅ VALIDATION COMPLETE")

if __name__ == "__main__":
    main()
