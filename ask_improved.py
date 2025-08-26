#!/usr/bin/env python3
"""Improved AI Assistant that prioritizes RAG for general questions."""

import sys
from pathlib import Path
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def ask_improved():
    """Interactive question asking with improved RAG priority."""
    
    try:
        from src.app_graph_simple import process_question
        
        print("🤖 AI Geothermal Assistant - Improved RAG Priority")
        print("=" * 70)
        print("Strategy: RAG First for General Questions, Digital Twin for Specific Model Data")
        print("Type 'quit' to exit")
        print()
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("❌ No OpenAI API key found. Please set OPENAI_API_KEY environment variable.")
            return
        
        while True:
            try:
                # Get user question
                question = input("\n🤔 Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q', 'keluar']:
                    print("👋 Goodbye!")
                    break
                
                if not question:
                    continue
                
                print("\n🔍 Processing with Improved Strategy...")
                print("=" * 70)
                
                # Process the question
                result = process_question(question, field="Semurup")
                
                # Safe display of results
                if result is None:
                    print("❌ No result returned from processing")
                    continue
                
                # Display framework routing information
                framework_debug = result.get('framework_debug', {})
                
                print("📋 FRAMEWORK ANALYSIS:")
                print("-" * 40)
                
                # Step 1: Question Matrix Mapping
                qm_mapping = framework_debug.get('qm_mapping', {})
                print(f"1️⃣ Question Matrix Mapping:")
                print(f"   ✅ Intent: {qm_mapping.get('intent', 'Unknown')}")
                print(f"   ✅ Requires Twin: {qm_mapping.get('requires_twin', False)}")
                print(f"   ✅ Confidence: {qm_mapping.get('confidence', 0):.3f}")
                
                # Step 2: LangGraph Decision
                lg_decision = framework_debug.get('langgraph_decision', {})
                print(f"\n2️⃣ LangGraph Decision:")
                print(f"   🧠 Use Twin: {lg_decision.get('use_twin', False)}")
                print(f"   💭 Reasoning: {lg_decision.get('reasoning', 'Unknown')}")
                
                # Step 3: Routing Strategy
                routing = framework_debug.get('routing_strategy', {})
                print(f"\n3️⃣ Routing Strategy:")
                print(f"   🔬 Twin Summary Used: {routing.get('twin_summary_used', False)}")
                print(f"   ⚡ Twin Live Query Used: {routing.get('twin_live_query_used', False)}")
                print(f"   🌡️ Temperature Data Extracted: {routing.get('temperature_data_extracted', False)}")
                
                # Display evidence sources
                evidence = framework_debug.get('evidence_sources', {})
                print(f"\n📊 EVIDENCE SOURCES:")
                print(f"   📄 Text Chunks: {evidence.get('text_chunks', 0)}")
                print(f"   🖼️  Figures: {evidence.get('figures', 0)}")
                print(f"   📍 Citations: {evidence.get('citations', 0)}")
                
                # Display temperature data if extracted
                temperature_data = result.get('temperature_data')
                if temperature_data and temperature_data.get('temperature_data'):
                    print(f"\n🌡️ TEMPERATURE DATA EXTRACTED:")
                    print(f"   📊 Total manifestations: {temperature_data['total_manifestations']}")
                    print(f"   🔥 Highest temperature: {temperature_data['highest_temperature']}°C")
                    print(f"   ❄️  Lowest temperature: {temperature_data['lowest_temperature']}°C")
                    
                    print(f"\n   📋 Top 5 Temperatures:")
                    for i, data in enumerate(temperature_data['temperature_data'][:5], 1):
                        print(f"      {i}. {data['location']}: {data['temperature']}°C ({data['manifestation_type']})")
                
                print("\n🤖 AI ANSWER:")
                print("=" * 70)
                answer = result.get('answer', 'No answer available')
                print(answer)
                print("=" * 70)
                
                # Strategy validation
                strategy_validation = framework_debug.get('strategy_validation', 'Unknown')
                print(f"\n✅ STRATEGY VALIDATION:")
                print(f"   🎯 {strategy_validation}")
                
                print(f"\n💡 TIP: For better answers, try:")
                print(f"   • General questions: 'Apa manifestasi di Semurup?', 'Bagaimana litologi?', 'Struktur geologi?'")
                print(f"   • Temperature questions: 'Urutkan manifestasi berdasarkan temperature', 'Manifestasi dengan suhu tertinggi'")
                print(f"   • Specific model data: 'Nilai resistivitas caprock?', 'Distribusi densitas 3D?', 'Range MT data?'")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error processing question: {e}")
                print("Please try a different question.")
                continue
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    ask_improved()
