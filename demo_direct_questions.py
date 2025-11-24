#!/usr/bin/env python3
"""Demonstration of direct user questions with integrated twin system."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.twin import twin_summary, twin_query, clarify_needed, should_use_twin_summary
from src.tools.qm import load_qm, infer_intent
from src.config import get_qa_paths

def demo_direct_questions():
    """Demonstrate answering direct user questions."""
    
    print("🌋 DEMONSTRASI PERTANYAAN LANGSUNG DENGAN INTEGRASI TWIN")
    print("=" * 70)
    
    # Load Question Matrix
    qm_path = get_qa_paths()['question_matrix']
    qm_rows = load_qm(str(qm_path))
    
    # Test questions in Indonesian
    test_questions = [
        "Dimana area caprock berdasarkan nilai resistivitas yang rendah?",
        "Bagaimana karakteristik reservoir di field Semurup?",
        "Dimana lokasi caprock dengan resistivitas < 5 ohm-m di Semurup?",
        "Berapa luas area reservoir di Semurup?",
        "Apa jenis batuan reservoir di Semurup?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. PERTANYAAN USER: {question}")
        print("-" * 60)
        
        try:
            # Step 1: Infer intent
            intent, confidence = infer_intent(question, qm_rows)
            print(f"🔍 Intent terdeteksi: {intent} (confidence: {confidence:.3f})")
            
            # Step 2: Check if twin is required
            intent_info = None
            for row in qm_rows:
                if row.get('intent_tag') == intent:
                    intent_info = row
                    break
            
            if intent_info and intent_info.get('requires_twin'):
                print(f"✅ Twin data diperlukan untuk intent ini")
                
                # Step 3: Determine approach (summary vs live query)
                use_summary = should_use_twin_summary(question, intent)
                print(f"📊 Menggunakan: {'Twin Summary' if use_summary else 'Live Twin Query'}")
                
                if use_summary:
                    # Use twin summary for general questions
                    twin_data = twin_summary("Semurup")
                    if "error" not in twin_data:
                        print(f"📋 Twin Summary berhasil dimuat")
                        
                        # Extract relevant information based on intent
                        if intent == "Caprock_Location":
                            caprock = twin_data.get('caprock', {})
                            print(f"📍 Area Caprock:")
                            print(f"   • Jumlah titik: {caprock.get('points', 0):,} titik")
                            print(f"   • Luas area: {caprock.get('xy_extent_km2', 0):.2f} km²")
                            print(f"   • Kedalaman: {caprock.get('depth_range', [0, 0])[0]:.1f} - {caprock.get('depth_range', [0, 0])[1]:.1f} m")
                            print(f"   • Resistivitas: {caprock.get('res_range', [0, 0])[0]:.2f} - {caprock.get('res_range', [0, 0])[1]:.2f} ohm-m")
                        
                        elif intent == "Reservoir_RockType":
                            reservoir = twin_data.get('reservoir', {})
                            print(f"🪨 Karakteristik Reservoir:")
                            print(f"   • Jenis batuan dominan: {reservoir.get('dominant_rock', 'Unknown')}")
                            print(f"   • Luas area: {reservoir.get('extent_km2', 0):.2f} km²")
                            print(f"   • Rentang resistivitas: {reservoir.get('res_range_ohmm', [0, 0])[0]:.1f} - {reservoir.get('res_range_ohmm', [0, 0])[1]:.1f} ohm-m")
                        
                        elif intent == "Reservoir_Analysis":
                            reservoir = twin_data.get('reservoir', {})
                            print(f"🔍 Analisis Reservoir:")
                            print(f"   • Luas area: {reservoir.get('extent_km2', 0):.2f} km²")
                            print(f"   • Connectivity score: {reservoir.get('connectivity_score', 0):.2f}")
                            print(f"   • Ketebalan rata-rata: {reservoir.get('thickness_stats', {}).get('mean', 0):.1f} m")
                    else:
                        print(f"❌ Error loading twin summary: {twin_data['error']}")
                
                else:
                    # Use live twin query for specific numeric questions
                    params = {}
                    if "resistivity" in question.lower() and "<" in question:
                        import re
                        match = re.search(r'< (\d+(?:\.\d+)?)', question)
                        if match:
                            params["res_threshold"] = float(match.group(1))
                            print(f"🎯 Threshold resistivitas: < {params['res_threshold']} ohm-m")
                    
                    twin_data = twin_query("Semurup", intent, question, params)
                    if "error" not in twin_data:
                        print(f"⚡ Live Twin Query berhasil dieksekusi")
                        print(f"   • Waktu eksekusi: {twin_data['execution_time_ms']:.1f} ms")
                        
                        metrics = twin_data['metrics']
                        if intent == "Caprock_Location":
                            print(f"📍 Hasil Caprock Analysis:")
                            print(f"   • Jumlah titik: {metrics.get('points', 0):,} titik")
                            print(f"   • Luas area: {metrics.get('xy_extent_km2', 0):.2f} km²")
                            print(f"   • Koordinat X: {metrics.get('xy_extent', [[0, 0], [0, 0]])[0][0]:.0f} - {metrics.get('xy_extent', [[0, 0], [0, 0]])[0][1]:.0f} m")
                            print(f"   • Koordinat Y: {metrics.get('xy_extent', [[0, 0], [0, 0]])[1][0]:.0f} - {metrics.get('xy_extent', [[0, 0], [0, 0]])[1][1]:.0f} m")
                    else:
                        print(f"❌ Error executing twin query: {twin_data['error']}")
                
                # Check for clarification needs
                clarification = clarify_needed(question, intent)
                if clarification:
                    print(f"⚠️  Klarifikasi diperlukan: {clarification}")
                else:
                    print(f"✅ Pertanyaan lengkap, tidak perlu klarifikasi")
                    
            else:
                print(f"ℹ️  Twin data tidak diperlukan untuk intent ini (menggunakan RAG)")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print()

if __name__ == "__main__":
    demo_direct_questions()
