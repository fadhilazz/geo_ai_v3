#!/usr/bin/env python3
"""Terminal-based interface untuk bertanya tentang Digital Twin data."""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.twin import twin_query, twin_summary

def ask_caprock_question():
    """Tanya tentang area caprock."""
    print("\n🌋 PERTANYAAN: Dimana area caprock berdasarkan nilai resistivitas yang rendah?")
    print("-" * 70)
    
    # Query caprock areas
    result = twin_query("Semurup", "Caprock_Location", "resistivity < 10 ohm-m")
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return
    
    metrics = result["metrics"]
    
    print("📊 JAWABAN:")
    print(f"   Area caprock terletak di:")
    print(f"   • Koordinat X: {metrics['xy_extent'][0][0]:.0f} - {metrics['xy_extent'][0][1]:.0f} m")
    print(f"   • Koordinat Y: {metrics['xy_extent'][1][0]:.0f} - {metrics['xy_extent'][1][1]:.0f} m")
    print(f"   • Luas area: {metrics['xy_extent_km2']:.2f} km²")
    print(f"   • Kedalaman: {metrics['depth_range'][0]:.1f} - {metrics['depth_range'][1]:.1f} m")
    print(f"   • Resistivitas: {metrics['res_range'][0]:.2f} - {metrics['res_range'][1]:.2f} ohm-m")
    print(f"   • Jumlah titik: {metrics['points']:,} titik")

def ask_reservoir_question():
    """Tanya tentang reservoir."""
    print("\n🌋 PERTANYAAN: Bagaimana karakteristik reservoir di field Semurup?")
    print("-" * 70)
    
    # Query reservoir analysis
    result = twin_query("Semurup", "Reservoir_Analysis", "resistivity 50-200 ohm-m")
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return
    
    metrics = result["metrics"]
    
    print("📊 JAWABAN:")
    print(f"   Karakteristik reservoir:")
    print(f"   • Luas area: {metrics.get('extent_km2', 0):.2f} km²")
    print(f"   • Jumlah titik: {metrics.get('points', 0):,} titik")
    print(f"   • Rentang resistivitas: {metrics.get('res_range', [0, 0])[0]:.1f} - {metrics.get('res_range', [0, 0])[1]:.1f} ohm-m")

def ask_summary_question():
    """Tanya tentang ringkasan field."""
    print("\n🌋 PERTANYAAN: Berikan ringkasan lengkap field Semurup")
    print("-" * 70)
    
    # Get field summary
    summary = twin_summary("Semurup")
    
    if "error" in summary:
        print(f"❌ Error: {summary['error']}")
        return
    
    print("📊 JAWABAN:")
    print(f"   Ringkasan Field Semurup:")
    print(f"   • Caprock: {summary['caprock']['points']:,} titik, {summary['caprock'].get('xy_extent_km2', 0):.2f} km²")
    print(f"   • Reservoir: {summary['reservoir'].get('extent_km2', 0):.2f} km²")
    print(f"   • Geochemistry: {summary['geochem']['n_samples']} sampel")
    print(f"   • Structure: {summary['structure']['n_features']} fitur")

def interactive_menu():
    """Menu interaktif untuk bertanya."""
    while True:
        print("\n" + "="*60)
        print("🌋 DIGITAL TWIN v2 - TERMINAL INTERFACE")
        print("="*60)
        print("Pilih pertanyaan:")
        print("1. Dimana area caprock berdasarkan nilai resistivitas rendah?")
        print("2. Bagaimana karakteristik reservoir di field Semurup?")
        print("3. Berikan ringkasan lengkap field Semurup")
        print("4. Keluar")
        
        choice = input("\nMasukkan pilihan (1-4): ").strip()
        
        if choice == "1":
            ask_caprock_question()
        elif choice == "2":
            ask_reservoir_question()
        elif choice == "3":
            ask_summary_question()
        elif choice == "4":
            print("👋 Terima kasih! Sampai jumpa!")
            break
        else:
            print("❌ Pilihan tidak valid. Silakan pilih 1-4.")
        
        input("\nTekan Enter untuk melanjutkan...")

if __name__ == "__main__":
    print("🌋 Digital Twin v2 Terminal Interface")
    print("Loading data...")
    interactive_menu()
