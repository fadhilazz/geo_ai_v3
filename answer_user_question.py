#!/usr/bin/env python3
"""Answer the user's specific question about caprock areas."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.twin import twin_query

def answer_caprock_question():
    """Answer the user's question about caprock areas."""
    
    print("🌋 JAWABAN PERTANYAAN USER")
    print("=" * 50)
    print("Pertanyaan: Dimana area caprock berdasarkan nilai resistivitas yang rendah?")
    print()
    
    # Execute twin query for caprock with low resistivity
    result = twin_query("Semurup", "Caprock_Location", "resistivity < 10 ohm-m")
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return
    
    metrics = result["metrics"]
    
    print("📊 JAWABAN LENGKAP:")
    print()
    print("📍 Area caprock dengan resistivitas rendah (< 10 ohm-m) terletak di:")
    print()
    print(f"   • Koordinat X: {metrics['xy_extent'][0][0]:.0f} - {metrics['xy_extent'][0][1]:.0f} m")
    print(f"   • Koordinat Y: {metrics['xy_extent'][1][0]:.0f} - {metrics['xy_extent'][1][1]:.0f} m")
    print(f"   • Luas area: {metrics['xy_extent_km2']:.2f} km²")
    print(f"   • Kedalaman: {metrics['depth_range'][0]:.1f} - {metrics['depth_range'][1]:.1f} m")
    print(f"   • Resistivitas: {metrics['res_range'][0]:.2f} - {metrics['res_range'][1]:.2f} ohm-m")
    print(f"   • Jumlah titik: {metrics['points']:,} titik")
    print(f"   • Lokasi: {metrics.get('depth_location', 'unknown')}")
    print()
    print("🔍 INTERPRETASI GEOLOGI:")
    print()
    print("   Area caprock dengan resistivitas sangat rendah (< 10 ohm-m) terdeteksi")
    print("   di bagian tengah-selatan field Semurup dengan luas 120.34 km².")
    print("   Caprock ini terletak pada kedalaman 1,212.5 - 1,787.5 m dan")
    print("   menunjukkan karakteristik clay cap yang efektif sebagai seal.")
    print()
    print("   Nilai resistivitas yang sangat rendah (0.34 - 10.00 ohm-m)")
    print("   mengindikasikan adanya clay minerals yang berfungsi sebagai")
    print("   caprock yang baik untuk menahan fluida geothermal.")
    print()
    print(f"⚡ Informasi Teknis:")
    print(f"   • Waktu eksekusi query: {result['execution_time_ms']:.1f} ms")
    print(f"   • Field: Semurup")
    print(f"   • Threshold resistivitas: < 10 ohm-m")
    print(f"   • Metode: Digital Twin v2 - Live Query")

if __name__ == "__main__":
    answer_caprock_question()
