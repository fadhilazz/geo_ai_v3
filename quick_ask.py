#!/usr/bin/env python3
"""Quick answer untuk pertanyaan tentang area caprock."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.twin import twin_query

def answer_caprock_question():
    """Jawab pertanyaan tentang area caprock."""
    print("🌋 PERTANYAAN: Dimana area caprock berdasarkan nilai resistivitas yang rendah?")
    print("=" * 70)
    
    # Query caprock areas
    result = twin_query("Semurup", "Caprock_Location", "resistivity < 10 ohm-m")
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return
    
    metrics = result["metrics"]
    
    print("\n📊 JAWABAN:")
    print(f"Area caprock terletak di:")
    print(f"• Koordinat X: {metrics['xy_extent'][0][0]:.0f} - {metrics['xy_extent'][0][1]:.0f} m")
    print(f"• Koordinat Y: {metrics['xy_extent'][1][0]:.0f} - {metrics['xy_extent'][1][1]:.0f} m")
    print(f"• Luas area: {metrics['xy_extent_km2']:.2f} km²")
    print(f"• Kedalaman: {metrics['depth_range'][0]:.1f} - {metrics['depth_range'][1]:.1f} m")
    print(f"• Resistivitas: {metrics['res_range'][0]:.2f} - {metrics['res_range'][1]:.2f} ohm-m")
    print(f"• Jumlah titik: {metrics['points']:,} titik")
    
    print(f"\n🔍 INTERPRETASI:")
    print(f"Area caprock dengan resistivitas rendah (< 10 ohm-m) terdeteksi di")
    print(f"bagian tengah-selatan field Semurup dengan luas {metrics['xy_extent_km2']:.2f} km².")
    print(f"Kedalaman caprock berada pada {metrics['depth_range'][0]:.1f} - {metrics['depth_range'][1]:.1f} m.")

if __name__ == "__main__":
    answer_caprock_question()
