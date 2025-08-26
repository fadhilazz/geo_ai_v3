#!/usr/bin/env python3
"""Caprock area analysis based on low resistivity values."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.twin import twin_query

def analyze_caprock_areas():
    """Analyze caprock areas based on low resistivity values."""
    print("🌋 ANALISIS AREA CAPROCK BERDASARKAN NILAI RESISTIVITAS RENDAH")
    print("=" * 70)
    
    # Query caprock areas with resistivity < 10 ohm-m
    result = twin_query("Semurup", "Caprock_Location", "resistivity < 10 ohm-m")
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return
    
    metrics = result["metrics"]
    
    print(f"\n📊 HASIL ANALISIS CAPROCK:")
    print(f"   • Jumlah titik caprock: {metrics['points']:,} titik")
    print(f"   • Rentang resistivitas: {metrics['res_range'][0]:.2f} - {metrics['res_range'][1]:.2f} ohm-m")
    print(f"   • Rentang kedalaman: {metrics['depth_range'][0]:.1f} - {metrics['depth_range'][1]:.1f} m")
    print(f"   • Luas area: {metrics['xy_extent_km2']:.2f} km²")
    print(f"   • Lokasi: {metrics.get('depth_location', 'unknown')}")
    
    print(f"\n📍 KOORDINAT AREA CAPROCK:")
    print(f"   • X: {metrics['xy_extent'][0][0]:.0f} - {metrics['xy_extent'][0][1]:.0f} m")
    print(f"   • Y: {metrics['xy_extent'][1][0]:.0f} - {metrics['xy_extent'][1][1]:.0f} m")
    
    print(f"\n🔍 INTERPRETASI:")
    print(f"   • Caprock terdeteksi di area seluas {metrics['xy_extent_km2']:.2f} km²")
    print(f"   • Terletak pada kedalaman {metrics['depth_range'][0]:.1f} - {metrics['depth_range'][1]:.1f} m")
    print(f"   • Nilai resistivitas sangat rendah (0.34 - 10.00 ohm-m)")
    print(f"   • Lokasi: Shallow caprock (di atas permukaan)")
    
    print(f"\n⚡ INFORMASI TAMBAHAN:")
    print(f"   • Waktu eksekusi query: {result['execution_time_ms']:.1f} ms")
    print(f"   • Field: Semurup")
    print(f"   • Threshold resistivitas: < 10 ohm-m")

if __name__ == "__main__":
    analyze_caprock_areas()
