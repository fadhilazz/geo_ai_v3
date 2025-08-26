#!/usr/bin/env python3
"""3D Model Visualization and Analysis for Digital Twin."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.twin.io import load_field_data
from src.config import TWIN_DATA_DIR_OBJ

def analyze_3d_model_data(field: str = "Semurup"):
    """Analyze and visualize 3D model data."""
    print(f"=== 3D Model Analysis for {field} ===")
    
    # Load field data
    field_data = load_field_data(field, TWIN_DATA_DIR_OBJ)
    model_df = field_data['model_df']
    
    if model_df.empty:
        print("❌ No 3D model data found!")
        return
    
    print(f"✅ Loaded {len(model_df):,} data points")
    print(f"📊 Columns: {list(model_df.columns)}")
    
    # Basic statistics
    print("\n📈 BASIC STATISTICS:")
    print("=" * 50)
    
    # Coordinate ranges
    print(f"X Range: {model_df['X'].min():.2f} to {model_df['X'].max():.2f}")
    print(f"Y Range: {model_df['Y'].min():.2f} to {model_df['Y'].max():.2f}")
    print(f"Z Range: {model_df['Z'].min():.2f} to {model_df['Z'].max():.2f}")
    
    # Calculate spatial extent
    x_extent = model_df['X'].max() - model_df['X'].min()
    y_extent = model_df['Y'].max() - model_df['Y'].min()
    area_km2 = (x_extent * y_extent) / 1_000_000  # Convert to km²
    print(f"📏 Spatial Extent: {x_extent:.0f}m x {y_extent:.0f}m ({area_km2:.1f} km²)")
    
    # Resistivity statistics
    if 'Resistivity' in model_df.columns:
        print(f"\n⚡ RESISTIVITY ANALYSIS:")
        print("-" * 30)
        res_stats = model_df['Resistivity'].describe()
        print(f"Min: {res_stats['min']:.2e} ohm.m")
        print(f"Max: {res_stats['max']:.2e} ohm.m")
        print(f"Mean: {res_stats['mean']:.2e} ohm.m")
        print(f"Std: {res_stats['std']:.2e} ohm.m")
        
        # Resistivity categories
        low_res = model_df[model_df['Resistivity'] < 10]
        medium_res = model_df[(model_df['Resistivity'] >= 10) & (model_df['Resistivity'] < 100)]
        high_res = model_df[model_df['Resistivity'] >= 100]
        
        print(f"\n📊 Resistivity Distribution:")
        print(f"Low (< 10 ohm.m): {len(low_res):,} points ({len(low_res)/len(model_df)*100:.1f}%)")
        print(f"Medium (10-100 ohm.m): {len(medium_res):,} points ({len(medium_res)/len(model_df)*100:.1f}%)")
        print(f"High (> 100 ohm.m): {len(high_res):,} points ({len(high_res)/len(model_df)*100:.1f}%)")
    
    # Density statistics
    if 'Density' in model_df.columns:
        print(f"\n⚖️ DENSITY ANALYSIS:")
        print("-" * 30)
        dens_stats = model_df['Density'].describe()
        print(f"Min: {dens_stats['min']:.3f} g/cm³")
        print(f"Max: {dens_stats['max']:.3f} g/cm³")
        print(f"Mean: {dens_stats['mean']:.3f} g/cm³")
        print(f"Std: {dens_stats['std']:.3f} g/cm³")
        
        # Density categories
        low_dens = model_df[model_df['Density'] < 2.0]
        medium_dens = model_df[(model_df['Density'] >= 2.0) & (model_df['Density'] < 2.5)]
        high_dens = model_df[model_df['Density'] >= 2.5]
        
        print(f"\n📊 Density Distribution:")
        print(f"Low (< 2.0 g/cm³): {len(low_dens):,} points ({len(low_dens)/len(model_df)*100:.1f}%)")
        print(f"Medium (2.0-2.5 g/cm³): {len(medium_dens):,} points ({len(medium_dens)/len(model_df)*100:.1f}%)")
        print(f"High (> 2.5 g/cm³): {len(high_dens):,} points ({len(high_dens)/len(model_df)*100:.1f}%)")
    
    # Depth analysis
    print(f"\n🏔️ DEPTH ANALYSIS:")
    print("-" * 30)
    depth_stats = model_df['Z'].describe()
    print(f"Shallowest: {depth_stats['max']:.1f} m")
    print(f"Deepest: {depth_stats['min']:.1f} m")
    print(f"Depth Range: {depth_stats['max'] - depth_stats['min']:.1f} m")
    
    # Sample data points
    print(f"\n📍 SAMPLE DATA POINTS:")
    print("-" * 30)
    sample_points = model_df.sample(min(5, len(model_df)))
    for i, (idx, row) in enumerate(sample_points.iterrows(), 1):
        print(f"{i}. X: {row['X']:.1f}, Y: {row['Y']:.1f}, Z: {row['Z']:.1f}m")
        if 'Resistivity' in row:
            print(f"   Resistivity: {row['Resistivity']:.2e} ohm.m")
        if 'Density' in row:
            print(f"   Density: {row['Density']:.3f} g/cm³")
        print()
    
    return model_df

def find_anomalies(model_df: pd.DataFrame):
    """Find geological anomalies in the 3D model."""
    print("🔍 GEOLOGICAL ANOMALY DETECTION:")
    print("=" * 50)
    
    anomalies = []
    
    # Low resistivity anomalies (potential clay cap)
    if 'Resistivity' in model_df.columns:
        low_res_threshold = 10  # ohm.m
        low_res_anomalies = model_df[model_df['Resistivity'] < low_res_threshold]
        
        if len(low_res_anomalies) > 0:
            print(f"⚡ Low Resistivity Anomalies (< {low_res_threshold} ohm.m):")
            print(f"   Found {len(low_res_anomalies):,} points ({len(low_res_anomalies)/len(model_df)*100:.1f}%)")
            print(f"   Range: {low_res_anomalies['Resistivity'].min():.2e} - {low_res_anomalies['Resistivity'].max():.2e} ohm.m")
            
            # Sample low resistivity points
            sample_low_res = low_res_anomalies.sample(min(3, len(low_res_anomalies)))
            for i, (idx, row) in enumerate(sample_low_res.iterrows(), 1):
                print(f"   {i}. X: {row['X']:.0f}, Y: {row['Y']:.0f}, Z: {row['Z']:.0f}m, R: {row['Resistivity']:.2e} ohm.m")
            print()
            
            anomalies.append({
                'type': 'low_resistivity',
                'count': len(low_res_anomalies),
                'percentage': len(low_res_anomalies)/len(model_df)*100,
                'data': low_res_anomalies
            })
    
    # Low density anomalies
    if 'Density' in model_df.columns:
        low_dens_threshold = 2.0  # g/cm³
        low_dens_anomalies = model_df[model_df['Density'] < low_dens_threshold]
        
        if len(low_dens_anomalies) > 0:
            print(f"⚖️ Low Density Anomalies (< {low_dens_threshold} g/cm³):")
            print(f"   Found {len(low_dens_anomalies):,} points ({len(low_dens_anomalies)/len(model_df)*100:.1f}%)")
            print(f"   Range: {low_dens_anomalies['Density'].min():.3f} - {low_dens_anomalies['Density'].max():.3f} g/cm³")
            
            # Sample low density points
            sample_low_dens = low_dens_anomalies.sample(min(3, len(low_dens_anomalies)))
            for i, (idx, row) in enumerate(sample_low_dens.iterrows(), 1):
                print(f"   {i}. X: {row['X']:.0f}, Y: {row['Y']:.0f}, Z: {row['Z']:.0f}m, ρ: {row['Density']:.3f} g/cm³")
            print()
            
            anomalies.append({
                'type': 'low_density',
                'count': len(low_dens_anomalies),
                'percentage': len(low_dens_anomalies)/len(model_df)*100,
                'data': low_dens_anomalies
            })
    
    # High resistivity anomalies (potential reservoir)
    if 'Resistivity' in model_df.columns:
        high_res_threshold = 100  # ohm.m
        high_res_anomalies = model_df[model_df['Resistivity'] > high_res_threshold]
        
        if len(high_res_anomalies) > 0:
            print(f"⚡ High Resistivity Anomalies (> {high_res_threshold} ohm.m):")
            print(f"   Found {len(high_res_anomalies):,} points ({len(high_res_anomalies)/len(model_df)*100:.1f}%)")
            print(f"   Range: {high_res_anomalies['Resistivity'].min():.2e} - {high_res_anomalies['Resistivity'].max():.2e} ohm.m")
            
            # Sample high resistivity points
            sample_high_res = high_res_anomalies.sample(min(3, len(high_res_anomalies)))
            for i, (idx, row) in enumerate(sample_high_res.iterrows(), 1):
                print(f"   {i}. X: {row['X']:.0f}, Y: {row['Y']:.0f}, Z: {row['Z']:.0f}m, R: {row['Resistivity']:.2e} ohm.m")
            print()
            
            anomalies.append({
                'type': 'high_resistivity',
                'count': len(high_res_anomalies),
                'percentage': len(high_res_anomalies)/len(model_df)*100,
                'data': high_res_anomalies
            })
    
    return anomalies

def create_summary_report(model_df: pd.DataFrame, anomalies: list):
    """Create a summary report for the 3D model."""
    print("📋 3D MODEL SUMMARY REPORT:")
    print("=" * 50)
    
    print(f"Field: Semurup")
    print(f"Total Data Points: {len(model_df):,}")
    print(f"Data Types: {', '.join([col for col in model_df.columns if col not in ['X', 'Y', 'Z']])}")
    
    print(f"\n🔍 Anomalies Detected:")
    for anomaly in anomalies:
        print(f"  • {anomaly['type'].replace('_', ' ').title()}: {anomaly['count']:,} points ({anomaly['percentage']:.1f}%)")
    
    print(f"\n💡 Geological Interpretation:")
    print("  • Low resistivity zones (< 10 ohm.m) may indicate clay cap or hydrothermal alteration")
    print("  • High resistivity zones (> 100 ohm.m) may indicate reservoir rocks")
    print("  • Low density zones (< 2.0 g/cm³) may indicate porous or fractured rocks")
    print("  • These anomalies are potential targets for geothermal exploration")

if __name__ == "__main__":
    print("3D Model Visualization and Analysis")
    print("=" * 60)
    
    # Analyze 3D model data
    model_df = analyze_3d_model_data("Semurup")
    
    if model_df is not None and not model_df.empty:
        # Find anomalies
        anomalies = find_anomalies(model_df)
        
        # Create summary report
        create_summary_report(model_df, anomalies)
        
        print(f"\n✅ Analysis complete! The 3D model data is ready for use in Digital Twin queries.")
    else:
        print("❌ No 3D model data available for analysis.")
