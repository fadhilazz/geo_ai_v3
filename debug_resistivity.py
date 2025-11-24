#!/usr/bin/env python3
"""Debug script to analyze resistivity data for caprock detection."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.twin.io import load_field_data
from src.config import TWIN_DATA_DIR_OBJ

def analyze_resistivity_data():
    """Analyze resistivity data to understand caprock detection."""
    print("=== Resistivity Data Analysis ===")
    
    # Load field data
    data_dict = load_field_data("Semurup", TWIN_DATA_DIR_OBJ)
    model_df = data_dict['model_df']
    
    if model_df.empty:
        print("No model data available")
        return
    
    print(f"Total points: {len(model_df)}")
    print(f"Columns: {list(model_df.columns)}")
    
    # Check resistivity statistics
    if 'Resistivity' in model_df.columns:
        res_data = model_df['Resistivity'].dropna()
        print(f"\nResistivity Statistics:")
        print(f"  Min: {res_data.min():.2f} ohm-m")
        print(f"  Max: {res_data.max():.2f} ohm-m")
        print(f"  Mean: {res_data.mean():.2f} ohm-m")
        print(f"  Median: {res_data.median():.2f} ohm-m")
        print(f"  Std: {res_data.std():.2f} ohm-m")
        
        # Check distribution around caprock threshold
        print(f"\nCaprock Analysis (threshold = 10 ohm-m):")
        below_10 = res_data[res_data < 10.0]
        print(f"  Points below 10 ohm-m: {len(below_10)}")
        
        if len(below_10) > 0:
            print(f"  Min below 10: {below_10.min():.2f} ohm-m")
            print(f"  Max below 10: {below_10.max():.2f} ohm-m")
            print(f"  Mean below 10: {below_10.mean():.2f} ohm-m")
        
        # Check depth distribution
        if 'Z' in model_df.columns:
            print(f"\nDepth Analysis:")
            z_data = model_df['Z'].dropna()
            print(f"  Z range: {z_data.min():.1f} to {z_data.max():.1f} m")
            print(f"  Z mean: {z_data.mean():.1f} m")
            print(f"  Z median: {z_data.median():.1f} m")
            
            # Check caprock points below surface
            below_surface = model_df[model_df['Z'] < 0]
            above_surface = model_df[model_df['Z'] >= 0]
            below_surface_low_res = below_surface[below_surface['Resistivity'] < 10.0]
            above_surface_low_res = above_surface[above_surface['Resistivity'] < 10.0]
            
            print(f"  Points below surface (Z < 0): {len(below_surface)}")
            print(f"  Points above surface (Z >= 0): {len(above_surface)}")
            print(f"  Low resistivity points below surface: {len(below_surface_low_res)}")
            print(f"  Low resistivity points above surface: {len(above_surface_low_res)}")
            
            # Check Z distribution for low resistivity points
            if len(above_surface_low_res) > 0:
                print(f"\nZ distribution for low resistivity points (< 10 ohm-m):")
                low_res_z = above_surface_low_res['Z']
                print(f"  Z range: {low_res_z.min():.1f} to {low_res_z.max():.1f} m")
                print(f"  Z mean: {low_res_z.mean():.1f} m")
                print(f"  Z median: {low_res_z.median():.1f} m")
                
                # Check if most are at shallow depths
                shallow_low_res = low_res_z[low_res_z < 100]  # Less than 100m depth
                print(f"  Low resistivity points at shallow depth (< 100m): {len(shallow_low_res)}")
        
        # Check different thresholds
        print(f"\nThreshold Analysis:")
        thresholds = [1, 5, 10, 20, 50, 100]
        for threshold in thresholds:
            count = len(res_data[res_data < threshold])
            percentage = (count / len(res_data)) * 100
            print(f"  < {threshold:3d} ohm-m: {count:6d} points ({percentage:5.2f}%)")
    
    # Check if there are any very low values
    print(f"\nLowest 10 resistivity values:")
    lowest_10 = res_data.nsmallest(10)
    for i, val in enumerate(lowest_10, 1):
        print(f"  {i:2d}. {val:.2f} ohm-m")

if __name__ == "__main__":
    analyze_resistivity_data()
