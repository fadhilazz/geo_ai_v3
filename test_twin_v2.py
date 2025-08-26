#!/usr/bin/env python3
"""Test script for Digital Twin v2 implementation."""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.twin import (
    load_field_data, build_summary, twin_summary, twin_query,
    register_field, get_field_data, list_available_fields
)
from src.config import TWIN_DATA_DIR_OBJ

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_data_loading():
    """Test data loading functionality."""
    print("\n=== Testing Data Loading ===")
    
    # Test loading Semurup data
    data_dict = load_field_data("Semurup", TWIN_DATA_DIR_OBJ)
    
    print(f"Model data points: {len(data_dict['model_df'])}")
    print(f"Geochemistry samples: {len(data_dict['geochem_df'])}")
    print(f"Has structures: {data_dict['structures_gdf'] is not None}")
    
    if not data_dict['model_df'].empty:
        print(f"Model columns: {list(data_dict['model_df'].columns)}")
        print(f"Model bounds: X={data_dict['model_df']['X'].min():.0f}-{data_dict['model_df']['X'].max():.0f}, "
              f"Y={data_dict['model_df']['Y'].min():.0f}-{data_dict['model_df']['Y'].max():.0f}")
    
    return data_dict

def test_summary_building(data_dict):
    """Test summary building functionality."""
    print("\n=== Testing Summary Building ===")
    
    # Build summary
    summary = build_summary(
        "Semurup",
        data_dict['model_df'],
        data_dict['geochem_df'],
        data_dict['structures_gdf']
    )
    
    print(f"Summary built for field: {summary['field']}")
    print(f"Caprock points: {summary['caprock']['points']}")
    print(f"Reservoir extent: {summary['reservoir'].get('extent_km2', 0):.2f} km²")
    print(f"Structure features: {summary['structure']['n_features']}")
    print(f"Geochemistry samples: {summary['geochem']['n_samples']}")
    
    return summary

def test_twin_functions():
    """Test twin summary and query functions."""
    print("\n=== Testing Twin Functions ===")
    
    # Test twin summary
    summary = twin_summary("Semurup")
    if "error" not in summary:
        print("✓ Twin summary loaded successfully")
        print(f"  - Caprock points: {summary['caprock']['points']}")
        print(f"  - Reservoir dominant rock: {summary['reservoir'].get('dominant_rock', 'Unknown')}")
    else:
        print(f"✗ Twin summary failed: {summary['error']}")
    
    # Test twin query
    query_result = twin_query("Semurup", "Caprock_Location", "resistivity < 10 ohm-m")
    if "error" not in query_result:
        print("✓ Twin query executed successfully")
        print(f"  - Execution time: {query_result['execution_time_ms']:.1f} ms")
        print(f"  - Caprock points found: {query_result['metrics']['points']}")
    else:
        print(f"✗ Twin query failed: {query_result['error']}")

def test_registry():
    """Test registry functionality."""
    print("\n=== Testing Registry ===")
    
    # Register field
    success = register_field("Semurup")
    if success:
        print("✓ Field registered successfully")
    else:
        print("✗ Field registration failed")
    
    # Get field data
    data_dict = get_field_data("Semurup")
    if data_dict:
        print("✓ Field data retrieved successfully")
        print(f"  - Has model data: {not data_dict['model_df'].empty}")
        print(f"  - Has summary: {data_dict['summary'] is not None}")
    else:
        print("✗ Field data retrieval failed")
    
    # List available fields
    fields = list_available_fields()
    print(f"Available fields: {fields}")

def main():
    """Run all tests."""
    print("Digital Twin v2 Test Suite")
    print("=" * 50)
    
    try:
        # Test data loading
        data_dict = test_data_loading()
        
        # Test summary building
        if not data_dict['model_df'].empty:
            test_summary_building(data_dict)
        
        # Test twin functions
        test_twin_functions()
        
        # Test registry
        test_registry()
        
        print("\n" + "=" * 50)
        print("✓ All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"\n✗ Test failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
