"""Digital Twin v2 summary generation and caching."""

import logging
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
from .metrics import (
    caprock_iso, reservoir_rocks_from_density, reservoir_iso, 
    analyze_geochemistry, distance_to_faults, calculate_connectivity_score
)
from .schema import TwinSummary, INTENT_TO_SUMMARY
from src.config import TWIN_SUMMARY_DIR_OBJ
from typing import List

logger = logging.getLogger(__name__)

def build_summary(field: str, model_df: pd.DataFrame, geochem_df: pd.DataFrame, 
                 structures_gdf=None) -> Dict:
    """
    Build a comprehensive summary for a field.
    
    Args:
        field: Field name
        model_df: 3D model DataFrame with X, Y, Z, Resistivity, Density
        geochem_df: Geochemistry DataFrame
        structures_gdf: Structural GeoDataFrame
    
    Returns:
        Dictionary with field summary
    """
    logger.info(f"Building summary for field: {field}")
    
    # Ensure summary directory exists
    TWIN_SUMMARY_DIR_OBJ.mkdir(parents=True, exist_ok=True)
    
    # Calculate bounds
    bounds = calculate_bounds(model_df, geochem_df)
    
    # Calculate caprock metrics
    caprock = caprock_iso(model_df, res_threshold=10.0, below_surface=True)
    
    # Calculate reservoir metrics
    reservoir = {}
    if not model_df.empty:
        # Reservoir isolation
        reservoir_iso_metrics = reservoir_iso(model_df, res_low=50.0, res_high=200.0)
        reservoir.update(reservoir_iso_metrics)
        
        # Rock type classification
        if 'Density' in model_df.columns:
            rock_metrics = reservoir_rocks_from_density(model_df)
            reservoir.update(rock_metrics)
        
        # Connectivity
        connectivity = calculate_connectivity_score(model_df, threshold=1000.0)
        reservoir['connectivity_score'] = connectivity
    
    # Hydrology (placeholder for future implementation)
    hydrology = {
        "flow_azimuth_deg": None,
        "upflow_hints": "TODO: Implement flow direction analysis",
        "gradient_analysis": None
    }
    
    # Structure analysis
    structure = analyze_structures(structures_gdf)
    
    # Geochemistry analysis
    geochem = analyze_geochemistry(geochem_df)
    
    # Distance to faults if available
    if structures_gdf is not None and not model_df.empty:
        fault_distances = distance_to_faults(model_df, structures_gdf)
        if fault_distances:
            structure.update(fault_distances)
    
    # Create summary
    summary = {
        "field": field,
        "bounds": bounds,
        "caprock": caprock,
        "reservoir": reservoir,
        "hydrology": hydrology,
        "structure": structure,
        "geochem": geochem,
        "last_updated": datetime.now().isoformat()
    }
    
    # Save to file
    summary_path = TWIN_SUMMARY_DIR_OBJ / f"{field}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"Saved summary to {summary_path}")
    return summary

def calculate_bounds(model_df: pd.DataFrame, geochem_df: pd.DataFrame) -> Dict:
    """Calculate field bounds from all available data."""
    bounds = {"x": [0, 0], "y": [0, 0], "z": [0, 0], "n": 0}
    
    all_x = []
    all_y = []
    all_z = []
    
    # Collect coordinates from model data
    if not model_df.empty and 'X' in model_df.columns and 'Y' in model_df.columns:
        all_x.extend(model_df['X'].dropna().tolist())
        all_y.extend(model_df['Y'].dropna().tolist())
        if 'Z' in model_df.columns:
            all_z.extend(model_df['Z'].dropna().tolist())
    
    # Collect coordinates from geochemistry data
    if not geochem_df.empty and 'X' in geochem_df.columns and 'Y' in geochem_df.columns:
        all_x.extend(geochem_df['X'].dropna().tolist())
        all_y.extend(geochem_df['Y'].dropna().tolist())
        if 'Z' in geochem_df.columns:
            all_z.extend(geochem_df['Z'].dropna().tolist())
    
    if all_x and all_y:
        bounds["x"] = [float(min(all_x)), float(max(all_x))]
        bounds["y"] = [float(min(all_y)), float(max(all_y))]
        bounds["n"] = len(set(zip(all_x, all_y)))  # Unique points
    
    if all_z:
        bounds["z"] = [float(min(all_z)), float(max(all_z))]
    
    return bounds

def analyze_structures(structures_gdf) -> Dict:
    """Analyze structural data."""
    if structures_gdf is None:
        return {"n_features": 0, "geom_types": [], "crs": "None"}
    
    geom_types = list(structures_gdf.geometry.geom_type.unique())
    
    # Calculate fault lengths and orientations if available
    fault_lengths = []
    fault_orientations = []
    
    for geom in structures_gdf.geometry:
        if geom.geom_type == 'LineString':
            length = geom.length
            fault_lengths.append(length)
            
            # Calculate orientation (simplified)
            coords = list(geom.coords)
            if len(coords) >= 2:
                dx = coords[-1][0] - coords[0][0]
                dy = coords[-1][1] - coords[0][1]
                angle = np.degrees(np.arctan2(dy, dx))
                fault_orientations.append(angle)
    
    return {
        "n_features": len(structures_gdf),
        "geom_types": geom_types,
        "crs": str(structures_gdf.crs) if structures_gdf.crs else "None",
        "fault_lengths": fault_lengths if fault_lengths else None,
        "fault_orientations": fault_orientations if fault_orientations else None
    }

def load_summary(field: str) -> Optional[Dict]:
    """
    Load cached summary for a field.
    
    Args:
        field: Field name
    
    Returns:
        Summary dictionary or None if not found
    """
    summary_path = TWIN_SUMMARY_DIR_OBJ / f"{field}.json"
    
    if not summary_path.exists():
        logger.warning(f"Summary not found for field: {field}")
        return None
    
    try:
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        logger.info(f"Loaded cached summary for field: {field}")
        return summary
        
    except Exception as e:
        logger.error(f"Failed to load summary for {field}: {e}")
        return None

def get_summary_sections(field: str, intent_tag: str) -> Dict:
    """
    Get relevant summary sections for a specific intent.
    
    Args:
        field: Field name
        intent_tag: Intent tag from Question Matrix
    
    Returns:
        Dictionary with relevant summary sections
    """
    summary = load_summary(field)
    if not summary:
        return {}
    
    # Get relevant sections based on intent
    relevant_keys = INTENT_TO_SUMMARY.get(intent_tag, [])
    
    if not relevant_keys:
        # Return basic info if no specific mapping
        return {
            "field": summary.get("field"),
            "bounds": summary.get("bounds"),
            "last_updated": summary.get("last_updated")
        }
    
    # Extract relevant sections
    result = {"field": summary.get("field")}
    for key in relevant_keys:
        if key in summary:
            result[key] = summary[key]
    
    result["last_updated"] = summary.get("last_updated")
    return result

def is_summary_fresh(field: str, max_age_hours: int = 24) -> bool:
    """
    Check if summary is fresh (recently updated).
    
    Args:
        field: Field name
        max_age_hours: Maximum age in hours
    
    Returns:
        True if summary is fresh
    """
    summary = load_summary(field)
    if not summary:
        return False
    
    try:
        last_updated = datetime.fromisoformat(summary.get("last_updated", ""))
        age_hours = (datetime.now() - last_updated).total_seconds() / 3600
        return age_hours <= max_age_hours
    except:
        return False

def list_available_summaries() -> List[str]:
    """List all available field summaries."""
    if not TWIN_SUMMARY_DIR_OBJ.exists():
        return []
    
    summaries = []
    for json_file in TWIN_SUMMARY_DIR_OBJ.glob("*.json"):
        summaries.append(json_file.stem)
    
    return summaries
