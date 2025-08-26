"""Digital Twin v2 metrics computation (data-only, no plotting)."""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from .schema import CaprockMetrics, ReservoirMetrics, StructureMetrics, GeochemMetrics
from shapely.geometry import Point

logger = logging.getLogger(__name__)

def caprock_iso(model_df: pd.DataFrame, res_threshold: float = 10.0, below_surface: bool = True) -> Dict:
    """
    Identify caprock based on resistivity threshold.
    
    Args:
        model_df: DataFrame with X, Y, Z, Resistivity columns
        res_threshold: Resistivity threshold in ohm-m
        below_surface: If True, only consider points below surface (Z < 0)
    
    Returns:
        Dictionary with caprock metrics
    """
    if model_df.empty or 'Resistivity' not in model_df.columns:
        return {"points": 0, "res_range": [0, 0], "depth_range": [0, 0], 
                "xy_extent": [(0, 0), (0, 0)], "sample": []}
    
    # Filter by resistivity threshold
    caprock_mask = model_df['Resistivity'] < res_threshold
    
    # First try with the original depth filter
    if below_surface:
        caprock_mask_depth = caprock_mask & (model_df['Z'] < 0)
        caprock_df = model_df[caprock_mask_depth].copy()
        
        # If no points found below surface, try above surface (shallow caprock)
        if caprock_df.empty:
            caprock_mask_shallow = caprock_mask & (model_df['Z'] >= 0)
            caprock_df = model_df[caprock_mask_shallow].copy()
            below_surface_actual = False
        else:
            below_surface_actual = True
    else:
        caprock_df = model_df[caprock_mask].copy()
        below_surface_actual = below_surface
    
    if caprock_df.empty:
        return {"points": 0, "res_range": [0, 0], "depth_range": [0, 0], 
                "xy_extent": [(0, 0), (0, 0)], "sample": []}
    
    # Calculate metrics
    res_range = [float(caprock_df['Resistivity'].min()), float(caprock_df['Resistivity'].max())]
    depth_range = [float(caprock_df['Z'].min()), float(caprock_df['Z'].max())]
    
    # Calculate XY extent in km2
    x_range = caprock_df['X'].max() - caprock_df['X'].min()
    y_range = caprock_df['Y'].max() - caprock_df['Y'].min()
    xy_extent_km2 = (x_range * y_range) / 1e6  # Convert m2 to km2
    
    # Sample 10 representative points
    sample_size = min(10, len(caprock_df))
    sample_df = caprock_df.sample(n=sample_size, random_state=42)
    sample_points = sample_df[['X', 'Y', 'Z', 'Resistivity']].to_dict('records')
    
    return {
        "points": len(caprock_df),
        "res_range": res_range,
        "depth_range": depth_range,
        "xy_extent_km2": xy_extent_km2,
        "xy_extent": [(float(caprock_df['X'].min()), float(caprock_df['X'].max())),
                     (float(caprock_df['Y'].min()), float(caprock_df['Y'].max()))],
        "sample": sample_points,
        "threshold_ohmm": res_threshold,
        "below_surface": below_surface_actual,
        "depth_location": "shallow" if not below_surface_actual else "deep"
    }

def reservoir_rocks_from_density(model_df: pd.DataFrame) -> Dict:
    """
    Classify reservoir rocks based on density values.
    
    Args:
        model_df: DataFrame with X, Y, Z, Density columns
    
    Returns:
        Dictionary with rock type classification and statistics
    """
    if model_df.empty or 'Density' not in model_df.columns:
        return {"dominant_rock": "Unknown", "density_stats": {}, "rock_counts": {}}
    
    # Define density ranges for rock types (g/cm3)
    rock_ranges = {
        "Sedimentary": (2.0, 2.8),
        "Volcanic": (2.3, 3.0),
        "Intrusive": (2.6, 3.3),
        "Metamorphic": (2.5, 3.2)
    }
    
    # Classify each point
    rock_types = []
    for density in model_df['Density']:
        if pd.isna(density):
            rock_types.append("Unknown")
            continue
            
        density = float(density)
        classified = False
        
        for rock_type, (min_dens, max_dens) in rock_ranges.items():
            if min_dens <= density <= max_dens:
                rock_types.append(rock_type)
                classified = True
                break
        
        if not classified:
            rock_types.append("Other")
    
    model_df['RockType'] = rock_types
    
    # Calculate statistics
    rock_counts = model_df['RockType'].value_counts()
    dominant_rock = rock_counts.index[0] if not rock_counts.empty else "Unknown"
    rock_counts_dict = rock_counts.to_dict()
    
    density_stats = {
        "mean": float(model_df['Density'].mean()),
        "std": float(model_df['Density'].std()),
        "min": float(model_df['Density'].min()),
        "max": float(model_df['Density'].max()),
        "q25": float(model_df['Density'].quantile(0.25)),
        "q75": float(model_df['Density'].quantile(0.75))
    }
    
    return {
        "dominant_rock": dominant_rock,
        "density_stats": density_stats,
        "rock_counts": rock_counts_dict,
        "total_points": len(model_df)
    }

def reservoir_iso(model_df: pd.DataFrame, res_low: float = 50.0, res_high: float = 200.0, 
                 rho_max: Optional[float] = None) -> Dict:
    """
    Identify reservoir zones based on resistivity and density criteria.
    
    Args:
        model_df: DataFrame with X, Y, Z, Resistivity columns (and optionally Density)
        res_low: Lower resistivity threshold (ohm-m)
        res_high: Upper resistivity threshold (ohm-m)
        rho_max: Maximum density threshold (g/cm3)
    
    Returns:
        Dictionary with reservoir metrics
    """
    if model_df.empty or 'Resistivity' not in model_df.columns:
        return {"points": 0, "extent_km2": 0, "thickness_stats": {}, "centroids": []}
    
    # Filter by resistivity
    reservoir_mask = (model_df['Resistivity'] >= res_low) & (model_df['Resistivity'] <= res_high)
    
    # Add density filter if available
    if rho_max is not None and 'Density' in model_df.columns:
        reservoir_mask &= model_df['Density'] <= rho_max
    
    reservoir_df = model_df[reservoir_mask].copy()
    
    if reservoir_df.empty:
        return {"points": 0, "extent_km2": 0, "thickness_stats": {}, "centroids": []}
    
    # Calculate extent
    x_range = reservoir_df['X'].max() - reservoir_df['X'].min()
    y_range = reservoir_df['Y'].max() - reservoir_df['Y'].min()
    extent_km2 = (x_range * y_range) / 1e6
    
    # Calculate thickness statistics if Z is available
    thickness_stats = {}
    if 'Z' in reservoir_df.columns:
        thickness_stats = {
            "mean_depth": float(reservoir_df['Z'].mean()),
            "min_depth": float(reservoir_df['Z'].min()),
            "max_depth": float(reservoir_df['Z'].max()),
            "depth_range": float(reservoir_df['Z'].max() - reservoir_df['Z'].min())
        }
    
    # Find centroids using clustering
    centroids = []
    if len(reservoir_df) > 10:
        try:
            # Use K-means to find clusters
            n_clusters = min(5, len(reservoir_df) // 20)
            if n_clusters > 1:
                coords = reservoir_df[['X', 'Y']].values
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(coords)
                
                for i in range(n_clusters):
                    cluster_points = reservoir_df[clusters == i]
                    centroid = [float(cluster_points['X'].mean()), float(cluster_points['Y'].mean())]
                    centroids.append(centroid)
            else:
                # Single centroid
                centroid = [float(reservoir_df['X'].mean()), float(reservoir_df['Y'].mean())]
                centroids.append(centroid)
        except Exception as e:
            logger.warning(f"Clustering failed: {e}")
            # Fallback to single centroid
            centroid = [float(reservoir_df['X'].mean()), float(reservoir_df['Y'].mean())]
            centroids.append(centroid)
    
    return {
        "points": len(reservoir_df),
        "extent_km2": extent_km2,
        "thickness_stats": thickness_stats,
        "centroids": centroids,
        "res_range": [res_low, res_high],
        "rho_max": rho_max
    }

def flow_direction_from_gradient(model_df: pd.DataFrame, surface_selector: str = "top") -> Optional[Dict]:
    """
    Calculate flow direction from gradient analysis.
    
    Args:
        model_df: DataFrame with X, Y, Z, Resistivity columns
        surface_selector: Method to select surface ("top", "bottom", "mean")
    
    Returns:
        Dictionary with flow direction metrics or None
    """
    # TODO: Implement flow direction calculation
    # This is a placeholder for future implementation
    logger.info("Flow direction calculation not yet implemented")
    return None

def distance_to_faults(model_df: pd.DataFrame, structures_gdf) -> Optional[Dict]:
    """
    Calculate distance to nearest faults for each point.
    
    Args:
        model_df: DataFrame with X, Y, Z columns
        structures_gdf: GeoDataFrame with fault geometries
    
    Returns:
        Dictionary with distance statistics or None
    """
    if structures_gdf is None or model_df.empty:
        return None
    
    try:
        # Extract fault lines
        fault_lines = []
        for geom in structures_gdf.geometry:
            if geom.geom_type == 'LineString':
                fault_lines.append(geom)
            elif geom.geom_type == 'MultiLineString':
                fault_lines.extend(list(geom.geoms))
        
        if not fault_lines:
            return None
        
        # Calculate distances
        distances = []
        for idx, row in model_df.iterrows():
            point = (row['X'], row['Y'])
            min_dist = float('inf')
            
            for fault in fault_lines:
                dist = fault.distance(Point(point))
                min_dist = min(min_dist, dist)
            
            if min_dist != float('inf'):
                distances.append(min_dist)
        
        if not distances:
            return None
        
        return {
            "mean_distance": float(np.mean(distances)),
            "min_distance": float(np.min(distances)),
            "max_distance": float(np.max(distances)),
            "std_distance": float(np.std(distances)),
            "points_with_faults": len(distances)
        }
        
    except Exception as e:
        logger.error(f"Distance calculation failed: {e}")
        return None

def calculate_connectivity_score(model_df: pd.DataFrame, threshold: float = 10.0) -> float:
    """
    Calculate connectivity score based on spatial distribution.
    
    Args:
        model_df: DataFrame with X, Y, Z columns
        threshold: Distance threshold for connectivity
    
    Returns:
        Connectivity score (0-1)
    """
    if model_df.empty or len(model_df) < 2:
        return 0.0
    
    try:
        # Sample points for efficiency
        sample_size = min(1000, len(model_df))
        sample_df = model_df.sample(n=sample_size, random_state=42)
        
        # Calculate pairwise distances
        coords = sample_df[['X', 'Y']].values
        distances = cdist(coords, coords)
        
        # Count connected points
        connected = (distances < threshold).sum()
        total_possible = len(distances) * (len(distances) - 1) / 2
        
        if total_possible == 0:
            return 0.0
        
        connectivity = connected / total_possible
        return float(connectivity)
        
    except Exception as e:
        logger.error(f"Connectivity calculation failed: {e}")
        return 0.0

def analyze_geochemistry(geochem_df: pd.DataFrame) -> Dict:
    """
    Analyze geochemical data for geothermal indicators.
    
    Args:
        geochem_df: DataFrame with geochemical measurements
    
    Returns:
        Dictionary with geochemical analysis results
    """
    if geochem_df.empty:
        return {"error": "No geochemical data available"}
    
    try:
        # Basic statistics
        results = {
            "sample_count": len(geochem_df),
            "temperature_range": None,
            "ph_range": None,
            "cl_content": None,
            "geothermometer_results": {}
        }
        
        # Temperature analysis if available
        if 'Temperature' in geochem_df.columns:
            temp_data = geochem_df['Temperature'].dropna()
            if not temp_data.empty:
                results["temperature_range"] = [float(temp_data.min()), float(temp_data.max())]
                results["avg_temperature"] = float(temp_data.mean())
        
        # pH analysis if available
        if 'pH' in geochem_df.columns:
            ph_data = geochem_df['pH'].dropna()
            if not ph_data.empty:
                results["ph_range"] = [float(ph_data.min()), float(ph_data.max())]
                results["avg_ph"] = float(ph_data.mean())
        
        # Chloride content if available
        if 'Cl' in geochem_df.columns:
            cl_data = geochem_df['Cl'].dropna()
            if not cl_data.empty:
                results["cl_content"] = {
                    "min": float(cl_data.min()),
                    "max": float(cl_data.max()),
                    "avg": float(cl_data.mean())
                }
        
        return results
        
    except Exception as e:
        logger.error(f"Error analyzing geochemistry: {e}")
        return {"error": str(e)}


def density_analysis(field_name: str) -> Dict:
    """
    Perform density analysis for a specific field using Digital Twin data.
    
    Args:
        field_name: Name of the geothermal field
        
    Returns:
        Dictionary with density analysis results
    """
    try:
        from .registry import TwinRegistry
        
        registry = TwinRegistry()
        model_data = registry.get(field_name)
        
        if not model_data or 'model_df' not in model_data:
            return {"error": f"No model data available for {field_name}"}
        
        density_df = model_data['model_df']
        
        if density_df.empty:
            return {"error": f"Empty model data for {field_name}"}
        
        # Check if density column exists
        if 'Density' not in density_df.columns:
            return {"error": f"No density data available for {field_name}"}
        
        # Verify coordinate columns exist
        required_cols = ['X', 'Y', 'Z', 'Density']
        missing_cols = [col for col in required_cols if col not in density_df.columns]
        if missing_cols:
            return {"error": f"Missing required columns: {missing_cols}"}
        
        # Basic density statistics
        results = {
            "field": field_name,
            "data_points": len(density_df),
            "density_range": [float(density_df['Density'].min()), float(density_df['Density'].max())],
            "avg_density": float(density_df['Density'].mean()),
            "depth_range": [float(density_df['Z'].min()), float(density_df['Z'].max())],
            "spatial_extent": {
                "x_range": [float(density_df['X'].min()), float(density_df['X'].max())],
                "y_range": [float(density_df['Y'].min()), float(density_df['Y'].max())]
            }
        }
        
        # Rock type classification
        rock_analysis = reservoir_rocks_from_density(density_df)
        results["rock_classification"] = rock_analysis
        
        # High density areas (>2.8 g/cm3)
        high_density_mask = density_df['Density'] > 2.8
        high_density_df = density_df[high_density_mask]
        
        if not high_density_df.empty:
            results["high_density_areas"] = {
                "count": len(high_density_df),
                "percentage": float(len(high_density_df) / len(density_df) * 100),
                "avg_density": float(high_density_df['Density'].mean()),
                "depth_range": [float(high_density_df['Z'].min()), float(high_density_df['Z'].max())]
            }
        
        return results
        
    except Exception as e:
        logger.error(f"Error in density analysis for {field_name}: {e}")
        return {"error": str(e)}


def resistivity_analysis(field_name: str) -> Dict:
    """
    Perform resistivity analysis for a specific field using Digital Twin data.
    
    Args:
        field_name: Name of the geothermal field
        
    Returns:
        Dictionary with resistivity analysis results
    """
    try:
        from .registry import TwinRegistry
        
        registry = TwinRegistry()
        model_data = registry.get(field_name)
        
        if not model_data or 'model_df' not in model_data:
            return {"error": f"No model data available for {field_name}"}
        
        resistivity_df = model_data['model_df']
        
        if resistivity_df.empty:
            return {"error": f"Empty model data for {field_name}"}
        
        # Check if resistivity column exists
        if 'Resistivity' not in resistivity_df.columns:
            return {"error": f"No resistivity data available for {field_name}"}
        
        # Verify coordinate columns exist
        required_cols = ['X', 'Y', 'Z', 'Resistivity']
        missing_cols = [col for col in required_cols if col not in resistivity_df.columns]
        if missing_cols:
            return {"error": f"Missing required columns: {missing_cols}"}
        
        # Basic resistivity statistics
        results = {
            "field": field_name,
            "data_points": len(resistivity_df),
            "resistivity_range": [float(resistivity_df['Resistivity'].min()), float(resistivity_df['Resistivity'].max())],
            "avg_resistivity": float(resistivity_df['Resistivity'].mean()),
            "depth_range": [float(resistivity_df['Z'].min()), float(resistivity_df['Z'].max())],
            "spatial_extent": {
                "x_range": [float(resistivity_df['X'].min()), float(resistivity_df['X'].max())],
                "y_range": [float(resistivity_df['Y'].min()), float(resistivity_df['Y'].max())]
            }
        }
        
        # Caprock analysis
        caprock_analysis = caprock_iso(resistivity_df, res_threshold=10.0)
        results["caprock_analysis"] = caprock_analysis
        
        # Reservoir analysis
        reservoir_analysis = reservoir_iso(resistivity_df)
        results["reservoir_analysis"] = reservoir_analysis
        
        # Low resistivity areas (<10 ohm-m)
        low_res_mask = resistivity_df['Resistivity'] < 10.0
        low_res_df = resistivity_df[low_res_mask]
        
        if not low_res_df.empty:
            results["low_resistivity_areas"] = {
                "count": len(low_res_df),
                "percentage": float(len(low_res_df) / len(resistivity_df) * 100),
                "avg_resistivity": float(low_res_df['Resistivity'].mean()),
                "depth_range": [float(low_res_df['Z'].min()), float(low_res_df['Z'].max())]
            }
        
        # High resistivity areas (>200 ohm-m)
        high_res_mask = resistivity_df['Resistivity'] > 200.0
        high_res_df = resistivity_df[high_res_mask]
        
        if not high_res_df.empty:
            results["high_resistivity_areas"] = {
                "count": len(high_res_df),
                "percentage": float(len(high_res_df) / len(resistivity_df) * 100),
                "avg_resistivity": float(high_res_df['Resistivity'].mean()),
                "depth_range": [float(high_res_df['Z'].min()), float(high_res_df['Z'].max())]
            }
        
        return results
        
    except Exception as e:
        logger.error(f"Error in resistivity analysis for {field_name}: {e}")
        return {"error": str(e)}
