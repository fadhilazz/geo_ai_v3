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
    rock_counts = model_df['RockType'].value_counts().to_dict()
    dominant_rock = rock_counts.most_common(1)[0][0] if rock_counts else "Unknown"
    
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
        "rock_counts": rock_counts,
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
    Analyze geochemistry data for anomalies and patterns.
    
    Args:
        geochem_df: DataFrame with geochemistry data
    
    Returns:
        Dictionary with geochemistry metrics
    """
    if geochem_df.empty:
        return {"n_samples": 0, "has_coords": 0.0, "anomaly_counts": {}}
    
    # Count samples with coordinates
    has_coords = geochem_df[['X', 'Y']].notna().all(axis=1).sum()
    coord_percentage = (has_coords / len(geochem_df)) * 100
    
    # Find potential anomalies (values > 2 std from mean)
    anomaly_counts = {}
    numeric_cols = geochem_df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col in ['X', 'Y', 'Z']:
            continue
        
        values = geochem_df[col].dropna()
        if len(values) > 10:
            mean_val = values.mean()
            std_val = values.std()
            threshold = mean_val + 2 * std_val
            
            anomalies = (values > threshold).sum()
            if anomalies > 0:
                anomaly_counts[col] = int(anomalies)
    
    # Temperature and pH ranges if available
    temp_range = None
    ph_range = None
    
    temp_cols = [col for col in geochem_df.columns if 'temp' in col.lower() or 't°' in col]
    ph_cols = [col for col in geochem_df.columns if 'ph' in col.lower()]
    
    if temp_cols:
        temp_values = geochem_df[temp_cols[0]].dropna()
        if len(temp_values) > 0:
            temp_range = [float(temp_values.min()), float(temp_values.max())]
    
    if ph_cols:
        ph_values = geochem_df[ph_cols[0]].dropna()
        if len(ph_values) > 0:
            ph_range = [float(ph_values.min()), float(ph_values.max())]
    
    return {
        "n_samples": len(geochem_df),
        "has_coords": coord_percentage,
        "anomaly_counts": anomaly_counts,
        "temperature_range": temp_range,
        "ph_range": ph_range
    }
