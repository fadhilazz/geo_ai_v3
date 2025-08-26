"""Coordinate normalization and CRS conversion for Digital Twin v2."""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from rapidfuzz import fuzz
import pyproj
from pyproj import CRS, Transformer
from src.config import DEFAULT_UTM_ZONE
from .schema import CoordMapping

logger = logging.getLogger(__name__)

# Coordinate column aliases for fuzzy matching
COORD_ALIASES = {
    'x': ['x', 'easting', 'utm_x', 'lon', 'longitude', 'long', 'east'],
    'y': ['y', 'northing', 'utm_y', 'lat', 'latitude', 'north'],
    'z': ['z', 'elev', 'rl', 'alt', 'altitude', 'depth', 'elevation']
}

def detect_coord_columns(df: pd.DataFrame) -> CoordMapping:
    """
    Detect coordinate columns with typos tolerated using fuzzy matching.
    
    Returns:
        CoordMapping with detected column names and confidence scores.
    """
    if df.empty:
        return CoordMapping(x_col="", y_col="", confidence=0.0)
    
    # Get all column names (case-insensitive)
    cols = [col.lower() for col in df.columns]
    original_cols = list(df.columns)
    
    # Initialize mapping
    mapping = CoordMapping(x_col="", y_col="", confidence=0.0)
    
    # Detect X coordinate
    x_scores = []
    for col, orig_col in zip(cols, original_cols):
        for alias in COORD_ALIASES['x']:
            score = fuzz.partial_ratio(col, alias)
            if score > 80:  # High confidence threshold
                x_scores.append((score, orig_col))
    
    if x_scores:
        best_x_score, best_x_col = max(x_scores, key=lambda x: x[0])
        mapping.x_col = best_x_col
        mapping.confidence = min(mapping.confidence + best_x_score / 100, 1.0)
    
    # Detect Y coordinate
    y_scores = []
    for col, orig_col in zip(cols, original_cols):
        for alias in COORD_ALIASES['y']:
            score = fuzz.partial_ratio(col, alias)
            if score > 80:  # High confidence threshold
                y_scores.append((score, orig_col))
    
    if y_scores:
        best_y_score, best_y_col = max(y_scores, key=lambda x: x[0])
        mapping.y_col = best_y_col
        mapping.confidence = min(mapping.confidence + best_y_score / 100, 1.0)
    
    # Detect Z coordinate (optional)
    z_scores = []
    for col, orig_col in zip(cols, original_cols):
        for alias in COORD_ALIASES['z']:
            score = fuzz.partial_ratio(col, alias)
            if score > 80:  # High confidence threshold
                z_scores.append((score, orig_col))
    
    if z_scores:
        best_z_score, best_z_col = max(z_scores, key=lambda x: x[0])
        mapping.z_col = best_z_col
    
    # Determine CRS hint based on column names
    if mapping.x_col and mapping.y_col:
        x_lower = mapping.x_col.lower()
        y_lower = mapping.y_col.lower()
        
        if any(term in x_lower for term in ['lon', 'long']) and any(term in y_lower for term in ['lat']):
            mapping.crs_hint = "EPSG:4326"  # WGS84 lat/lon
        elif any(term in x_lower for term in ['east', 'utm']) and any(term in y_lower for term in ['north']):
            mapping.crs_hint = f"EPSG:326{DEFAULT_UTM_ZONE}"  # UTM zone
    
    logger.info(f"Detected coordinates: X={mapping.x_col}, Y={mapping.y_col}, Z={mapping.z_col}, confidence={mapping.confidence:.2f}")
    return mapping

def to_utm_xyz(df: pd.DataFrame, zone: Optional[int] = None, crs_hint: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Normalize coordinates to X,Y,Z in meters (UTM).
    
    Args:
        df: Input DataFrame with coordinate columns
        zone: UTM zone (defaults to DEFAULT_UTM_ZONE)
        crs_hint: CRS hint from detect_coord_columns
    
    Returns:
        Tuple of (normalized_df, needs_georef_df)
    """
    if df.empty:
        return df.copy(), pd.DataFrame()
    
    # Detect coordinate columns
    mapping = detect_coord_columns(df)
    
    if not mapping.x_col or not mapping.y_col:
        # Only warn if this looks like it should have coordinates (has location-related columns)
        location_indicators = ['location', 'site', 'sample', 'well', 'borehole', 'station']
        has_location_data = any(indicator in ' '.join(df.columns).lower() for indicator in location_indicators)
        
        if has_location_data:
            logger.warning("No coordinate columns detected in location data")
        else:
            logger.debug("No coordinate columns detected (expected for non-spatial data)")
        return df.copy(), df.copy()
    
    # Create output DataFrame
    result_df = df.copy()
    needs_georef = []
    
    # Check if we need coordinate conversion
    x_lower = mapping.x_col.lower()
    y_lower = mapping.y_col.lower()
    
    is_latlon = any(term in x_lower for term in ['lon', 'long']) and any(term in y_lower for term in ['lat'])
    
    if is_latlon and crs_hint != "EPSG:4326":
        crs_hint = "EPSG:4326"
    
    if is_latlon:
        # Convert lat/lon to UTM
        target_zone = zone or DEFAULT_UTM_ZONE
        target_crs = f"EPSG:326{target_zone}"
        
        # Create transformer
        try:
            transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
            
            # Convert coordinates
            x_coords = []
            y_coords = []
            
            for idx, row in df.iterrows():
                try:
                    lon = float(row[mapping.x_col])
                    lat = float(row[mapping.y_col])
                    
                    # Check if coordinates are valid
                    if -180 <= lon <= 180 and -90 <= lat <= 90:
                        x, y = transformer.transform(lon, lat)
                        x_coords.append(x)
                        y_coords.append(y)
                    else:
                        x_coords.append(np.nan)
                        y_coords.append(np.nan)
                        needs_georef.append(idx)
                        
                except (ValueError, TypeError):
                    x_coords.append(np.nan)
                    y_coords.append(np.nan)
                    needs_georef.append(idx)
            
            # Update DataFrame
            result_df['X'] = x_coords
            result_df['Y'] = y_coords
            
            # Handle Z coordinate if present
            if mapping.z_col:
                result_df['Z'] = df[mapping.z_col]
            
            logger.info(f"Converted {len(df) - len(needs_georef)} coordinates from lat/lon to UTM zone {target_zone}")
            
        except Exception as e:
            logger.error(f"Coordinate conversion failed: {e}")
            return df.copy(), df.copy()
    
    else:
        # Already in UTM or similar projection, just rename columns
        result_df['X'] = df[mapping.x_col]
        result_df['Y'] = df[mapping.y_col]
        if mapping.z_col:
            result_df['Z'] = df[mapping.z_col]
        
        # Check for invalid coordinates
        for idx, row in result_df.iterrows():
            try:
                x = float(row['X'])
                y = float(row['Y'])
                if np.isnan(x) or np.isnan(y):
                    needs_georef.append(idx)
            except (ValueError, TypeError):
                needs_georef.append(idx)
    
    # Create needs_georef DataFrame
    if needs_georef:
        needs_georef_df = df.iloc[needs_georef].copy()
        # Only warn if this is significant (more than 10% of data)
        if len(needs_georef) > len(df) * 0.1:
            logger.warning(f"Found {len(needs_georef)} rows without valid coordinates ({len(needs_georef)/len(df)*100:.1f}%)")
        else:
            logger.debug(f"Found {len(needs_georef)} rows without valid coordinates (minor issue)")
    else:
        needs_georef_df = pd.DataFrame()
    
    return result_df, needs_georef_df

def save_needs_georef_report(needs_georef_df: pd.DataFrame, field: str, cache_dir: Path) -> Optional[Path]:
    """Save rows needing georeferencing to Excel file."""
    if needs_georef_df.empty:
        return None
    
    output_path = cache_dir / f"{field}_needs_georef.xlsx"
    needs_georef_df.to_excel(output_path, index=False)
    logger.info(f"Saved {len(needs_georef_df)} rows needing georeferencing to {output_path}")
    return output_path
