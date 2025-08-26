"""Data I/O loaders for Digital Twin v2."""

import logging
import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from .normalize import to_utm_xyz, save_needs_georef_report
from src.config import TWIN_CACHE_DIR_OBJ

logger = logging.getLogger(__name__)

def load_xyz_dat(path: Path, value_name: str = "Value", max_rows: int = 100000) -> pd.DataFrame:
    """
    Load XYZ grid data from .dat file format.
    
    Expected format: "X Y Z Value" space-separated values.
    
    Args:
        path: Path to .dat file
        value_name: Name for the value column
        max_rows: Maximum number of rows to load (for large files)
    
    Returns:
        DataFrame with X, Y, Z, value_name columns
    """
    try:
        # Check file size
        file_size_mb = path.stat().st_size / (1024 * 1024)
        logger.info(f"Loading {path.name} ({file_size_mb:.1f} MB)")
        
        # For large files, sample rows for efficiency
        if file_size_mb > 10:  # Files larger than 10MB
            logger.info(f"Large file detected, sampling {max_rows} rows for efficiency")
            # Read first few lines to get column names
            df = pd.read_csv(path, sep=r'\s+', header=None, 
                           names=['X', 'Y', 'Z', value_name],
                           engine='python', nrows=max_rows)
        else:
            # Read entire file for smaller files
            df = pd.read_csv(path, sep=r'\s+', header=None, 
                           names=['X', 'Y', 'Z', value_name],
                           engine='python')
        
        # Ensure coordinate columns are numeric
        for col in ['X', 'Y', 'Z']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with invalid coordinates
        initial_count = len(df)
        df = df.dropna(subset=['X', 'Y', 'Z'])
        final_count = len(df)
        
        if final_count < initial_count:
            logger.warning(f"Removed {initial_count - final_count} rows with invalid coordinates")
        
        # Check if we have valid coordinates
        if len(df) == 0:
            logger.error(f"No valid coordinate data found in {path}")
            return pd.DataFrame()
        
        # Verify coordinate ranges are reasonable (not all zeros or extreme values)
        x_range = df['X'].max() - df['X'].min()
        y_range = df['Y'].max() - df['Y'].min()
        
        if x_range < 1e-6 or y_range < 1e-6:
            logger.warning(f"Very small coordinate ranges detected: X={x_range:.6f}, Y={y_range:.6f}")
            logger.warning("This might indicate coordinate system issues")
        
        logger.info(f"Loaded {len(df)} valid points from {path}")
        logger.info(f"Coordinate ranges: X={df['X'].min():.2f} to {df['X'].max():.2f}, Y={df['Y'].min():.2f} to {df['Y'].max():.2f}")
        
        # Verify we have the expected columns
        expected_cols = ['X', 'Y', 'Z', value_name]
        if not all(col in df.columns for col in expected_cols):
            logger.error(f"Missing required columns. Expected: {expected_cols}, Got: {list(df.columns)}")
            return pd.DataFrame()
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
        return pd.DataFrame()

def load_structures(path: Path) -> Optional[gpd.GeoDataFrame]:
    """
    Load structural data from shapefile.
    
    Args:
        path: Path to shapefile (.shp)
    
    Returns:
        GeoDataFrame or None if loading fails
    """
    try:
        if not path.exists():
            logger.warning(f"Shapefile not found: {path}")
            return None
        
        # Load shapefile
        gdf = gpd.read_file(path)
        
        # Ensure it has a CRS
        if gdf.crs is None:
            logger.warning(f"No CRS found in {path}, assuming WGS84")
            gdf.set_crs("EPSG:4326", inplace=True)
        
        # Convert to UTM if needed
        if gdf.crs.is_geographic:
            from src.config import DEFAULT_UTM_ZONE
            target_crs = f"EPSG:326{DEFAULT_UTM_ZONE}"
            gdf = gdf.to_crs(target_crs)
            logger.info(f"Converted shapefile to UTM zone {DEFAULT_UTM_ZONE}")
        
        logger.info(f"Loaded {len(gdf)} features from {path}")
        return gdf
        
    except Exception as e:
        logger.error(f"Failed to load shapefile {path}: {e}")
        return None

def load_geochem_csv_tsv(path: Path) -> pd.DataFrame:
    """
    Load geochemistry data from CSV/TSV file with auto-detection.
    
    Args:
        path: Path to CSV/TSV file
    
    Returns:
        Normalized DataFrame with X, Y, Z columns
    """
    try:
        # Try different separators
        separators = [',', '\t', ';', '|']
        
        for sep in separators:
            try:
                df = pd.read_csv(path, sep=sep)
                if len(df.columns) > 1:  # Valid CSV/TSV
                    logger.info(f"Successfully loaded with separator '{sep}'")
                    break
            except:
                continue
        else:
            logger.error(f"Could not parse {path} with any separator")
            return pd.DataFrame()
        
        # Normalize coordinates
        normalized_df, needs_georef_df = to_utm_xyz(df)
        
        # Save needs_georef report if any
        if not needs_georef_df.empty:
            field_name = path.stem
            save_needs_georef_report(needs_georef_df, field_name, TWIN_CACHE_DIR_OBJ)
        
        logger.info(f"Loaded {len(normalized_df)} geochemistry samples from {path}")
        return normalized_df
        
    except Exception as e:
        logger.error(f"Failed to load geochemistry file {path}: {e}")
        return pd.DataFrame()

def load_geochem_xlsx(path: Path) -> Dict[str, pd.DataFrame]:
    """
    Load geochemistry data from Excel file with multiple sheets.
    
    Args:
        path: Path to Excel file
    
    Returns:
        Dictionary of sheet_name -> normalized DataFrame
    """
    try:
        # Read all sheets
        excel_file = pd.ExcelFile(path)
        sheets = {}
        
        for sheet_name in excel_file.sheet_names:
            try:
                df = pd.read_excel(path, sheet_name=sheet_name)
                
                if df.empty:
                    logger.warning(f"Empty sheet: {sheet_name}")
                    continue
                
                # Normalize coordinates
                normalized_df, needs_georef_df = to_utm_xyz(df)
                
                # Save needs_georef report if any
                if not needs_georef_df.empty:
                    field_name = f"{path.stem}_{sheet_name}"
                    save_needs_georef_report(needs_georef_df, field_name, TWIN_CACHE_DIR_OBJ)
                
                sheets[sheet_name] = normalized_df
                logger.info(f"Loaded sheet '{sheet_name}' with {len(normalized_df)} samples")
                
            except Exception as e:
                logger.error(f"Failed to load sheet '{sheet_name}': {e}")
                continue
        
        return sheets
        
    except Exception as e:
        logger.error(f"Failed to load Excel file {path}: {e}")
        return {}

def load_field_data(field: str, data_dir: Path) -> Dict[str, Union[pd.DataFrame, gpd.GeoDataFrame, Dict]]:
    """
    Load all data for a specific field.
    
    Args:
        field: Field name (e.g., "Semurup")
        data_dir: Path to data directory
    
    Returns:
        Dictionary with 'model_df', 'geochem_df', 'structures_gdf' keys
    """
    result = {
        'model_df': pd.DataFrame(),
        'geochem_df': pd.DataFrame(),
        'structures_gdf': None
    }
    
    # Load 3D model data (resistivity/density)
    model_dir = data_dir / "3d_models"
    if model_dir.exists():
        logger.info(f"Loading 3D models from {model_dir}")
        
        # Look for any .dat files in the 3d_models directory
        model_files = list(model_dir.glob("*.dat"))
        logger.info(f"Found {len(model_files)} model files: {[f.name for f in model_files]}")
        
        resistivity_df = pd.DataFrame()
        density_df = pd.DataFrame()
        
        for model_file in model_files:
            if "res" in model_file.name.lower():
                logger.info(f"Loading resistivity model: {model_file.name}")
                df = load_xyz_dat(model_file, "Resistivity")
                if not df.empty:
                    resistivity_df = df
                    logger.info(f"Loaded resistivity model: {len(df)} points")
                    logger.info(f"Resistivity range: {df['Resistivity'].min():.2e} - {df['Resistivity'].max():.2e}")
                else:
                    logger.error(f"Failed to load resistivity model: {model_file.name}")
                    
            elif "dens" in model_file.name.lower():
                logger.info(f"Loading density model: {model_file.name}")
                df = load_xyz_dat(model_file, "Density")
                if not df.empty:
                    density_df = df
                    logger.info(f"Loaded density model: {len(df)} points")
                    logger.info(f"Density range: {df['Density'].min():.2f} - {df['Density'].max():.2f}")
                else:
                    logger.error(f"Failed to load density model: {model_file.name}")
        
        # Merge resistivity and density data
        if not resistivity_df.empty and not density_df.empty:
            logger.info("Merging resistivity and density data")
            # Merge on X, Y, Z coordinates
            result['model_df'] = resistivity_df.merge(density_df, on=['X', 'Y', 'Z'], how='outer')
            logger.info(f"Merged model data: {len(result['model_df'])} points")
            logger.info(f"Final columns: {list(result['model_df'].columns)}")
        elif not resistivity_df.empty:
            logger.info("Using resistivity data only")
            result['model_df'] = resistivity_df
        elif not density_df.empty:
            logger.info("Using density data only")
            result['model_df'] = density_df
        else:
            logger.warning("No valid model data loaded")
    
    # Load geochemistry data
    geochem_dir = data_dir / "geochem"
    if geochem_dir.exists():
        geochem_dfs = []
        
        for geochem_file in geochem_dir.glob(f"*{field}*.xlsx"):
            sheets = load_geochem_xlsx(geochem_file)
            for sheet_name, df in sheets.items():
                if not df.empty:
                    geochem_dfs.append(df)
        
        for geochem_file in geochem_dir.glob(f"*{field}*.csv"):
            df = load_geochem_csv_tsv(geochem_file)
            if not df.empty:
                geochem_dfs.append(df)
        
        # Combine all geochemistry data
        if geochem_dfs:
            result['geochem_df'] = pd.concat(geochem_dfs, ignore_index=True)
            logger.info(f"Combined geochemistry data: {len(result['geochem_df'])} samples")
    
    # Load structural data
    struct_dir = data_dir / "shapefiles"
    if struct_dir.exists():
        for shp_file in struct_dir.glob(f"*{field}*.shp"):
            gdf = load_structures(shp_file)
            if gdf is not None:
                result['structures_gdf'] = gdf
                break
    
    return result

def validate_loaded_data(data_dict: Dict) -> bool:
    """
    Validate that loaded data has required coordinate columns.
    
    Args:
        data_dict: Dictionary from load_field_data
    
    Returns:
        True if data is valid
    """
    from .schema import validate_coordinates
    
    # Check model data
    if not data_dict['model_df'].empty:
        if not validate_coordinates(data_dict['model_df']):
            logger.error("Model data missing X, Y coordinates")
            return False
    
    # Check geochemistry data
    if not data_dict['geochem_df'].empty:
        if not validate_coordinates(data_dict['geochem_df']):
            logger.error("Geochemistry data missing X, Y coordinates")
            return False
    
    return True
