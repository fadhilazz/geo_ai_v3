"""Digital Twin v2 registry for managing multiple fields."""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
import geopandas as gpd
from .io import load_field_data, validate_loaded_data
from .summary import build_summary, load_summary, is_summary_fresh
from src.config import TWIN_DATA_DIR_OBJ, TWIN_CACHE_DIR_OBJ

logger = logging.getLogger(__name__)

class TwinRegistry:
    """Registry for managing multiple field twins."""
    
    def __init__(self):
        self.fields: Dict[str, Dict] = {}
        self.last_loaded: Dict[str, float] = {}
        self.cache_duration = 3600  # 1 hour cache
    
    def register_field(self, field: str, model_paths: List[Path] = None, 
                      struct_path: Path = None, geochem_paths: List[Path] = None) -> bool:
        """
        Register a field with its data paths.
        
        Args:
            field: Field name
            model_paths: List of model file paths (.dat files)
            struct_path: Structure file path (.shp file)
            geochem_paths: List of geochemistry file paths
        
        Returns:
            True if registration successful
        """
        try:
            logger.info(f"Registering field: {field}")
            
            # Load field data
            data_dict = load_field_data(field, TWIN_DATA_DIR_OBJ)
            
            # Validate loaded data
            if not validate_loaded_data(data_dict):
                logger.error(f"Data validation failed for field: {field}")
                return False
            
            # Store in registry
            self.fields[field] = data_dict
            self.last_loaded[field] = time.time()
            
            # Build summary
            summary = build_summary(
                field, 
                data_dict['model_df'], 
                data_dict['geochem_df'], 
                data_dict['structures_gdf']
            )
            
            logger.info(f"Successfully registered field: {field}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register field {field}: {e}")
            return False
    
    def get(self, field: str) -> Optional[Dict]:
        """
        Get field data and summary.
        
        Args:
            field: Field name
        
        Returns:
            Dictionary with model_df, geochem_df, structures_gdf, summary
        """
        # Check if field is cached and fresh
        if field in self.fields and self._is_cache_fresh(field):
            logger.info(f"Using cached data for field: {field}")
            data_dict = self.fields[field].copy()
            data_dict['summary'] = load_summary(field)
            return data_dict
        
        # Try to load from disk
        summary = load_summary(field)
        if summary:
            logger.info(f"Loaded summary from disk for field: {field}")
            # Load actual data if summary exists
            try:
                data_dict = load_field_data(field, TWIN_DATA_DIR_OBJ)
                data_dict['summary'] = summary
                # Cache the loaded data
                self.fields[field] = data_dict
                self.last_loaded[field] = time.time()
                return data_dict
            except Exception as e:
                logger.warning(f"Failed to load data for field {field}: {e}")
                return {
                    'model_df': pd.DataFrame(),
                    'geochem_df': pd.DataFrame(),
                    'structures_gdf': None,
                    'summary': summary
                }
        
        logger.warning(f"Field not found: {field}")
        return None
    
    def get_model_data(self, field: str) -> Optional[pd.DataFrame]:
        """Get 3D model data for a field."""
        data_dict = self.get(field)
        if data_dict:
            return data_dict.get('model_df', pd.DataFrame())
        return None
    
    def get_geochem_data(self, field: str) -> Optional[pd.DataFrame]:
        """Get geochemistry data for a field."""
        data_dict = self.get(field)
        if data_dict:
            return data_dict.get('geochem_df', pd.DataFrame())
        return None
    
    def get_structures(self, field: str) -> Optional[gpd.GeoDataFrame]:
        """Get structural data for a field."""
        data_dict = self.get(field)
        if data_dict:
            return data_dict.get('structures_gdf')
        return None
    
    def get_summary(self, field: str) -> Optional[Dict]:
        """Get summary for a field."""
        data_dict = self.get(field)
        if data_dict:
            return data_dict.get('summary')
        return None
    
    def list_fields(self) -> List[str]:
        """List all registered fields."""
        return list(self.fields.keys())
    
    def list_available_fields(self) -> List[str]:
        """List all available fields (including those with summaries on disk)."""
        # Get registered fields
        registered = set(self.fields.keys())
        
        # Get fields with summaries on disk
        summary_dir = TWIN_CACHE_DIR_OBJ.parent / "summaries"
        if summary_dir.exists():
            for json_file in summary_dir.glob("*.json"):
                registered.add(json_file.stem)
        
        return sorted(list(registered))
    
    def refresh_field(self, field: str) -> bool:
        """
        Refresh field data and summary.
        
        Args:
            field: Field name
        
        Returns:
            True if refresh successful
        """
        logger.info(f"Refreshing field: {field}")
        return self.register_field(field)
    
    def remove_field(self, field: str) -> bool:
        """
        Remove field from registry.
        
        Args:
            field: Field name
        
        Returns:
            True if removal successful
        """
        if field in self.fields:
            del self.fields[field]
            del self.last_loaded[field]
            logger.info(f"Removed field from registry: {field}")
            return True
        return False
    
    def _is_cache_fresh(self, field: str) -> bool:
        """Check if cached data is fresh."""
        if field not in self.last_loaded:
            return False
        
        age = time.time() - self.last_loaded[field]
        return age < self.cache_duration
    
    def watch_directories(self, callback=None) -> None:
        """
        Watch data directories for changes (optional).
        
        Args:
            callback: Function to call when changes detected
        """
        # TODO: Implement file watching with watchdog
        logger.info("File watching not yet implemented")
        pass
    
    def get_field_stats(self, field: str) -> Dict:
        """
        Get statistics for a field.
        
        Args:
            field: Field name
        
        Returns:
            Dictionary with field statistics
        """
        data_dict = self.get(field)
        if not data_dict:
            return {"error": f"Field not found: {field}"}
        
        stats = {
            "field": field,
            "model_points": len(data_dict.get('model_df', pd.DataFrame())),
            "geochem_samples": len(data_dict.get('geochem_df', pd.DataFrame())),
            "has_structures": data_dict.get('structures_gdf') is not None,
            "has_summary": data_dict.get('summary') is not None,
            "cache_fresh": self._is_cache_fresh(field) if field in self.fields else False
        }
        
        # Add structure stats if available
        if data_dict.get('structures_gdf') is not None:
            gdf = data_dict['structures_gdf']
            stats["structure_features"] = len(gdf)
            stats["structure_types"] = list(gdf.geometry.geom_type.unique())
        
        return stats
    
    def list_available_fields(self) -> List[str]:
        """
        List all available fields.
        
        Returns:
            List of field names
        """
        # Return fields from registry
        registry_fields = list(self.fields.keys())
        
        # Also check for summaries on disk
        from .summary import list_available_summaries
        summary_fields = list_available_summaries()
        
        # Combine and deduplicate
        all_fields = list(set(registry_fields + summary_fields))
        return all_fields

# Global registry instance
_twin_registry = None

def get_twin_registry() -> TwinRegistry:
    """Get the global twin registry instance."""
    global _twin_registry
    if _twin_registry is None:
        _twin_registry = TwinRegistry()
    return _twin_registry

def register_field(field: str) -> bool:
    """Register a field using the global registry."""
    registry = get_twin_registry()
    return registry.register_field(field)

def get_field_data(field: str) -> Optional[Dict]:
    """Get field data using the global registry."""
    registry = get_twin_registry()
    return registry.get(field)

def list_available_fields() -> List[str]:
    """List available fields using the global registry."""
    registry = get_twin_registry()
    return registry.list_available_fields()
