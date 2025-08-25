"""Digital Twin v2 caching for improved performance."""

import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
import geopandas as gpd
from src.config import TWIN_CACHE_DIR_OBJ

logger = logging.getLogger(__name__)

class TwinCache:
    """Cache manager for Digital Twin v2 data."""
    
    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or TWIN_CACHE_DIR_OBJ
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load cache metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load cache metadata: {e}")
        return {}
    
    def _save_metadata(self):
        """Save cache metadata."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")
    
    def cache_model_data(self, field: str, model_df: pd.DataFrame) -> bool:
        """
        Cache 3D model data as parquet.
        
        Args:
            field: Field name
            model_df: Model DataFrame
        
        Returns:
            True if caching successful
        """
        try:
            if model_df.empty:
                return False
            
            cache_file = self.cache_dir / f"{field}_model.parquet"
            model_df.to_parquet(cache_file, index=False)
            
            # Update metadata
            self.metadata[f"{field}_model"] = {
                "file": str(cache_file),
                "timestamp": time.time(),
                "rows": len(model_df),
                "columns": list(model_df.columns)
            }
            self._save_metadata()
            
            logger.info(f"Cached model data for {field}: {len(model_df)} rows")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache model data for {field}: {e}")
            return False
    
    def cache_geochem_data(self, field: str, geochem_df: pd.DataFrame) -> bool:
        """
        Cache geochemistry data as parquet.
        
        Args:
            field: Field name
            geochem_df: Geochemistry DataFrame
        
        Returns:
            True if caching successful
        """
        try:
            if geochem_df.empty:
                return False
            
            cache_file = self.cache_dir / f"{field}_geochem.parquet"
            geochem_df.to_parquet(cache_file, index=False)
            
            # Update metadata
            self.metadata[f"{field}_geochem"] = {
                "file": str(cache_file),
                "timestamp": time.time(),
                "rows": len(geochem_df),
                "columns": list(geochem_df.columns)
            }
            self._save_metadata()
            
            logger.info(f"Cached geochemistry data for {field}: {len(geochem_df)} rows")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache geochemistry data for {field}: {e}")
            return False
    
    def cache_structures(self, field: str, structures_gdf: gpd.GeoDataFrame) -> bool:
        """
        Cache structural data as GeoJSON.
        
        Args:
            field: Field name
            structures_gdf: Structural GeoDataFrame
        
        Returns:
            True if caching successful
        """
        try:
            if structures_gdf is None or structures_gdf.empty:
                return False
            
            cache_file = self.cache_dir / f"{field}_structures.geojson"
            structures_gdf.to_file(cache_file, driver='GeoJSON')
            
            # Update metadata
            self.metadata[f"{field}_structures"] = {
                "file": str(cache_file),
                "timestamp": time.time(),
                "features": len(structures_gdf),
                "crs": str(structures_gdf.crs) if structures_gdf.crs else "None"
            }
            self._save_metadata()
            
            logger.info(f"Cached structures for {field}: {len(structures_gdf)} features")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache structures for {field}: {e}")
            return False
    
    def load_cached_model_data(self, field: str) -> Optional[pd.DataFrame]:
        """
        Load cached 3D model data.
        
        Args:
            field: Field name
        
        Returns:
            Model DataFrame or None
        """
        try:
            cache_file = self.cache_dir / f"{field}_model.parquet"
            if cache_file.exists():
                df = pd.read_parquet(cache_file)
                logger.info(f"Loaded cached model data for {field}: {len(df)} rows")
                return df
        except Exception as e:
            logger.error(f"Failed to load cached model data for {field}: {e}")
        return None
    
    def load_cached_geochem_data(self, field: str) -> Optional[pd.DataFrame]:
        """
        Load cached geochemistry data.
        
        Args:
            field: Field name
        
        Returns:
            Geochemistry DataFrame or None
        """
        try:
            cache_file = self.cache_dir / f"{field}_geochem.parquet"
            if cache_file.exists():
                df = pd.read_parquet(cache_file)
                logger.info(f"Loaded cached geochemistry data for {field}: {len(df)} rows")
                return df
        except Exception as e:
            logger.error(f"Failed to load cached geochemistry data for {field}: {e}")
        return None
    
    def load_cached_structures(self, field: str) -> Optional[gpd.GeoDataFrame]:
        """
        Load cached structural data.
        
        Args:
            field: Field name
        
        Returns:
            Structural GeoDataFrame or None
        """
        try:
            cache_file = self.cache_dir / f"{field}_structures.geojson"
            if cache_file.exists():
                gdf = gpd.read_file(cache_file)
                logger.info(f"Loaded cached structures for {field}: {len(gdf)} features")
                return gdf
        except Exception as e:
            logger.error(f"Failed to load cached structures for {field}: {e}")
        return None
    
    def is_cache_fresh(self, field: str, data_type: str, max_age_hours: int = 24) -> bool:
        """
        Check if cached data is fresh.
        
        Args:
            field: Field name
            data_type: Type of data ('model', 'geochem', 'structures')
            max_age_hours: Maximum age in hours
        
        Returns:
            True if cache is fresh
        """
        key = f"{field}_{data_type}"
        if key not in self.metadata:
            return False
        
        timestamp = self.metadata[key].get("timestamp", 0)
        age_hours = (time.time() - timestamp) / 3600
        return age_hours <= max_age_hours
    
    def clear_cache(self, field: str = None):
        """
        Clear cache for a field or all fields.
        
        Args:
            field: Field name (None for all fields)
        """
        try:
            if field:
                # Clear specific field
                patterns = [f"{field}_*"]
            else:
                # Clear all
                patterns = ["*_model.parquet", "*_geochem.parquet", "*_structures.geojson"]
            
            for pattern in patterns:
                for cache_file in self.cache_dir.glob(pattern):
                    cache_file.unlink()
                    logger.info(f"Cleared cache file: {cache_file}")
            
            # Update metadata
            if field:
                keys_to_remove = [k for k in self.metadata.keys() if k.startswith(f"{field}_")]
                for key in keys_to_remove:
                    del self.metadata[key]
            else:
                self.metadata.clear()
            
            self._save_metadata()
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        stats = {
            "cache_dir": str(self.cache_dir),
            "total_files": 0,
            "total_size_mb": 0,
            "fields": {}
        }
        
        try:
            for cache_file in self.cache_dir.glob("*"):
                if cache_file.is_file() and cache_file.suffix in ['.parquet', '.geojson']:
                    stats["total_files"] += 1
                    stats["total_size_mb"] += cache_file.stat().st_size / (1024 * 1024)
            
            # Field-specific stats
            for key, info in self.metadata.items():
                field = key.split('_')[0]
                if field not in stats["fields"]:
                    stats["fields"][field] = {}
                
                data_type = key.split('_', 1)[1]
                stats["fields"][field][data_type] = {
                    "rows": info.get("rows", 0),
                    "timestamp": info.get("timestamp", 0),
                    "age_hours": (time.time() - info.get("timestamp", 0)) / 3600
                }
        
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
        
        return stats

# Global cache instance
_twin_cache = None

def get_twin_cache() -> TwinCache:
    """Get the global twin cache instance."""
    global _twin_cache
    if _twin_cache is None:
        _twin_cache = TwinCache()
    return _twin_cache
