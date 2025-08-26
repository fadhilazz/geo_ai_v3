"""Digital Twin v2 schema definitions and data models."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from pydantic import BaseModel
import pandas as pd
import geopandas as gpd

# Intent to summary mapping for focused retrieval
INTENT_TO_SUMMARY = {
    "Caprock_Location": ["caprock", "structure"],
    "Reservoir_RockType": ["reservoir", "geochem"],
    "Hydrology_Direction": ["hydrology", "caprock", "reservoir"],
    "Wells_Targeting": ["reservoir", "caprock", "structure", "geochem"],
    "Temperature_Distribution": ["geochem", "reservoir"],
    "Structure_Analysis": ["structure", "caprock"],
    "Geochemistry_Analysis": ["geochem", "reservoir"],
    "Resistivity_Analysis": ["caprock", "reservoir"],
    "Density_Analysis": ["reservoir", "structure"]
}

@dataclass
class CoordMapping:
    """Coordinate column mapping with confidence scores."""
    x_col: str
    y_col: str
    z_col: Optional[str] = None
    confidence: float = 1.0
    crs_hint: Optional[str] = None

@dataclass
class TwinMetrics:
    """Standardized metrics output from twin analysis."""
    points: int
    extent_km2: float
    depth_range: Optional[Tuple[float, float]] = None
    value_range: Optional[Tuple[float, float]] = None
    centroids: Optional[List[Tuple[float, float]]] = None
    dominant_type: Optional[str] = None
    connectivity_score: Optional[float] = None

@dataclass
class CaprockMetrics(TwinMetrics):
    """Caprock-specific metrics."""
    threshold_ohmm: float = 10.0
    below_surface: bool = True
    clay_cap_interpretation: bool = True

@dataclass
class ReservoirMetrics(TwinMetrics):
    """Reservoir-specific metrics."""
    res_low_ohmm: float = 50.0
    res_high_ohmm: float = 200.0
    rho_max: Optional[float] = None
    dominant_rock: Optional[str] = None
    density_stats: Optional[Dict] = None

@dataclass
class StructureMetrics:
    """Structure analysis metrics."""
    n_features: int
    geom_types: List[str]
    crs: str
    fault_lengths: Optional[List[float]] = None
    fault_orientations: Optional[List[float]] = None

@dataclass
class GeochemMetrics:
    """Geochemistry analysis metrics."""
    n_samples: int
    has_coords: float  # percentage
    anomaly_counts: Dict[str, int]
    temperature_range: Optional[Tuple[float, float]] = None
    ph_range: Optional[Tuple[float, float]] = None

class TwinSummary(BaseModel):
    """Complete twin summary for a field."""
    field: str
    bounds: Dict[str, Union[List[float], int]]
    caprock: Dict
    reservoir: Dict
    hydrology: Dict
    structure: Dict
    geochem: Dict
    last_updated: str

class TwinQueryRequest(BaseModel):
    """Request model for twin queries."""
    field: str
    intent_tag: str
    params: Optional[Dict] = None

class TwinQueryResponse(BaseModel):
    """Response model for twin queries."""
    field: str
    intent_tag: str
    metrics: Dict
    execution_time_ms: float
    cache_hit: bool = False

def validate_coordinates(df: pd.DataFrame) -> bool:
    """Validate that DataFrame has proper coordinate columns."""
    required_cols = ['X', 'Y']
    return all(col in df.columns for col in required_cols)

def validate_numeric_coordinates(df: pd.DataFrame) -> bool:
    """Validate that coordinate columns are numeric."""
    if not validate_coordinates(df):
        return False
    return df['X'].dtype in ['float64', 'int64'] and df['Y'].dtype in ['float64', 'int64']
