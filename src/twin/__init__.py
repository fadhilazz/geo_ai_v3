"""Digital Twin v2 - Data-only engine with summaries and LangGraph adapter."""

from .schema import (
    CoordMapping, TwinMetrics, CaprockMetrics, ReservoirMetrics,
    StructureMetrics, GeochemMetrics, TwinSummary, TwinQueryRequest, TwinQueryResponse,
    INTENT_TO_SUMMARY, validate_coordinates, validate_numeric_coordinates
)

from .normalize import (
    detect_coord_columns, to_utm_xyz, save_needs_georef_report
)

from .io import (
    load_xyz_dat, load_structures, load_geochem_csv_tsv, load_geochem_xlsx,
    load_field_data, validate_loaded_data
)

from .metrics import (
    caprock_iso, reservoir_rocks_from_density, reservoir_iso,
    flow_direction_from_gradient, distance_to_faults,
    calculate_connectivity_score, analyze_geochemistry
)

from .summary import (
    build_summary, load_summary, get_summary_sections,
    is_summary_fresh, list_available_summaries
)

from .registry import (
    TwinRegistry, get_twin_registry, register_field,
    get_field_data, list_available_fields
)

from .adapter import (
    twin_summary, twin_query, clarify_needed, get_twin_context,
    is_numeric_question, should_use_twin_summary
)

from .cache import (
    TwinCache, get_twin_cache
)

__version__ = "2.0.0"
__all__ = [
    # Schema
    "CoordMapping", "TwinMetrics", "CaprockMetrics", "ReservoirMetrics",
    "StructureMetrics", "GeochemMetrics", "TwinSummary", "TwinQueryRequest", "TwinQueryResponse",
    "INTENT_TO_SUMMARY", "validate_coordinates", "validate_numeric_coordinates",
    
    # Normalize
    "detect_coord_columns", "to_utm_xyz", "save_needs_georef_report",
    
    # I/O
    "load_xyz_dat", "load_structures", "load_geochem_csv_tsv", "load_geochem_xlsx",
    "load_field_data", "validate_loaded_data",
    
    # Metrics
    "caprock_iso", "reservoir_rocks_from_density", "reservoir_iso",
    "flow_direction_from_gradient", "distance_to_faults",
    "calculate_connectivity_score", "analyze_geochemistry",
    
    # Summary
    "build_summary", "load_summary", "get_summary_sections",
    "is_summary_fresh", "list_available_summaries",
    
    # Registry
    "TwinRegistry", "get_twin_registry", "register_field",
    "get_field_data", "list_available_fields",
    
    # Adapter
    "twin_summary", "twin_query", "clarify_needed", "get_twin_context",
    "is_numeric_question", "should_use_twin_summary",
    
    # Cache
    "TwinCache", "get_twin_cache"
]
