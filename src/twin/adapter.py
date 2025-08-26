"""Digital Twin v2 LangGraph adapter for agent-friendly queries."""

import logging
import re
import time
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from .registry import get_twin_registry, get_field_data
from .summary import get_summary_sections, load_summary
from .metrics import caprock_iso, reservoir_iso, reservoir_rocks_from_density
from src.config import ENABLE_TWIN_SUMMARY, ENABLE_TWIN_LIVE_QUERIES
import re

logger = logging.getLogger(__name__)

def twin_summary(field: str) -> Dict:
    """
    Get twin summary for a field (for general questions).
    
    Args:
        field: Field name
    
    Returns:
        Summary dictionary
    """
    if not ENABLE_TWIN_SUMMARY:
        return {"error": "Twin summary is disabled"}
    
    try:
        summary = load_summary(field)
        if summary:
            return summary
        else:
            return {"error": f"No summary found for field: {field}"}
    except Exception as e:
        logger.error(f"Error getting twin summary for {field}: {e}")
        return {"error": f"Failed to load summary: {str(e)}"}

def twin_query(field: str, intent_tag: str, twin_query_template: str, 
               params: Optional[Dict] = None) -> Dict:
    """
    Execute a twin query based on intent and template.
    
    Args:
        field: Field name
        intent_tag: Intent tag from Question Matrix
        twin_query_template: Query template (e.g., "resistivity < 10 ohm-m")
        params: Additional parameters
    
    Returns:
        Query results dictionary
    """
    if not ENABLE_TWIN_LIVE_QUERIES:
        return {"error": "Twin live queries are disabled"}
    
    start_time = time.time()
    
    try:
        # Get field data
        data_dict = get_field_data(field)
        if not data_dict or data_dict.get('model_df', pd.DataFrame()).empty:
            return {"error": f"No model data available for field: {field}"}
        
        model_df = data_dict['model_df']
        
        # Parse query template and execute appropriate metric function
        if intent_tag == "Caprock_Location":
            threshold = _extract_resistivity_threshold(twin_query_template, default=10.0)
            result = caprock_iso(model_df, res_threshold=threshold, below_surface=True)
            
        elif intent_tag == "Reservoir_RockType":
            result = reservoir_rocks_from_density(model_df)
            
        elif intent_tag == "Reservoir_Analysis":
            res_low = params.get('res_low', 50.0) if params else 50.0
            res_high = params.get('res_high', 200.0) if params else 200.0
            rho_max = params.get('rho_max') if params else None
            result = reservoir_iso(model_df, res_low=res_low, res_high=res_high, rho_max=rho_max)
            
        else:
            # Generic query - try to extract parameters from template
            result = _execute_generic_query(model_df, twin_query_template, params)
        
        execution_time_ms = (time.time() - start_time) * 1000
        
        return {
            "field": field,
            "intent_tag": intent_tag,
            "metrics": result,
            "execution_time_ms": execution_time_ms,
            "cache_hit": False  # TODO: Implement caching
        }
        
    except Exception as e:
        logger.error(f"Error executing twin query for {field}: {e}")
        return {"error": f"Query execution failed: {str(e)}"}

def clarify_needed(question: str, intent_tag: str) -> Optional[str]:
    """
    Check if clarification is needed for a twin query.
    
    Args:
        question: User question
        intent_tag: Intent tag from Question Matrix
    
    Returns:
        Clarification message or None
    """
    # Check for missing thresholds in caprock queries
    if intent_tag == "Caprock_Location":
        if not re.search(r'\d+\s*(?:ohm|Ω|ohm-m|Ω·m)', question, re.IGNORECASE):
            return "Please specify the resistivity threshold (e.g., 'resistivity < 10 ohm-m')"
    
    # Check for missing field specification
    if not re.search(r'\b(?:semurup|field|area)\b', question, re.IGNORECASE):
        return "Please specify which field you're asking about"
    
    # Check for missing parameters in reservoir queries
    if intent_tag == "Reservoir_Analysis":
        if not re.search(r'\d+\s*(?:ohm|Ω|ohm-m|Ω·m)', question, re.IGNORECASE):
            return "Please specify resistivity range (e.g., 'resistivity 50-200 ohm-m')"
    
    return None

def get_twin_context(field: str, intent_tag: str) -> Dict:
    """
    Get relevant twin context for an intent.
    
    Args:
        field: Field name
        intent_tag: Intent tag from Question Matrix
    
    Returns:
        Context dictionary
    """
    if not ENABLE_TWIN_SUMMARY:
        return {}
    
    try:
        # Get relevant summary sections
        context = get_summary_sections(field, intent_tag)
        
        # Add field info
        context["field"] = field
        context["intent_tag"] = intent_tag
        
        return context
        
    except Exception as e:
        logger.error(f"Error getting twin context for {field}: {e}")
        return {"error": f"Failed to get context: {str(e)}"}

def _extract_resistivity_threshold(template: str, default: float = 10.0) -> float:
    """Extract resistivity threshold from query template."""
    # Look for patterns like "resistivity < 10 ohm-m", "res < 5 Ω", etc.
    patterns = [
        r'resistivity\s*[<≤]\s*(\d+(?:\.\d+)?)\s*(?:ohm|Ω|ohm-m|Ω·m)',
        r'res\s*[<≤]\s*(\d+(?:\.\d+)?)\s*(?:ohm|Ω|ohm-m|Ω·m)',
        r'(\d+(?:\.\d+)?)\s*(?:ohm|Ω|ohm-m|Ω·m)\s*[<≤]',
        r'[<≤]\s*(\d+(?:\.\d+)?)\s*(?:ohm|Ω|ohm-m|Ω·m)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, template, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    
    return default

def _execute_generic_query(model_df: pd.DataFrame, template: str, params: Optional[Dict]) -> Dict:
    """Execute a generic query based on template parsing."""
    template_lower = template.lower()
    
    # Check for resistivity queries
    if 'resistivity' in template_lower or 'res' in template_lower:
        threshold = _extract_resistivity_threshold(template)
        return caprock_iso(model_df, res_threshold=threshold, below_surface=True)
    
    # Check for density queries
    if 'density' in template_lower or 'rho' in template_lower:
        return reservoir_rocks_from_density(model_df)
    
    # Default to basic statistics
    return {
        "total_points": len(model_df),
        "bounds": {
            "x": [float(model_df['X'].min()), float(model_df['X'].max())],
            "y": [float(model_df['Y'].min()), float(model_df['Y'].max())],
            "z": [float(model_df['Z'].min()), float(model_df['Z'].max())] if 'Z' in model_df.columns else [0, 0]
        },
        "available_columns": list(model_df.columns)
    }

def is_numeric_question(question: str) -> bool:
    """
    Determine if a question requires numeric twin analysis.
    
    Args:
        question: User question
    
    Returns:
        True if question requires numeric analysis
    """
    numeric_indicators = [
        r'\d+\s*(?:ohm|Ω|ohm-m|Ω·m)',  # Resistivity thresholds
        r'\d+\s*(?:g/cm3|g/cc)',       # Density thresholds
        r'\d+\s*(?:m|km|ft)',          # Distance/depth thresholds
        r'\d+\s*(?:°c|celsius|f)',     # Temperature thresholds
        r'(?:less than|greater than|above|below)\s*\d+',  # Comparative thresholds
        r'(?:threshold|limit|minimum|maximum)',           # Threshold keywords
        r'(?:area|extent|volume|thickness)',              # Spatial measurements
    ]
    
    question_lower = question.lower()
    for pattern in numeric_indicators:
        if re.search(pattern, question_lower, re.IGNORECASE):
            return True
    
    return False

def should_use_twin_summary(question: str, intent_tag: str) -> bool:
    """
    Determine if twin summary should be used instead of live query.
    
    Args:
        question: User question
        intent_tag: Intent tag from Question Matrix
    
    Returns:
        True if summary should be used
    """
    # Use summary for general questions
    if not is_numeric_question(question):
        return True
    
    # Use summary for overview questions
    overview_keywords = ['overview', 'summary', 'general', 'describe', 'what is', 'tell me about']
    question_lower = question.lower()
    if any(keyword in question_lower for keyword in overview_keywords):
        return True
    
    # Use live query for specific numeric questions
    return False
