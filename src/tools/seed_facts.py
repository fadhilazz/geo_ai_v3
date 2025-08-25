"""Seed facts loader for field-specific knowledge."""

import logging
import yaml
import os
from pathlib import Path
from typing import Dict, List, Optional
from src.config import FACTS_DIR

logger = logging.getLogger(__name__)

def load_facts(field: str) -> List[Dict]:
    """Load facts for a specific field from YAML file.
    
    Args:
        field: Field name (e.g., "Semurup")
        
    Returns:
        List of fact dictionaries
    """
    p = Path(FACTS_DIR) / f"{field.lower()}_facts.yaml"
    if not p.exists(): 
        return []
    return yaml.safe_load(open(p, "r")).get("facts", [])

def as_evidence_bullets(facts: List[Dict]) -> List[str]:
    """Convert facts list to evidence bullet points.
    
    Args:
        facts: List of fact dictionaries
        
    Returns:
        List of evidence bullet points
    """
    out = []
    for f in facts:
        cite = f"[semurup_facts:{f.get('source_doc_id')} p.{f.get('page')}]"
        out.append(f"{f['key']}: {f['value']} {cite}")
    return out

def get_field_facts(field: str) -> List[str]:
    """Get facts for a field as evidence bullets.
    
    Args:
        field: Field name
        
    Returns:
        List of evidence bullet points
    """
    facts = load_facts(field)
    if facts:
        return as_evidence_bullets(facts)
    return []
