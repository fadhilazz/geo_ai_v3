"""Seed facts loader for field-specific knowledge."""

import logging
import yaml
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

def load_facts(field: str, facts_dir: str) -> Optional[Dict]:
    """Load facts for a specific field from YAML file.
    
    Args:
        field: Field name (e.g., "Semurup")
        facts_dir: Directory containing facts files
        
    Returns:
        Dictionary of facts or None if not found
    """
    try:
        facts_path = Path(facts_dir) / f"{field.lower()}_facts.yaml"
        
        if not facts_path.exists():
            logger.debug(f"Facts file not found: {facts_path}")
            return None
        
        with open(facts_path, 'r', encoding='utf-8') as f:
            facts = yaml.safe_load(f)
        
        logger.info(f"Loaded facts for {field}: {len(facts) if facts else 0} fact categories")
        return facts
        
    except Exception as e:
        logger.warning(f"Error loading facts for {field}: {e}")
        return None

def as_evidence_bullets(facts: Dict) -> List[str]:
    """Convert facts dictionary to evidence bullet points.
    
    Args:
        facts: Facts dictionary loaded from YAML
        
    Returns:
        List of evidence bullet points
    """
    bullets = []
    
    if not facts:
        return bullets
    
    try:
        for category, data in facts.items():
            if isinstance(data, dict):
                # Handle structured data
                if 'summary' in data:
                    bullets.append(f"• {category}: {data['summary']}")
                elif 'value' in data:
                    bullets.append(f"• {category}: {data['value']}")
                else:
                    # Convert dict to bullet points
                    for key, value in data.items():
                        if isinstance(value, (str, int, float)):
                            bullets.append(f"• {category} - {key}: {value}")
                        elif isinstance(value, list):
                            bullets.append(f"• {category} - {key}: {', '.join(map(str, value))}")
            elif isinstance(data, (str, int, float)):
                # Handle simple values
                bullets.append(f"• {category}: {data}")
            elif isinstance(data, list):
                # Handle lists
                bullets.append(f"• {category}: {', '.join(map(str, data))}")
    
    except Exception as e:
        logger.warning(f"Error converting facts to bullets: {e}")
    
    logger.info(f"Generated {len(bullets)} evidence bullets from facts")
    return bullets

def get_field_facts(field: str, facts_dir: str) -> List[str]:
    """Get facts for a field as evidence bullets.
    
    Args:
        field: Field name
        facts_dir: Facts directory path
        
    Returns:
        List of evidence bullet points
    """
    facts = load_facts(field, facts_dir)
    if facts:
        return as_evidence_bullets(facts)
    return []
