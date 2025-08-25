"""Field detection from Chroma metadata using fuzzy matching."""

import logging
from typing import Optional, Tuple, Set
from rapidfuzz import fuzz, process
import chromadb

logger = logging.getLogger(__name__)

def get_available_fields(chroma_dir: str) -> Set[str]:
    """Build set of available fields from Chroma metadata.
    
    Args:
        chroma_dir: Path to Chroma database directory
        
    Returns:
        Set of unique field names
    """
    try:
        client = chromadb.PersistentClient(path=chroma_dir)
        collection = client.get_collection("text_emb")
        
        # Get all metadata
        results = collection.get(include=['metadatas'])
        metadatas = results['metadatas']
        
        fields = set()
        for metadata in metadatas:
            if metadata and 'field' in metadata:
                field = metadata['field']
                if field and field != 'NONE':
                    fields.add(str(field))
        
        logger.info(f"Found {len(fields)} unique fields in Chroma: {sorted(fields)}")
        return fields
        
    except Exception as e:
        logger.warning(f"Error getting fields from Chroma: {e}")
        return set()

def detect_field(question: str, available_fields: Set[str], cutoff: int = 85) -> Tuple[Optional[str], float]:
    """Detect field from question using fuzzy matching.
    
    Args:
        question: User question
        available_fields: Set of available field names
        cutoff: Minimum similarity score (0-100)
        
    Returns:
        Tuple of (detected_field, confidence_score)
    """
    if not available_fields:
        return None, 0.0
    
    question_lower = question.lower()
    
    # Try exact matches first
    for field in available_fields:
        if field.lower() in question_lower:
            logger.info(f"Exact field match found: {field}")
            return field, 100.0
    
    # Use fuzzy matching
    try:
        # Find best match using rapidfuzz
        best_match = process.extractOne(
            question, 
            available_fields,
            scorer=fuzz.partial_ratio,
            score_cutoff=cutoff
        )
        
        if best_match:
            field, score = best_match
            logger.info(f"Fuzzy field match: '{field}' (score: {score})")
            return field, score
        else:
            logger.info(f"No field detected (best score below cutoff {cutoff})")
            return None, 0.0
            
    except Exception as e:
        logger.warning(f"Error in field detection: {e}")
        return None, 0.0

def is_general_question(question: str) -> bool:
    """Check if question is general (not field-specific).
    
    Args:
        question: User question
        
    Returns:
        True if question appears to be general
    """
    general_indicators = [
        'geothermal', 'energy', 'power', 'generation', 'technology',
        'what is', 'how does', 'explain', 'describe', 'overview',
        'general', 'basics', 'fundamentals', 'principles'
    ]
    
    question_lower = question.lower()
    for indicator in general_indicators:
        if indicator in question_lower:
            return True
    
    return False
