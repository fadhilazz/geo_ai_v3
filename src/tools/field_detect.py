"""Field detection with typo tolerance for QA engine."""

import logging
import re
from typing import List, Optional, Tuple, Union
from pathlib import Path

from rapidfuzz import process, fuzz

try:
    from ..config import FIELD_DETECTION_SCORE_CUTOFF
except ImportError:
    from src.config import FIELD_DETECTION_SCORE_CUTOFF
from .rag_text import get_text_rag
from .rag_image import get_image_rag

logger = logging.getLogger(__name__)


class FieldDetector:
    """Typo-tolerant field detection from user questions."""
    
    def __init__(self):
        """Initialize field detector."""
        self._known_fields = None
        self._field_aliases = {
            # Common variations and typos
            'semurup': 'Semurup',
            'semuru': 'Semurup', 
            'semurap': 'Semurup',
            'kamojang': 'Kamojang',
            'kamojng': 'Kamojang',
            'wayang': 'Wayang',
            'wayng': 'Wayang',
            'dieng': 'Dieng',
            'deng': 'Dieng',
            'sarulla': 'Sarulla',
            'sarula': 'Sarulla',
            'lahendong': 'Lahendong',
            'lahendng': 'Lahendong',
            'ulubelu': 'Ulubelu',
            'ulublu': 'Ulubelu',
        }
        
    def _load_known_fields(self) -> List[str]:
        """Load known fields from Chroma databases.
        
        Returns:
            List of unique field names
        """
        if self._known_fields is not None:
            return self._known_fields
            
        fields = set()
        
        try:
            # Get fields from text RAG
            text_rag = get_text_rag()
            text_fields = text_rag.get_all_fields()
            fields.update(text_fields)
            logger.debug(f"Found {len(text_fields)} fields from text collection")
            
        except Exception as e:
            logger.warning(f"Could not get fields from text collection: {e}")
            
        try:
            # Get fields from image RAG
            image_rag = get_image_rag()
            image_fields = image_rag.get_all_fields()
            fields.update(image_fields)
            logger.debug(f"Found {len(image_fields)} fields from image collection")
            
        except Exception as e:
            logger.warning(f"Could not get fields from image collection: {e}")
            
        # Add known aliases
        fields.update(self._field_aliases.values())
        
        self._known_fields = sorted(list(fields))
        logger.info(f"Loaded {len(self._known_fields)} known fields: {self._known_fields}")
        
        return self._known_fields
        
    def _extract_field_candidates(self, question: str) -> List[str]:
        """Extract potential field names from question.
        
        Args:
            question: User's question
            
        Returns:
            List of potential field name candidates
        """
        candidates = []
        
        # Convert to lowercase for processing
        question_lower = question.lower()
        
        # Look for field-like patterns
        patterns = [
            # Direct field mentions
            r'\b(semurup|kamojang|wayang|dieng|sarulla|lahendong|ulubelu)\b',
            # "in X field" patterns
            r'\bin\s+(\w+)\s+field\b',
            # "at X" patterns  
            r'\bat\s+(\w+)\b',
            # "X area/region" patterns
            r'\b(\w+)\s+(?:area|region|field|site)\b',
            # Capitalized words that might be field names
            r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)?)\b'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, question_lower)
            if matches:
                if isinstance(matches[0], tuple):
                    candidates.extend([match for match in matches[0] if match])
                else:
                    candidates.extend(matches)
                    
        # Also check original case for capitalized field names
        words = question.split()
        for word in words:
            # Look for capitalized words that might be field names
            if word and word[0].isupper() and len(word) > 3:
                candidates.append(word.lower())
                
        # Remove duplicates and filter
        candidates = list(set(candidates))
        candidates = [c for c in candidates if len(c) > 2 and c.isalpha()]
        
        logger.debug(f"Extracted field candidates from '{question}': {candidates}")
        return candidates
        
    def detect_field(self, question: str) -> Tuple[Optional[str], int]:
        """Detect field name from user question with typo tolerance.
        
        Args:
            question: User's question
            
        Returns:
            Tuple of (field_name, confidence_score) or (None, 0)
        """
        known_fields = self._load_known_fields()
        
        if not known_fields:
            logger.warning("No known fields available for detection")
            return None, 0
            
        # Extract candidates from question
        candidates = self._extract_field_candidates(question)
        
        if not candidates:
            logger.debug(f"No field candidates found in question: '{question}'")
            return None, 0
            
        best_field = None
        best_score = 0
        
        # Check each candidate against known fields
        for candidate in candidates:
            # First check direct aliases
            if candidate in self._field_aliases:
                canonical_field = self._field_aliases[candidate]
                logger.info(f"Found field via alias: '{candidate}' -> '{canonical_field}'")
                return canonical_field, 100
                
            # Then try fuzzy matching
            matches = process.extract(
                candidate,
                known_fields,
                scorer=fuzz.WRatio,
                score_cutoff=FIELD_DETECTION_SCORE_CUTOFF,
                limit=3
            )
            
            for match, score, _ in matches:
                if score > best_score:
                    best_field = match
                    best_score = score
                    logger.debug(f"Fuzzy match: '{candidate}' -> '{match}' (score: {score})")
                    
        if best_field and best_score >= FIELD_DETECTION_SCORE_CUTOFF:
            logger.info(f"Detected field: '{best_field}' (confidence: {best_score})")
            return best_field, best_score
        else:
            logger.debug(f"No field detected with sufficient confidence. Best: '{best_field}' (score: {best_score})")
            return None, best_score
            
    def get_known_fields(self) -> List[str]:
        """Get list of all known fields.
        
        Returns:
            List of known field names
        """
        return self._load_known_fields()
        
    def add_field_alias(self, alias: str, canonical: str) -> None:
        """Add a field alias mapping.
        
        Args:
            alias: Alias/variation of field name
            canonical: Canonical field name
        """
        self._field_aliases[alias.lower()] = canonical
        logger.info(f"Added field alias: '{alias}' -> '{canonical}'")
        
    def reload_fields(self) -> None:
        """Force reload of known fields from databases."""
        self._known_fields = None
        self._load_known_fields()


# Global instance for caching
_field_detector_instance = None


def get_field_detector() -> FieldDetector:
    """Get cached FieldDetector instance.
    
    Returns:
        FieldDetector instance
    """
    global _field_detector_instance
    
    if _field_detector_instance is None:
        _field_detector_instance = FieldDetector()
        
    return _field_detector_instance


def detect_field_from_question(question: str) -> Tuple[Optional[str], int]:
    """Convenience function to detect field from question.
    
    Args:
        question: User's question
        
    Returns:
        Tuple of (field_name, confidence_score) or (None, 0)
    """
    detector = get_field_detector()
    return detector.detect_field(question)
