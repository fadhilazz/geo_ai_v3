"""Question Matrix loader and routing for QA engine."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from ..config import (
        DEFAULT_TEXT_MODEL, QM_INTENT_CONFIDENCE_THRESHOLD,
        QUESTION_MATRIX_PATH_OBJ
    )
except ImportError:
    from src.config import (
        DEFAULT_TEXT_MODEL, QM_INTENT_CONFIDENCE_THRESHOLD,
        QUESTION_MATRIX_PATH_OBJ
    )

logger = logging.getLogger(__name__)


class QuestionMatrix:
    """Question Matrix loader and intent router."""
    
    def __init__(self, matrix_path: Union[str, Path] = None):
        """Initialize Question Matrix.
        
        Args:
            matrix_path: Path to Question Matrix Excel/CSV file
        """
        self.matrix_path = Path(matrix_path) if matrix_path else QUESTION_MATRIX_PATH_OBJ
        self.rows = []
        self.by_intent = {}
        self.aspects = set()
        self.methods = set()
        self.topics = set()
        self.embedder = None
        self.question_embeddings = None
        
    def _load_embedder(self):
        """Lazy load the sentence transformer model."""
        if self.embedder is None:
            logger.info(f"Loading intent detection model: {DEFAULT_TEXT_MODEL}")
            self.embedder = SentenceTransformer(DEFAULT_TEXT_MODEL)
            
    def _normalize_header(self, header: str) -> str:
        """Normalize column header."""
        if pd.isna(header):
            return ""
        return str(header).strip().lower().replace(' ', '_').replace('-', '_')
        
    def _parse_json_field(self, field_value: str) -> Union[Dict, List, str]:
        """Parse JSON-like field or return as string."""
        if pd.isna(field_value) or not field_value:
            return ""
            
        field_str = str(field_value).strip()
        if not field_str:
            return ""
            
        # Try to parse as JSON
        try:
            return json.loads(field_str)
        except (json.JSONDecodeError, ValueError):
            # Return as string if not valid JSON
            return field_str
            
    def _extract_terms(self, field_value: str) -> List[str]:
        """Extract terms from comma-separated or JSON field."""
        if pd.isna(field_value) or not field_value:
            return []
            
        field_str = str(field_value).strip()
        if not field_str:
            return []
            
        # Try JSON first
        try:
            parsed = json.loads(field_str)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if item]
            elif isinstance(parsed, str):
                return [parsed.strip()]
            else:
                return [str(parsed).strip()]
        except (json.JSONDecodeError, ValueError):
            # Fall back to comma-separated
            return [item.strip() for item in field_str.split(',') if item.strip()]
            
    def load_matrix(self) -> None:
        """Load Question Matrix from Excel or CSV file."""
        logger.info(f"Loading Question Matrix from {self.matrix_path}")
        
        if not self.matrix_path.exists():
            logger.warning(f"Question Matrix file not found: {self.matrix_path}")
            logger.info("Creating empty Question Matrix")
            return
            
        try:
            # Determine file type and load
            if self.matrix_path.suffix.lower() == '.csv':
                df = pd.read_csv(self.matrix_path)
            else:
                df = pd.read_excel(self.matrix_path)
                
            # Normalize column names
            df.columns = [self._normalize_header(col) for col in df.columns]
            
            # Expected columns (with normalized names)
            expected_cols = {
                'id', 'user_question', 'intent_tag', 'primary_aspect', 
                'secondary_aspects', 'discipline_hints', 'method_or_topic_hints',
                'requires_twin', 'twin_query_template', 'retrieval_filter_hint',
                'expected_outputs', 'eval_keywords', 'priority'
            }
            
            # Process each row
            for _, row in df.iterrows():
                if pd.isna(row.get('user_question', '')) or pd.isna(row.get('intent_tag', '')):
                    continue
                    
                # Build row dict with normalized structure
                row_data = {
                    'id': str(row.get('id', '')).strip(),
                    'user_question': str(row.get('user_question', '')).strip(),
                    'intent_tag': str(row.get('intent_tag', '')).strip(),
                    'primary_aspect': str(row.get('primary_aspect', '')).strip(),
                    'secondary_aspects': self._extract_terms(row.get('secondary_aspects', '')),
                    'discipline_hints': self._extract_terms(row.get('discipline_hints', '')),
                    'method_or_topic_hints': self._extract_terms(row.get('method_or_topic_hints', '')),
                    'requires_twin': str(row.get('requires_twin', 'false')).lower() == 'true',
                    'twin_query_template': str(row.get('twin_query_template', '')).strip(),
                    'retrieval_filter_hint': self._parse_json_field(row.get('retrieval_filter_hint', '')),
                    'expected_outputs': self._extract_terms(row.get('expected_outputs', '')),
                    'eval_keywords': self._extract_terms(row.get('eval_keywords', '')),
                    'priority': int(row.get('priority', 1)) if pd.notna(row.get('priority')) else 1
                }
                
                self.rows.append(row_data)
                
                # Build intent index
                intent = row_data['intent_tag']
                if intent not in self.by_intent:
                    self.by_intent[intent] = []
                self.by_intent[intent].append(row_data)
                
                # Collect unique terms
                if row_data['primary_aspect']:
                    self.aspects.add(row_data['primary_aspect'])
                self.aspects.update(row_data['secondary_aspects'])
                
                for hint in row_data['method_or_topic_hints']:
                    # Simple heuristic: if it looks like a method (short, uppercase)
                    if len(hint) <= 5 and hint.isupper():
                        self.methods.add(hint)
                    else:
                        self.topics.add(hint)
                        
            logger.info(f"Loaded {len(self.rows)} question patterns")
            logger.info(f"Found {len(self.by_intent)} intent categories")
            logger.info(f"Extracted {len(self.aspects)} aspects, {len(self.methods)} methods, {len(self.topics)} topics")
            
            # Pre-compute embeddings for intent detection
            self._compute_question_embeddings()
            
        except Exception as e:
            logger.error(f"Error loading Question Matrix: {e}")
            raise
            
    def _compute_question_embeddings(self) -> None:
        """Pre-compute embeddings for all user questions."""
        if not self.rows:
            return
            
        self._load_embedder()
        
        questions = [row['user_question'] for row in self.rows]
        logger.info(f"Computing embeddings for {len(questions)} question examples")
        
        self.question_embeddings = self.embedder.encode(questions, convert_to_numpy=True)
        
    def infer_intent(self, question: str) -> Tuple[Optional[str], float]:
        """Infer intent from user question using semantic similarity.
        
        Args:
            question: User's question
            
        Returns:
            Tuple of (intent_tag, confidence_score)
        """
        if not self.rows or self.question_embeddings is None:
            return None, 0.0
            
        self._load_embedder()
        
        # Encode the user question
        question_embed = self.embedder.encode([question], convert_to_numpy=True)[0]
        
        # Compute similarities with all example questions
        similarities = np.dot(self.question_embeddings, question_embed)
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        best_intent = self.rows[best_idx]['intent_tag']
        
        # Also check for keyword matches to boost confidence
        question_lower = question.lower()
        keyword_boost = 0.0
        
        for row in self.rows:
            if row['intent_tag'] == best_intent:
                for keyword in row['eval_keywords']:
                    if keyword.lower() in question_lower:
                        keyword_boost += 0.1
                        
        final_confidence = min(best_score + keyword_boost, 1.0)
        
        logger.debug(f"Intent inference: '{question}' -> {best_intent} (confidence: {final_confidence:.3f})")
        
        if final_confidence >= QM_INTENT_CONFIDENCE_THRESHOLD:
            return best_intent, final_confidence
        else:
            return None, final_confidence
            
    def filters_for_intent(self, intent_tag: str, field: Optional[str] = None) -> Dict:
        """Build retrieval filters for given intent and field.
        
        Args:
            intent_tag: Intent category
            field: Optional field name (e.g., "Semurup")
            
        Returns:
            Dictionary of filters for Chroma where clause
        """
        filters = {}
        
        # Add field filter if provided
        if field:
            filters['field'] = field
            
        # Get intent-specific filters
        if intent_tag in self.by_intent:
            # Use the first (highest priority) row for this intent
            intent_rows = sorted(self.by_intent[intent_tag], key=lambda x: x['priority'])
            primary_row = intent_rows[0]
            
            # Parse retrieval filter hint
            filter_hint = primary_row['retrieval_filter_hint']
            if isinstance(filter_hint, dict):
                filters.update(filter_hint)
            elif isinstance(filter_hint, str) and filter_hint:
                # Try to parse as simple key:value pairs
                try:
                    hint_dict = json.loads(filter_hint)
                    if isinstance(hint_dict, dict):
                        filters.update(hint_dict)
                except (json.JSONDecodeError, ValueError):
                    # Parse simple format like "aspect:Caprock,method:MT"
                    for pair in filter_hint.split(','):
                        if ':' in pair:
                            key, value = pair.split(':', 1)
                            filters[key.strip()] = value.strip()
                            
            # Add aspect filters
            if primary_row['primary_aspect']:
                if 'aspect' not in filters:
                    filters['aspect'] = primary_row['primary_aspect']
                    
            # Add discipline filters
            if primary_row['discipline_hints']:
                if 'discipline' not in filters:
                    filters['discipline'] = primary_row['discipline_hints'][0]
                    
            # Add method/topic filters
            if primary_row['method_or_topic_hints']:
                hint = primary_row['method_or_topic_hints'][0]
                if hint in self.methods and 'method' not in filters:
                    filters['method'] = hint
                elif hint in self.topics and 'topic' not in filters:
                    filters['topic'] = hint
                    
        logger.debug(f"Filters for intent '{intent_tag}', field '{field}': {filters}")
        return filters
        
    def get_intent_info(self, intent_tag: str) -> Optional[Dict]:
        """Get information about a specific intent.
        
        Args:
            intent_tag: Intent category
            
        Returns:
            Dictionary with intent information or None
        """
        if intent_tag not in self.by_intent:
            return None
            
        intent_rows = sorted(self.by_intent[intent_tag], key=lambda x: x['priority'])
        primary_row = intent_rows[0]
        
        return {
            'intent_tag': intent_tag,
            'requires_twin': primary_row['requires_twin'],
            'expected_outputs': primary_row['expected_outputs'],
            'eval_keywords': primary_row['eval_keywords'],
            'example_questions': [row['user_question'] for row in intent_rows[:3]]
        }


# Global instance for caching
_qm_instance = None


def get_question_matrix(matrix_path: Union[str, Path] = None) -> QuestionMatrix:
    """Get cached Question Matrix instance.
    
    Args:
        matrix_path: Path to matrix file (uses default if None)
        
    Returns:
        QuestionMatrix instance
    """
    global _qm_instance
    
    if _qm_instance is None:
        _qm_instance = QuestionMatrix(matrix_path)
        _qm_instance.load_matrix()
        
    return _qm_instance


def reload_question_matrix(matrix_path: Union[str, Path] = None) -> QuestionMatrix:
    """Force reload of Question Matrix.
    
    Args:
        matrix_path: Path to matrix file
        
    Returns:
        New QuestionMatrix instance
    """
    global _qm_instance
    _qm_instance = QuestionMatrix(matrix_path)
    _qm_instance.load_matrix()
    return _qm_instance
