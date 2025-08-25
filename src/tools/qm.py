"""Question Matrix loader with xlsx/csv autodetect and intent inference."""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)

def load_qm(path: str) -> pd.DataFrame:
    """Load Question Matrix from xlsx or csv file with autodetect.
    
    Args:
        path: Path to question matrix file (.xlsx or .csv)
        
    Returns:
        DataFrame with normalized headers
    """
    path_obj = Path(path)
    
    if not path_obj.exists():
        raise FileNotFoundError(f"Question Matrix file not found: {path}")
    
    # Autodetect file type
    if path_obj.suffix.lower() == '.xlsx':
        logger.info(f"Loading Question Matrix from Excel: {path}")
        df = pd.read_excel(path)
    elif path_obj.suffix.lower() == '.csv':
        logger.info(f"Loading Question Matrix from CSV: {path}")
        df = pd.read_csv(path)
    else:
        # Try both formats
        try:
            logger.info(f"Trying Excel format for: {path}")
            df = pd.read_excel(path)
        except Exception:
            try:
                logger.info(f"Trying CSV format for: {path}")
                df = pd.read_csv(path)
            except Exception as e:
                raise ValueError(f"Could not load Question Matrix from {path}. Tried both xlsx and csv formats. Error: {e}")
    
    # Normalize headers (lowercase, replace spaces with underscores)
    df.columns = [col.lower().replace(' ', '_').replace('-', '_') for col in df.columns]
    
    logger.info(f"Loaded Question Matrix with {len(df)} rows and columns: {list(df.columns)}")
    return df

def by_intent(rows: pd.DataFrame) -> Dict[str, List[Dict]]:
    """Group matrix rows by intent.
    
    Args:
        rows: Question Matrix DataFrame
        
    Returns:
        Dictionary mapping intent to list of row data
    """
    if 'intent' not in rows.columns:
        logger.warning("No 'intent' column found in Question Matrix")
        return {}
    
    intent_groups = {}
    for _, row in rows.iterrows():
        intent = row.get('intent', 'unknown')
        if intent not in intent_groups:
            intent_groups[intent] = []
        intent_groups[intent].append(row.to_dict())
    
    logger.info(f"Grouped Question Matrix by {len(intent_groups)} intents")
    return intent_groups

def infer_intent(question: str, rows: pd.DataFrame) -> Tuple[str, float]:
    """Infer intent from question using E5 exemplar + keywords.
    
    Args:
        question: User question
        rows: Question Matrix DataFrame
        
    Returns:
        Tuple of (intent, confidence_score)
    """
    if rows.empty:
        return "unknown", 0.0
    
    # Extract exemplar questions and keywords
    exemplars = []
    keywords = []
    
    for _, row in rows.iterrows():
        if 'exemplar_question' in row and pd.notna(row['exemplar_question']):
            exemplars.append(row['exemplar_question'])
        
        # Collect keywords from various columns
        keyword_cols = ['keywords', 'retrieval_hint', 'question_pattern']
        for col in keyword_cols:
            if col in row and pd.notna(row[col]):
                keywords.extend(str(row[col]).split(','))
    
    # Use E5 model for similarity
    try:
        model = SentenceTransformer("intfloat/e5-large-v2")
        
        # Encode question and exemplars
        question_embedding = model.encode([question])
        exemplar_embeddings = model.encode(exemplars) if exemplars else np.array([])
        
        if len(exemplar_embeddings) > 0:
            # Calculate similarities
            similarities = np.dot(exemplar_embeddings, question_embedding.T).flatten()
            best_idx = np.argmax(similarities)
            best_score = similarities[best_idx]
            
            # Find corresponding intent
            exemplar_questions = [row.get('exemplar_question', '') for _, row in rows.iterrows() if pd.notna(row.get('exemplar_question'))]
            if exemplar_questions and best_idx < len(exemplar_questions):
                matching_rows = rows[rows['exemplar_question'] == exemplar_questions[best_idx]]
                if not matching_rows.empty:
                    intent = matching_rows.iloc[0].get('intent', 'unknown')
                    return intent, float(best_score)
        
        # Fallback: keyword matching
        question_lower = question.lower()
        keyword_scores = {}
        
        for _, row in rows.iterrows():
            intent = row.get('intent', 'unknown')
            if intent not in keyword_scores:
                keyword_scores[intent] = 0
            
            # Check keywords
            for col in ['keywords', 'retrieval_hint', 'question_pattern']:
                if col in row and pd.notna(row[col]):
                    row_keywords = str(row[col]).lower().split(',')
                    for keyword in row_keywords:
                        keyword = keyword.strip()
                        if keyword in question_lower:
                            keyword_scores[intent] += 1
        
        if keyword_scores:
            best_intent = max(keyword_scores, key=keyword_scores.get)
            best_score = keyword_scores[best_intent] / max(1, len(keywords))
            return best_intent, best_score
        
    except Exception as e:
        logger.warning(f"Error in intent inference: {e}")
    
    return "unknown", 0.0

def filters_for_intent(intent: str, rows: pd.DataFrame, field: Optional[str] = None) -> Dict:
    """Generate filters for intent with optional field filter.
    
    Args:
        intent: Detected intent
        rows: Question Matrix DataFrame
        field: Optional field to filter by
        
    Returns:
        Dictionary of filters for Chroma query
    """
    filters = {}
    
    # Find rows matching intent
    intent_rows = rows[rows['intent'] == intent]
    
    if not intent_rows.empty:
        row = intent_rows.iloc[0]
        
        # Add retrieval filter hints
        if 'retrieval_filter_hint' in row and pd.notna(row['retrieval_filter_hint']):
            filter_hint = str(row['retrieval_filter_hint'])
            # Parse filter hint (simple key=value format)
            for part in filter_hint.split(','):
                if '=' in part:
                    key, value = part.strip().split('=', 1)
                    filters[key.strip()] = value.strip()
    
    # Add field filter if provided
    if field:
        filters['field'] = field
    
    logger.info(f"Generated filters for intent '{intent}': {filters}")
    return filters
