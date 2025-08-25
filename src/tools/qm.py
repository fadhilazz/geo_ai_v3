"""Question Matrix loader with xlsx/csv autodetect and intent inference."""

import logging
import pandas as pd
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)

QM_COLS = ["id","user_question","intent_tag","primary_aspect","secondary_aspects",
           "discipline_hints","method_or_topic_hints","requires_twin","twin_query_template",
           "retrieval_filter_hint","expected_outputs","eval_keywords","priority"]

def load_qm(path: str) -> List[Dict]:
    """Load Question Matrix from xlsx or csv file with autodetect.
    
    Args:
        path: Path to question matrix file (.xlsx or .csv)
        
    Returns:
        List of dictionaries with normalized headers
    """
    p = Path(path)
    if not p.exists():
        # try alternate extension
        alt = p.with_suffix(".csv") if p.suffix.lower()==".xlsx" else p.with_suffix(".xlsx")
        if alt.exists(): 
            p = alt
        else: 
            raise FileNotFoundError(f"QM not found: {path}")
    
    df = pd.read_excel(p) if p.suffix.lower()==".xlsx" else pd.read_csv(p)
    
    # normalize headers
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    
    # coerce missing required columns
    missing = [c for c in QM_COLS if c not in df.columns]
    if missing: 
        raise ValueError(f"QM missing columns: {missing}")
    
    # unify types
    df["requires_twin"] = df["requires_twin"].astype(str).str.upper().str.contains("YES|TRUE")
    
    logger.info(f"Loaded Question Matrix with {len(df)} rows and columns: {list(df.columns)}")
    return df[QM_COLS].to_dict(orient="records")

def by_intent_df(rows: pd.DataFrame) -> Dict[str, List[Dict]]:
    """Group matrix rows by intent (DataFrame version).
    
    Args:
        rows: Question Matrix DataFrame
        
    Returns:
        Dictionary mapping intent to list of row data
    """
    # Check for different possible intent column names
    intent_column = None
    for col in ['intent', 'intent_tag']:
        if col in rows.columns:
            intent_column = col
            break
    
    if not intent_column:
        logger.warning("No 'intent' or 'intent_tag' column found in Question Matrix")
        return {}
    
    intent_groups = {}
    for _, row in rows.iterrows():
        intent = row.get(intent_column, 'unknown')
        if intent not in intent_groups:
            intent_groups[intent] = []
        intent_groups[intent].append(row.to_dict())
    
    logger.info(f"Grouped Question Matrix by {len(intent_groups)} intents using column '{intent_column}'")
    return intent_groups

def infer_intent(question: str, rows: List[Dict]) -> Tuple[str, float]:
    """Infer intent from question using E5 exemplar + keywords.
    
    Args:
        question: User question
        rows: Question Matrix rows as list of dicts
        
    Returns:
        Tuple of (intent, confidence_score)
    """
    if not rows:
        return "unknown", 0.0
    
    # Find intent column name
    intent_column = None
    for col in ['intent', 'intent_tag']:
        if col in rows[0].keys():
            intent_column = col
            break
    
    if not intent_column:
        return "unknown", 0.0
    
    # Extract exemplar questions and keywords
    exemplars = []
    keywords = []
    
    for row in rows:
        # Check for exemplar question in different possible columns
        for col in ['exemplar_question', 'user_question']:
            if col in row and row[col]:
                exemplars.append(row[col])
                break
        
        # Collect keywords from various columns
        keyword_cols = ['keywords', 'retrieval_hint', 'question_pattern', 'eval_keywords']
        for col in keyword_cols:
            if col in row and row[col]:
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
            exemplar_questions = []
            for row in rows:
                for col in ['exemplar_question', 'user_question']:
                    if col in row and row[col]:
                        exemplar_questions.append(row[col])
                        break
            
            if exemplar_questions and best_idx < len(exemplar_questions):
                # Find matching row
                for row in rows:
                    for col in ['exemplar_question', 'user_question']:
                        if col in row and row[col] and row[col] == exemplar_questions[best_idx]:
                            intent = row.get(intent_column, 'unknown')
                            return intent, float(best_score)
        
        # Fallback: keyword matching
        question_lower = question.lower()
        keyword_scores = {}
        
        for row in rows:
            intent = row.get(intent_column, 'unknown')
            if intent not in keyword_scores:
                keyword_scores[intent] = 0
            
            # Check keywords
            for col in ['keywords', 'retrieval_hint', 'question_pattern', 'eval_keywords']:
                if col in row and row[col]:
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

def filters_for_intent(intent: str, rows: List[Dict], field: Optional[str] = None) -> Dict:
    """Generate filters for intent with optional field filter.
    
    Args:
        intent: Detected intent
        rows: Question Matrix rows as list of dicts
        field: Optional field to filter by
        
    Returns:
        Dictionary of filters for Chroma query
    """
    filters = {}
    
    # Find intent column name
    intent_column = None
    for col in ['intent', 'intent_tag']:
        if col in rows[0].keys():
            intent_column = col
            break
    
    if not intent_column:
        return filters
    
    # Find rows matching intent
    intent_rows = [row for row in rows if row.get(intent_column) == intent]
    
    if intent_rows:
        row = intent_rows[0]
        
        # Add retrieval filter hints
        if 'retrieval_filter_hint' in row and row['retrieval_filter_hint']:
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

def by_intent(rows: List[Dict]) -> Dict[str, List[Dict]]:
    """Group matrix rows by intent.
    
    Args:
        rows: Question Matrix rows as list of dicts
        
    Returns:
        Dictionary mapping intent to list of row data
    """
    d = {}
    for r in rows:
        d.setdefault(r["intent_tag"], []).append(r)
    return d

def filters_for_row(row: Dict, field: str = None) -> Dict:
    """Generate filters for a specific row with optional field filter.
    
    Args:
        row: Question Matrix row as dict
        field: Optional field to filter by
        
    Returns:
        Dictionary of filters for Chroma query
    """
    # row["retrieval_filter_hint"] is JSON-like; tolerate single quotes
    hint = row.get("retrieval_filter_hint") or "{}"
    hint = hint.replace("'", '"')
    try: 
        w = json.loads(hint)
    except: 
        w = {}
    if field: 
        w = {**w, "field": field}
    return w
