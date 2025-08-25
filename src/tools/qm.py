"""Question Matrix loader and intent inference."""

import logging
import json
import re
from typing import List, Dict, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

# Question Matrix columns
QM_COLS = [
    'id', 'user_question', 'intent_tag', 'primary_aspect', 'secondary_aspects',
    'discipline_hints', 'method_or_topic_hints', 'requires_twin', 'twin_query_template',
    'retrieval_filter_hint', 'expected_outputs', 'eval_keywords', 'priority'
]

# Global variables for semantic index
_qm_index = None
_qm_rows = None
_embedding_model = None

def load_qm(qm_path: str) -> List[Dict]:
    """Load Question Matrix from CSV file."""
    import pandas as pd
    
    try:
        df = pd.read_csv(qm_path)
        logger.info(f"Loaded QM from {qm_path}: {len(df)} rows")
    except Exception as e:
        logger.error(f"Failed to load QM from {qm_path}: {e}")
        return []
    
    # Check required columns
    missing = [col for col in QM_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"QM missing columns: {missing}")
    
    # Normalize requires_twin to bool
    df["requires_twin"] = df["requires_twin"].astype(str).str.upper().str.contains("YES|TRUE")
    
    # Parse retrieval_filter_hint JSON (tolerate single quotes)
    def parse_filter_hint(hint_str):
        if pd.isna(hint_str) or hint_str == "":
            return {}
        try:
            # Replace single quotes with double quotes for JSON parsing
            hint_str = str(hint_str).replace("'", '"')
            return json.loads(hint_str)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse retrieval_filter_hint: {hint_str}")
            return {}
    
    df["retrieval_filter_hint"] = df["retrieval_filter_hint"].apply(parse_filter_hint)
    
    logger.info(f"Loaded Question Matrix with {len(df)} rows and columns: {list(df.columns)}")
    return df[QM_COLS].to_dict(orient="records")

def build_qm_index(rows: List[Dict]) -> Dict:
    """
    Build semantic index over Question Matrix rows.
    
    Args:
        rows: List of Question Matrix rows
        
    Returns:
        Dict with embeddings and rows: {"emb": np.array(n,d), "rows": rows}
    """
    global _embedding_model
    
    if _embedding_model is None:
        try:
            _embedding_model = SentenceTransformer('intfloat/e5-large-v2')
            logger.info("Loaded E5-large-v2 embedding model for QM indexing")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            return {"emb": np.array([]), "rows": rows}
    
    # Extract user questions for embedding
    questions = [row['user_question'] for row in rows]
    
    try:
        # Generate embeddings
        embeddings = _embedding_model.encode(questions, normalize_embeddings=True)
        logger.info(f"Generated embeddings for {len(questions)} QM rows")
        
        return {
            "emb": embeddings,
            "rows": rows
        }
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        return {"emb": np.array([]), "rows": rows}

def match_rows(question: str, k: int = 5) -> List[Tuple[Dict, float]]:
    """
    Match question against Question Matrix using semantic similarity.
    
    Args:
        question: User question
        k: Number of top matches to return
        
    Returns:
        List of tuples: (row, similarity_score)
    """
    global _qm_index, _qm_rows
    
    if _qm_index is None:
        # Initialize index on first call
        try:
            from ..config import get_qa_paths
        except ImportError:
            from config import get_qa_paths
        
        qm_path = get_qa_paths()['question_matrix']
        _qm_rows = load_qm(str(qm_path))
        _qm_index = build_qm_index(_qm_rows)
    
    if _qm_index["emb"].size == 0:
        # Fallback to fuzzy matching if embeddings failed
        return _fuzzy_match_rows(question, _qm_rows, k)
    
    try:
        # Encode the question
        question_emb = _embedding_model.encode([question], normalize_embeddings=True)[0]
        
        # Compute cosine similarities
        similarities = np.dot(_qm_index["emb"], question_emb)
        
        # Get top-k matches
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score > 0.1:  # Minimum similarity threshold
                results.append((_qm_rows[idx], score))
        
        logger.info(f"Semantic match for '{question}': {len(results)} matches, top score: {results[0][1] if results else 0:.3f}")
        return results
        
    except Exception as e:
        logger.error(f"Failed to match rows semantically: {e}")
        return _fuzzy_match_rows(question, _qm_rows, k)

def _fuzzy_match_rows(question: str, rows: List[Dict], k: int) -> List[Tuple[Dict, float]]:
    """Fallback fuzzy matching for Question Matrix rows."""
    results = []
    
    for row in rows:
        # Match against user_question
        score1 = fuzz.ratio(question.lower(), row['user_question'].lower()) / 100.0
        
        # Match against eval_keywords
        keywords = row.get('eval_keywords', '[]')
        if keywords and keywords != '[]':
            try:
                keyword_list = json.loads(keywords.replace("'", '"'))
                keyword_text = ' '.join(keyword_list)
                score2 = fuzz.ratio(question.lower(), keyword_text.lower()) / 100.0
            except:
                score2 = 0.0
        else:
            score2 = 0.0
        
        # Match against expected_outputs
        outputs = row.get('expected_outputs', '[]')
        if outputs and outputs != '[]':
            try:
                output_list = json.loads(outputs.replace("'", '"'))
                output_text = ' '.join(output_list)
                score3 = fuzz.ratio(question.lower(), output_text.lower()) / 100.0
            except:
                score3 = 0.0
        else:
            score3 = 0.0
        
        # Take the best score
        best_score = max(score1, score2, score3)
        if best_score > 0.3:  # Minimum fuzzy match threshold
            results.append((row, best_score))
    
    # Sort by score and return top-k
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:k]

def infer_intent(question: str, qm_rows: List[Dict]) -> Tuple[str, float]:
    """Infer intent from question using semantic matching."""
    matches = match_rows(question, k=3)
    
    if not matches:
        return "general_inquiry", 0.0
    
    # Return the best match
    best_match, confidence = matches[0]
    return best_match['intent_tag'], confidence

def get_union_strategy(matches: List[Tuple[Dict, float]]) -> Dict:
    """
    Build union strategy from top-k Question Matrix matches.
    
    Args:
        matches: List of (row, score) tuples from match_rows()
        
    Returns:
        Dict with unified strategy: {
            'requires_twin': bool,
            'filters': dict,
            'intent_tag': str,
            'confidence': float
        }
    """
    if not matches:
        return {
            'requires_twin': False,
            'filters': {},
            'intent_tag': 'general_inquiry',
            'confidence': 0.0
        }
    
    # Union requires_twin
    requires_twin = any(row['requires_twin'] for row, _ in matches)
    
    # Union filters (dedupe arrays)
    all_filters = {}
    for row, _ in matches:
        filters = row.get('retrieval_filter_hint', {})
        for key, value in filters.items():
            if key not in all_filters:
                all_filters[key] = value
            elif isinstance(value, list) and isinstance(all_filters[key], list):
                # Dedupe arrays
                all_filters[key] = list(set(all_filters[key] + value))
    
    # Choose dominant intent (mode of intent_tag, tie → highest score)
    intent_counts = {}
    for row, score in matches:
        intent = row['intent_tag']
        if intent not in intent_counts:
            intent_counts[intent] = {'count': 0, 'max_score': 0}
        intent_counts[intent]['count'] += 1
        intent_counts[intent]['max_score'] = max(intent_counts[intent]['max_score'], score)
    
    # Find intent with highest count, then highest score
    dominant_intent = max(intent_counts.keys(), 
                         key=lambda x: (intent_counts[x]['count'], intent_counts[x]['max_score']))
    
    # Overall confidence is max score
    confidence = max(score for _, score in matches)
    
    return {
        'requires_twin': requires_twin,
        'filters': all_filters,
        'intent_tag': dominant_intent,
        'confidence': confidence
    }
