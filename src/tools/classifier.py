"""Typo-tolerant classification using taxonomy and fuzzy matching."""

import logging
import re
import unicodedata
from typing import Dict, List, Optional, Set

from rapidfuzz import process, fuzz

try:
    from ..config import DISCIPLINE_FOLDERS, FUZZY_SCORE_CUTOFF
except ImportError:
    from src.config import DISCIPLINE_FOLDERS, FUZZY_SCORE_CUTOFF
from .taxonomy import TaxonomyLoader

logger = logging.getLogger(__name__)


class TypoTolerantClassifier:
    """Classifier with fuzzy matching for typo tolerance."""
    
    def __init__(self, taxonomy: TaxonomyLoader):
        """Initialize classifier with taxonomy.
        
        Args:
            taxonomy: Loaded taxonomy data
        """
        self.taxonomy = taxonomy
        
    def _normalize_token(self, token: str) -> str:
        """Normalize token for fuzzy matching.
        
        Args:
            token: Input token
            
        Returns:
            Normalized token (lowercase, no accents, collapsed punctuation)
        """
        if not token:
            return ""
            
        # Convert to lowercase
        token = token.lower()
        
        # Remove accents/diacritics
        token = unicodedata.normalize('NFD', token)
        token = ''.join(c for c in token if unicodedata.category(c) != 'Mn')
        
        # Collapse dashes and underscores
        token = re.sub(r'[-_]+', '_', token)
        
        # Remove extra whitespace
        token = re.sub(r'\s+', ' ', token).strip()
        
        return token
        
    def _extract_tokens(self, text: str) -> Set[str]:
        """Extract and normalize tokens from text.
        
        Args:
            text: Input text
            
        Returns:
            Set of normalized tokens
        """
        if not text:
            return set()
            
        # Split on common delimiters
        tokens = set()
        for delimiter in [' ', ',', ';', '\n', '\t', '(', ')', '[', ']', '{', '}']:
            text = text.replace(delimiter, ' ')
            
        for token in text.split():
            normalized = self._normalize_token(token)
            if len(normalized) > 2:  # Skip very short tokens
                tokens.add(normalized)
                
        return tokens
        
    def _fuzzy_match_keywords(self, tokens: Set[str], keyword_set: Set[str], 
                             score_cutoff: int = FUZZY_SCORE_CUTOFF) -> List[str]:
        """Fuzzy match tokens against keyword set.
        
        Args:
            tokens: Set of normalized tokens to match
            keyword_set: Set of keywords to match against
            score_cutoff: Minimum fuzzy match score (default 80 for 1-2 typos)
            
        Returns:
            List of matched keywords
        """
        if not tokens or not keyword_set:
            return []
            
        matches = []
        normalized_keywords = [self._normalize_token(kw) for kw in keyword_set]
        
        for token in tokens:
            # Try exact match first
            if token in keyword_set:
                matches.append(token)
                continue
                
            # Try fuzzy matching
            fuzzy_matches = process.extract(
                token, 
                normalized_keywords, 
                scorer=fuzz.WRatio,
                score_cutoff=score_cutoff,
                limit=3
            )
            
            for match, score, _ in fuzzy_matches:
                # Find original keyword
                for orig_kw in keyword_set:
                    if self._normalize_token(orig_kw) == match:
                        matches.append(orig_kw)
                        logger.debug(f"Fuzzy matched '{token}' -> '{orig_kw}' (score: {score})")
                        break
                        
        return list(set(matches))  # Remove duplicates
        
    def _classify_from_folder_hints(self, folder_hints: List[str]) -> List[str]:
        """Extract discipline from folder hints.
        
        Args:
            folder_hints: List of folder names from path
            
        Returns:
            List of matched disciplines
        """
        disciplines = []
        
        for folder in folder_hints:
            folder_norm = self._normalize_token(folder)
            
            # Direct mapping from config
            for key, discipline in DISCIPLINE_FOLDERS.items():
                if key in folder_norm:
                    disciplines.append(discipline)
                    
        return list(set(disciplines))
        
    def _apply_synonym_mapping(self, tokens: Set[str]) -> Set[str]:
        """Apply synonym mapping to tokens.
        
        Args:
            tokens: Set of tokens to map
            
        Returns:
            Set of tokens with synonyms mapped to canonical forms
        """
        mapped_tokens = set()
        
        for token in tokens:
            # Check if token has synonym mapping
            canonical = self.taxonomy.synonym_map.get(token.lower())
            if canonical:
                mapped_tokens.add(canonical)
                logger.debug(f"Mapped synonym '{token}' -> '{canonical}'")
            else:
                mapped_tokens.add(token)
                
        return mapped_tokens
        
    def classify_text(self, text_sample: str, folder_hints: List[str] = None,
                     caption: Optional[str] = None) -> Dict:
        """Classify text content using taxonomy.
        
        Args:
            text_sample: Text content to classify (title + first page)
            folder_hints: List of folder names from file path
            caption: Optional figure caption for image classification
            
        Returns:
            Classification dict with discipline, aspect, method, topic, keywords
        """
        if folder_hints is None:
            folder_hints = []
            
        # Initialize result
        result = {
            "discipline": [],
            "aspect": [],
            "method": [],
            "topic": [],
            "figure_type": None,
            "keywords": []
        }
        
        # Extract and normalize tokens
        all_text = text_sample or ""
        if caption:
            all_text += " " + caption
            
        tokens = self._extract_tokens(all_text)
        
        # Apply synonym mapping
        tokens = self._apply_synonym_mapping(tokens)
        
        # 1. Classify from folder hints
        result["discipline"] = self._classify_from_folder_hints(folder_hints)
        
        # 2. Fuzzy match against taxonomy keywords
        
        # Match aspects
        aspect_matches = self._fuzzy_match_keywords(tokens, self.taxonomy.aspect_keywords)
        for match in aspect_matches:
            for aspect_key, aspect_data in self.taxonomy.aspects.items():
                if match in aspect_data['keywords']:
                    result["aspect"].append(aspect_data['name'])
                    
        # Match methods (geophysics)
        method_matches = self._fuzzy_match_keywords(tokens, self.taxonomy.method_keywords)
        for match in method_matches:
            for method_key, method_data in self.taxonomy.methods.items():
                if match in method_data['keywords']:
                    result["method"].append(method_data['name'])
                    
        # Match topics (geochemistry)
        topic_matches = self._fuzzy_match_keywords(tokens, self.taxonomy.topic_keywords)
        for match in topic_matches:
            for topic_key, topic_data in self.taxonomy.topics.items():
                if match in topic_data['keywords']:
                    result["topic"].append(topic_data['name'])
                    
        # 3. For images, match figure types from caption
        if caption:
            caption_tokens = self._extract_tokens(caption)
            caption_tokens = self._apply_synonym_mapping(caption_tokens)
            
            figtype_matches = self._fuzzy_match_keywords(caption_tokens, self.taxonomy.figtype_keywords)
            if figtype_matches:
                # Find the best matching figure type
                best_score = 0
                best_figtype = None
                
                for match in figtype_matches:
                    for figtype_key, figtype_data in self.taxonomy.figtypes.items():
                        if match in figtype_data['keywords']:
                            # Use the figtype key as the figure_type
                            result["figure_type"] = figtype_key
                            break
                            
        # 4. Collect all matched keywords
        all_matches = aspect_matches + method_matches + topic_matches
        if caption:
            all_matches.extend(figtype_matches)
        result["keywords"] = list(set(all_matches))
        
        # Remove duplicates from lists
        result["discipline"] = list(set(result["discipline"]))
        result["aspect"] = list(set(result["aspect"]))
        result["method"] = list(set(result["method"]))
        result["topic"] = list(set(result["topic"]))
        
        return result
        
    def classify_chunk(self, chunk_text: str, doc_context: str, 
                      folder_hints: List[str] = None) -> Dict:
        """Classify a text chunk with document context.
        
        Args:
            chunk_text: Text chunk to classify
            doc_context: Document context (title + first page)
            folder_hints: List of folder names from file path
            
        Returns:
            Classification dict
        """
        # Combine chunk with document context for better classification
        combined_text = f"{doc_context}\n\n{chunk_text}"
        return self.classify_text(combined_text, folder_hints)
        
    def classify_figure(self, caption: str, doc_context: str,
                       folder_hints: List[str] = None) -> Dict:
        """Classify a figure with its caption.
        
        Args:
            caption: Figure caption
            doc_context: Document context (title + first page)
            folder_hints: List of folder names from file path
            
        Returns:
            Classification dict with figure_type
        """
        return self.classify_text(doc_context, folder_hints, caption)


def create_classifier(taxonomy: TaxonomyLoader) -> TypoTolerantClassifier:
    """Create classifier instance with loaded taxonomy.
    
    Args:
        taxonomy: Loaded taxonomy data
        
    Returns:
        Configured classifier instance
    """
    return TypoTolerantClassifier(taxonomy)
