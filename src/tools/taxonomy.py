"""Taxonomy loader for Excel-based classification system."""

import logging
from pathlib import Path
from typing import Dict, List, Set, Union

import pandas as pd

logger = logging.getLogger(__name__)


class TaxonomyLoader:
    """Loads and processes taxonomy data from Excel file."""
    
    def __init__(self, xlsx_path: Union[str, Path]):
        """Initialize taxonomy loader.
        
        Args:
            xlsx_path: Path to Excel taxonomy file
        """
        self.xlsx_path = Path(xlsx_path)
        self.aspects = {}
        self.methods = {}
        self.topics = {}
        self.figtypes = {}
        self.synonyms = {}
        
        # Fast lookup dictionaries
        self.synonym_map = {}
        self.aspect_keywords = set()
        self.method_keywords = set()
        self.topic_keywords = set()
        self.figtype_keywords = set()
        
    def _normalize_header(self, header: str) -> str:
        """Normalize Excel column header.
        
        Args:
            header: Original header string
            
        Returns:
            Normalized header (lowercase, stripped)
        """
        if pd.isna(header):
            return ""
        return str(header).strip().lower()
        
    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract and normalize keywords from text.
        
        Args:
            text: Input text containing keywords
            
        Returns:
            Set of normalized keywords
        """
        if pd.isna(text) or not text:
            return set()
            
        # Split by common delimiters and normalize
        keywords = set()
        for item in str(text).split(','):
            item = item.strip().lower()
            if item:
                keywords.add(item)
                # Also add individual words for better matching
                for word in item.split():
                    word = word.strip()
                    if len(word) > 2:  # Skip very short words
                        keywords.add(word)
                        
        return keywords
        
    def _load_sheet_safe(self, sheet_name: str) -> pd.DataFrame:
        """Safely load Excel sheet with error handling.
        
        Args:
            sheet_name: Name of Excel sheet to load
            
        Returns:
            DataFrame or empty DataFrame if sheet not found
        """
        try:
            df = pd.read_excel(self.xlsx_path, sheet_name=sheet_name)
            # Normalize column names
            df.columns = [self._normalize_header(col) for col in df.columns]
            # Strip whitespace from string columns
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].astype(str).str.strip()
            return df
        except Exception as e:
            logger.warning(f"Could not load sheet '{sheet_name}': {e}")
            return pd.DataFrame()
            
    def load_aspects(self) -> Dict:
        """Load aspects taxonomy sheet.
        
        Returns:
            Dictionary of aspect data
        """
        df = self._load_sheet_safe('Aspects')
        aspects = {}
        
        if df.empty:
            logger.warning("Aspects sheet not found or empty")
            return aspects
            
        for _, row in df.iterrows():
            # Try common column name variations
            aspect_col = None
            keywords_col = None
            
            for col in df.columns:
                if 'aspect' in col or 'name' in col:
                    aspect_col = col
                if 'keyword' in col or 'term' in col or 'tag' in col:
                    keywords_col = col
                    
            if aspect_col and not pd.isna(row[aspect_col]):
                aspect_name = str(row[aspect_col]).strip()
                keywords = set()
                
                if keywords_col and not pd.isna(row[keywords_col]):
                    keywords = self._extract_keywords(row[keywords_col])
                    
                aspects[aspect_name.lower()] = {
                    'name': aspect_name,
                    'keywords': keywords
                }
                self.aspect_keywords.update(keywords)
                
        logger.info(f"Loaded {len(aspects)} aspects with {len(self.aspect_keywords)} keywords")
        return aspects
        
    def load_methods(self) -> Dict:
        """Load geophysics methods taxonomy sheet.
        
        Returns:
            Dictionary of method data
        """
        df = self._load_sheet_safe('Geophysics_Methods')
        methods = {}
        
        if df.empty:
            logger.warning("Geophysics_Methods sheet not found or empty")
            return methods
            
        for _, row in df.iterrows():
            # Try common column name variations
            method_col = None
            keywords_col = None
            
            for col in df.columns:
                if 'method' in col or 'name' in col or 'technique' in col:
                    method_col = col
                if 'keyword' in col or 'term' in col or 'tag' in col:
                    keywords_col = col
                    
            if method_col and not pd.isna(row[method_col]):
                method_name = str(row[method_col]).strip()
                keywords = set()
                
                if keywords_col and not pd.isna(row[keywords_col]):
                    keywords = self._extract_keywords(row[keywords_col])
                    
                methods[method_name.lower()] = {
                    'name': method_name,
                    'keywords': keywords
                }
                self.method_keywords.update(keywords)
                
        logger.info(f"Loaded {len(methods)} methods with {len(self.method_keywords)} keywords")
        return methods
        
    def load_topics(self) -> Dict:
        """Load geochemistry topics taxonomy sheet.
        
        Returns:
            Dictionary of topic data
        """
        df = self._load_sheet_safe('Geochemistry_Topics')
        topics = {}
        
        if df.empty:
            logger.warning("Geochemistry_Topics sheet not found or empty")
            return topics
            
        for _, row in df.iterrows():
            # Try common column name variations
            topic_col = None
            keywords_col = None
            
            for col in df.columns:
                if 'topic' in col or 'name' in col or 'subject' in col:
                    topic_col = col
                if 'keyword' in col or 'term' in col or 'tag' in col:
                    keywords_col = col
                    
            if topic_col and not pd.isna(row[topic_col]):
                topic_name = str(row[topic_col]).strip()
                keywords = set()
                
                if keywords_col and not pd.isna(row[keywords_col]):
                    keywords = self._extract_keywords(row[keywords_col])
                    
                topics[topic_name.lower()] = {
                    'name': topic_name,
                    'keywords': keywords
                }
                self.topic_keywords.update(keywords)
                
        logger.info(f"Loaded {len(topics)} topics with {len(self.topic_keywords)} keywords")
        return topics
        
    def load_figtypes(self) -> Dict:
        """Load figure types taxonomy sheet.
        
        Returns:
            Dictionary of figure type data
        """
        df = self._load_sheet_safe('Figure_Types')
        figtypes = {}
        
        if df.empty:
            logger.warning("Figure_Types sheet not found or empty")
            return figtypes
            
        for _, row in df.iterrows():
            # Try common column name variations
            type_col = None
            keywords_col = None
            
            for col in df.columns:
                if 'type' in col or 'name' in col or 'figure' in col:
                    type_col = col
                if 'keyword' in col or 'term' in col or 'tag' in col:
                    keywords_col = col
                    
            if type_col and not pd.isna(row[type_col]):
                type_name = str(row[type_col]).strip()
                keywords = set()
                
                if keywords_col and not pd.isna(row[keywords_col]):
                    keywords = self._extract_keywords(row[keywords_col])
                    
                figtypes[type_name.lower()] = {
                    'name': type_name,
                    'keywords': keywords
                }
                self.figtype_keywords.update(keywords)
                
        logger.info(f"Loaded {len(figtypes)} figure types with {len(self.figtype_keywords)} keywords")
        return figtypes
        
    def load_synonyms(self) -> Dict:
        """Load synonyms mapping sheet.
        
        Returns:
            Dictionary of synonym mappings
        """
        df = self._load_sheet_safe('Synonyms_Map')
        synonyms = {}
        
        if df.empty:
            logger.warning("Synonyms_Map sheet not found or empty")
            return synonyms
            
        for _, row in df.iterrows():
            # Try common column name variations
            synonym_col = None
            canonical_col = None
            
            for col in df.columns:
                if 'synonym' in col or 'alias' in col or 'alternative' in col:
                    synonym_col = col
                if 'canonical' in col or 'standard' in col or 'main' in col or 'primary' in col:
                    canonical_col = col
                    
            if synonym_col and canonical_col:
                if not pd.isna(row[synonym_col]) and not pd.isna(row[canonical_col]):
                    synonym = str(row[synonym_col]).strip().lower()
                    canonical = str(row[canonical_col]).strip().lower()
                    synonyms[synonym] = canonical
                    self.synonym_map[synonym] = canonical
                    
        logger.info(f"Loaded {len(synonyms)} synonym mappings")
        return synonyms
        
    def load_all(self) -> Dict:
        """Load all taxonomy sheets.
        
        Returns:
            Dictionary containing all taxonomy data
        """
        logger.info(f"Loading taxonomy from {self.xlsx_path}")
        
        if not self.xlsx_path.exists():
            raise FileNotFoundError(f"Taxonomy file not found: {self.xlsx_path}")
            
        self.aspects = self.load_aspects()
        self.methods = self.load_methods()
        self.topics = self.load_topics()
        self.figtypes = self.load_figtypes()
        self.synonyms = self.load_synonyms()
        
        return {
            'aspects': self.aspects,
            'methods': self.methods,
            'topics': self.topics,
            'figtypes': self.figtypes,
            'synonyms': self.synonyms
        }


def load_taxonomy(xlsx_path: Union[str, Path]) -> TaxonomyLoader:
    """Load taxonomy from Excel file.
    
    Args:
        xlsx_path: Path to Excel taxonomy file
        
    Returns:
        Loaded TaxonomyLoader instance
    """
    loader = TaxonomyLoader(xlsx_path)
    loader.load_all()
    return loader
