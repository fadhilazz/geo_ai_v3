"""Image RAG operations for QA engine."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import chromadb
import numpy as np
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer

try:
    from ..config import (
        CHROMA_IMAGE_DIR_OBJ, IMAGE_RETRIEVAL_TOP_K, 
        DEFAULT_TEXT_MODEL, IMAGES_BASE_DIR_OBJ, ENABLE_CAPTION_FIRST
    )
except ImportError:
    from src.config import (
        CHROMA_IMAGE_DIR_OBJ, IMAGE_RETRIEVAL_TOP_K, 
        DEFAULT_TEXT_MODEL, IMAGES_BASE_DIR_OBJ, ENABLE_CAPTION_FIRST
    )

logger = logging.getLogger(__name__)


@dataclass
class ImageFigure:
    """Image figure with metadata."""
    caption: str
    score: float
    doc_id: str
    page: int
    filename: str
    source_path: str
    image_path: str
    discipline: List[str]
    aspect: List[str]
    method: List[str]
    topic: List[str]
    figure_type: Optional[str]
    keywords: List[str]
    width: int
    height: int
    field: Optional[str] = None
    source: Optional[str] = None  # "caption" or "clip"
    
    @classmethod
    def from_chroma_result(cls, document: str, metadata: dict, distance: float) -> 'ImageFigure':
        """Create ImageFigure from Chroma search result.
        
        Args:
            document: Document text (caption)
            metadata: Metadata dictionary
            distance: Distance score (lower is better)
            
        Returns:
            ImageFigure instance
        """
        # Convert distance to similarity score (higher is better)
        score = 1.0 - distance if distance <= 1.0 else 1.0 / (1.0 + distance)
        
        # Parse list fields that are stored as strings
        def parse_list_field(field_value):
            if isinstance(field_value, str):
                # Remove brackets and quotes, split by comma
                cleaned = field_value.strip("[]'\"")
                if cleaned:
                    return [item.strip().strip("'\"") for item in cleaned.split(',') if item.strip()]
            elif isinstance(field_value, list):
                return field_value
            return []
        
        return cls(
            caption=document,
            score=score,
            doc_id=metadata.get('doc_id', ''),
            page=int(metadata.get('page', 0)),
            filename=metadata.get('filename', ''),
            source_path=metadata.get('source_path', ''),
            image_path=metadata.get('image_path', ''),
            discipline=parse_list_field(metadata.get('discipline', [])),
            aspect=parse_list_field(metadata.get('aspect', [])),
            method=parse_list_field(metadata.get('method', [])),
            topic=parse_list_field(metadata.get('topic', [])),
            figure_type=metadata.get('figure_type'),
            keywords=parse_list_field(metadata.get('keywords', [])),
            width=int(metadata.get('width', 0)),
            height=int(metadata.get('height', 0)),
            field=metadata.get('field')
        )
        
    def get_relative_path(self, base_dir: Union[str, Path] = None) -> str:
        """Get relative path for UI display.
        
        Args:
            base_dir: Base directory to calculate relative path from
            
        Returns:
            Relative path string
        """
        if not base_dir:
            base_dir = IMAGES_BASE_DIR_OBJ
            
        try:
            image_path = Path(self.image_path)
            base_path = Path(base_dir)
            return str(image_path.relative_to(base_path))
        except ValueError:
            # If not relative, return filename
            return Path(self.image_path).name


class ImageRAG:
    """Image retrieval and search operations."""
    
    def __init__(self, chroma_dir: Union[str, Path] = None, collection_name: str = "image_emb"):
        """Initialize Image RAG.
        
        Args:
            chroma_dir: Directory containing Chroma database
            collection_name: Name of the collection
        """
        self.chroma_dir = Path(chroma_dir) if chroma_dir else CHROMA_IMAGE_DIR_OBJ
        self.collection_name = collection_name
        self.collection = None
        self.text_embedder = None
        
    def _get_collection(self) -> chromadb.Collection:
        """Get or create Chroma collection."""
        if self.collection is None:
            try:
                client = chromadb.PersistentClient(path=str(self.chroma_dir))
                self.collection = client.get_collection(name=self.collection_name)
                logger.info(f"Connected to image collection '{self.collection_name}' with {self.collection.count()} items")
            except Exception as e:
                logger.error(f"Failed to connect to image collection: {e}")
                raise
                
        return self.collection
        
    def _get_text_embedder(self) -> SentenceTransformer:
        """Get text embedder for caption search."""
        if self.text_embedder is None:
            self.text_embedder = SentenceTransformer(DEFAULT_TEXT_MODEL)
        return self.text_embedder
        
    def search_by_caption(self, query: str, where: Optional[Dict] = None, top_k: int = IMAGE_RETRIEVAL_TOP_K) -> List[ImageFigure]:
        """Search images by caption text similarity.
        
        Args:
            query: Search query
            where: Optional filters for metadata
            top_k: Number of results to return
            
        Returns:
            List of ImageFigure results
        """
        collection = self._get_collection()
        
        try:
            # Build the search parameters
            search_params = {
                'query_texts': [query],
                'n_results': top_k,
                'include': ['documents', 'metadatas', 'distances']
            }
            
            if where:
                # Convert filters to Chroma format
                chroma_where = self._build_chroma_filters(where)
                if chroma_where:
                    search_params['where'] = chroma_where
                    
            # Perform search
            results = collection.query(**search_params)
            
            # Convert to ImageFigure objects
            figures = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i]
                    distance = results['distances'][0][i]
                    
                    figure = ImageFigure.from_chroma_result(doc, metadata, distance)
                    figures.append(figure)
                    
            logger.info(f"Caption search for '{query}' with filters {where}: found {len(figures)} results")
            return figures
            
        except Exception as e:
            logger.error(f"Error searching image collection by caption: {e}")
            return []
            
    def search_by_embedding(self, query_embedding: np.ndarray, where: Optional[Dict] = None, top_k: int = IMAGE_RETRIEVAL_TOP_K) -> List[ImageFigure]:
        """Search images by CLIP embedding similarity.
        
        Args:
            query_embedding: Pre-computed CLIP embedding
            where: Optional filters for metadata
            top_k: Number of results to return
            
        Returns:
            List of ImageFigure results
        """
        collection = self._get_collection()
        
        try:
            # Build the search parameters
            search_params = {
                'query_embeddings': [query_embedding.tolist()],
                'n_results': top_k,
                'include': ['documents', 'metadatas', 'distances']
            }
            
            if where:
                # Convert filters to Chroma format
                chroma_where = self._build_chroma_filters(where)
                if chroma_where:
                    search_params['where'] = chroma_where
                    
            # Perform search
            results = collection.query(**search_params)
            
            # Convert to ImageFigure objects
            figures = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i]
                    distance = results['distances'][0][i]
                    
                    figure = ImageFigure.from_chroma_result(doc, metadata, distance)
                    figures.append(figure)
                    
            logger.info(f"CLIP embedding search with filters {where}: found {len(figures)} results")
            return figures
            
        except Exception as e:
            logger.error(f"Error searching image collection by embedding: {e}")
            return []
            
    def search(self, query: str, where: Optional[Dict] = None, top_k: int = IMAGE_RETRIEVAL_TOP_K) -> List[ImageFigure]:
        """Search images with caption-first strategy when enabled.
        
        Args:
            query: Search query
            where: Optional filters for metadata
            top_k: Number of results to return
            
        Returns:
            List of ImageFigure results with source indication
        """
        if ENABLE_CAPTION_FIRST:
            # Try caption search first (fast)
            figures = self.search_by_caption(query, where, top_k)
            
            if figures:
                logger.info(f"Caption-first search found {len(figures)} results (source: caption)")
                # Add source indicator to figures
                for figure in figures:
                    figure.source = "caption"
                return figures
                
            logger.info("Caption-first search found no results, falling back to CLIP")
            
            # Fall back to CLIP embedding search
            try:
                embedder = self._get_text_embedder()
                query_embedding = embedder.encode([query], convert_to_numpy=True)[0]
                figures = self.search_by_embedding(query_embedding, where, top_k)
                # Add source indicator
                for figure in figures:
                    figure.source = "clip"
                return figures
            except Exception as e:
                logger.error(f"Error in CLIP embedding search: {e}")
                return []
        else:
            # Original behavior: try caption first, then CLIP
            figures = self.search_by_caption(query, where, top_k)
            
            if figures:
                logger.debug(f"Caption search found {len(figures)} results, skipping CLIP search")
                return figures
                
            logger.debug("Caption search found no results, trying CLIP embedding search")
            
            try:
                embedder = self._get_text_embedder()
                query_embedding = embedder.encode([query], convert_to_numpy=True)[0]
                return self.search_by_embedding(query_embedding, where, top_k)
            except Exception as e:
                logger.error(f"Error in CLIP embedding search: {e}")
                return []
            
    def _build_chroma_filters(self, where: Dict) -> Dict:
        """Build Chroma-compatible filter dictionary.
        
        Args:
            where: Filter dictionary
            
        Returns:
            Chroma-compatible filter dict
        """
        if not where:
            return {}
            
        # Chroma expects filters to be combined with $and if multiple conditions
        filter_conditions = []
        
        for key, value in where.items():
            if value is None:
                continue
                
            # Handle list fields that are stored as strings
            if key in ['discipline', 'aspect', 'method', 'topic', 'keywords']:
                # For list fields stored as strings, we'll use $eq and rely on the string representation
                # This is a simplified approach - in production you might want more sophisticated matching
                filter_conditions.append({key: {"$eq": str([value])}})
            else:
                # Direct equality for other fields
                filter_conditions.append({key: {"$eq": value}})
                
        # Return appropriate format based on number of conditions
        if len(filter_conditions) == 0:
            return {}
        elif len(filter_conditions) == 1:
            return filter_conditions[0]
        else:
            return {"$and": filter_conditions}
        
    def search_with_fallback(self, query: str, where: Optional[Dict] = None, top_k: int = IMAGE_RETRIEVAL_TOP_K) -> List[ImageFigure]:
        """Search with progressive filter relaxation if no results found.
        
        Args:
            query: Search query
            where: Optional filters for metadata
            top_k: Number of results to return
            
        Returns:
            List of ImageFigure results
        """
        if not where:
            return self.search(query, where, top_k)
            
        # Try with full filters first
        results = self.search(query, where, top_k)
        if results:
            return results
            
        # Progressive relaxation: drop topic -> method -> aspect -> discipline
        relaxation_order = ['topic', 'method', 'aspect', 'discipline']
        
        for field_to_drop in relaxation_order:
            if field_to_drop in where:
                relaxed_where = where.copy()
                del relaxed_where[field_to_drop]
                
                logger.info(f"Relaxing image filters: dropped '{field_to_drop}'")
                results = self.search(query, relaxed_where, top_k)
                if results:
                    return results
                    
        # Final fallback: no filters except field
        final_where = {}
        if 'field' in where:
            final_where['field'] = where['field']
            
        logger.info("Final image fallback: using minimal filters")
        return self.search(query, final_where, top_k)
        
    def get_all_fields(self) -> List[str]:
        """Get all unique field values from the collection.
        
        Returns:
            List of unique field names
        """
        collection = self._get_collection()
        
        try:
            # Get all documents with metadata
            results = collection.get(include=['metadatas'])
            
            fields = set()
            for metadata in results['metadatas']:
                field = metadata.get('field')
                if field and field.strip():
                    fields.add(field.strip())
                    
            return sorted(list(fields))
            
        except Exception as e:
            logger.error(f"Error getting fields from image collection: {e}")
            return []


# Global instance for caching
_image_rag_instance = None


def get_image_rag(chroma_dir: Union[str, Path] = None) -> ImageRAG:
    """Get cached ImageRAG instance.
    
    Args:
        chroma_dir: Directory containing Chroma database
        
    Returns:
        ImageRAG instance
    """
    global _image_rag_instance
    
    if _image_rag_instance is None:
        _image_rag_instance = ImageRAG(chroma_dir)
        
    return _image_rag_instance
