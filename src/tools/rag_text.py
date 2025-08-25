"""Text RAG operations for QA engine."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import chromadb
from dataclasses import dataclass

try:
    from ..config import CHROMA_TEXT_DIR_OBJ, TEXT_RETRIEVAL_TOP_K, ENABLE_PROGRESSIVE_RELAX
except ImportError:
    from src.config import CHROMA_TEXT_DIR_OBJ, TEXT_RETRIEVAL_TOP_K, ENABLE_PROGRESSIVE_RELAX

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """Text chunk with metadata."""
    text: str
    score: float
    doc_id: str
    page: int
    filename: str
    source_path: str
    discipline: List[str]
    aspect: List[str]
    method: List[str]
    topic: List[str]
    keywords: List[str]
    field: Optional[str] = None
    
    @classmethod
    def from_chroma_result(cls, document: str, metadata: dict, distance: float) -> 'TextChunk':
        """Create TextChunk from Chroma search result.
        
        Args:
            document: Document text
            metadata: Metadata dictionary
            distance: Distance score (lower is better)
            
        Returns:
            TextChunk instance
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
            text=document,
            score=score,
            doc_id=metadata.get('doc_id', ''),
            page=int(metadata.get('page', 0)),
            filename=metadata.get('filename', ''),
            source_path=metadata.get('source_path', ''),
            discipline=parse_list_field(metadata.get('discipline', [])),
            aspect=parse_list_field(metadata.get('aspect', [])),
            method=parse_list_field(metadata.get('method', [])),
            topic=parse_list_field(metadata.get('topic', [])),
            keywords=parse_list_field(metadata.get('keywords', [])),
            field=metadata.get('field')
        )


class TextRAG:
    """Text retrieval and search operations."""
    
    def __init__(self, chroma_dir: Union[str, Path] = None, collection_name: str = "text_emb"):
        """Initialize Text RAG.
        
        Args:
            chroma_dir: Directory containing Chroma database
            collection_name: Name of the collection
        """
        self.chroma_dir = Path(chroma_dir) if chroma_dir else CHROMA_TEXT_DIR_OBJ
        self.collection_name = collection_name
        self.collection = None
        self.embedder = None
        
    def _get_collection(self) -> chromadb.Collection:
        """Get or create Chroma collection."""
        if self.collection is None:
            try:
                client = chromadb.PersistentClient(path=str(self.chroma_dir))
                self.collection = client.get_collection(name=self.collection_name)
                logger.info(f"Connected to text collection '{self.collection_name}' with {self.collection.count()} items")
            except Exception as e:
                logger.error(f"Failed to connect to text collection: {e}")
                raise
                
        return self.collection
        
    def _get_embedder(self):
        """Get or create sentence transformer embedder."""
        if self.embedder is None:
            from sentence_transformers import SentenceTransformer
            try:
                from ..config import DEFAULT_TEXT_MODEL
            except ImportError:
                from src.config import DEFAULT_TEXT_MODEL
            self.embedder = SentenceTransformer(DEFAULT_TEXT_MODEL)
            logger.info(f"Loaded text embedder: {DEFAULT_TEXT_MODEL}")
        return self.embedder
        
    def search(self, query: str, where: Optional[Dict] = None, top_k: int = TEXT_RETRIEVAL_TOP_K) -> List[TextChunk]:
        """Search for relevant text chunks.
        
        Args:
            query: Search query
            where: Optional filters for metadata
            top_k: Number of results to return
            
        Returns:
            List of TextChunk results
        """
        collection = self._get_collection()
        
        try:
            # Get embedder and encode query
            embedder = self._get_embedder()
            query_embedding = embedder.encode([query])
            
            # Build the search parameters with correct embedding
            search_params = {
                'query_embeddings': [query_embedding[0].tolist()],
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
            
            # Convert to TextChunk objects
            chunks = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i]
                    distance = results['distances'][0][i]
                    
                    chunk = TextChunk.from_chroma_result(doc, metadata, distance)
                    chunks.append(chunk)
                    
            logger.info(f"Text search for '{query}' with filters {where}: found {len(chunks)} results")
            return chunks
            
        except Exception as e:
            logger.error(f"Error searching text collection: {e}")
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
        
    def progressive_query(self, query: str, base_where: Optional[Dict] = None, 
                         relax_order: tuple = ("topic", "method", "aspect"), 
                         top_k: int = TEXT_RETRIEVAL_TOP_K) -> List[TextChunk]:
        """Progressive filter relaxation for better retrieval.
        
        Args:
            query: Search query
            base_where: Base filters to start with
            relax_order: Order of fields to relax
            top_k: Number of results to return
            
        Returns:
            List of TextChunk results
        """
        if not ENABLE_PROGRESSIVE_RELAX:
            return self.search(query, base_where, top_k)
        
        if not base_where:
            return self.search(query, None, top_k)
        
        # Try with base filters first
        results = self.search(query, base_where, top_k)
        if results:
            logger.info("Progressive query: found results with base filters")
            return results
        
        # Progressive relaxation
        current_where = base_where.copy()
        
        for field_to_drop in relax_order:
            if field_to_drop in current_where:
                del current_where[field_to_drop]
                logger.info(f"Progressive relax: dropped '{field_to_drop}' (relax level: {relax_order.index(field_to_drop) + 1})")
                
                results = self.search(query, current_where, top_k)
                if results:
                    return results
        
        # Final fallback: minimal filters
        final_where = {}
        if 'field' in base_where:
            final_where['field'] = base_where['field']
        
        logger.info("Progressive relax: final fallback with minimal filters")
        return self.search(query, final_where, top_k)
        
    def search_with_fallback(self, query: str, where: Optional[Dict] = None, top_k: int = TEXT_RETRIEVAL_TOP_K) -> List[TextChunk]:
        """Search with progressive filter relaxation if no results found.
        
        Args:
            query: Search query
            where: Optional filters for metadata
            top_k: Number of results to return
            
        Returns:
            List of TextChunk results
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
                
                logger.info(f"Relaxing filters: dropped '{field_to_drop}'")
                results = self.search(query, relaxed_where, top_k)
                if results:
                    return results
                    
        # Final fallback: no filters except field
        final_where = {}
        if 'field' in where:
            final_where['field'] = where['field']
            
        logger.info("Final fallback: using minimal filters")
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
            logger.error(f"Error getting fields from text collection: {e}")
            return []


# Global instance for caching
_text_rag_instance = None


def get_text_rag(chroma_dir: Union[str, Path] = None) -> TextRAG:
    """Get cached TextRAG instance.
    
    Args:
        chroma_dir: Directory containing Chroma database
        
    Returns:
        TextRAG instance
    """
    global _text_rag_instance
    
    if _text_rag_instance is None:
        _text_rag_instance = TextRAG(chroma_dir)
        
    return _text_rag_instance
