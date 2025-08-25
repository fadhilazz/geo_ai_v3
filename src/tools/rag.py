"""Text RAG with Chroma vector store and Sentence Transformers."""

import logging
import re
from typing import Dict, List, Optional, Union

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from ..config import DEFAULT_TEXT_MODEL, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP
except ImportError:
    from src.config import DEFAULT_TEXT_MODEL, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP
from .utils_io import sha1

logger = logging.getLogger(__name__)

# Try to import tiktoken for better token counting, fallback to simple estimation
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not available, using simple token estimation")


class TextEmbedder:
    """Text embedding using Sentence Transformers."""
    
    def __init__(self, model_name: str = DEFAULT_TEXT_MODEL):
        """Initialize text embedder.
        
        Args:
            model_name: Name of sentence transformer model
        """
        self.model_name = model_name
        self.model = None
        
    def _load_model(self):
        """Lazy load the embedding model."""
        if self.model is None:
            try:
                logger.info(f"Loading text embedding model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                # Try to use GPU if available, fallback to CPU
                if hasattr(self.model, 'to'):
                    try:
                        import torch
                        if torch.cuda.is_available():
                            self.model = self.model.to('cuda')
                            logger.info("Using GPU for text embeddings")
                        else:
                            self.model = self.model.to('cpu')
                            logger.info("Using CPU for text embeddings")
                    except ImportError:
                        logger.info("PyTorch not available, using CPU for text embeddings")
            except Exception as e:
                logger.error(f"Failed to load text embedding model: {e}")
                raise
                
    def embed(self, texts: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """Embed text(s) into vector space.
        
        Args:
            texts: Single text or list of texts to embed
            
        Returns:
            Embedding vector(s)
        """
        self._load_model()
        
        if isinstance(texts, str):
            texts = [texts]
            return_single = True
        else:
            return_single = False
            
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            
            if return_single:
                return embeddings[0]
            return embeddings
            
        except Exception as e:
            logger.error(f"Error embedding texts: {e}")
            raise


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens in text.
    
    Args:
        text: Input text
        encoding_name: Tiktoken encoding name
        
    Returns:
        Token count
    """
    if TIKTOKEN_AVAILABLE:
        try:
            encoding = tiktoken.get_encoding(encoding_name)
            return len(encoding.encode(text))
        except Exception:
            pass
            
    # Fallback: simple word-based estimation
    # Rough approximation: 1 token ≈ 0.75 words for English
    words = len(text.split())
    return int(words / 0.75)


def chunk_text(paged_text: List[Dict], chunk_size: int = DEFAULT_CHUNK_SIZE,
               overlap: int = DEFAULT_OVERLAP) -> List[Dict]:
    """Chunk text from paged documents.
    
    Args:
        paged_text: List of dicts with doc_id, page, text
        chunk_size: Target chunk size in tokens
        overlap: Overlap size in tokens
        
    Returns:
        List of text chunks with metadata
    """
    chunks = []
    
    for page_data in paged_text:
        doc_id = page_data['doc_id']
        page_num = page_data['page']
        text = page_data['text']
        
        if not text.strip():
            continue
            
        # Split text into sentences for better chunking
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        current_chunk = ""
        current_tokens = 0
        chunk_start = 0
        
        for i, sentence in enumerate(sentences):
            sentence_tokens = count_tokens(sentence)
            
            # If adding this sentence would exceed chunk size
            if current_tokens + sentence_tokens > chunk_size and current_chunk:
                # Create chunk
                chunk_id = sha1(f"{doc_id}:{page_num}:{chunk_start}:{current_chunk[:64]}")[:16]
                
                chunks.append({
                    'chunk_id': chunk_id,
                    'doc_id': doc_id,
                    'page': page_num,
                    'text': current_chunk.strip(),
                    'start_sentence': chunk_start,
                    'end_sentence': i - 1,
                    'token_count': current_tokens
                })
                
                # Start new chunk with overlap
                if overlap > 0:
                    # Find overlap point
                    overlap_text = ""
                    overlap_tokens = 0
                    overlap_start = max(0, i - 3)  # Go back a few sentences for overlap
                    
                    for j in range(overlap_start, i):
                        if j < len(sentences):
                            sent_tokens = count_tokens(sentences[j])
                            if overlap_tokens + sent_tokens <= overlap:
                                overlap_text += sentences[j] + ". "
                                overlap_tokens += sent_tokens
                            else:
                                break
                                
                    current_chunk = overlap_text + sentence + ". "
                    current_tokens = count_tokens(current_chunk)
                    chunk_start = overlap_start
                else:
                    current_chunk = sentence + ". "
                    current_tokens = sentence_tokens
                    chunk_start = i
            else:
                current_chunk += sentence + ". "
                current_tokens += sentence_tokens
                
        # Add final chunk if any content remains
        if current_chunk.strip():
            chunk_id = sha1(f"{doc_id}:{page_num}:{chunk_start}:{current_chunk[:64]}")[:16]
            
            chunks.append({
                'chunk_id': chunk_id,
                'doc_id': doc_id,
                'page': page_num,
                'text': current_chunk.strip(),
                'start_sentence': chunk_start,
                'end_sentence': len(sentences) - 1,
                'token_count': current_tokens
            })
            
    logger.info(f"Created {len(chunks)} text chunks")
    return chunks


def get_text_store(persist_dir: str, collection_name: str = "text_emb") -> chromadb.Collection:
    """Get or create Chroma text collection.
    
    Args:
        persist_dir: Directory for persistent storage
        collection_name: Name of the collection
        
    Returns:
        Chroma collection instance
    """
    try:
        client = chromadb.PersistentClient(path=persist_dir)
        
        # Get or create collection
        try:
            collection = client.get_collection(name=collection_name)
            logger.info(f"Loaded existing collection '{collection_name}' with {collection.count()} items")
        except Exception:
            # Collection doesn't exist, create it
            collection = client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            logger.info(f"Created new collection '{collection_name}'")
            
        return collection
        
    except Exception as e:
        logger.error(f"Error setting up text store: {e}")
        raise


def get_text_embedder(model_name: str = DEFAULT_TEXT_MODEL) -> TextEmbedder:
    """Get text embedder instance.
    
    Args:
        model_name: Name of sentence transformer model
        
    Returns:
        TextEmbedder instance
    """
    return TextEmbedder(model_name)


def upsert_text_chunks(collection: chromadb.Collection, chunks: List[Dict],
                      embedder: TextEmbedder, force: bool = False) -> Dict[str, int]:
    """Upsert text chunks into Chroma collection.
    
    Args:
        collection: Chroma collection
        chunks: List of text chunks with metadata
        embedder: Text embedder instance
        force: Force re-embedding of existing chunks
        
    Returns:
        Dict with counts of new and skipped items
    """
    if not chunks:
        return {"new": 0, "skipped": 0}
        
    # Check which chunks already exist
    existing_ids = set()
    if not force:
        try:
            chunk_ids = [chunk['chunk_id'] for chunk in chunks]
            existing_results = collection.get(ids=chunk_ids)
            existing_ids = set(existing_results['ids'])
        except Exception as e:
            logger.warning(f"Error checking existing chunks: {e}")
            
    # Filter new chunks
    new_chunks = [chunk for chunk in chunks if chunk['chunk_id'] not in existing_ids or force]
    skipped_count = len(chunks) - len(new_chunks)
    
    if not new_chunks:
        logger.info(f"All {len(chunks)} chunks already exist, skipping")
        return {"new": 0, "skipped": skipped_count}
        
    # Embed new chunks
    logger.info(f"Embedding {len(new_chunks)} new text chunks")
    texts = [chunk['text'] for chunk in new_chunks]
    embeddings = embedder.embed(texts)
    
    # Prepare data for upsert
    ids = [chunk['chunk_id'] for chunk in new_chunks]
    metadatas = []
    
    for chunk in new_chunks:
        metadata = {k: v for k, v in chunk.items() if k not in ['chunk_id', 'text']}
        # Convert lists to strings for Chroma compatibility
        for key, value in metadata.items():
            if isinstance(value, list):
                metadata[key] = str(value)
        metadatas.append(metadata)
    
    # Upsert to collection
    try:
        collection.upsert(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas
        )
        logger.info(f"Upserted {len(new_chunks)} text chunks to collection")
        
    except Exception as e:
        logger.error(f"Error upserting text chunks: {e}")
        raise
        
    return {"new": len(new_chunks), "skipped": skipped_count}
