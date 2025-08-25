"""Image embedding with Chroma vector store and OpenCLIP."""

import logging
from pathlib import Path
from typing import Dict, List, Union

import chromadb
import numpy as np
from PIL import Image

try:
    from ..config import DEFAULT_IMAGE_MODEL, DEFAULT_IMAGE_PRETRAINED
except ImportError:
    from src.config import DEFAULT_IMAGE_MODEL, DEFAULT_IMAGE_PRETRAINED
from .utils_io import sha1

logger = logging.getLogger(__name__)

# Try to import OpenCLIP, handle gracefully if not available
try:
    import open_clip
    import torch
    OPENCLIP_AVAILABLE = True
except ImportError:
    OPENCLIP_AVAILABLE = False
    logger.warning("OpenCLIP not available, image embedding will not work")


class ImageEmbedder:
    """Image embedding using OpenCLIP."""
    
    def __init__(self, model_name: str = DEFAULT_IMAGE_MODEL, 
                 pretrained: str = DEFAULT_IMAGE_PRETRAINED):
        """Initialize image embedder.
        
        Args:
            model_name: OpenCLIP model name (e.g., 'ViT-L-14')
            pretrained: Pretrained weights (e.g., 'openai')
        """
        if not OPENCLIP_AVAILABLE:
            raise ImportError("OpenCLIP is not available. Please install: pip install open_clip_torch")
            
        self.model_name = model_name
        self.pretrained = pretrained
        self.model = None
        self.preprocess = None
        self.device = None
        
    def _load_model(self):
        """Lazy load the embedding model."""
        if self.model is None:
            try:
                logger.info(f"Loading image embedding model: {self.model_name} ({self.pretrained})")
                
                # Determine device (CPU fallback)
                if torch.cuda.is_available():
                    self.device = torch.device('cuda')
                    logger.info("Using GPU for image embeddings")
                else:
                    self.device = torch.device('cpu')
                    logger.info("Using CPU for image embeddings")
                
                # Load model and preprocessing
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    self.model_name, 
                    pretrained=self.pretrained,
                    device=self.device
                )
                self.model.eval()
                
            except Exception as e:
                logger.error(f"Failed to load image embedding model: {e}")
                raise
                
    def _load_image(self, image_path: Union[str, Path]) -> Image.Image:
        """Load and validate image file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            PIL Image object
        """
        try:
            image = Image.open(image_path)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            return image
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            raise
            
    def embed_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """Embed single image into vector space.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Embedding vector
        """
        self._load_model()
        
        try:
            # Load and preprocess image
            image = self._load_image(image_path)
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Generate embedding
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
            return image_features.cpu().numpy()[0]
            
        except Exception as e:
            logger.error(f"Error embedding image {image_path}: {e}")
            raise
            
    def embed_image_batch(self, image_paths: List[Union[str, Path]], 
                         batch_size: int = 16) -> List[np.ndarray]:
        """Embed multiple images in batches.
        
        Args:
            image_paths: List of image file paths
            batch_size: Batch size for processing
            
        Returns:
            List of embedding vectors
        """
        self._load_model()
        
        embeddings = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            
            try:
                # Load and preprocess batch
                batch_tensors = []
                valid_indices = []
                
                for j, image_path in enumerate(batch_paths):
                    try:
                        image = self._load_image(image_path)
                        image_tensor = self.preprocess(image)
                        batch_tensors.append(image_tensor)
                        valid_indices.append(i + j)
                    except Exception as e:
                        logger.warning(f"Skipping invalid image {image_path}: {e}")
                        embeddings.append(None)  # Placeholder for failed image
                        
                if not batch_tensors:
                    continue
                    
                # Stack tensors and move to device
                batch_tensor = torch.stack(batch_tensors).to(self.device)
                
                # Generate embeddings
                with torch.no_grad():
                    batch_features = self.model.encode_image(batch_tensor)
                    batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
                    
                # Add embeddings to results
                batch_embeddings = batch_features.cpu().numpy()
                for j, embedding in enumerate(batch_embeddings):
                    # Insert at correct position (accounting for failed images)
                    while len(embeddings) <= valid_indices[j]:
                        embeddings.append(None)
                    embeddings[valid_indices[j]] = embedding
                    
            except Exception as e:
                logger.error(f"Error processing image batch: {e}")
                # Add None placeholders for failed batch
                for _ in batch_paths:
                    embeddings.append(None)
                    
        logger.info(f"Embedded {sum(1 for e in embeddings if e is not None)}/{len(image_paths)} images")
        return embeddings


def get_image_store(persist_dir: str, collection_name: str = "image_emb") -> chromadb.Collection:
    """Get or create Chroma image collection.
    
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
        logger.error(f"Error setting up image store: {e}")
        raise


def get_image_embedder(model: str = DEFAULT_IMAGE_MODEL,
                      pretrained: str = DEFAULT_IMAGE_PRETRAINED) -> ImageEmbedder:
    """Get image embedder instance.
    
    Args:
        model: OpenCLIP model name
        pretrained: Pretrained weights name
        
    Returns:
        ImageEmbedder instance
    """
    return ImageEmbedder(model, pretrained)


def create_image_id(doc_id: str, page: int, path: str) -> str:
    """Create idempotent ID for image.
    
    Args:
        doc_id: Document ID
        page: Page number
        path: Image file path
        
    Returns:
        Unique image ID
    """
    return sha1(f"{doc_id}:{page}:{path}")[:16]


def upsert_image_figures(collection: chromadb.Collection, figures: List[Dict],
                        embedder: ImageEmbedder, force: bool = False) -> Dict[str, int]:
    """Upsert image figures into Chroma collection.
    
    Args:
        collection: Chroma collection
        figures: List of figure dicts with metadata
        embedder: Image embedder instance
        force: Force re-embedding of existing figures
        
    Returns:
        Dict with counts of new and skipped items
    """
    if not figures:
        return {"new": 0, "skipped": 0}
        
    # Create image IDs
    for figure in figures:
        if 'image_id' not in figure:
            figure['image_id'] = create_image_id(
                figure['doc_id'], 
                figure['page'], 
                figure['path']
            )
    
    # Check which images already exist
    existing_ids = set()
    if not force:
        try:
            image_ids = [figure['image_id'] for figure in figures]
            existing_results = collection.get(ids=image_ids)
            existing_ids = set(existing_results['ids'])
        except Exception as e:
            logger.warning(f"Error checking existing images: {e}")
            
    # Filter new figures
    new_figures = [figure for figure in figures if figure['image_id'] not in existing_ids or force]
    skipped_count = len(figures) - len(new_figures)
    
    if not new_figures:
        logger.info(f"All {len(figures)} images already exist, skipping")
        return {"new": 0, "skipped": skipped_count}
        
    # Embed new images
    logger.info(f"Embedding {len(new_figures)} new images")
    
    # Extract paths and embed in batches
    image_paths = [figure['path'] for figure in new_figures]
    embeddings = embedder.embed_image_batch(image_paths)
    
    # Filter out failed embeddings
    valid_figures = []
    valid_embeddings = []
    
    for figure, embedding in zip(new_figures, embeddings):
        if embedding is not None:
            valid_figures.append(figure)
            valid_embeddings.append(embedding)
        else:
            logger.warning(f"Failed to embed image: {figure['path']}")
            
    if not valid_figures:
        logger.warning("No valid image embeddings generated")
        return {"new": 0, "skipped": len(figures)}
        
    # Prepare data for upsert
    ids = [figure['image_id'] for figure in valid_figures]
    documents = [figure['caption'] for figure in valid_figures]  # Use caption as document
    metadatas = []
    
    for figure in valid_figures:
        metadata = {k: v for k, v in figure.items() if k not in ['image_id', 'path']}
        # Convert lists and tuples to strings for Chroma compatibility
        for key, value in metadata.items():
            if isinstance(value, (list, tuple)):
                metadata[key] = str(value)
        # Keep path as metadata for reference
        metadata['image_path'] = figure['path']
        metadatas.append(metadata)
    
    # Upsert to collection
    try:
        collection.upsert(
            ids=ids,
            embeddings=[emb.tolist() for emb in valid_embeddings],
            documents=documents,
            metadatas=metadatas
        )
        logger.info(f"Upserted {len(valid_figures)} image embeddings to collection")
        
    except Exception as e:
        logger.error(f"Error upserting image embeddings: {e}")
        raise
        
    failed_count = len(new_figures) - len(valid_figures)
    return {"new": len(valid_figures), "skipped": skipped_count + failed_count}
