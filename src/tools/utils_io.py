"""File I/O utilities for PDF processing and text extraction."""

import hashlib
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

import fitz  # PyMuPDF
from PIL import Image

logger = logging.getLogger(__name__)


def sha1(obj: Union[str, bytes]) -> str:
    """Generate SHA1 hash of string or bytes object.
    
    Args:
        obj: String or bytes to hash
        
    Returns:
        Hexadecimal SHA1 hash string
    """
    if isinstance(obj, str):
        obj = obj.encode('utf-8')
    return hashlib.sha1(obj).hexdigest()


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if necessary.
    
    Args:
        path: Directory path to create
        
    Returns:
        Path object of created directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitize_filename(filename: str) -> str:
    """Sanitize filename by removing/replacing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename safe for filesystem
    """
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove multiple underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Strip leading/trailing whitespace and dots
    sanitized = sanitized.strip(' .')
    return sanitized


def read_pdf_pages(pdf_path: Union[str, Path], max_pages: int = 0) -> List[Dict]:
    """Extract text from PDF pages using PyMuPDF.
    
    Args:
        pdf_path: Path to PDF file
        max_pages: Maximum pages to process (0 = all pages)
        
    Returns:
        List of dicts with doc_id, page, text
    """
    pdf_path = Path(pdf_path)
    
    # Generate doc_id: sanitized_stem + first 8 chars of file hash
    with open(pdf_path, 'rb') as f:
        file_hash = sha1(f.read())
    
    sanitized_stem = sanitize_filename(pdf_path.stem)
    doc_id = f"{sanitized_stem}-{file_hash[:8]}"
    
    pages = []
    
    try:
        doc = fitz.open(pdf_path)
        page_count = len(doc) if max_pages == 0 else min(len(doc), max_pages)
        
        for page_num in range(page_count):
            page = doc.load_page(page_num)
            text = page.get_text()
            
            pages.append({
                'doc_id': doc_id,
                'page': page_num + 1,  # 1-based page numbering
                'text': text.strip()
            })
            
        doc.close()
        logger.info(f"Extracted text from {page_count} pages of {pdf_path.name}")
        
    except Exception as e:
        logger.error(f"Error reading PDF {pdf_path}: {e}")
        raise
        
    return pages


def extract_caption_from_text(text: str, page_num: int) -> Optional[str]:
    """Extract figure caption from page text.
    
    Looks for lines starting with Fig, Figure, Gambar, Gbr.
    
    Args:
        text: Page text content
        page_num: Page number for context
        
    Returns:
        Extracted caption or None if not found
    """
    if not text:
        return None
        
    lines = text.split('\n')
    caption_patterns = [
        r'^\s*Fig\.?\s*\d*[:\.]?\s*(.+)',
        r'^\s*Figure\s*\d*[:\.]?\s*(.+)',
        r'^\s*Gambar\s*\d*[:\.]?\s*(.+)',
        r'^\s*Gbr\.?\s*\d*[:\.]?\s*(.+)'
    ]
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        for pattern in caption_patterns:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                caption = match.group(1).strip()
                if len(caption) > 10:  # Reasonable caption length
                    return caption
                    
    return None


def extract_figures(pdf_path: Union[str, Path], out_dir: Union[str, Path], 
                   max_pages: int = 0) -> List[Dict]:
    """Extract figures from PDF pages and save as images.
    
    Args:
        pdf_path: Path to PDF file
        out_dir: Output directory for extracted images
        max_pages: Maximum pages to process (0 = all pages)
        
    Returns:
        List of dicts with figure metadata
    """
    pdf_path = Path(pdf_path)
    out_dir = ensure_dir(out_dir)
    
    # Generate same doc_id as read_pdf_pages
    with open(pdf_path, 'rb') as f:
        file_hash = sha1(f.read())
    
    sanitized_stem = sanitize_filename(pdf_path.stem)
    doc_id = f"{sanitized_stem}-{file_hash[:8]}"
    
    figures = []
    
    try:
        doc = fitz.open(pdf_path)
        page_count = len(doc) if max_pages == 0 else min(len(doc), max_pages)
        
        for page_num in range(page_count):
            page = doc.load_page(page_num)
            
            # Get page text for caption extraction
            page_text = page.get_text()
            
            # Extract images from page
            image_list = page.get_images()
            
            for img_idx, img in enumerate(image_list):
                try:
                    # Get image data
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    # Skip if not RGB/RGBA
                    if pix.n - pix.alpha < 3:
                        pix = None
                        continue
                        
                    # Convert CMYK to RGB if needed
                    if pix.colorspace and pix.colorspace.n == 4:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    
                    # Save image
                    img_filename = f"{doc_id}_p{page_num + 1}_{img_idx}.png"
                    img_path = out_dir / img_filename
                    
                    if pix.alpha:
                        pix.pil_save(img_path, format="PNG")
                    else:
                        pix.pil_save(img_path, format="PNG")
                    
                    # Try to extract caption
                    caption = extract_caption_from_text(page_text, page_num + 1)
                    if not caption:
                        caption = "Figure (no caption found)"
                    
                    # Get image dimensions
                    with Image.open(img_path) as pil_img:
                        width, height = pil_img.size
                    
                    figures.append({
                        'doc_id': doc_id,
                        'page': page_num + 1,
                        'path': str(img_path),
                        'caption': caption,
                        'bbox': img[1:5] if len(img) > 4 else None,  # x0, y0, x1, y1
                        'width': width,
                        'height': height
                    })
                    
                    pix = None  # Free memory
                    
                except Exception as e:
                    logger.warning(f"Error extracting image {img_idx} from page {page_num + 1} of {pdf_path.name}: {e}")
                    continue
                    
        doc.close()
        logger.info(f"Extracted {len(figures)} figures from {pdf_path.name}")
        
    except Exception as e:
        logger.error(f"Error extracting figures from {pdf_path}: {e}")
        raise
        
    return figures


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
