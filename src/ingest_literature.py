"""CLI entry point for literature ingestion pipeline."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from .config import (
    CORPUS_DIR_PATH, TAXONOMY_XLSX_PATH, PERSIST_TEXT_PATH, PERSIST_IMAGE_PATH,
    IMAGES_OUT_PATH, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP
)
from .tools.utils_io import ensure_dir, read_pdf_pages, extract_figures, setup_logging
from .tools.taxonomy import load_taxonomy
from .tools.classifier import create_classifier
from .tools.rag import get_text_store, get_text_embedder, chunk_text, upsert_text_chunks
from .tools.vision import get_image_store, get_image_embedder, upsert_image_figures

logger = logging.getLogger(__name__)


def extract_folder_hints(pdf_path: Path, corpus_dir: Path) -> List[str]:
    """Extract folder hints from PDF path relative to corpus directory.
    
    Args:
        pdf_path: Full path to PDF file
        corpus_dir: Base corpus directory
        
    Returns:
        List of folder names for classification hints
    """
    try:
        # Get relative path from corpus dir
        rel_path = pdf_path.relative_to(corpus_dir)
        # Get parent directories (exclude filename)
        folder_parts = rel_path.parent.parts
        # Return last 2-3 folder names
        return list(folder_parts[-3:]) if len(folder_parts) >= 3 else list(folder_parts)
    except ValueError:
        # Path is not relative to corpus_dir
        return []


def process_pdf_text(pdf_path: Path, classifier, text_store, text_embedder,
                    folder_hints: List[str], chunk_size: int, overlap: int,
                    max_pages: int, force: bool) -> Dict[str, int]:
    """Process PDF for text extraction and embedding.
    
    Args:
        pdf_path: Path to PDF file
        classifier: Text classifier instance
        text_store: Chroma text collection
        text_embedder: Text embedding model
        folder_hints: Folder hints for classification
        chunk_size: Text chunk size
        overlap: Text chunk overlap
        max_pages: Maximum pages to process (0 = all)
        force: Force re-processing of existing chunks
        
    Returns:
        Dict with processing counts
    """
    try:
        # Extract text from PDF
        logger.info(f"Processing text from {pdf_path.name}")
        paged_text = read_pdf_pages(pdf_path, max_pages)
        
        if not paged_text:
            logger.warning(f"No text extracted from {pdf_path.name}")
            return {"new": 0, "skipped": 0}
            
        # Create document context for classification (title + first page)
        doc_context = ""
        if paged_text:
            first_page = paged_text[0]['text'][:1000]  # First 1000 chars
            doc_context = f"{pdf_path.stem}\n\n{first_page}"
            
        # Chunk text
        chunks = chunk_text(paged_text, chunk_size, overlap)
        
        if not chunks:
            logger.warning(f"No chunks created from {pdf_path.name}")
            return {"new": 0, "skipped": 0}
            
        # Classify and add metadata to chunks
        for chunk in chunks:
            # Classify chunk
            classification = classifier.classify_chunk(
                chunk['text'], 
                doc_context, 
                folder_hints
            )
            
            # Add classification metadata
            chunk.update({
                'filename': pdf_path.name,
                'source_path': str(pdf_path),
                'discipline': classification['discipline'],
                'aspect': classification['aspect'],
                'method': classification['method'],
                'topic': classification['topic'],
                'keywords': classification['keywords']
            })
            
        # Upsert chunks to vector store
        result = upsert_text_chunks(text_store, chunks, text_embedder, force)
        logger.info(f"Text processing complete for {pdf_path.name}: {result['new']} new, {result['skipped']} skipped")
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing text from {pdf_path}: {e}")
        return {"new": 0, "skipped": 0}


def process_pdf_images(pdf_path: Path, classifier, image_store, image_embedder,
                      folder_hints: List[str], images_out_dir: Path, 
                      max_pages: int, force: bool) -> Dict[str, int]:
    """Process PDF for image extraction and embedding.
    
    Args:
        pdf_path: Path to PDF file
        classifier: Text classifier instance
        image_store: Chroma image collection
        image_embedder: Image embedding model
        folder_hints: Folder hints for classification
        images_out_dir: Directory to save extracted images
        max_pages: Maximum pages to process (0 = all)
        force: Force re-processing of existing images
        
    Returns:
        Dict with processing counts
    """
    try:
        # Extract figures from PDF
        logger.info(f"Processing images from {pdf_path.name}")
        figures = extract_figures(pdf_path, images_out_dir, max_pages)
        
        if not figures:
            logger.info(f"No figures extracted from {pdf_path.name}")
            return {"new": 0, "skipped": 0}
            
        # Create document context for classification
        try:
            paged_text = read_pdf_pages(pdf_path, 1)  # Just first page for context
            doc_context = ""
            if paged_text:
                doc_context = f"{pdf_path.stem}\n\n{paged_text[0]['text'][:1000]}"
        except Exception as e:
            logger.warning(f"Could not extract text context from {pdf_path}: {e}")
            doc_context = pdf_path.stem
            
        # Classify and add metadata to figures
        for figure in figures:
            # Classify figure
            classification = classifier.classify_figure(
                figure['caption'],
                doc_context,
                folder_hints
            )
            
            # Add classification metadata
            figure.update({
                'filename': pdf_path.name,
                'source_path': str(pdf_path),
                'discipline': classification['discipline'],
                'aspect': classification['aspect'],
                'method': classification['method'],
                'topic': classification['topic'],
                'figure_type': classification['figure_type'],
                'keywords': classification['keywords']
            })
            
        # Upsert figures to vector store
        result = upsert_image_figures(image_store, figures, image_embedder, force)
        logger.info(f"Image processing complete for {pdf_path.name}: {result['new']} new, {result['skipped']} skipped")
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing images from {pdf_path}: {e}")
        return {"new": 0, "skipped": 0}


def print_summary(text_results: Dict[str, int], image_results: Dict[str, int],
                 text_store, image_store, pdf_count: int):
    """Print processing summary.
    
    Args:
        text_results: Text processing results
        image_results: Image processing results
        text_store: Text Chroma collection
        image_store: Image Chroma collection
        pdf_count: Number of PDFs processed
    """
    print("\n" + "="*60)
    print("INGESTION SUMMARY")
    print("="*60)
    
    print(f"PDFs processed: {pdf_count}")
    
    # Text results
    print(f"\nText chunks:")
    print(f"  - New: {text_results['new']}")
    print(f"  - Skipped: {text_results['skipped']}")
    if text_store:
        print(f"  - Total in store: {text_store.count()}")
        
        # Show sample text metadata
        if text_results['new'] > 0:
            try:
                sample = text_store.get(limit=1)
                if sample['ids']:
                    metadata = sample['metadatas'][0]
                    print(f"  - Sample metadata: {dict(list(metadata.items())[:6])}")
            except Exception as e:
                logger.warning(f"Could not fetch sample text metadata: {e}")
    else:
        print(f"  - Total in store: N/A (text processing disabled)")
            
    # Image results
    print(f"\nImage figures:")
    print(f"  - New: {image_results['new']}")
    print(f"  - Skipped: {image_results['skipped']}")
    if image_store:
        print(f"  - Total in store: {image_store.count()}")
        
        # Show sample image metadata
        if image_results['new'] > 0:
            try:
                sample = image_store.get(limit=1)
                if sample['ids']:
                    metadata = sample['metadatas'][0]
                    print(f"  - Sample metadata: {dict(list(metadata.items())[:6])}")
            except Exception as e:
                logger.warning(f"Could not fetch sample image metadata: {e}")
    else:
        print(f"  - Total in store: N/A (image processing disabled)")
            
    print("="*60)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest literature PDFs into vector stores",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--corpus_dir", 
        type=str, 
        default=str(CORPUS_DIR_PATH),
        help="Directory containing PDF files to process"
    )
    
    parser.add_argument(
        "--taxonomy_xlsx",
        type=str,
        default=str(TAXONOMY_XLSX_PATH),
        help="Path to taxonomy Excel file"
    )
    
    parser.add_argument(
        "--max_pages",
        type=int,
        default=0,
        help="Maximum pages per PDF to process (0 = all pages)"
    )
    
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Text chunk size in tokens"
    )
    
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP,
        help="Text chunk overlap in tokens"
    )
    
    parser.add_argument(
        "--text",
        action="store_true",
        default=True,
        help="Process text content"
    )
    
    parser.add_argument(
        "--images",
        action="store_true",
        default=True,
        help="Process image content"
    )
    
    parser.add_argument(
        "--no-text",
        action="store_false",
        dest="text",
        help="Skip text processing"
    )
    
    parser.add_argument(
        "--no-images",
        action="store_false",
        dest="images",
        help="Skip image processing"
    )
    
    parser.add_argument(
        "--force_text",
        action="store_true",
        help="Force re-processing of existing text chunks"
    )
    
    parser.add_argument(
        "--force_image",
        action="store_true",
        help="Force re-processing of existing images"
    )
    
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Validate paths
    corpus_dir = Path(args.corpus_dir)
    taxonomy_path = Path(args.taxonomy_xlsx)
    
    if not corpus_dir.exists():
        logger.error(f"Corpus directory not found: {corpus_dir}")
        sys.exit(1)
        
    if not taxonomy_path.exists():
        logger.error(f"Taxonomy file not found: {taxonomy_path}")
        sys.exit(1)
        
    # Create output directories
    ensure_dir(PERSIST_TEXT_PATH)
    ensure_dir(PERSIST_IMAGE_PATH)
    ensure_dir(IMAGES_OUT_PATH)
    
    print("="*60)
    print("LITERATURE INGESTION PIPELINE")
    print("="*60)
    print(f"Corpus directory: {corpus_dir}")
    print(f"Taxonomy file: {taxonomy_path}")
    print(f"Text store: {PERSIST_TEXT_PATH}")
    print(f"Image store: {PERSIST_IMAGE_PATH}")
    print(f"Images output: {IMAGES_OUT_PATH}")
    print(f"Processing: text={args.text}, images={args.images}")
    print("="*60)
    
    try:
        # Load taxonomy
        logger.info("Loading taxonomy...")
        taxonomy = load_taxonomy(taxonomy_path)
        classifier = create_classifier(taxonomy)
        
        # Initialize stores and embedders
        text_store = None
        text_embedder = None
        image_store = None
        image_embedder = None
        
        if args.text:
            logger.info("Initializing text processing...")
            text_store = get_text_store(str(PERSIST_TEXT_PATH))
            text_embedder = get_text_embedder()
            
        if args.images:
            logger.info("Initializing image processing...")
            image_store = get_image_store(str(PERSIST_IMAGE_PATH))
            image_embedder = get_image_embedder()
            
        # Find all PDF files
        pdf_files = list(corpus_dir.rglob("*.pdf"))
        
        if not pdf_files:
            print("\nNo PDF files found in corpus directory.")
            print("Please check the path and ensure PDF files are present.")
            return
            
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Process PDFs
        total_text_results = {"new": 0, "skipped": 0}
        total_image_results = {"new": 0, "skipped": 0}
        
        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            try:
                # Extract folder hints
                folder_hints = extract_folder_hints(pdf_path, corpus_dir)
                
                # Process text
                if args.text and text_store and text_embedder:
                    text_result = process_pdf_text(
                        pdf_path, classifier, text_store, text_embedder,
                        folder_hints, args.chunk_size, args.overlap,
                        args.max_pages, args.force_text
                    )
                    total_text_results["new"] += text_result["new"]
                    total_text_results["skipped"] += text_result["skipped"]
                    
                # Process images
                if args.images and image_store and image_embedder:
                    image_result = process_pdf_images(
                        pdf_path, classifier, image_store, image_embedder,
                        folder_hints, IMAGES_OUT_PATH, args.max_pages, args.force_image
                    )
                    total_image_results["new"] += image_result["new"]
                    total_image_results["skipped"] += image_result["skipped"]
                    
            except Exception as e:
                logger.error(f"Error processing {pdf_path.name}: {e}")
                continue
                
        # Print summary
        if args.text and text_store:
            if args.images and image_store:
                print_summary(total_text_results, total_image_results, 
                            text_store, image_store, len(pdf_files))
            else:
                print_summary(total_text_results, {"new": 0, "skipped": 0},
                            text_store, None, len(pdf_files))
        elif args.images and image_store:
            print_summary({"new": 0, "skipped": 0}, total_image_results,
                        None, image_store, len(pdf_files))
                        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
