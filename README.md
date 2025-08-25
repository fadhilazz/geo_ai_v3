# Geo AI v3 - Literature Ingestion Pipeline

A robust ingestion pipeline that loads geothermal literature taxonomy from Excel, extracts text chunks and figure images from PDFs, classifies content using typo-tolerant matching, and embeds into persistent Chroma vector stores.

## Features

- **Windows-optimized paths** with proper raw string handling
- **Typo-tolerant classification** using RapidFuzz (1-2 typo tolerance)
- **Dual embedding stores**: Text (E5-large-v2) + Images (OpenCLIP ViT-L/14)
- **Folder-based discipline hints** from directory structure
- **Idempotent processing** - skip already processed items
- **CPU fallback** for all models
- **Comprehensive logging** and progress tracking

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Install in development mode
pip install -e .
```

## Usage

### Basic Usage

```bash
python -m src.ingest_literature
```

### Advanced Usage

```bash
python -m src.ingest_literature \
    --corpus_dir "D:\Work\geo_ai_v3\knowledge" \
    --taxonomy_xlsx "D:\Work\geo_ai_v3\ingest_guide\Geothermal Taxonomy.xlsx" \
    --max_pages 10 \
    --chunk_size 500 \
    --overlap 80 \
    --text --images \
    --log_level INFO
```

### Command Line Options

- `--corpus_dir`: Directory containing PDF files (default: from config)
- `--taxonomy_xlsx`: Path to taxonomy Excel file (default: from config)
- `--max_pages`: Maximum pages per PDF (0 = all pages)
- `--chunk_size`: Text chunk size in tokens (default: 500)
- `--overlap`: Text chunk overlap in tokens (default: 80)
- `--text/--no-text`: Enable/disable text processing
- `--images/--no-images`: Enable/disable image processing
- `--force_text`: Force re-processing of existing text chunks
- `--force_image`: Force re-processing of existing images
- `--log_level`: Logging level (DEBUG, INFO, WARNING, ERROR)

## Configuration

Edit `src/config.py` to customize paths and parameters:

```python
# Windows paths (use raw strings)
TAXONOMY_XLSX = r"D:\Work\geo_ai_v3\ingest_guide\Geothermal Taxonomy.xlsx"
CORPUS_DIR = r"D:\Work\geo_ai_v3\knowledge"
PERSIST_TEXT = r"D:\Work\geo_ai_v3\knowledge\text_emb"
PERSIST_IMAGE = r"D:\Work\geo_ai_v3\knowledge\image_emb"
IMAGES_OUT = r"D:\Work\geo_ai_v3\knowledge\images"
```

## Taxonomy Excel Format

The pipeline expects an Excel file with these sheets:

- **Aspects**: Geothermal aspects with keywords
- **Geophysics_Methods**: Geophysical methods with keywords
- **Geochemistry_Topics**: Geochemical topics with keywords
- **Figure_Types**: Figure type classifications with keywords
- **Synonyms_Map**: Synonym mappings (Indonesian/English)
- **Intent_Tags**: Additional intent classifications

Each sheet should have columns for names/types and associated keywords.

## Output

The pipeline creates:

1. **Text embeddings** in `PERSIST_TEXT/text_emb` collection
2. **Image embeddings** in `PERSIST_IMAGE/image_emb` collection
3. **Extracted figures** saved in `IMAGES_OUT` directory

### Metadata Structure

Both text and image items include:
- `doc_id`: Unique document identifier
- `filename`: Original PDF filename
- `page`: Page number
- `discipline[]`: Classified disciplines
- `aspect[]`: Classified aspects
- `method[]`: Classified methods
- `topic[]`: Classified topics (geochemistry)
- `figure_type`: Figure classification (images only)
- `keywords[]`: Matched keywords
- `source_path`: Original file path

## Architecture

```
src/
├── config.py              # Path constants and configuration
├── ingest_literature.py   # CLI entry point
└── tools/
    ├── utils_io.py        # PDF processing and file utilities
    ├── taxonomy.py        # Excel taxonomy loader
    ├── classifier.py      # Typo-tolerant classification
    ├── rag.py            # Text embedding and Chroma store
    └── vision.py         # Image embedding and Chroma store
```

## Dependencies

- **chromadb**: Vector database
- **sentence-transformers**: Text embeddings
- **open_clip_torch**: Image embeddings
- **pymupdf**: PDF processing
- **rapidfuzz**: Fuzzy string matching
- **pandas**: Excel processing
- **pillow**: Image processing
- **tqdm**: Progress bars

## Notes

- All models automatically fallback to CPU if GPU unavailable
- Processing is idempotent - rerunning skips existing items
- Folder names (`Geology`, `Geophysics`, `Geochemistry`, `Geothermal`, `Spesific`) influence classification
- Supports 1-2 typo tolerance with 80% fuzzy match threshold
- Caption extraction attempts to find "Fig", "Figure", "Gambar", "Gbr" patterns
