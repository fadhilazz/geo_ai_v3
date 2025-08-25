# Geo AI v3 - Complete Geothermal AI Agent

AI agent for geothermal geoscientist interpretation with comprehensive RAG system, literature ingestion pipeline, and Semurup data integration.

## Features

### 🚀 **Complete RAG System**
- **Question Matrix Integration**: Intent-driven retrieval strategies
- **Field Detection**: Automatic geothermal field identification
- **Multi-modal Retrieval**: Text + Image evidence synthesis
- **Comprehensive Analysis**: Up to 12 text chunks + 6 images
- **Direct Data Integration**: Semurup-specific fallback data

### 📚 **Literature Ingestion Pipeline**
- **Windows-optimized paths** with proper raw string handling
- **Typo-tolerant classification** using RapidFuzz (1-2 typo tolerance)
- **Dual embedding stores**: Text (E5-large-v2) + Images (OpenCLIP ViT-L/14)
- **Folder-based discipline hints** from directory structure
- **Idempotent processing** - skip already processed items
- **CPU fallback** for all models
- **Comprehensive logging** and progress tracking

### 🌡️ **Semurup Geothermal Data**
- **Reservoir Temperature**: 229-239°C (geothermometer analysis)
- **Upflow Zone**: Dusun Baru (95.7°C) - Primary discharge
- **Outflow Zone**: Mukai Pintu (39°C) - Diluted fluids
- **Location**: Jambi Province, Indonesia (NOT West Java)
- **Geochemical Analysis**: High Cl content, minimal fluid-rock interaction

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Install in development mode
pip install -e .
```

## Usage

### 🎯 **QA System (Primary)**

```bash
# Start the API server
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000

# Test with curl
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Dimana outflow zone di Semurup?", "field": "Semurup"}'
```

### 📖 **Literature Ingestion**

```bash
# Basic ingestion
python -m src.ingest_literature

# Advanced usage
python -m src.ingest_literature \
    --corpus_dir "D:\Work\geo_ai_v3\knowledge" \
    --taxonomy_xlsx "D:\Work\geo_ai_v3\ingest_guide\Geothermal Taxonomy.xlsx" \
    --max_pages 10 \
    --chunk_size 500 \
    --overlap 80 \
    --text --images \
    --log_level INFO
```

## API Endpoints

- `POST /ask` - Main QA endpoint
- `GET /fields` - List available geothermal fields
- `GET /health` - Health check
- `GET /stats` - System statistics

## Configuration

Edit `src/config.py` to customize paths and parameters:

```python
# Windows paths (use raw strings)
TAXONOMY_XLSX = r"D:\Work\geo_ai_v3\ingest_guide\Geothermal Taxonomy.xlsx"
CORPUS_DIR = r"D:\Work\geo_ai_v3\knowledge"
PERSIST_TEXT = r"D:\Work\geo_ai_v3\knowledge\text_emb"
PERSIST_IMAGE = r"D:\Work\geo_ai_v3\knowledge\image_emb"
IMAGES_OUT = r"D:\Work\geo_ai_v3\knowledge\images"

# QA System Configuration
QUESTION_MATRIX_PATH = r"D:\Work\geo_ai_v3\ingest_guide\Question Matrix.csv"
CHROMA_TEXT_DIR = r"D:\Work\geo_ai_v3\knowledge\text_emb"
CHROMA_IMAGE_DIR = r"D:\Work\geo_ai_v3\knowledge\image_emb"
```

## Architecture

```
src/
├── config.py              # Path constants and configuration
├── api.py                 # FastAPI server
├── app_graph.py           # LangGraph QA workflow
├── ingest_literature.py   # CLI entry point for ingestion
├── prompts/
│   └── system_prompt.py   # LLM system prompts
└── tools/
    ├── utils_io.py        # PDF processing and file utilities
    ├── taxonomy.py        # Excel taxonomy loader
    ├── classifier.py      # Typo-tolerant classification
    ├── rag.py            # Text embedding and Chroma store
    ├── vision.py         # Image embedding and Chroma store
    ├── qm.py             # Question Matrix loader
    ├── rag_text.py       # Text RAG with fallback
    ├── rag_image.py      # Image RAG with CLIP
    ├── field_detect.py   # Field detection
    └── semurup_data.py   # Direct Semurup data integration
```

## Dependencies

- **fastapi**: API framework
- **langchain-openai**: LLM integration
- **langgraph**: Workflow orchestration
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
- Direct Semurup data integration provides fallback for critical queries
