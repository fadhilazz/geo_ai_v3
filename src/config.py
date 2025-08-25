"""Configuration constants for geo_ai_v3 ingestion and QA pipeline."""

import os
from pathlib import Path

# Fixed Windows paths (use raw strings) - Ingestion
TAXONOMY_XLSX = r"D:\Work\geo_ai_v3\ingest_guide\Geothermal Taxonomy.xlsx"
CORPUS_DIR = r"D:\Work\geo_ai_v3\knowledge"
PERSIST_TEXT = r"D:\Work\geo_ai_v3\knowledge\text_emb"
PERSIST_IMAGE = r"D:\Work\geo_ai_v3\knowledge\image_emb"
IMAGES_OUT = r"D:\Work\geo_ai_v3\knowledge\images"

# QA Engine paths (Windows defaults, overridable by CLI/env)
QUESTION_MATRIX_PATH = r"D:\Work\geo_ai_v3\ingest_guide\Question Matrix.csv"
CHROMA_TEXT_DIR = r"D:\Work\geo_ai_v3\knowledge\text_emb"
CHROMA_IMAGE_DIR = r"D:\Work\geo_ai_v3\knowledge\image_emb"
IMAGES_BASE_DIR = r"D:\Work\geo_ai_v3\knowledge\images"
FACTS_DIR = r"D:\Work\geo_ai_v3\data"
STAMP_PATH = r"D:\Work\geo_ai_v3\knowledge\last_ingest.stamp"

# Convert to Path objects for cross-platform compatibility
TAXONOMY_XLSX_PATH = Path(TAXONOMY_XLSX)
CORPUS_DIR_PATH = Path(CORPUS_DIR)
PERSIST_TEXT_PATH = Path(PERSIST_TEXT)
PERSIST_IMAGE_PATH = Path(PERSIST_IMAGE)
IMAGES_OUT_PATH = Path(IMAGES_OUT)

# QA Engine Path objects
QUESTION_MATRIX_PATH_OBJ = Path(QUESTION_MATRIX_PATH)
CHROMA_TEXT_DIR_OBJ = Path(CHROMA_TEXT_DIR)
CHROMA_IMAGE_DIR_OBJ = Path(CHROMA_IMAGE_DIR)
IMAGES_BASE_DIR_OBJ = Path(IMAGES_BASE_DIR)
FACTS_DIR_OBJ = Path(FACTS_DIR)
STAMP_PATH_OBJ = Path(STAMP_PATH)

# Environment variable overrides for QA paths
def get_qa_paths():
    """Get QA paths with environment variable overrides."""
    return {
        'question_matrix': Path(os.getenv('QM_PATH', QUESTION_MATRIX_PATH)),
        'text_db': Path(os.getenv('TEXT_DB_PATH', CHROMA_TEXT_DIR)),
        'image_db': Path(os.getenv('IMAGE_DB_PATH', CHROMA_IMAGE_DIR)),
        'images_dir': Path(os.getenv('IMAGES_DIR_PATH', IMAGES_BASE_DIR))
    }

# Folder hints for discipline classification
DISCIPLINE_FOLDERS = {
    "geology": "Geology",
    "geophysics": "Geophysics", 
    "geochemistry": "Geochemistry",
    "geothermal": "Geothermal",
    "spesific": "Field_Specific",  # Handle typo in folder name
    "specific": "Field_Specific"
}

# Default model configurations - Ingestion
DEFAULT_TEXT_MODEL = "intfloat/e5-large-v2"
DEFAULT_IMAGE_MODEL = "ViT-L-14"
DEFAULT_IMAGE_PRETRAINED = "openai"

# LLM Provider settings for QA
LLM_PROVIDER = "openai"
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.5
LLM_FREQUENCY_PENALTY = 0.2
LLM_MAX_TOKENS = 2000

# QA Engine parameters
QM_INTENT_CONFIDENCE_THRESHOLD = 0.7
FIELD_DETECTION_SCORE_CUTOFF = 85
TEXT_RETRIEVAL_TOP_K = 12
IMAGE_RETRIEVAL_TOP_K = 6

# Feature flags (new, default OFF for backward compatibility)
ENABLE_PROGRESSIVE_RELAX = False
ENABLE_CAPTION_FIRST = False
ENABLE_SEED_FACTS = False
STRUCTURED_LOGS = False

# Chunking parameters
DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 80

# Classification parameters
FUZZY_SCORE_CUTOFF = 80  # For 1-2 typo tolerance
