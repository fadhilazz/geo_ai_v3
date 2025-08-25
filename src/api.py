"""FastAPI wrapper for QA engine."""

import logging
import os
import json
import time
from typing import Dict, List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from .app_graph import get_qa_workflow
from .tools.field_detect import get_field_detector

try:
    from .config import STRUCTURED_LOGS, STAMP_PATH_OBJ, FACTS_DIR_OBJ
except ImportError:
    from src.config import STRUCTURED_LOGS, STAMP_PATH_OBJ, FACTS_DIR_OBJ

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = FastAPI(
    title="Geothermal QA Engine",
    description="AI-powered question answering for geothermal energy literature",
    version="1.0.0"
)

# Enable CORS for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    """Request model for asking questions."""
    question: str
    field: Optional[str] = None


class QuestionResponse(BaseModel):
    """Response model for question answers."""
    answer: str
    citations: List[str]
    figures: List[Dict]
    field: Optional[str]
    intent: Optional[str]
    confidence: str
    text_chunks_found: int
    figures_found: int
    error: Optional[str] = None


class FieldsResponse(BaseModel):
    """Response model for available fields."""
    fields: List[str]


def get_api_key():
    """Get OpenAI API key from environment."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OpenAI API key not configured. Please set OPENAI_API_KEY environment variable."
        )
    return api_key


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Geothermal QA Engine API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "ask": "POST /ask - Ask a question",
            "fields": "GET /fields - Get available fields"
        }
    }


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest, api_key: str = Depends(get_api_key)):
    """Ask a question to the QA engine.
    
    Args:
        request: Question request with question and optional field
        api_key: OpenAI API key from dependency
        
    Returns:
        Question response with answer and metadata
    """
    start_time = time.time()
    
    try:
        logger.info(f"Received question: '{request.question}' (field: {request.field})")
        
        # Get QA workflow
        workflow = get_qa_workflow(api_key)
        
        # If no field provided, try to detect it
        field = request.field
        if not field:
            field_detector = get_field_detector()
            detected_field, confidence = field_detector.detect_field(request.question)
            if detected_field and confidence >= 85:  # High confidence threshold
                field = detected_field
                logger.info(f"Auto-detected field: '{field}' (confidence: {confidence})")
            else:
                logger.info("No field detected or low confidence, using general mode")
        
        # Run the workflow
        result = workflow.run(request.question, field)
        
        # Calculate latency
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Structured logging if enabled
        if STRUCTURED_LOGS:
            log_data = {
                "timestamp": time.time(),
                "question": request.question,
                "field": field,
                "intent": result.get('intent'),
                "confidence": result.get('confidence'),
                "text_chunks_found": result.get('text_chunks_found', 0),
                "figures_found": result.get('figures_found', 0),
                "latency_ms": latency_ms,
                "citations_count": len(result.get('citations', [])),
                "error": None
            }
            logger.info(f"STRUCTURED_LOG: {json.dumps(log_data)}")
        
        # Log the result
        logger.info(f"Generated response: {result['text_chunks_found']} text chunks, "
                   f"{result['figures_found']} figures, confidence: {result['confidence']}, "
                   f"latency: {latency_ms}ms")
        
        return QuestionResponse(**result)
        
    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Structured error logging if enabled
        if STRUCTURED_LOGS:
            log_data = {
                "timestamp": time.time(),
                "question": request.question,
                "field": request.field,
                "error": str(e),
                "latency_ms": latency_ms
            }
            logger.error(f"STRUCTURED_ERROR: {json.dumps(log_data)}")
        
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {e}")


@app.get("/fields", response_model=FieldsResponse)
async def get_fields():
    """Get list of available fields from the knowledge base.
    
    Returns:
        List of available field names
    """
    try:
        field_detector = get_field_detector()
        fields = field_detector.get_known_fields()
        
        logger.info(f"Retrieved {len(fields)} available fields")
        return FieldsResponse(fields=fields)
        
    except Exception as e:
        logger.error(f"Error getting fields: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting fields: {e}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Basic health checks
        checks = {
            "api": "ok",
            "openai_key": "configured" if os.getenv("OPENAI_API_KEY") else "missing"
        }
        
        # Try to load field detector (tests database connections)
        try:
            field_detector = get_field_detector()
            field_count = len(field_detector.get_known_fields())
            checks["database"] = f"ok ({field_count} fields)"
        except Exception as e:
            checks["database"] = f"error: {e}"
            
        # Determine overall status
        all_ok = all(
            status == "ok" or status.startswith("ok (") or status == "configured"
            for status in checks.values()
        )
        
        status_code = 200 if all_ok else 503
        
        return {
            "status": "healthy" if all_ok else "unhealthy",
            "checks": checks
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@app.get("/stats")
async def get_stats():
    """Get knowledge base statistics."""
    try:
        from .tools.rag_text import get_text_rag
        from .tools.rag_image import get_image_rag
        from .tools.field_detect import get_available_fields
        from .tools.seed_facts import load_facts
        import chromadb
        
        stats = {}
        
        # Text collection stats
        try:
            text_rag = get_text_rag()
            text_collection = text_rag._get_collection()
            stats["text_chunks"] = text_collection.count()
            
            # Get per-field counts
            results = text_collection.get(include=['metadatas'])
            field_counts = {}
            for metadata in results['metadatas']:
                if metadata and 'field' in metadata:
                    field = metadata['field']
                    if field and field != 'NONE':
                        field_counts[field] = field_counts.get(field, 0) + 1
            stats["field_counts"] = field_counts
            
        except Exception as e:
            stats["text_chunks"] = f"error: {e}"
            
        # Image collection stats
        try:
            image_rag = get_image_rag()
            image_collection = image_rag._get_collection()
            stats["image_figures"] = image_collection.count()
        except Exception as e:
            stats["image_figures"] = f"error: {e}"
            
        # Last ingest time
        try:
            if STAMP_PATH_OBJ.exists():
                stats["last_ingest_time"] = STAMP_PATH_OBJ.stat().st_mtime
            else:
                stats["last_ingest_time"] = "unknown"
        except Exception as e:
            stats["last_ingest_time"] = f"error: {e}"
            
        # Top aspects (from text metadata)
        try:
            results = text_collection.get(include=['metadatas'])
            aspect_counts = {}
            for metadata in results['metadatas']:
                if metadata and 'aspect' in metadata:
                    aspects = metadata['aspect']
                    if isinstance(aspects, str):
                        # Parse string representation
                        aspects = aspects.strip("[]'").split(',')
                    if isinstance(aspects, list):
                        for aspect in aspects:
                            aspect = aspect.strip().strip("'\"")
                            if aspect:
                                aspect_counts[aspect] = aspect_counts.get(aspect, 0) + 1
            
            # Get top 5 aspects
            top_aspects = sorted(aspect_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            stats["top_aspects"] = dict(top_aspects)
            
        except Exception as e:
            stats["top_aspects"] = f"error: {e}"
            
        # Available fields
        try:
            available_fields = get_available_fields(str(text_rag.chroma_dir))
            stats["available_fields"] = list(available_fields)
        except Exception as e:
            stats["available_fields"] = f"error: {e}"
            
        return {
            "knowledge_base": stats,
            "api_version": "1.0.0",
            "feature_flags": {
                "progressive_relax": os.getenv("ENABLE_PROGRESSIVE_RELAX", "False"),
                "caption_first": os.getenv("ENABLE_CAPTION_FIRST", "False"),
                "seed_facts": os.getenv("ENABLE_SEED_FACTS", "False"),
                "structured_logs": os.getenv("STRUCTURED_LOGS", "False")
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting stats: {e}")


if __name__ == "__main__":
    # For development - use uvicorn command for production
    uvicorn.run(
        "src.api:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
