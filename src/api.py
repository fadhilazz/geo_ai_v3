"""FastAPI wrapper for QA engine."""

import logging
import os
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from .app_graph import get_qa_workflow
from .tools.field_detect import get_field_detector

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
        
        # Log the result
        logger.info(f"Generated response: {result['text_chunks_found']} text chunks, "
                   f"{result['figures_found']} figures, confidence: {result['confidence']}")
        
        return QuestionResponse(**result)
        
    except Exception as e:
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
        from .tools.qm import get_question_matrix
        
        stats = {}
        
        # Text collection stats
        try:
            text_rag = get_text_rag()
            text_collection = text_rag._get_collection()
            stats["text_chunks"] = text_collection.count()
        except Exception as e:
            stats["text_chunks"] = f"error: {e}"
            
        # Image collection stats
        try:
            image_rag = get_image_rag()
            image_collection = image_rag._get_collection()
            stats["image_figures"] = image_collection.count()
        except Exception as e:
            stats["image_figures"] = f"error: {e}"
            
        # Question Matrix stats
        try:
            qm = get_question_matrix()
            stats["question_patterns"] = len(qm.rows)
            stats["intent_categories"] = len(qm.by_intent)
        except Exception as e:
            stats["question_matrix"] = f"error: {e}"
            
        # Field stats
        try:
            field_detector = get_field_detector()
            stats["known_fields"] = len(field_detector.get_known_fields())
        except Exception as e:
            stats["known_fields"] = f"error: {e}"
            
        return {
            "knowledge_base": stats,
            "api_version": "1.0.0"
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
