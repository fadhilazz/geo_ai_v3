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
from .tools.field_detect import get_available_fields, detect_field
from .twin.adapter import twin_summary, twin_query
from .twin.registry import list_available_fields as list_twin_fields

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
    needs_clarification: bool = False
    clarification_options: List[str] = []
    plan_debug: Optional[Dict] = None
    error: Optional[str] = None


class FieldsResponse(BaseModel):
    """Response model for available fields."""
    fields: List[str]


class TwinSummaryResponse(BaseModel):
    """Response model for twin summary."""
    field: str
    summary: Dict
    error: Optional[str] = None


class TwinQueryRequest(BaseModel):
    """Request model for twin queries."""
    field: str
    intent_tag: str
    params: Optional[Dict] = None


class TwinQueryResponse(BaseModel):
    """Response model for twin queries."""
    field: str
    intent_tag: str
    metrics: Dict
    execution_time_ms: float
    cache_hit: bool = False
    error: Optional[str] = None


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
            "fields": "GET /fields - Get available fields",
            "twin_summary": "GET /twin/summary?field=<field> - Get twin summary",
            "twin_query": "POST /twin/query - Execute twin query",
            "twin_fields": "GET /twin/fields - Get available twin fields"
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
        
        # Detect field if not provided
        field = request.field
        if not field:
            # Get available fields from Chroma
            from .tools.rag_text import get_text_rag
            text_rag = get_text_rag()
            available_fields = get_available_fields(str(text_rag.chroma_dir))
            
            # Detect field using fuzzy matching
            detected_field, confidence = detect_field(request.question, available_fields)
            if detected_field and confidence >= 85:  # High confidence threshold
                field = detected_field
                logger.info(f"Auto-detected field: '{field}' (confidence: {confidence})")
            else:
                logger.info("No field detected or low confidence, using general mode")
        
        # Run the workflow with new state-based approach
        from .app_graph import QAState
        
        # Initialize state
        initial_state = QAState(request.question, field)
        
        # Run workflow
        final_state = workflow.invoke(initial_state)
        
        # Calculate latency
        latency_ms = int((time.time() - start_time) * 1000)
        
        # Check for clarification needs
        if hasattr(final_state, 'clarification_needed') and final_state.clarification_needed:
            return QuestionResponse(
                answer=final_state.answer,
                citations=final_state.citations,
                figures=[],
                field=field,
                intent=final_state.intent,
                confidence=f"{final_state.intent_confidence:.3f}",
                text_chunks_found=0,
                figures_found=0,
                needs_clarification=True,
                clarification_options=getattr(final_state, 'clarification_options', []),
                plan_debug=getattr(final_state, 'plan_debug', {})
            )
        
        # Structured logging if enabled
        if STRUCTURED_LOGS:
            log_data = {
                "timestamp": time.time(),
                "question": request.question,
                "field": field,
                "intent": final_state.intent,
                "confidence": final_state.intent_confidence,
                "text_chunks_found": len(final_state.text_chunks),
                "figures_found": len(final_state.figures),
                "latency_ms": latency_ms,
                "citations_count": len(final_state.citations),
                "error": None
            }
            logger.info(f"STRUCTURED_LOG: {json.dumps(log_data)}")
        
        # Log the result
        logger.info(f"Generated response: {len(final_state.text_chunks)} text chunks, "
                   f"{len(final_state.figures)} figures, confidence: {final_state.intent_confidence:.3f}, "
                   f"latency: {latency_ms}ms")
        
        return QuestionResponse(
            answer=final_state.answer,
            citations=final_state.citations,
            figures=[],  # Convert figures to dict format if needed
            field=field,
            intent=final_state.intent,
            confidence=f"{final_state.intent_confidence:.3f}",
            text_chunks_found=len(final_state.text_chunks),
            figures_found=len(final_state.figures),
            needs_clarification=False,
            clarification_options=[],
            plan_debug=getattr(final_state, 'plan_debug', {})
        )
        
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
            logger.error(f"STRUCTURED_LOG: {json.dumps(log_data)}")
        
        logger.error(f"Error processing question: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )


@app.get("/fields", response_model=FieldsResponse)
async def get_fields():
    """Get available geothermal fields."""
    try:
        fields = get_available_fields()
        return FieldsResponse(fields=fields)
    except Exception as e:
        logger.error(f"Error getting fields: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting fields: {e}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}


@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    try:
        stats = {}
        
        # Text RAG stats
        try:
            from .tools.rag_text import get_text_rag
            text_rag = get_text_rag()
            text_collection = text_rag._get_collection()
            stats["text_chunks"] = text_collection.count()
        except Exception as e:
            stats["text_chunks"] = f"Error: {e}"
        
        # Image RAG stats
        try:
            from .tools.rag_image import get_image_rag
            image_rag = get_image_rag()
            image_collection = image_rag._get_collection()
            stats["image_figures"] = image_collection.count()
        except Exception as e:
            stats["image_figures"] = f"Error: {e}"
        
        # Question Matrix stats
        try:
            from .config import get_qa_paths
            from .tools.qm import load_qm
            qm_path = get_qa_paths()['question_matrix']
            qm_rows = load_qm(str(qm_path))
            stats["question_matrix_rows"] = len(qm_rows)
        except Exception as e:
            stats["question_matrix_rows"] = f"Error: {e}"
        
        # Twin stats
        try:
            twin_fields = list_twin_fields()
            stats["twin_fields"] = len(twin_fields)
        except Exception as e:
            stats["twin_fields"] = f"Error: {e}"
        
        # System info
        stats["timestamp"] = time.time()
        stats["version"] = "1.0.0"
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting stats: {e}")


@app.get("/twin/summary", response_model=TwinSummaryResponse)
async def get_twin_summary(field: str):
    """Get twin summary for a field."""
    try:
        summary = twin_summary(field)
        if "error" in summary:
            return TwinSummaryResponse(field=field, error=summary["error"])
        return TwinSummaryResponse(field=field, summary=summary)
    except Exception as e:
        logger.error(f"Error getting twin summary for {field}: {e}")
        return TwinSummaryResponse(field=field, error=f"Error getting twin summary: {str(e)}")


@app.post("/twin/query", response_model=TwinQueryResponse)
async def execute_twin_query(request: TwinQueryRequest):
    """Execute a twin query."""
    try:
        result = twin_query(request.field, request.intent_tag, "", request.params)
        if "error" in result:
            return TwinQueryResponse(
                field=request.field,
                intent_tag=request.intent_tag,
                metrics={},
                execution_time_ms=0,
                error=result["error"]
            )
        return TwinQueryResponse(
            field=result["field"],
            intent_tag=result["intent_tag"],
            metrics=result["metrics"],
            execution_time_ms=result["execution_time_ms"],
            cache_hit=result.get("cache_hit", False)
        )
    except Exception as e:
        logger.error(f"Error executing twin query: {e}")
        return TwinQueryResponse(
            field=request.field,
            intent_tag=request.intent_tag,
            metrics={},
            execution_time_ms=0,
            error=f"Error executing twin query: {str(e)}"
        )


@app.get("/twin/fields", response_model=FieldsResponse)
async def get_twin_fields():
    """Get available twin fields."""
    try:
        fields = list_twin_fields()
        return FieldsResponse(fields=fields)
    except Exception as e:
        logger.error(f"Error getting twin fields: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting twin fields: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
