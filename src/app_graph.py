"""LangGraph workflow for QA engine."""

import logging
import time
from typing import Dict, List, Optional, Any
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from .tools.qm import match_rows, get_union_strategy
from .tools.rag_text import get_text_rag
from .tools.rag_image import get_image_rag
from .tools.field_detect import detect_field
from .twin.adapter import twin_summary, twin_query, should_use_twin_summary, clarify_needed
from .prompts.system_prompt import (
    QA_SYSTEM_PROMPT, CLARIFIER_PROMPT, CLARIFIER_RESPONSE_PROMPT,
    TWIN_SUMMARY_PROMPT, TWIN_QUERY_PROMPT
)

logger = logging.getLogger(__name__)

# State definition
class QAState:
    """State for QA workflow."""
    def __init__(self, question: str, field: Optional[str] = None):
        self.question = question
        self.field = field
        self.intent = None
        self.intent_confidence = 0.0
        self.filters = {}
        self.numeric_ctx = None
        self.clarification_needed = None
        self.clarification_options = []
        self.selected_aspect = None
        self.text_chunks = []
        self.figures = []
        self.answer = ""
        self.citations = []
        self.error = None
        self.plan_debug = {}

def get_qa_workflow(api_key: str):
    """Get the QA workflow."""
    
    # Initialize LLM with config-driven parameters
    llm = ChatOpenAI(
        model="gpt-4-turbo-preview",
        temperature=0.5,
        frequency_penalty=0.2,
        api_key=api_key
    )
    
    # Initialize RAG components
    text_rag = get_text_rag()
    image_rag = get_image_rag()
    
    # Define nodes
    def route_by_qm_node(state: QAState) -> Dict:
        """Route question using semantic Question Matrix matching."""
        try:
            from .tools.qm import match_rows, get_union_strategy
            from .config import get_qa_paths
            
            # Load Question Matrix
            qm_path = get_qa_paths()['question_matrix']
            qm_rows = load_qm(str(qm_path))
            
            # Get semantic matches
            matches = match_rows(state.question, k=5)
            
            if not matches:
                return {
                    "intent": "general_inquiry",
                    "intent_confidence": 0.0,
                    "filters": {},
                    "numeric_ctx": None,
                    "error": "No Question Matrix matches found"
                }
            
            # Build union strategy
            strategy = get_union_strategy(matches)
            
            # Check for low confidence (τ_low = 0.35)
            if strategy['confidence'] < 0.35:
                # Generate clarification options
                aspects = [
                    "Caprock", "Reservoir", "Hydrology", "Heat Source", 
                    "Recharge", "Wells/Targeting", "Drilling Risk"
                ]
                
                return {
                    "intent": strategy['intent_tag'],
                    "intent_confidence": strategy['confidence'],
                    "filters": strategy['filters'],
                    "numeric_ctx": None,
                    "clarification_needed": True,
                    "clarification_options": aspects
                }
            
            # Check if twin data is required
            numeric_ctx = None
            if strategy['requires_twin'] and state.field:
                try:
                    # Determine approach (summary vs live query)
                    if should_use_twin_summary(state.question, strategy['intent_tag']):
                        # Use twin summary for general questions
                        twin_data = twin_summary(state.field)
                        if "error" not in twin_data:
                            numeric_ctx = {
                                "type": "twin_summary",
                                "field": state.field,
                                "intent": strategy['intent_tag'],
                                "data": twin_data
                            }
                    else:
                        # Use live twin query for specific numeric questions
                        params = {}
                        if "resistivity" in state.question.lower() and "<" in state.question:
                            import re
                            match = re.search(r'< (\d+(?:\.\d+)?)', state.question)
                            if match:
                                params["res_threshold"] = float(match.group(1))
                        
                        twin_data = twin_query(state.field, strategy['intent_tag'], state.question, params)
                        if "error" not in twin_data:
                            numeric_ctx = {
                                "type": "twin_query",
                                "field": state.field,
                                "intent": strategy['intent_tag'],
                                "data": twin_data["metrics"],
                                "execution_time_ms": twin_data["execution_time_ms"]
                            }
                    
                    # Check for clarification needs
                    clarification = clarify_needed(state.question, strategy['intent_tag'])
                    if clarification:
                        return {
                            "intent": strategy['intent_tag'],
                            "intent_confidence": strategy['confidence'],
                            "filters": strategy['filters'],
                            "numeric_ctx": None,
                            "clarification_needed": clarification
                        }
                        
                except Exception as e:
                    logger.error(f"Error getting twin data: {e}")
            
            # Add field to filters if detected
            if state.field:
                strategy['filters']['field'] = state.field
            
            return {
                "intent": strategy['intent_tag'],
                "intent_confidence": strategy['confidence'],
                "filters": strategy['filters'],
                "numeric_ctx": numeric_ctx,
                "plan_debug": {
                    "top_matches": [(m[0]['intent_tag'], m[1]) for m in matches[:3]],
                    "union_strategy": strategy,
                    "field_detected": state.field
                }
            }
            
        except Exception as e:
            logger.error(f"QM routing error: {e}")
            return {
                "intent": "general_inquiry",
                "intent_confidence": 0.0,
                "filters": {},
                "numeric_ctx": None,
                "error": f"QM routing error: {e}"
            }
    
    def retrieve_text_node(state: QAState) -> Dict:
        """Retrieve relevant text chunks."""
        try:
            # Progressive relaxation if filters return 0 hits
            relax_level = 0
            max_relax = 3
            
            while relax_level <= max_relax:
                # Apply filters with relaxation
                filters = state.filters.copy()
                if relax_level > 0:
                    # Remove field filter first, then aspect filters
                    if relax_level == 1 and 'field' in filters:
                        del filters['field']
                    elif relax_level == 2 and 'aspect' in filters:
                        del filters['aspect']
                    elif relax_level == 3:
                        filters = {}  # No filters
                
                # Retrieve text chunks
                chunks = text_rag.retrieve(
                    state.question,
                    filters=filters,
                    k=12
                )
                
                if chunks:
                    return {
                        "text_chunks": chunks,
                        "plan_debug": {
                            **state.plan_debug,
                            "text_relax_level": relax_level,
                            "text_filters_applied": filters
                        }
                    }
                
                relax_level += 1
            
            # If still no chunks, return empty
            return {
                "text_chunks": [],
                "plan_debug": {
                    **state.plan_debug,
                    "text_relax_level": relax_level,
                    "text_filters_applied": {}
                }
            }
            
        except Exception as e:
            logger.error(f"Text retrieval error: {e}")
            return {
                "text_chunks": [],
                "error": f"Text retrieval error: {e}"
            }
    
    def retrieve_image_node(state: QAState) -> Dict:
        """Retrieve relevant images."""
        try:
            # Progressive relaxation for images too
            relax_level = 0
            max_relax = 3
            
            while relax_level <= max_relax:
                filters = state.filters.copy()
                if relax_level > 0:
                    if relax_level == 1 and 'field' in filters:
                        del filters['field']
                    elif relax_level == 2 and 'aspect' in filters:
                        del filters['aspect']
                    elif relax_level == 3:
                        filters = {}
                
                figures = image_rag.retrieve(
                    state.question,
                    filters=filters,
                    k=6
                )
                
                if figures:
                    return {
                        "figures": figures,
                        "plan_debug": {
                            **state.plan_debug,
                            "image_relax_level": relax_level,
                            "image_filters_applied": filters
                        }
                    }
                
                relax_level += 1
            
            return {
                "figures": [],
                "plan_debug": {
                    **state.plan_debug,
                    "image_relax_level": relax_level,
                    "image_filters_applied": {}
                }
            }
            
        except Exception as e:
            logger.error(f"Image retrieval error: {e}")
            return {
                "figures": [],
                "error": f"Image retrieval error: {e}"
            }
    
    def generate_answer_node(state: QAState) -> Dict:
        """Generate final answer."""
        try:
            # Build context
            context_parts = []
            
            # Add twin data if available
            if state.numeric_ctx:
                if state.numeric_ctx["type"] == "twin_summary":
                    context_parts.append(TWIN_SUMMARY_PROMPT.format(
                        summary_data=str(state.numeric_ctx["data"]),
                        question=state.question
                    ))
                elif state.numeric_ctx["type"] == "twin_query":
                    context_parts.append(TWIN_QUERY_PROMPT.format(
                        query_data=str(state.numeric_ctx["data"]),
                        question=state.question
                    ))
            
            # Add text chunks
            if state.text_chunks:
                context_parts.append(f"Text Evidence:\n" + "\n\n".join([
                    f"[{chunk.metadata.get('doc_id', 'unknown')}:{chunk.metadata.get('page', 'unknown')}] {chunk.page_content}"
                    for chunk in state.text_chunks
                ]))
            
            # Add images
            if state.figures:
                context_parts.append(f"Image Evidence:\n" + "\n".join([
                    f"[fig:{fig.metadata.get('doc_id', 'unknown')}:{fig.metadata.get('page', 'unknown')}] {fig.metadata.get('caption', 'No caption')}"
                    for fig in state.figures
                ]))
            
            # Build full prompt
            system_prompt = QA_SYSTEM_PROMPT
            if state.selected_aspect:
                system_prompt = CLARIFIER_RESPONSE_PROMPT.format(
                    selected_aspect=state.selected_aspect,
                    original_question=state.question
                )
            
            full_prompt = f"{system_prompt}\n\nQuestion: {state.question}\n\nContext:\n" + "\n\n".join(context_parts)
            
            # Generate answer
            response = llm.invoke([HumanMessage(content=full_prompt)])
            answer = response.content
            
            # Extract citations
            citations = []
            for chunk in state.text_chunks:
                doc_id = chunk.metadata.get('doc_id', 'unknown')
                page = chunk.metadata.get('page', 'unknown')
                citations.append(f"{doc_id}:{page}")
            
            for fig in state.figures:
                doc_id = fig.metadata.get('doc_id', 'unknown')
                page = fig.metadata.get('page', 'unknown')
                citations.append(f"fig:{doc_id}:{page}")
            
            return {
                "answer": answer,
                "citations": list(set(citations)),  # Dedupe
                "text_chunks_found": len(state.text_chunks),
                "figures_found": len(state.figures)
            }
            
        except Exception as e:
            logger.error(f"Answer generation error: {e}")
            return {
                "answer": f"Error generating answer: {e}",
                "citations": [],
                "text_chunks_found": 0,
                "figures_found": 0,
                "error": str(e)
            }
    
    def generate_clarifier_node(state: QAState) -> Dict:
        """Generate clarification question."""
        try:
            response = llm.invoke([
                HumanMessage(content=f"{CLARIFIER_PROMPT}\n\nUser question: {state.question}")
            ])
            
            return {
                "answer": response.content,
                "clarification_needed": True,
                "citations": [],
                "text_chunks_found": 0,
                "figures_found": 0
            }
            
        except Exception as e:
            logger.error(f"Clarifier generation error: {e}")
            return {
                "answer": f"Error generating clarifier: {e}",
                "clarification_needed": True,
                "citations": [],
                "text_chunks_found": 0,
                "figures_found": 0,
                "error": str(e)
            }
    
    # Build workflow
    workflow = StateGraph(QAState)
    
    # Add nodes
    workflow.add_node("route_by_qm", route_by_qm_node)
    workflow.add_node("retrieve_text", retrieve_text_node)
    workflow.add_node("retrieve_image", retrieve_image_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("generate_clarifier", generate_clarifier_node)
    
    # Define edges
    workflow.set_entry_point("route_by_qm")
    
    # Conditional routing based on clarification needs
    def route_after_qm(state: QAState) -> str:
        if state.clarification_needed:
            return "generate_clarifier"
        else:
            return "retrieve_text"
    
    workflow.add_conditional_edges("route_by_qm", route_after_qm)
    
    # Parallel retrieval
    workflow.add_edge("retrieve_text", "retrieve_image")
    workflow.add_edge("retrieve_image", "generate_answer")
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("generate_clarifier", END)
    
    return workflow.compile()

def load_qm(qm_path: str):
    """Load Question Matrix."""
    from .tools.qm import load_qm as load_qm_func
    return load_qm_func(qm_path)
