"""Simplified QA workflow that works with Question Matrix, RAG, and Digital Twin."""

import logging
import time
from typing import Dict, List, Optional, Any
from langchain_openai import ChatOpenAI
import re

from .tools.qm import match_rows, get_union_strategy
from .tools.rag_text import get_text_rag
from .tools.rag_image import get_image_rag
from .tools.field_detect import detect_field, get_available_fields
from .prompts.system_prompt import build_user_prompt, get_system_prompt, get_fallback_prompt

logger = logging.getLogger(__name__)

def extract_temperature_data(chunks: List[Any]) -> Dict[str, Any]:
    """Extract temperature data from chunks for manifestation queries."""
    
    temperature_data = []
    
    for chunk in chunks:
        text = chunk.text
        # Look for temperature patterns
        temp_pattern = r'(\d+\.?\d*)\s*[o°]?C'
        matches = re.findall(temp_pattern, text)
        
        if matches:
            for match in matches:
                try:
                    temp_value = float(match)
                    # Only include reasonable temperature values (above 0°C and below 300°C)
                    if 0 < temp_value < 300:
                        # Try to find location context
                        location = "Unknown"
                        if "Dusun Baru" in text:
                            location = "Dusun Baru"
                        elif "Sungai Tutung" in text:
                            location = "Sungai Tutung"
                        elif "Pungut Mudik" in text:
                            location = "Pungut Mudik"
                        elif "Lubuk Larangan" in text:
                            location = "Lubuk Larangan"
                        elif "Mukai Pintu" in text:
                            location = "Mukai Pintu"
                        
                        # Try to find manifestation type
                        manifestation_type = "Unknown"
                        if "mata air panas" in text.lower():
                            manifestation_type = "Mata Air Panas"
                        elif "steam vent" in text.lower():
                            manifestation_type = "Steam Vent"
                        elif "sumur" in text.lower():
                            manifestation_type = "Sumur"
                        elif "sungai" in text.lower():
                            manifestation_type = "Sungai"
                        
                        temperature_data.append({
                            'temperature': temp_value,
                            'location': location,
                            'manifestation_type': manifestation_type,
                            'source': f"{chunk.filename} (page {chunk.page})",
                            'context': text[:100] + "..."
                        })
                except ValueError:
                    continue
    
    # Remove duplicates and sort by temperature
    unique_data = []
    seen = set()
    for data in temperature_data:
        key = (data['temperature'], data['location'], data['manifestation_type'])
        if key not in seen:
            seen.add(key)
            unique_data.append(data)
    
    unique_data.sort(key=lambda x: x['temperature'], reverse=True)
    
    return {
        'temperature_data': unique_data,
        'total_manifestations': len(unique_data),
        'highest_temperature': unique_data[0]['temperature'] if unique_data else None,
        'lowest_temperature': unique_data[-1]['temperature'] if unique_data else None
    }

def should_use_twin(question: str, intent_info: Dict) -> bool:
    """Determine if Digital Twin should be used based on question and intent."""
    
    # Check if QM explicitly requires twin
    if intent_info.get('requires_twin', False):
        return True
    
    # Check for specific model data keywords
    model_keywords = [
        'nilai resistivitas', 'resistivitas rendah', 'resistivitas tinggi',
        'g/cm3', 'densitas rendah', 'densitas tinggi',
        '3d model data', 'model data', 'grid data',
        'xyz', 'shapefile', 'geochemistry',
        'kedalaman', 'luas', 'area', 'estimasi', 'kalkulasi'
    ]
    
    # Check for structure-related keywords that need Digital Twin data
    structure_keywords = [
        'kontras densitas', 'batas struktur', 'struktur geologi', 'fault', 'sesar',
        'density contrast', 'structural boundary', 'geological structure'
    ]
    
    question_lower = question.lower()
    if any(keyword in question_lower for keyword in model_keywords):
        return True
    
    # Check for structure keywords that need Digital Twin
    if any(keyword in question_lower for keyword in structure_keywords):
        return True
    
    # Check for specific intents that require twin
    twin_intents = ['caprock_location', 'reservoir_rocktype', 'well_targeting', 'geological_structures', 'density_contrast']
    if intent_info.get('intent') in twin_intents:
        return True
    
    return False

def process_question(question: str, field: str = "Semurup") -> Dict:
    """Process a question using the improved framework that ALWAYS uses RAG and combines with Digital Twin when needed."""
    
    framework_debug = {
        'qm_mapping': {},
        'langgraph_decision': {},
        'routing_strategy': {},
        'evidence_sources': {},
        'strategy_validation': {}
    }
    
    try:
        # 1. Question Matrix Mapping
        from .tools.qm import load_qm, match_question
        qm = load_qm()
        intent_info = match_question(question, qm)
        
        framework_debug['qm_mapping'] = {
            'intent': intent_info.get('intent'),
            'requires_twin': intent_info.get('requires_twin', False),
            'confidence': intent_info.get('confidence', 0)
        }
        
        # 2. LangGraph Decision (ALWAYS use RAG, optionally add Digital Twin)
        use_twin = should_use_twin(question, intent_info)
        
        framework_debug['langgraph_decision'] = {
            'use_twin': use_twin,
            'use_rag': True,  # Always true
            'reasoning': f"RAG: Always used for knowledge base. Twin: QM requires twin: {intent_info.get('requires_twin', False)}, Model keywords: {any(keyword in question.lower() for keyword in ['nilai resistivitas', 'densitas', '3d model'])}"
        }
        
        # 3. ALWAYS get RAG data first (knowledge base is mandatory)
        from .tools.rag_text import get_text_rag
        from .config import CHROMA_TEXT_DIR_OBJ
        
        text_rag = get_text_rag(CHROMA_TEXT_DIR_OBJ)
        text_results = text_rag.search_with_fallback(question, None, top_k=12)
        
        # 4. Get Digital Twin data if needed
        twin_data = None
        if use_twin:
            try:
                from .twin.registry import TwinRegistry
                from .twin.summary import load_summary
                
                # Load twin summary
                twin_summary = load_summary(field)
                
                # Try live twin query for specific model data
                live_query_results = None
                if any(keyword in question.lower() for keyword in ['resistivitas', 'densitas', 'g/cm3', 'ohm.m', 'struktur', 'kontras', 'kedalaman', 'luas', 'area']):
                    try:
                        from .twin.metrics import density_analysis, resistivity_analysis
                        if 'densitas' in question.lower() or 'kontras' in question.lower():
                            live_query_results = density_analysis(field)
                        elif 'resistivitas' in question.lower():
                            live_query_results = resistivity_analysis(field)
                        # For area/volume calculations, use both analyses
                        elif any(keyword in question.lower() for keyword in ['luas', 'area', 'volume', 'estimasi']):
                            density_result = density_analysis(field)
                            resistivity_result = resistivity_analysis(field)
                            live_query_results = {
                                'density_analysis': density_result,
                                'resistivity_analysis': resistivity_result
                            }
                    except Exception as e:
                        logger.warning(f"Live twin query failed: {e}")
                
                twin_data = {
                    'summary': twin_summary,
                    'live_query': live_query_results
                }
            except Exception as e:
                logger.warning(f"Digital Twin data loading failed: {e}")
                twin_data = {'summary': None, 'live_query': None}
        
        # 5. Extract temperature data if this is a temperature-related query
        temperature_info = None
        if any(keyword in question.lower() for keyword in ['temperature', 'suhu', 'tertinggi', 'terendah', 'urutkan', 'panas', 'manifestasi']):
            temperature_info = extract_temperature_data(text_results)
        
        framework_debug['routing_strategy'] = {
            'rag_always_used': True,
            'twin_summary_used': twin_data is not None and twin_data['summary'] is not None,
            'twin_live_query_used': twin_data is not None and twin_data['live_query'] is not None,
            'temperature_data_extracted': temperature_info is not None,
            'combination_strategy': 'RAG + Twin' if use_twin else 'RAG only'
        }
        
        # 6. Evidence Sources
        framework_debug['evidence_sources'] = {
            'text_chunks': len(text_results),
            'figures': 0,  # Image search disabled due to embedding issues
            'citations': len(text_results),
            'twin_summary_available': twin_data is not None and twin_data['summary'] is not None,
            'twin_live_data_available': twin_data is not None and twin_data['live_query'] is not None
        }
        
        # 7. Build prompt with BOTH RAG and Twin data, or open-domain fallback if no evidence
        from .prompts.system_prompt import get_system_prompt, build_user_prompt, get_fallback_prompt
        
        system_prompt = get_system_prompt()
        
        evidence_available = (len(text_results) > 0) or (twin_data is not None and (twin_data.get('summary') is not None or twin_data.get('live_query') is not None))
        if evidence_available:
            # Combine both knowledge base and model data
            context_data = {
                'text_chunks': text_results,
                'twin_data': twin_data,
                'temperature_info': temperature_info,
                'knowledge_base_available': len(text_results) > 0,
                'model_data_available': twin_data is not None and (twin_data['summary'] is not None or twin_data['live_query'] is not None)
            }
            user_prompt = build_user_prompt(question, context_data)
        else:
            # Open-domain fallback when no specific evidence found
            user_prompt = get_fallback_prompt(question, intent_info)
        
        # 8. Get AI response
        from .tools.llm import get_llm_response
        response = get_llm_response(system_prompt, user_prompt)
        
        # 9. Strategy validation
        if not evidence_available:
            strategy_validation = "Open-domain fallback (no specific evidence)"
        elif use_twin and twin_data and (twin_data['summary'] or twin_data['live_query']):
            strategy_validation = "Combined RAG (knowledge) + Digital Twin (model data)"
        elif use_twin:
            strategy_validation = "RAG (knowledge) + Digital Twin attempted but failed"
        else:
            strategy_validation = "RAG only (knowledge base)"
        
        if temperature_info:
            strategy_validation += " + Temperature data extracted"
        
        framework_debug['strategy_validation'] = strategy_validation
        
        return {
            'answer': response,
            'framework_debug': framework_debug,
            'temperature_data': temperature_info,
            'data_sources': {
                'knowledge_base_chunks': len(text_results),
                'twin_summary_used': twin_data is not None and twin_data['summary'] is not None,
                'twin_live_data_used': twin_data is not None and twin_data['live_query'] is not None
            }
        }
        
    except Exception as e:
        logging.error(f"Error processing question: {e}")
        return {
            'answer': f"Error processing question: {str(e)}",
            'framework_debug': framework_debug,
            'temperature_data': None,
            'data_sources': {}
        }
