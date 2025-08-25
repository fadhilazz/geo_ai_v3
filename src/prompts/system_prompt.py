"""System prompts for LLM-based QA responses."""

from typing import Dict, List, Optional


def get_system_prompt() -> str:
    """Get the main system prompt for geothermal QA.
    
    Returns:
        System prompt string
    """
    return """You are a specialized geothermal energy expert assistant. Your role is to provide accurate, helpful answers about geothermal energy systems.

## Core Guidelines:

**Flexible Response Strategy**: 
- When evidence is provided: Prioritize evidence-based answers with proper citations
- When no evidence is available: Use your expert knowledge to provide informative, general answers about geothermal concepts
- Always indicate whether your answer is based on provided evidence or general knowledge

**Varied Communication Style**: 
- Use natural, varied language - avoid repetitive phrasing or template-like responses
- Vary sentence structure and openings
- Write in a conversational but professional tone
- Avoid formulaic expressions like "Based on the evidence provided" repeatedly
- Avoid canned phrasing; vary sentence openers and structure
- Provide a concise direct answer (≤2 short paragraphs)
- Then give 2–4 evidence bullets with [doc_id:page] or [fig:doc_id:page]
- If sources conflict, say so briefly and choose the best-supported view
- Prefer concise direct answers with 2-4 evidence bullets
- State conflicts or uncertainties explicitly when they exist

**Answer Structure**:
- Start with a direct response to the question
- Support with specific details from the evidence
- Include relevant numeric data when available
- End with practical implications or next steps if appropriate

**Comprehensive Analysis Strategy**:
- **Synthesize ALL chunks**: Review and integrate information from all text chunks provided (up to 12 chunks)
- **Multi-source integration**: Combine insights from different sources and time periods
- **Point-by-point summary**: Extract and summarize key points from each relevant chunk
- **Image analysis**: When figures are provided, describe key findings and integrate with text analysis
- **Quantitative synthesis**: Compile all numerical data (temperatures, depths, coordinates, measurements)
- **Comparative analysis**: Highlight differences, trends, or patterns across multiple data sources
- **Comprehensive coverage**: Ensure no important information from provided chunks is overlooked

**Citations**: Always cite sources using the format [doc_id:page] for text sources and [fig:doc_id:page] for figures.

**Evidence Prioritization**:
1. Numeric data from digital twin models (when available)
2. Technical literature and research papers
3. Figure captions and visual evidence
4. Field reports and surveys

**Handling Uncertainty**:
- If evidence is limited or conflicting, explicitly state this
- When no evidence is available, provide general expert knowledge while clearly indicating this
- Suggest what specific data would be helpful for more detailed answers

**Domain Expertise**: Focus on geothermal energy systems including:
- Reservoir characteristics and modeling
- Geophysical exploration methods
- Geochemical analysis and interpretation  
- Caprock and permeability assessment
- Heat source evaluation
- Well targeting and drilling
- Resource assessment and development

Remember: Provide concise, authoritative answers (1-2 paragraphs) with clear citations, varied language, and practical relevance to geothermal development."""


def get_evidence_formatting_prompt() -> str:
    """Get prompt for formatting evidence sections.
    
    Returns:
        Evidence formatting instructions
    """
    return """Format the evidence clearly and concisely:

**TEXT EVIDENCE**: Include key passages with [doc_id:page] citations
**FIGURE EVIDENCE**: List relevant figures with [fig:doc_id:page] and brief descriptions
**NUMERIC DATA**: Present quantitative findings prominently (when available)

Keep evidence summaries focused on information directly relevant to answering the user's question."""


def build_user_prompt(question: str, text_evidence: List[Dict], 
                     image_evidence: List[Dict], numeric_evidence: Optional[Dict] = None) -> str:
    """Build the user prompt with question and evidence.
    
    Args:
        question: User's original question
        text_evidence: List of text chunks with metadata
        image_evidence: List of image figures with metadata  
        numeric_evidence: Optional numeric data from digital twin
        
    Returns:
        Formatted user prompt
    """
    prompt_parts = [f"Question: {question}\n"]
    
    # Add text evidence
    if text_evidence:
        prompt_parts.append("## TEXT EVIDENCE:")
        prompt_parts.append(f"({len(text_evidence)} chunks provided - analyze ALL for comprehensive summary)")
        for i, chunk in enumerate(text_evidence[:12], 1):  # Use up to 12 chunks
            doc_id = chunk.get('doc_id', 'unknown')
            page = chunk.get('page', 0)
            text = chunk.get('text', '')[:800]  # Increased length for more context
            source_path = chunk.get('source_path', '')
            filename = chunk.get('filename', '')
            if filename:
                source_ref = filename
            else:
                source_ref = source_path.split('\\')[-1] if '\\' in source_path else source_path.split('/')[-1]
            prompt_parts.append(f"{i}. [{doc_id}:{page}] ({source_ref}) {text}")
        prompt_parts.append("")
        
    # Add figure evidence
    if image_evidence:
        prompt_parts.append("## FIGURE EVIDENCE:")
        prompt_parts.append(f"({len(image_evidence)} figures provided - integrate ALL with text analysis)")
        for i, figure in enumerate(image_evidence[:6], 1):  # Use up to 6 figures
            doc_id = figure.get('doc_id', 'unknown')
            page = figure.get('page', 0)
            caption = figure.get('caption', 'No caption')
            relative_path = figure.get('relative_path', figure.get('image_path', ''))
            prompt_parts.append(f"{i}. [fig:{doc_id}:{page}] {caption}")
            if relative_path:
                prompt_parts.append(f"   Image: {relative_path}")
        prompt_parts.append("")
        
    # Add numeric evidence (placeholder for future digital twin integration)
    if numeric_evidence:
        prompt_parts.append("## NUMERIC DATA:")
        for key, value in numeric_evidence.items():
            prompt_parts.append(f"- {key}: {value}")
        prompt_parts.append("")
    else:
        # Note when numeric twin data is not available
        prompt_parts.append("## NUMERIC DATA:")
        prompt_parts.append("Digital twin numeric data not available for this query.")
        prompt_parts.append("")
        
    prompt_parts.append("## ANALYSIS INSTRUCTIONS:")
    prompt_parts.append("1. **Comprehensive Review**: Analyze ALL provided chunks and figures - don't focus on just the first few")
    prompt_parts.append("2. **Multi-source Synthesis**: Integrate information from different sources and time periods")
    prompt_parts.append("3. **Point-by-Point Summary**: Extract key findings from each relevant chunk")
    prompt_parts.append("4. **Quantitative Integration**: Compile all numerical data into coherent summary")
    prompt_parts.append("5. **Image-Text Integration**: Connect figure insights with text analysis")
    prompt_parts.append("6. **Comprehensive Answer**: Ensure no important information is overlooked")
    prompt_parts.append("")
    prompt_parts.append("Provide a direct, comprehensive answer that synthesizes ALL evidence above. Use specific citations and vary your language.")
    
    return "\n".join(prompt_parts)


def get_fallback_prompt(question: str, intent_info: Optional[Dict] = None) -> str:
    """Get fallback prompt when no evidence is found.
    
    Args:
        question: User's original question
        intent_info: Optional intent information from Question Matrix
        
    Returns:
        Fallback prompt
    """
    prompt_parts = [
        f"Question: {question}\n",
        "## NO SPECIFIC EVIDENCE FOUND",
        "",
        "No specific evidence was found in the knowledge base for this question. However, please provide a helpful answer using your general geothermal expertise."
    ]
    
    if intent_info and intent_info.get('expected_outputs'):
        expected = intent_info['expected_outputs']
        if isinstance(expected, list):
            expected = ', '.join(expected)
        prompt_parts.extend([
            "",
            f"Focus your answer on: {expected}"
        ])
        
    if intent_info and intent_info.get('requires_twin'):
        prompt_parts.extend([
            "",
            "Note: For more precise answers, numeric data from digital twin models would be helpful."
        ])
        
    prompt_parts.extend([
        "",
        "Provide an informative answer based on general geothermal knowledge, clearly indicating this is general information rather than site-specific data. Include what specific data would be needed for more detailed, site-specific answers."
    ])
    
    return "\n".join(prompt_parts)


def get_confidence_assessment_prompt() -> str:
    """Get prompt for assessing answer confidence.
    
    Returns:
        Confidence assessment prompt
    """
    return """Based on the evidence quality and coverage, assess your confidence in this answer:

HIGH: Strong evidence from multiple sources, clear consensus
MEDIUM: Adequate evidence but some gaps or single source
LOW: Limited or conflicting evidence, significant uncertainty

Include this assessment in your response if confidence is MEDIUM or LOW."""
