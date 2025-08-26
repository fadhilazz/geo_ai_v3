"""LLM tools for AI responses."""

import logging
import os
from typing import Optional
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

def get_llm_response(system_prompt: str, user_prompt: str, api_key: Optional[str] = None) -> str:
    """Get response from LLM using system and user prompts."""
    
    try:
        # Get API key
        if not api_key:
            api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            raise ValueError("No OpenAI API key provided or found in environment")
        
        # Initialize LLM
        llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.5,
            frequency_penalty=0.2,
            api_key=api_key
        )
        
        # Create messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Get response
        response = llm.invoke(messages)
        return response.content
        
    except Exception as e:
        logger.error(f"Error getting LLM response: {e}")
        return f"Error generating response: {str(e)}"
