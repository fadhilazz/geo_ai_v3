"""Smoke tests for QA engine with Question Matrix integration."""

import asyncio
import json
import logging
import time
from typing import Dict, List

import httpx
import pytest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration
API_BASE_URL = "http://127.0.0.1:8000"
TIMEOUT = 30

# Test questions covering different intents
TEST_QUESTIONS = [
    # Temperature/reservoir questions
    {
        "question": "What is the reservoir temperature of Semurup?",
        "field": "Semurup",
        "expected_intent": "temperature_inquiry",
        "requires_twin": False
    },
    {
        "question": "Dimana lokasi manifestasi dengan temperature tertinggi di Semurup?",
        "field": "Semurup", 
        "expected_intent": "temperature_inquiry",
        "requires_twin": False
    },
    # Location questions
    {
        "question": "Where is the Semurup geothermal field located?",
        "field": "Semurup",
        "expected_intent": "location_inquiry", 
        "requires_twin": False
    },
    # Geology questions
    {
        "question": "What is the geology of Semurup area?",
        "field": "Semurup",
        "expected_intent": "geology_inquiry",
        "requires_twin": False
    },
    # General questions (no field)
    {
        "question": "How does geothermal energy work?",
        "field": None,
        "expected_intent": "general_inquiry",
        "requires_twin": False
    },
    # Questions that might require twin data
    {
        "question": "What is the current production capacity of Semurup?",
        "field": "Semurup",
        "expected_intent": "production_inquiry",
        "requires_twin": True
    }
]

async def test_api_health():
    """Test API health endpoint."""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.get(f"{API_BASE_URL}/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        logger.info("✅ Health check passed")

async def test_api_stats():
    """Test API stats endpoint."""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.get(f"{API_BASE_URL}/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "knowledge_base" in data
        assert "text_chunks" in data["knowledge_base"]
        assert "image_figures" in data["knowledge_base"]
        
        logger.info(f"✅ Stats endpoint: {data['knowledge_base']['text_chunks']} text chunks, "
                   f"{data['knowledge_base']['image_figures']} figures")

async def test_question_with_field(question_data: Dict):
    """Test a specific question with field."""
    question = question_data["question"]
    field = question_data["field"]
    expected_intent = question_data["expected_intent"]
    requires_twin = question_data["requires_twin"]
    
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        payload = {
            "question": question,
            "field": field
        }
        
        logger.info(f"Testing: '{question}' (field: {field})")
        
        response = await client.post(
            f"{API_BASE_URL}/ask",
            json=payload,
            timeout=TIMEOUT
        )
        
        assert response.status_code == 200, f"Request failed: {response.text}"
        
        data = response.json()
        
        # Basic response validation
        assert "answer" in data
        assert "citations" in data
        assert "text_chunks_found" in data
        assert "figures_found" in data
        
        # Content validation
        assert len(data["answer"]) > 0, "Answer should not be empty"
        assert data["text_chunks_found"] >= 0, "Should have non-negative text chunks"
        assert data["figures_found"] >= 0, "Should have non-negative figures"
        
        # Intent validation (if available)
        if "intent" in data and data["intent"]:
            logger.info(f"Detected intent: {data['intent']} (expected: {expected_intent})")
        
        # Twin data handling
        if requires_twin:
            # Check if answer mentions missing twin data
            answer_lower = data["answer"].lower()
            twin_indicators = ["twin", "production", "capacity", "current", "real-time"]
            has_twin_content = any(indicator in answer_lower for indicator in twin_indicators)
            
            if not has_twin_content:
                logger.warning(f"Question may need twin data but answer doesn't mention it: {question}")
        
        # Citations validation
        assert len(data["citations"]) >= 1, f"Should have at least 1 citation for: {question}"
        
        logger.info(f"✅ Question passed: {len(data['citations'])} citations, "
                   f"{data['text_chunks_found']} chunks, {data['figures_found']} figures")
        
        return data

async def test_field_detection():
    """Test automatic field detection."""
    question = "What is the temperature of Semurup reservoir?"
    
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        payload = {
            "question": question,
            "field": None  # Let API detect field
        }
        
        logger.info(f"Testing field detection: '{question}'")
        
        response = await client.post(
            f"{API_BASE_URL}/ask",
            json=payload,
            timeout=TIMEOUT
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Should still work even without explicit field
        assert len(data["answer"]) > 0
        assert data["text_chunks_found"] >= 0
        
        logger.info(f"✅ Field detection passed: detected field = {data.get('field', 'None')}")

async def test_general_question():
    """Test general question without specific field."""
    question = "How does geothermal energy generation work?"
    
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        payload = {
            "question": question,
            "field": None
        }
        
        logger.info(f"Testing general question: '{question}'")
        
        response = await client.post(
            f"{API_BASE_URL}/ask",
            json=payload,
            timeout=TIMEOUT
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Should provide general knowledge answer
        assert len(data["answer"]) > 0
        assert "geothermal" in data["answer"].lower() or "energy" in data["answer"].lower()
        
        logger.info("✅ General question passed")

async def run_all_tests():
    """Run all smoke tests."""
    logger.info("🚀 Starting QA Engine smoke tests...")
    
    # Test API health
    await test_api_health()
    
    # Test stats endpoint
    await test_api_stats()
    
    # Test field detection
    await test_field_detection()
    
    # Test general question
    await test_general_question()
    
    # Test specific questions
    for i, question_data in enumerate(TEST_QUESTIONS):
        logger.info(f"\n--- Test {i+1}/{len(TEST_QUESTIONS)} ---")
        await test_question_with_field(question_data)
        await asyncio.sleep(1)  # Brief pause between requests
    
    logger.info("\n🎉 All smoke tests passed!")

if __name__ == "__main__":
    # Run tests
    asyncio.run(run_all_tests())
