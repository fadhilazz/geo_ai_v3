"""Smoke tests for Digital Twin v2."""

import asyncio
import logging
import httpx
import pytest
from typing import Dict, List

logger = logging.getLogger(__name__)

API_BASE_URL = "http://127.0.0.1:8000"
TIMEOUT = 30

async def test_twin_summary_endpoint():
    """Test /twin/summary endpoint."""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        response = await client.get(f"{API_BASE_URL}/twin/summary?field=Semurup")
        assert response.status_code == 200
        
        data = response.json()
        assert "field" in data
        assert data["field"] == "Semurup"
        assert "caprock" in data
        assert "reservoir" in data
        assert "geochem" in data
        
        logger.info(f"✓ Twin summary endpoint working")

async def test_twin_query_endpoint():
    """Test /twin/query endpoint."""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        query_data = {
            "field": "Semurup",
            "intent_tag": "Caprock_Location",
            "params": {"res_threshold": 50}
        }
        
        response = await client.post(f"{API_BASE_URL}/twin/query", json=query_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "field" in data
        assert data["field"] == "Semurup"
        assert "metrics" in data
        assert "execution_time_ms" in data
        
        logger.info(f"✓ Twin query endpoint working")

async def test_ask_with_twin_integration():
    """Test /ask endpoint with twin integration."""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        questions = [
            "Where is the caprock in Semurup?",
            "What is the reservoir extent in Semurup?",
            "What is the likely reservoir rock type in Semurup?"
        ]
        
        for question in questions:
            response = await client.post(f"{API_BASE_URL}/ask", json={"question": question})
            assert response.status_code == 200
            
            data = response.json()
            assert "answer" in data
            assert "citations" in data
            assert len(data["answer"]) > 0
            
            logger.info(f"✓ Question answered: {question[:50]}...")

async def test_twin_context_in_ask():
    """Test that twin context is included in ask responses."""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        question = "Where is the caprock in Semurup?"
        response = await client.post(f"{API_BASE_URL}/ask", json={"question": question})
        
        assert response.status_code == 200
        data = response.json()
        
        # Check if twin context is mentioned in the answer
        answer_lower = data["answer"].lower()
        assert any(term in answer_lower for term in ["semurup", "caprock", "resistivity", "ohm"])
        
        logger.info(f"✓ Twin context found in answer")

async def test_twin_query_with_threshold():
    """Test twin query with specific resistivity threshold."""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        query_data = {
            "field": "Semurup",
            "intent_tag": "Caprock_Location",
            "params": {"res_threshold": 50}
        }
        
        response = await client.post(f"{API_BASE_URL}/twin/query", json=query_data)
        assert response.status_code == 200
        
        data = response.json()
        metrics = data["metrics"]
        
        # Should find caprock points with 50 ohm-m threshold
        assert metrics["points"] > 0
        assert "res_range" in metrics
        assert "depth_range" in metrics
        
        logger.info(f"✓ Found {metrics['points']} caprock points with 50 ohm-m threshold")

async def test_twin_registry_endpoints():
    """Test twin registry endpoints."""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        # Test available fields
        response = await client.get(f"{API_BASE_URL}/fields")
        assert response.status_code == 200
        
        data = response.json()
        assert "fields" in data
        assert "Semurup" in data["fields"]
        
        logger.info(f"✓ Available fields: {data['fields']}")

async def test_twin_performance():
    """Test twin query performance."""
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        query_data = {
            "field": "Semurup",
            "intent_tag": "Reservoir_Analysis",
            "params": {"res_low": 50, "res_high": 200}
        }
        
        response = await client.post(f"{API_BASE_URL}/twin/query", json=query_data)
        assert response.status_code == 200
        
        data = response.json()
        execution_time = data["execution_time_ms"]
        
        # Should complete within reasonable time
        assert execution_time < 5000  # 5 seconds
        
        logger.info(f"✓ Query completed in {execution_time:.1f} ms")

async def run_all_tests():
    """Run all twin smoke tests."""
    logger.info("Starting Digital Twin v2 smoke tests...")
    
    tests = [
        test_twin_summary_endpoint,
        test_twin_query_endpoint,
        test_ask_with_twin_integration,
        test_twin_context_in_ask,
        test_twin_query_with_threshold,
        test_twin_registry_endpoints,
        test_twin_performance
    ]
    
    for test in tests:
        try:
            await test()
        except Exception as e:
            logger.error(f"Test {test.__name__} failed: {e}")
            raise
    
    logger.info("✓ All Digital Twin v2 smoke tests passed!")

if __name__ == "__main__":
    asyncio.run(run_all_tests())
