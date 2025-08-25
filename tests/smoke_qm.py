"""Smoke tests for QA engine acceptance testing."""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import httpx
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.app_graph import get_qa_workflow

logger = logging.getLogger(__name__)


# Test questions covering different aspects
TEST_QUESTIONS = [
    {
        "question": "Where is the caprock and how thick is it?",
        "field": "Semurup",
        "expected_aspects": ["caprock", "thickness"],
        "requires_evidence": True
    },
    {
        "question": "What is the reservoir temperature based on geothermometers?",
        "field": "Semurup", 
        "expected_aspects": ["reservoir", "temperature", "geothermometer"],
        "requires_evidence": True
    },
    {
        "question": "How does MT data show the subsurface structure?",
        "field": "Semurup",
        "expected_aspects": ["mt", "magnetotelluric", "structure"],
        "requires_evidence": True
    },
    {
        "question": "What does the hydrology tell us about the geothermal system?",
        "field": "Semurup",
        "expected_aspects": ["hydrology", "water", "system"],
        "requires_evidence": True
    },
    {
        "question": "Where should we target wells for development?",
        "field": "Semurup",
        "expected_aspects": ["wells", "targeting", "development"],
        "requires_evidence": True
    },
    {
        "question": "What is the heat source for this geothermal field?",
        "field": "Semurup",
        "expected_aspects": ["heat", "source"],
        "requires_evidence": True
    },
    {
        "question": "Explain typical caprock lithologies in Indonesian volcanic settings.",
        "field": None,  # General question, no specific field
        "expected_aspects": ["caprock", "lithology", "volcanic"],
        "requires_evidence": True
    },
    {
        "question": "How do geothermometers work in volcanic geothermal systems?",
        "field": None,  # General question
        "expected_aspects": ["geothermometer", "volcanic", "geothermal"],
        "requires_evidence": True
    }
]


class QASmokeTest:
    """Smoke test suite for QA engine."""
    
    def __init__(self, api_key: str = None, base_url: str = "http://127.0.0.1:8000"):
        """Initialize smoke test.
        
        Args:
            api_key: OpenAI API key
            base_url: Base URL for API testing
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        self.results = []
        
    async def test_api_endpoints(self) -> bool:
        """Test API endpoints are accessible.
        
        Returns:
            True if all endpoints are accessible
        """
        print("\n=== Testing API Endpoints ===")
        
        try:
            async with httpx.AsyncClient() as client:
                # Test root endpoint
                response = await client.get(f"{self.base_url}/")
                assert response.status_code == 200
                print("✓ Root endpoint accessible")
                
                # Test health endpoint
                response = await client.get(f"{self.base_url}/health")
                print(f"✓ Health check: {response.json()}")
                
                # Test fields endpoint
                response = await client.get(f"{self.base_url}/fields")
                assert response.status_code == 200
                fields_data = response.json()
                print(f"✓ Fields endpoint: {len(fields_data['fields'])} fields available")
                
                # Test stats endpoint
                response = await client.get(f"{self.base_url}/stats")
                assert response.status_code == 200
                stats_data = response.json()
                print(f"✓ Stats endpoint: {stats_data}")
                
                return True
                
        except Exception as e:
            print(f"✗ API endpoint test failed: {e}")
            return False
            
    async def test_question_via_api(self, question_data: dict) -> dict:
        """Test a question via API endpoint.
        
        Args:
            question_data: Question test data
            
        Returns:
            Test result dictionary
        """
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                request_data = {
                    "question": question_data["question"],
                    "field": question_data["field"]
                }
                
                response = await client.post(
                    f"{self.base_url}/ask",
                    json=request_data,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code != 200:
                    return {
                        "question": question_data["question"],
                        "field": question_data["field"],
                        "success": False,
                        "error": f"HTTP {response.status_code}: {response.text}",
                        "answer": None,
                        "citations": [],
                        "figures": []
                    }
                    
                result = response.json()
                
                return {
                    "question": question_data["question"],
                    "field": question_data["field"],
                    "success": True,
                    "error": result.get("error"),
                    "answer": result.get("answer", ""),
                    "citations": result.get("citations", []),
                    "figures": result.get("figures", []),
                    "confidence": result.get("confidence", "UNKNOWN"),
                    "text_chunks_found": result.get("text_chunks_found", 0),
                    "figures_found": result.get("figures_found", 0),
                    "intent": result.get("intent")
                }
                
        except Exception as e:
            return {
                "question": question_data["question"],
                "field": question_data["field"],
                "success": False,
                "error": str(e),
                "answer": None,
                "citations": [],
                "figures": []
            }
            
    def test_question_direct(self, question_data: dict) -> dict:
        """Test a question via direct workflow call.
        
        Args:
            question_data: Question test data
            
        Returns:
            Test result dictionary
        """
        try:
            if not self.api_key:
                return {
                    "question": question_data["question"],
                    "field": question_data["field"],
                    "success": False,
                    "error": "No API key provided",
                    "answer": None,
                    "citations": [],
                    "figures": []
                }
                
            workflow = get_qa_workflow(self.api_key)
            result = workflow.run(question_data["question"], question_data["field"])
            
            return {
                "question": question_data["question"],
                "field": question_data["field"],
                "success": True,
                "error": result.get("error"),
                "answer": result.get("answer", ""),
                "citations": result.get("citations", []),
                "figures": result.get("figures", []),
                "confidence": result.get("confidence", "UNKNOWN"),
                "text_chunks_found": result.get("text_chunks_found", 0),
                "figures_found": result.get("figures_found", 0),
                "intent": result.get("intent")
            }
            
        except Exception as e:
            return {
                "question": question_data["question"],
                "field": question_data["field"],
                "success": False,
                "error": str(e),
                "answer": None,
                "citations": [],
                "figures": []
            }
            
    def validate_result(self, result: dict, expected_data: dict) -> list:
        """Validate a test result against expectations.
        
        Args:
            result: Test result
            expected_data: Expected test data
            
        Returns:
            List of validation issues (empty if all good)
        """
        issues = []
        
        # Check if test succeeded
        if not result["success"]:
            issues.append(f"Test failed: {result.get('error', 'Unknown error')}")
            return issues
            
        # Check if answer is non-empty
        answer = result.get("answer", "")
        if not answer or len(answer.strip()) < 10:
            issues.append("Answer is too short or empty")
            
        # Check for citations if evidence is required
        citations = result.get("citations", [])
        if expected_data.get("requires_evidence") and len(citations) == 0:
            issues.append("No citations found but evidence was expected")
            
        # Check for numeric twin fallback message if needed
        if "numeric" in expected_data.get("expected_aspects", []):
            if "numeric twin not available" not in answer.lower() and "digital twin" not in answer.lower():
                # This is OK if we actually found numeric data, but warn if not
                if result.get("text_chunks_found", 0) == 0:
                    issues.append("Expected numeric twin fallback message but not found")
                    
        # Check confidence level
        confidence = result.get("confidence", "UNKNOWN")
        if confidence == "LOW" and len(citations) > 0:
            issues.append("Confidence is LOW despite having citations")
            
        return issues
        
    async def run_smoke_tests(self, test_api: bool = True, test_direct: bool = True) -> dict:
        """Run all smoke tests.
        
        Args:
            test_api: Whether to test API endpoints
            test_direct: Whether to test direct workflow calls
            
        Returns:
            Test results summary
        """
        print("=" * 60)
        print("GEOTHERMAL QA ENGINE - SMOKE TESTS")
        print("=" * 60)
        
        summary = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "api_accessible": False,
            "results": []
        }
        
        # Test API endpoints if requested
        if test_api:
            summary["api_accessible"] = await self.test_api_endpoints()
            
        # Test each question
        print(f"\n=== Testing {len(TEST_QUESTIONS)} Questions ===")
        
        for i, question_data in enumerate(TEST_QUESTIONS, 1):
            print(f"\n[{i}/{len(TEST_QUESTIONS)}] Testing: {question_data['question'][:50]}...")
            if question_data["field"]:
                print(f"    Field: {question_data['field']}")
            else:
                print("    Field: General (no specific field)")
                
            test_results = []
            
            # Test via API if available
            if test_api and summary["api_accessible"]:
                print("  → Testing via API...")
                api_result = await self.test_question_via_api(question_data)
                api_result["method"] = "API"
                test_results.append(api_result)
                
            # Test via direct call
            if test_direct:
                print("  → Testing via direct call...")
                direct_result = self.test_question_direct(question_data)
                direct_result["method"] = "Direct"
                test_results.append(direct_result)
                
            # Validate results
            for result in test_results:
                summary["total_tests"] += 1
                issues = self.validate_result(result, question_data)
                
                if issues:
                    summary["failed"] += 1
                    result["validation_issues"] = issues
                    print(f"    ✗ {result['method']}: FAILED")
                    for issue in issues:
                        print(f"      - {issue}")
                else:
                    summary["passed"] += 1
                    result["validation_issues"] = []
                    print(f"    ✓ {result['method']}: PASSED")
                    
                # Show key metrics
                if result["success"]:
                    print(f"      Citations: {len(result.get('citations', []))}")
                    print(f"      Text chunks: {result.get('text_chunks_found', 0)}")
                    print(f"      Figures: {result.get('figures_found', 0)}")
                    print(f"      Confidence: {result.get('confidence', 'UNKNOWN')}")
                    if result.get('intent'):
                        print(f"      Intent: {result.get('intent')}")
                        
                summary["results"].append(result)
                
        # Print summary
        print("\n" + "=" * 60)
        print("SMOKE TEST SUMMARY")
        print("=" * 60)
        print(f"Total tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Success rate: {summary['passed']/summary['total_tests']*100:.1f}%" if summary['total_tests'] > 0 else "N/A")
        print(f"API accessible: {'Yes' if summary['api_accessible'] else 'No'}")
        
        # Show sample answers
        print(f"\n=== Sample Answers ===")
        for i, result in enumerate(summary["results"][:3], 1):
            if result["success"] and result.get("answer"):
                print(f"\n{i}. Q: {result['question'][:60]}...")
                print(f"   A: {result['answer'][:200]}...")
                if result.get("citations"):
                    print(f"   Citations: {', '.join(result['citations'][:3])}")
                    
        return summary


async def main():
    """Main test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run QA engine smoke tests")
    parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument("--no-api", action="store_true", help="Skip API tests")
    parser.add_argument("--no-direct", action="store_true", help="Skip direct tests")
    parser.add_argument("--output", help="Save results to JSON file")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise
    
    # Run tests
    tester = QASmokeTest(args.api_key, args.base_url)
    results = await tester.run_smoke_tests(
        test_api=not args.no_api,
        test_direct=not args.no_direct
    )
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
        
    # Exit with appropriate code
    if results["failed"] == 0:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print(f"\n❌ {results['failed']} tests failed")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
