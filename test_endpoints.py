#!/usr/bin/env python3
"""
Test runner for SAM3 Agent Modal endpoints.

This script tests the deployed SAM3 agent endpoints, similar to the 
sam3_agent.ipynb notebook example.

Usage:
    python test_endpoints.py
"""

import os
import sys
import json
import base64
import requests
from pathlib import Path
from typing import Dict, Any, Optional

# Default endpoint URL (update if your deployment URL is different)
DEFAULT_ENDPOINT_URL = "https://animesh-raj--sam3-agent-sam3-segment.modal.run"

# Test image path (relative to repo root)
TEST_IMAGE_PATH = "assets/images/test_image.jpg"


def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def test_segment_endpoint(
    endpoint_url: str,
    prompt: str,
    image_path: Optional[str] = None,
    image_url: Optional[str] = None,
    image_b64: Optional[str] = None,
    llm_profile: str = "openai-gpt4o",
    debug: bool = True,
) -> Dict[str, Any]:
    """
    Test the /sam3/segment endpoint.
    
    Args:
        endpoint_url: The Modal endpoint URL
        prompt: Text prompt for segmentation
        image_path: Local path to image file (will be converted to base64)
        image_url: URL to an image
        image_b64: Base64 encoded image string
        llm_profile: LLM profile to use (openai-gpt4o, vllm-local, vllm-modal)
        debug: Whether to return debug visualization
        
    Returns:
        Response JSON from the endpoint
    """
    # Prepare request body
    body: Dict[str, Any] = {
        "prompt": prompt,
        "llm_profile": llm_profile,
        "debug": debug,
    }
    
    # Add image in one of the supported formats
    if image_path:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        body["image_b64"] = encode_image_to_base64(image_path)
    elif image_url:
        body["image_url"] = image_url
    elif image_b64:
        body["image_b64"] = image_b64
    else:
        raise ValueError("Must provide one of: image_path, image_url, or image_b64")
    
    # Make request
    print(f"\n{'='*60}")
    print(f"Testing endpoint: {endpoint_url}")
    print(f"Prompt: {prompt}")
    print(f"LLM Profile: {llm_profile}")
    print(f"Debug: {debug}")
    print(f"{'='*60}\n")
    
    try:
        response = requests.post(endpoint_url, json=body, timeout=600)
        response.raise_for_status()
        result = response.json()
        
        print(f"✓ Request successful!")
        print(f"Status: {result.get('status', 'unknown')}")
        
        if result.get("status") == "success":
            print(f"Summary: {result.get('summary', 'N/A')}")
            regions = result.get("regions", [])
            print(f"Regions found: {len(regions)}")
            
            if debug and result.get("debug_image_b64"):
                print(f"Debug image: Available (base64 encoded)")
            
            if result.get("raw_sam3_json"):
                print(f"Raw SAM3 JSON: Available")
        else:
            print(f"Error: {result.get('message', 'Unknown error')}")
            if result.get("traceback"):
                print(f"\nTraceback:\n{result['traceback']}")
        
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_body = e.response.json()
                print(f"Error response: {json.dumps(error_body, indent=2)}")
            except:
                print(f"Error response: {e.response.text}")
        raise


def run_all_tests(endpoint_url: str = DEFAULT_ENDPOINT_URL):
    """Run all test cases."""
    
    print("="*60)
    print("SAM3 Agent Endpoint Test Runner")
    print("="*60)
    print(f"Endpoint URL: {endpoint_url}\n")
    
    # Get repo root
    repo_root = Path(__file__).resolve().parent
    test_image = repo_root / TEST_IMAGE_PATH
    
    # Check if test image exists
    if not test_image.exists():
        print(f"⚠ Warning: Test image not found at {test_image}")
        print("Creating a dummy test image...")
        from PIL import Image
        test_image.parent.mkdir(parents=True, exist_ok=True)
        img = Image.new('RGB', (100, 100), color='red')
        img.save(test_image)
        print(f"✓ Created dummy image at {test_image}\n")
    
    test_results = []
    
    # Test 1: Basic segmentation with local image (base64)
    print("\n" + "="*60)
    print("TEST 1: Basic segmentation with local image (base64)")
    print("="*60)
    try:
        result = test_segment_endpoint(
            endpoint_url=endpoint_url,
            prompt="segment all objects",
            image_path=str(test_image),
            llm_profile="openai-gpt4o",
            debug=True,
        )
        test_results.append(("Test 1: Basic segmentation", "PASS" if result.get("status") == "success" else "FAIL"))
    except Exception as e:
        print(f"✗ Test 1 failed: {e}")
        test_results.append(("Test 1: Basic segmentation", f"FAIL: {e}"))
    
    # Test 2: Complex prompt (like notebook example)
    print("\n" + "="*60)
    print("TEST 2: Complex prompt (notebook example)")
    print("="*60)
    try:
        result = test_segment_endpoint(
            endpoint_url=endpoint_url,
            prompt="the leftmost child wearing blue vest",
            image_path=str(test_image),
            llm_profile="openai-gpt4o",
            debug=True,
        )
        test_results.append(("Test 2: Complex prompt", "PASS" if result.get("status") == "success" else "FAIL"))
    except Exception as e:
        print(f"✗ Test 2 failed: {e}")
        test_results.append(("Test 2: Complex prompt", f"FAIL: {e}"))
    
    # Test 3: Image URL (if you have a test image URL)
    print("\n" + "="*60)
    print("TEST 3: Image URL (optional - skip if no URL)")
    print("="*60)
    test_image_url = "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba"  # Example cat image
    try:
        result = test_segment_endpoint(
            endpoint_url=endpoint_url,
            prompt="segment the cat",
            image_url=test_image_url,
            llm_profile="openai-gpt4o",
            debug=False,
        )
        test_results.append(("Test 3: Image URL", "PASS" if result.get("status") == "success" else "FAIL"))
    except Exception as e:
        print(f"⚠ Test 3 skipped or failed: {e}")
        test_results.append(("Test 3: Image URL", f"SKIP: {e}"))
    
    # Test 4: vLLM profile (if configured)
    print("\n" + "="*60)
    print("TEST 4: vLLM profile (optional - skip if not configured)")
    print("="*60)
    try:
        result = test_segment_endpoint(
            endpoint_url=endpoint_url,
            prompt="find all objects",
            image_path=str(test_image),
            llm_profile="vllm-local",  # Update this if you have vLLM configured
            debug=True,
        )
        test_results.append(("Test 4: vLLM profile", "PASS" if result.get("status") == "success" else "FAIL"))
    except Exception as e:
        print(f"⚠ Test 4 skipped or failed: {e}")
        test_results.append(("Test 4: vLLM profile", f"SKIP: {e}"))
    
    # Test 5: Error handling - missing prompt
    print("\n" + "="*60)
    print("TEST 5: Error handling - missing prompt")
    print("="*60)
    try:
        body = {"image_b64": encode_image_to_base64(str(test_image))}
        response = requests.post(endpoint_url, json=body, timeout=60)
        result = response.json()
        if result.get("status") == "error" and "prompt" in result.get("message", "").lower():
            print("✓ Correctly returned error for missing prompt")
            test_results.append(("Test 5: Error handling", "PASS"))
        else:
            print(f"✗ Unexpected response: {result}")
            test_results.append(("Test 5: Error handling", "FAIL"))
    except Exception as e:
        print(f"✗ Test 5 failed: {e}")
        test_results.append(("Test 5: Error handling", f"FAIL: {e}"))
    
    # Test 6: Error handling - missing image
    print("\n" + "="*60)
    print("TEST 6: Error handling - missing image")
    print("="*60)
    try:
        body = {"prompt": "segment objects"}
        response = requests.post(endpoint_url, json=body, timeout=60)
        result = response.json()
        if result.get("status") == "error" and ("image" in result.get("message", "").lower()):
            print("✓ Correctly returned error for missing image")
            test_results.append(("Test 6: Error handling (no image)", "PASS"))
        else:
            print(f"✗ Unexpected response: {result}")
            test_results.append(("Test 6: Error handling (no image)", "FAIL"))
    except Exception as e:
        print(f"✗ Test 6 failed: {e}")
        test_results.append(("Test 6: Error handling (no image)", f"FAIL: {e}"))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    passed = sum(1 for _, status in test_results if status == "PASS")
    failed = sum(1 for _, status in test_results if status.startswith("FAIL"))
    skipped = sum(1 for _, status in test_results if status.startswith("SKIP"))
    
    for test_name, status in test_results:
        status_symbol = "✓" if status == "PASS" else "✗" if status.startswith("FAIL") else "⚠"
        print(f"{status_symbol} {test_name}: {status}")
    
    print(f"\nTotal: {len(test_results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print("="*60)
    
    return test_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test SAM3 Agent Modal endpoints")
    parser.add_argument(
        "--endpoint-url",
        type=str,
        default=DEFAULT_ENDPOINT_URL,
        help=f"Modal endpoint URL (default: {DEFAULT_ENDPOINT_URL})"
    )
    parser.add_argument(
        "--test-image",
        type=str,
        default=TEST_IMAGE_PATH,
        help=f"Path to test image (default: {TEST_IMAGE_PATH})"
    )
    parser.add_argument(
        "--single-test",
        type=str,
        choices=["basic", "complex", "url", "vllm", "error"],
        help="Run a single test instead of all tests"
    )
    
    args = parser.parse_args()
    
    if args.single_test:
        # Run single test
        repo_root = Path(__file__).resolve().parent
        test_image = repo_root / args.test_image
        
        if args.single_test == "basic":
            test_segment_endpoint(
                endpoint_url=args.endpoint_url,
                prompt="segment all objects",
                image_path=str(test_image),
                llm_profile="openai-gpt4o",
                debug=True,
            )
        elif args.single_test == "complex":
            test_segment_endpoint(
                endpoint_url=args.endpoint_url,
                prompt="the leftmost child wearing blue vest",
                image_path=str(test_image),
                llm_profile="openai-gpt4o",
                debug=True,
            )
        elif args.single_test == "url":
            test_segment_endpoint(
                endpoint_url=args.endpoint_url,
                prompt="segment the cat",
                image_url="https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba",
                llm_profile="openai-gpt4o",
                debug=False,
            )
        elif args.single_test == "vllm":
            test_segment_endpoint(
                endpoint_url=args.endpoint_url,
                prompt="find all objects",
                image_path=str(test_image),
                llm_profile="vllm-local",
                debug=True,
            )
        elif args.single_test == "error":
            # Test error handling
            body = {"prompt": "test"}
            response = requests.post(args.endpoint_url, json=body, timeout=60)
            print(json.dumps(response.json(), indent=2))
    else:
        # Run all tests
        run_all_tests(endpoint_url=args.endpoint_url)

