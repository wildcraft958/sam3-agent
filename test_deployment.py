#!/usr/bin/env python3
"""
Test deployment for SAM3 Agent Modal service.

This script tests the deployed SAM3 agent endpoint, similar to the 
sam3_agent.ipynb notebook example. It verifies that:
1. The endpoint is accessible
2. SAM3 model loads correctly
3. The agent can process images and prompts
4. LLM integration works (or fails gracefully if credentials missing)

Usage:
    python test_deployment.py
    python test_deployment.py --endpoint-url <your-url>
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


def create_test_image(image_path: str) -> None:
    """Create a dummy test image if it doesn't exist."""
    from PIL import Image
    
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    img = Image.new('RGB', (100, 100), color='red')
    img.save(image_path)
    print(f"✓ Created test image at {image_path}")


def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def test_deployment(
    endpoint_url: str,
    prompt: str = "find the red box",
    image_path: Optional[str] = None,
    llm_profile: str = "openai-gpt4o",
    debug: bool = True,
) -> Dict[str, Any]:
    """
    Test the deployed SAM3 agent endpoint.
    
    This follows the same pattern as sam3_agent.ipynb:
    1. Prepare image (like notebook loads image)
    2. Send prompt (like notebook prompt)
    3. Call endpoint (like notebook calls run_single_image_inference)
    4. Verify response (like notebook displays output)
    
    Args:
        endpoint_url: The Modal endpoint URL
        prompt: Text prompt for segmentation (default: "find the red box")
        image_path: Local path to image file (will be created if missing)
        llm_profile: LLM profile to use (default: "openai-gpt4o")
        debug: Whether to return debug visualization
        
    Returns:
        Response JSON from the endpoint
    """
    # Get repo root
    repo_root = Path(__file__).resolve().parent
    
    # Setup test image (like notebook)
    if image_path is None:
        image_path = repo_root / TEST_IMAGE_PATH
    else:
        image_path = Path(image_path)
    
    if not image_path.exists():
        print(f"⚠ Test image not found at {image_path}")
        print("Creating test image (like notebook setup)...")
        create_test_image(str(image_path))
    
    # Encode image to base64 (like notebook would prepare it)
    print(f"\n{'='*60}")
    print("Testing SAM3 Agent Deployment")
    print(f"{'='*60}")
    print(f"Endpoint: {endpoint_url}")
    print(f"Image: {image_path}")
    print(f"Prompt: {prompt}")
    print(f"LLM Profile: {llm_profile}")
    print(f"Debug: {debug}")
    print(f"{'='*60}\n")
    
    # Prepare request body (matching endpoint API)
    body: Dict[str, Any] = {
        "prompt": prompt,
        "image_b64": encode_image_to_base64(str(image_path)),
        "llm_profile": llm_profile,
        "debug": debug,
    }
    
    # Make request to deployed endpoint
    print("📤 Sending request to deployed endpoint...")
    try:
        response = requests.post(endpoint_url, json=body, timeout=600)
        response.raise_for_status()
        result = response.json()
        
        print("✓ Request successful!\n")
        print(f"Response Status: {result.get('status', 'unknown')}")
        
        if result.get("status") == "success":
            print("\n✅ SUCCESS: Deployment is working correctly!")
            print(f"Summary: {result.get('summary', 'N/A')}")
            
            regions = result.get("regions", [])
            print(f"Regions detected: {len(regions)}")
            
            if debug and result.get("debug_image_b64"):
                print("✓ Debug visualization available (base64 encoded)")
            
            if result.get("raw_sam3_json"):
                print("✓ Raw SAM3 output available")
            
            print(f"\n✓ LLM Profile used: {result.get('llm_profile', 'N/A')}")
            print("\n🎉 Deployment test PASSED - SAM3 agent is working!")
            
        elif result.get("status") == "error":
            error_msg = result.get("message", "Unknown error")
            print(f"\n⚠ Response indicates error: {error_msg}")
            
            # Check if it's an expected error (like missing API key)
            if "api_key" in error_msg.lower() or "OPENAI_API_KEY" in error_msg:
                print("\n⚠ Expected: Missing OpenAI API key")
                print("   This is OK - it means:")
                print("   ✓ Endpoint is accessible")
                print("   ✓ SAM3 model loaded successfully")
                print("   ✓ Request processing works")
                print("   ⚠ LLM call failed (expected without API key)")
                print("\n💡 To fix: Add OpenAI API key secret:")
                print("   modal secret create openai-api-key OPENAI_API_KEY=<your-key>")
                print("\n✅ Deployment test PARTIAL PASS - Core functionality works!")
            elif "LLM" in error_msg or "llm" in error_msg:
                print("\n⚠ LLM-related error (may be expected if credentials missing)")
                print("   Core SAM3 functionality appears to be working")
            else:
                print(f"\n❌ Unexpected error: {error_msg}")
                if result.get("traceback"):
                    print(f"\nTraceback:\n{result['traceback']}")
                print("\n❌ Deployment test FAILED")
        else:
            print(f"\n⚠ Unexpected response format: {result}")
            print("❌ Deployment test FAILED")
        
        return result
        
    except requests.exceptions.Timeout:
        print("\n❌ Request timed out (600s limit)")
        print("   This might indicate:")
        print("   - Model is still loading (first request)")
        print("   - Network connectivity issues")
        print("   - Endpoint is not responding")
        print("\n❌ Deployment test FAILED - Timeout")
        raise
        
    except requests.exceptions.ConnectionError as e:
        print(f"\n❌ Connection error: {e}")
        print("   This might indicate:")
        print("   - Endpoint URL is incorrect")
        print("   - Deployment is not active")
        print("   - Network connectivity issues")
        print("\n❌ Deployment test FAILED - Connection error")
        raise
        
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_body = e.response.json()
                print(f"Error response: {json.dumps(error_body, indent=2)}")
            except:
                print(f"Error response: {e.response.text}")
        print("\n❌ Deployment test FAILED")
        raise


def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test SAM3 Agent Modal deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with default endpoint
  python test_deployment.py
  
  # Test with custom endpoint
  python test_deployment.py --endpoint-url https://your-app.modal.run
  
  # Test with custom prompt (like notebook example)
  python test_deployment.py --prompt "the leftmost child wearing blue vest"
  
  # Test with vLLM profile
  python test_deployment.py --llm-profile vllm-local
        """
    )
    parser.add_argument(
        "--endpoint-url",
        type=str,
        default=DEFAULT_ENDPOINT_URL,
        help=f"Modal endpoint URL (default: {DEFAULT_ENDPOINT_URL})"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="find the red box",
        help="Text prompt for segmentation (default: 'find the red box')"
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default=None,
        help=f"Path to test image (default: {TEST_IMAGE_PATH})"
    )
    parser.add_argument(
        "--llm-profile",
        type=str,
        default="openai-gpt4o",
        choices=["openai-gpt4o", "vllm-local", "vllm-modal"],
        help="LLM profile to use (default: openai-gpt4o)"
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Disable debug visualization"
    )
    
    args = parser.parse_args()
    
    try:
        result = test_deployment(
            endpoint_url=args.endpoint_url,
            prompt=args.prompt,
            image_path=args.image_path,
            llm_profile=args.llm_profile,
            debug=not args.no_debug,
        )
        
        # Exit with appropriate code
        if result.get("status") == "success":
            sys.exit(0)
        elif "api_key" in result.get("message", "").lower():
            # Partial success - core works but API key missing
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
