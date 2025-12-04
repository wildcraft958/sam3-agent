#!/usr/bin/env python3
"""
Test script for SAM3 Agent on Modal

Usage:
    python test_sam3_agent.py --prompt "segment all buildings" --image-url "https://example.com/image.jpg"
    python test_sam3_agent.py --prompt "segment cars" --image-path "./local_image.jpg"
    
Environment variables (optional):
    SAM3_ENDPOINT: SAM3 agent URL
    VLLM_ENDPOINT: vLLM server URL
    MODEL_NAME: Model name
"""

import argparse
import base64
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import requests

# ============================================================================
# Default Configuration
# ============================================================================

DEFAULT_CONFIG = {
    "sam3_endpoint": "https://aryan-don357--sam3-agent-sam3-segment.modal.run",
    "vllm_endpoint": "https://aryan-don357--qwen3-vl-vllm-server-30b-vllm-server.modal.run/v1",
    "model_name": "Qwen/Qwen2.5-VL-32B-Instruct",
    "api_key": "dummy",  # vLLM doesn't require real key
    "max_tokens": 4096,
}


def load_image_as_base64(image_path: str) -> str:
    """Load a local image file and convert to base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def save_debug_image(b64_data: str, output_path: str = None) -> str:
    """Save base64 image data to file."""
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"sam3_result_{timestamp}.png"
    
    with open(output_path, "wb") as f:
        f.write(base64.b64decode(b64_data))
    
    return output_path


def test_sam3_agent(
    prompt: str,
    image_url: str = None,
    image_path: str = None,
    sam3_endpoint: str = None,
    vllm_endpoint: str = None,
    model_name: str = None,
    api_key: str = None,
    debug: bool = True,
    save_image: bool = True,
    output_path: str = None,
    verbose: bool = True,
) -> dict:
    """
    Test SAM3 agent with given prompt and image.
    
    Args:
        prompt: Text prompt for segmentation
        image_url: URL of the image (mutually exclusive with image_path)
        image_path: Local path to image (mutually exclusive with image_url)
        sam3_endpoint: SAM3 agent endpoint URL
        vllm_endpoint: vLLM server endpoint URL
        model_name: Model name to use
        api_key: API key (usually not required for vLLM)
        debug: Whether to request debug visualization
        save_image: Whether to save debug image to disk
        output_path: Path for output image (auto-generated if None)
        verbose: Print progress messages
        
    Returns:
        Response dictionary from SAM3 agent
    """
    # Use defaults if not specified
    sam3_endpoint = sam3_endpoint or os.environ.get("SAM3_ENDPOINT", DEFAULT_CONFIG["sam3_endpoint"])
    vllm_endpoint = vllm_endpoint or os.environ.get("VLLM_ENDPOINT", DEFAULT_CONFIG["vllm_endpoint"])
    model_name = model_name or os.environ.get("MODEL_NAME", DEFAULT_CONFIG["model_name"])
    api_key = api_key or os.environ.get("API_KEY", DEFAULT_CONFIG["api_key"])
    
    # Validate inputs
    if not image_url and not image_path:
        raise ValueError("Must provide either image_url or image_path")
    if image_url and image_path:
        raise ValueError("Cannot provide both image_url and image_path")
    
    if verbose:
        print("=" * 50)
        print("🚀 SAM3 Agent Test")
        print("=" * 50)
        print(f"SAM3 Endpoint: {sam3_endpoint}")
        print(f"vLLM Endpoint: {vllm_endpoint}")
        print(f"Model: {model_name}")
        print(f"Prompt: {prompt}")
        if image_url:
            print(f"Image URL: {image_url}")
        else:
            print(f"Image Path: {image_path}")
        print(f"Debug: {debug}")
        print("=" * 50)
        print()
    
    # Build request payload
    payload = {
        "prompt": prompt,
        "llm_config": {
            "base_url": vllm_endpoint,
            "model": model_name,
            "api_key": api_key,
            "max_tokens": DEFAULT_CONFIG["max_tokens"],
        },
        "debug": debug,
    }
    
    if image_url:
        payload["image_url"] = image_url
    else:
        payload["image_b64"] = load_image_as_base64(image_path)
    
    if verbose:
        print("📡 Sending request to SAM3 agent...")
        print()
    
    # Make request
    try:
        response = requests.post(
            sam3_endpoint,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=300,  # 5 minute timeout for long inference
        )
        response.raise_for_status()
        result = response.json()
    except requests.exceptions.Timeout:
        print("❌ Request timed out (5 minutes)")
        return {"status": "error", "error": "timeout"}
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return {"status": "error", "error": str(e)}
    
    if verbose:
        print("📋 Response:")
        print("=" * 50)
    
    # Handle response
    status = result.get("status", "unknown")
    
    if status == "success":
        if verbose:
            print("✅ Success!")
            if "summary" in result:
                print(f"📝 Summary: {result['summary']}")
            if "regions" in result:
                print(f"🎯 Regions found: {len(result['regions'])}")
        
        # Save debug image if present
        if save_image and debug and result.get("debug_image_b64"):
            saved_path = save_debug_image(result["debug_image_b64"], output_path)
            if verbose:
                print(f"🖼️  Debug image saved to: {saved_path}")
            result["saved_image_path"] = saved_path
    else:
        if verbose:
            print("❌ Request failed")
            error = result.get("error", result.get("message", "Unknown error"))
            print(f"Error: {error}")
    
    if verbose:
        print()
        print("=" * 50)
        
        # Print full response in verbose mode
        print()
        print("Full response:")
        print(json.dumps({k: v for k, v in result.items() if k != "debug_image_b64"}, indent=2))
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Test SAM3 Agent on Modal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # With image URL
    python test_sam3_agent.py --prompt "segment all buildings" --image-url "https://example.com/satellite.jpg"
    
    # With local image
    python test_sam3_agent.py --prompt "segment cars" --image-path "./image.jpg"
    
    # Custom endpoints
    python test_sam3_agent.py --prompt "segment water" --image-url "https://..." \\
        --sam3-endpoint "https://custom-sam3.modal.run" \\
        --vllm-endpoint "https://custom-vllm.modal.run/v1"
        
    # Save to specific file
    python test_sam3_agent.py --prompt "segment roads" --image-url "https://..." --output result.png
""",
    )
    
    # Required arguments
    parser.add_argument("--prompt", "-p", required=True, help="Text prompt for segmentation")
    
    # Image source (one required)
    image_group = parser.add_mutually_exclusive_group(required=True)
    image_group.add_argument("--image-url", "-u", help="URL of the image")
    image_group.add_argument("--image-path", "-i", help="Local path to image file")
    
    # Optional configuration
    parser.add_argument("--sam3-endpoint", help=f"SAM3 agent URL (default: {DEFAULT_CONFIG['sam3_endpoint']})")
    parser.add_argument("--vllm-endpoint", help=f"vLLM server URL (default: {DEFAULT_CONFIG['vllm_endpoint']})")
    parser.add_argument("--model", help=f"Model name (default: {DEFAULT_CONFIG['model_name']})")
    parser.add_argument("--api-key", help="API key (default: dummy)")
    
    # Output options
    parser.add_argument("--output", "-o", help="Output path for debug image")
    parser.add_argument("--no-debug", action="store_true", help="Don't request debug visualization")
    parser.add_argument("--no-save", action="store_true", help="Don't save debug image to disk")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")
    parser.add_argument("--json", action="store_true", help="Output raw JSON response")
    
    args = parser.parse_args()
    
    result = test_sam3_agent(
        prompt=args.prompt,
        image_url=args.image_url,
        image_path=args.image_path,
        sam3_endpoint=args.sam3_endpoint,
        vllm_endpoint=args.vllm_endpoint,
        model_name=args.model,
        api_key=args.api_key,
        debug=not args.no_debug,
        save_image=not args.no_save,
        output_path=args.output,
        verbose=not args.quiet and not args.json,
    )
    
    if args.json:
        # Remove large base64 data for cleaner output
        clean_result = {k: v for k, v in result.items() if k != "debug_image_b64"}
        print(json.dumps(clean_result, indent=2))
    
    # Exit with appropriate code
    sys.exit(0 if result.get("status") == "success" else 1)


if __name__ == "__main__":
    main()

