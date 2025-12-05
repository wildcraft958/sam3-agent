#!/usr/bin/env python3
"""
Example usage of SAM3 Agent Modal endpoint with OpenAI.

This script demonstrates how to use the deployed SAM3 agent endpoint
with complete LLM configuration passed in the request.

Usage:
    python example_usage.py
    python example_usage.py --endpoint-url <your-url> --image <path-to-image>
    python example_usage.py --api-key sk-your-key --base-url https://api.openai.com/v1
"""

import argparse
import base64
import json
import os
import requests
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Example SAM3 Agent usage")
    parser.add_argument(
        "--endpoint-url",
        type=str,
        default="https://srinjoy59--sam3-agent-sam3-segment.modal.run",
        help="Modal endpoint URL"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="assets/images/test_image.jpg",
        help="Path to image file"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="segment all objects",
        help="Text prompt for segmentation"
    )
    parser.add_argument(
        "--llm-profile",
        type=str,
        default="openai-gpt4o",
        choices=["openai-gpt4o", "vllm-local", "vllm-modal"],
        help="LLM profile preset to use (or use --api-key/--base-url for custom)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (overrides profile default)"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL for LLM API (overrides profile default)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (overrides profile default)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=True,
        help="Get debug visualization"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_result.json",
        help="Output file for results"
    )
    
    args = parser.parse_args()
    
    # Load and encode image
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        print("Creating a test image...")
        from PIL import Image
        image_path.parent.mkdir(parents=True, exist_ok=True)
        img = Image.new('RGB', (100, 100), color='red')
        img.save(image_path)
        print(f"✓ Created test image at {image_path}")
    
    print(f"📷 Loading image: {image_path}")
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")
    
    # Build LLM config - allow full customization via arguments
    if args.llm_profile == "openai-gpt4o":
        llm_config = {
            "base_url": args.base_url or "https://api.openai.com/v1",
            "model": args.model or "gpt-4o",
            "api_key": args.api_key or os.environ.get("OPENAI_API_KEY", ""),
            "name": "openai-gpt4o",
        }
    elif args.llm_profile == "vllm-local":
        llm_config = {
            "base_url": args.base_url or "http://localhost:8001/v1",
            "model": args.model or "Qwen/Qwen3-VL-8B-Thinking",
            "api_key": args.api_key or "",
            "name": "vllm-local",
        }
    else:
        # Custom config - use provided args or defaults
        llm_config = {
            "base_url": args.base_url or "https://api.openai.com/v1",
            "model": args.model or "gpt-4o",
            "api_key": args.api_key or os.environ.get("OPENAI_API_KEY", ""),
            "name": args.model or "custom",
        }
    
    # Validate required fields
    if not llm_config["api_key"] and args.llm_profile == "openai-gpt4o":
        print("⚠ Warning: No API key provided. Set OPENAI_API_KEY env var or use --api-key")
    
    # Prepare request
    print(f"\n🚀 Sending request to: {args.endpoint_url}")
    print(f"📝 Prompt: {args.prompt}")
    print(f"🤖 LLM Config:")
    print(f"   - Base URL: {llm_config['base_url']}")
    print(f"   - Model: {llm_config['model']}")
    print(f"   - API Key: {'*' * 10 if llm_config['api_key'] else '(not set)'}")
    print(f"🐛 Debug: {args.debug}\n")
    
    payload = {
        "prompt": args.prompt,
        "image_b64": image_b64,
        "llm_config": llm_config,  # Pass complete config
        "debug": args.debug,
    }
    
    # Make request
    try:
        print("⏳ Processing (this may take 30-60 seconds for first request)...")
        response = requests.post(
            args.endpoint_url,
            json=payload,
            timeout=600
        )
        response.raise_for_status()
        result = response.json()
        
        # Save results
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"✓ Saved results to {args.output}\n")
        
        # Display results
        print("="*60)
        print("RESULTS")
        print("="*60)
        print(json.dumps(result, indent=2))
        print("="*60)
        
        if result.get("status") == "success":
            regions = result.get("regions", [])
            print(f"\n✅ Success! Found {len(regions)} regions")
            print(f"Summary: {result.get('summary', 'N/A')}")
            
            if args.debug and result.get("debug_image_b64"):
                debug_path = "output_debug.png"
                with open(debug_path, "wb") as f:
                    f.write(base64.b64decode(result["debug_image_b64"]))
                print(f"✓ Saved debug visualization to {debug_path}")
        else:
            error_msg = result.get("message", "Unknown error")
            print(f"\n❌ Error: {error_msg}")
            
            if "api_key" in error_msg.lower() or "llm_config" in error_msg.lower():
                print("\n💡 Tip: Provide complete llm_config in request:")
                print("   {")
                print('     "llm_config": {')
                print('       "base_url": "https://api.openai.com/v1",')
                print('       "model": "gpt-4o",')
                print('       "api_key": "sk-your-key-here"')
                print("     }")
                print("   }")
            
            if result.get("traceback"):
                print(f"\nTraceback:\n{result['traceback']}")
        
    except requests.exceptions.Timeout:
        print("❌ Request timed out (600s limit)")
        print("   First request takes longer due to model loading")
        print("   Try again in a few minutes")
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_body = e.response.json()
                print(f"Error: {json.dumps(error_body, indent=2)}")
            except:
                print(f"Error: {e.response.text}")


if __name__ == "__main__":
    main()

