# modal_agent.py
#
# SAM3 Agent Microservice on Modal - LLM Provider Agnostic
#
# This microservice provides SAM3 agent functionality with complete LLM provider
# flexibility. All LLM configuration is passed in API requests - no hardcoded
# providers or required LLM secrets.
#
# HTTP API (after `modal deploy modal_agent.py`):
#
#   POST /sam3/segment
#   Body (JSON):
#   {
#     "prompt": "segment all ships",
#     "image_url": "https://example.com/image.jpg",   # or "image_b64": "..."
#     "llm_config": {                                 # Required: complete LLM config
#       "base_url": "https://api.openai.com/v1",      # Any OpenAI-compatible API
#       "model": "gpt-4o",                            # Any model name
#       "api_key": "sk-...",                          # API key (can be empty for some backends)
#       "name": "openai-gpt4o",                       # Optional: for output files
#       "max_tokens": 4096                            # Optional: default 4096
#     },
#     "debug": true                                   # Optional: get visualization
#   }
#
#   Response (JSON):
#   {
#     "status": "success",
#     "summary": "...",
#     "regions": [...],
#     "debug_image_b64": "...",      # only if debug=true
#     "raw_sam3_json": {...},
#     "llm_config": {...}
#   }
#
# Supports any OpenAI-compatible LLM provider:
#   - OpenAI (GPT-4o, GPT-4, etc.)
#   - Anthropic (Claude)
#   - vLLM servers
#   - Custom OpenAI-compatible APIs

import os
import sys
import json
import base64
import tempfile
from functools import partial
from pathlib import Path
from typing import Dict, Any

import modal

# ------------------------------------------------------------------------------
# Modal app + image
# ------------------------------------------------------------------------------

app = modal.App("sam3-agent")

REPO_ROOT = Path(__file__).resolve().parent
LOCAL_SAM3_DIR = REPO_ROOT / "sam3"
LOCAL_ASSETS_DIR = REPO_ROOT / "assets"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "wget", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch",
        "torchvision",
        "timm",
        "numpy<2.0",  # sam3 requires numpy<2.0
        "tqdm",
        "ftfy",
        "regex",
        "iopath",
        "typing_extensions",
        "huggingface_hub",
        "opencv-python",
        "pycocotools",
        "matplotlib",
        "scikit-image",
        "openai",
        "pillow",
        "einops",
        "decord",
        "scikit-learn",
        "psutil",
        "pandas",
        "scipy",
        "fastapi",  # Required for fastapi_endpoint decorator
        "requests",  # For downloading images from URLs
    )
    .env({"PYTHONPATH": "/root/sam3"})
    # use repo-relative paths so deploy works regardless of cwd
    .add_local_dir(str(LOCAL_SAM3_DIR), remote_path="/root/sam3/sam3")
    .add_local_dir(str(LOCAL_ASSETS_DIR), remote_path="/root/sam3/assets")
)

# ------------------------------------------------------------------------------
# LLM configuration helpers
# ------------------------------------------------------------------------------

def validate_llm_config(llm_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize LLM config from request.
    
    Required fields:
    - base_url: API endpoint URL
    - model: Model name/identifier
    - api_key: API key (can be empty for some backends)
    
    Optional fields:
    - name: Name for output files (defaults to model name)
    - provider: Provider type (defaults to "openai-compatible")
    - max_tokens: Maximum tokens (defaults to 4096)
    """
    if not isinstance(llm_config, dict):
        raise ValueError("llm_config must be a dictionary")
    
    # Required fields
    if "base_url" not in llm_config:
        raise ValueError("llm_config must include 'base_url'")
    if "model" not in llm_config:
        raise ValueError("llm_config must include 'model'")
    
    # Set defaults
    normalized = {
        "base_url": str(llm_config["base_url"]),
        "model": str(llm_config["model"]),
        "api_key": str(llm_config.get("api_key", "")),
        "name": llm_config.get("name", llm_config["model"]),  # Default to model name
        "provider": llm_config.get("provider", "openai-compatible"),
        "max_tokens": int(llm_config.get("max_tokens", 4096)),
    }
    
    return normalized


# ------------------------------------------------------------------------------
# GPU-backed SAM3 model class
# ------------------------------------------------------------------------------

@app.cls(
    gpu="A100",
    timeout=600,
    image=image,
    # Optional secrets - only needed if SAM3 model is gated on HuggingFace:
    #   - "hf-token" containing key HF_TOKEN (optional - only if model requires authentication)
    # To add secret: modal secret create hf-token HF_TOKEN=<your-token>
    # Note: LLM configuration is passed in API requests, no LLM secrets needed
    secrets=[
        # modal.Secret.from_name("hf-token"),  # Optional - uncomment only if SAM3 model is gated
    ],
)
class SAM3Model:
    @modal.enter()
    def setup(self):
        """Runs once per container: load SAM3 model + processor into GPU."""
        from huggingface_hub import login
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        if "/root/sam3" not in sys.path:
            sys.path.append("/root/sam3")

        # HF_TOKEN is optional - only needed if SAM3 model is gated
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            try:
                login(token=hf_token)
                print("✓ Authenticated with HuggingFace")
            except Exception as e:
                print(f"⚠ Warning: Failed to login to HuggingFace: {e}")
                print("   Continuing without authentication - model may fail to load if gated")
        else:
            print("⚠ No HF_TOKEN found - attempting model load without authentication")
            print("   If model is gated, create secret: modal secret create hf-token HF_TOKEN=<your-token>")

        print("Loading SAM3 model...")
        bpe_path = "/root/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
        self.model = build_sam3_image_model(bpe_path=bpe_path)
        self.processor = Sam3Processor(self.model, confidence_threshold=0.5)
        print("SAM3 model loaded successfully.")

    @modal.method()
    def sam3_infer_only(
        self,
        image_bytes: bytes,
        text_prompt: str,
    ) -> Dict[str, Any]:
        """
        Pure SAM3 inference only (no LLM/agent).
        Returns the same format as sam3_inference function.
        """
        from PIL import Image
        import io
        import torch
        from sam3.model.box_ops import box_xyxy_to_xywh
        from sam3.train.masks_ops import rle_encode
        
        # Load image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        orig_img_w, orig_img_h = image.size
        
        # Run SAM3 inference
        inference_state = self.processor.set_image(image)
        inference_state = self.processor.set_text_prompt(
            state=inference_state, prompt=text_prompt
        )
        
        # Format and assemble outputs (same as sam3_inference function)
        pred_boxes_xyxy = torch.stack(
            [
                inference_state["boxes"][:, 0] / orig_img_w,
                inference_state["boxes"][:, 1] / orig_img_h,
                inference_state["boxes"][:, 2] / orig_img_w,
                inference_state["boxes"][:, 3] / orig_img_h,
            ],
            dim=-1,
        )  # normalized in range [0, 1]
        pred_boxes_xywh = box_xyxy_to_xywh(pred_boxes_xyxy).tolist()
        pred_masks = rle_encode(inference_state["masks"].squeeze(1))
        pred_masks = [m["counts"] for m in pred_masks]
        
        outputs = {
            "status": "success",
            "orig_img_h": orig_img_h,
            "orig_img_w": orig_img_w,
            "pred_boxes": pred_boxes_xywh,
            "pred_masks": pred_masks,
            "pred_scores": inference_state["scores"].tolist(),
        }
        
        return outputs

    @modal.method()
    def infer(
        self,
        image_bytes: bytes,
        prompt: str,
        llm_config: Dict[str, Any],
        debug: bool = False,
    ) -> Dict[str, Any]:
        """
        Core GPU inference method - LLM provider agnostic.
        
        This method accepts any LLM configuration and uses it to call the LLM API.
        No hardcoded provider assumptions - works with any OpenAI-compatible API.
        
        Args:
            image_bytes: raw bytes of the input image
            prompt: natural language query for segmentation
            llm_config: Complete LLM configuration dict (provider-agnostic):
                - base_url: API endpoint URL (required) - any OpenAI-compatible endpoint
                - model: Model name/identifier (required) - any model name
                - api_key: API key (required, can be empty string for backends without auth)
                - name: Name for output files (optional, defaults to model name)
                - max_tokens: Maximum tokens (optional, defaults to 4096)
            debug: whether to return a visualization image (base64)
        
        Returns:
            Dict with status, regions, summary, and optional debug visualization
        """
        try:
            from sam3.agent.inference import run_single_image_inference
            from sam3.agent.client_llm import (
                send_generate_request as send_generate_request_orig,
            )
            from sam3.agent.client_sam3 import (
                call_sam_service as call_sam_service_orig,
            )
        except ImportError as e:
            return {
                "status": "error",
                "message": f"Failed to import SAM3 agent modules: {e}",
            }

        # Validate and normalize LLM config
        try:
            llm_config = validate_llm_config(llm_config)
        except Exception as e:
            return {"status": "error", "message": f"Invalid llm_config: {str(e)}"}

        send_generate_request = partial(
            send_generate_request_orig,
            server_url=llm_config["base_url"],
            model=llm_config["model"],
            api_key=llm_config["api_key"],
            max_tokens=llm_config.get("max_tokens", 4096),
        )

        call_sam_service = partial(
            call_sam_service_orig,
            sam3_processor=self.processor,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, "input_image.jpg")
            with open(image_path, "wb") as f:
                f.write(image_bytes)

            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(output_dir, exist_ok=True)

            try:
                output_image_path = run_single_image_inference(
                    image_path,
                    prompt,
                    llm_config,
                    send_generate_request,
                    call_sam_service,
                    debug=debug,
                    output_dir=output_dir,
                )

                if not output_image_path or not os.path.exists(output_image_path):
                    return {
                        "status": "error",
                        "message": "No output image generated by SAM3.",
                    }

                # Debug visualization image
                debug_image_b64 = None
                if debug:
                    with open(output_image_path, "rb") as f:
                        debug_image_b64 = base64.b64encode(f.read()).decode("ascii")

                # SAM3 JSON output (whatever the agent produced)
                json_path = output_image_path.replace(".png", ".json")
                raw_json: Dict[str, Any] = {}
                if os.path.exists(json_path):
                    with open(json_path, "r") as f:
                        raw_json = json.load(f)

                # Try to normalize "regions" field if present, otherwise just pass raw
                regions = (
                    raw_json.get("regions")
                    or raw_json.get("objects")
                    or raw_json.get("instances")
                    or []
                )

                summary = (
                    f"SAM3 returned {len(regions)} regions for prompt: {prompt}"
                )

                return {
                    "status": "success",
                    "summary": summary,
                    "regions": regions,
                    "debug_image_b64": debug_image_b64,
                    "raw_sam3_json": raw_json,
                    "llm_config": {
                        "name": llm_config["name"],
                        "model": llm_config["model"],
                        "base_url": llm_config["base_url"],
                    },
                }

            except Exception as e:
                import traceback

                return {
                    "status": "error",
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                    "llm_config": {
                        "name": llm_config.get("name", "unknown"),
                        "model": llm_config.get("model", "unknown"),
                    },
                }


# ------------------------------------------------------------------------------
# HTTP endpoint: /sam3/infer (SAM3-only inference, no LLM/agent)
# ------------------------------------------------------------------------------

from modal import fastapi_endpoint  # lightweight JSON endpoint decorator


@app.function(timeout=600, image=image)
@fastapi_endpoint(method="POST")
def sam3_infer(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    HTTP endpoint for pure SAM3 inference (no LLM/agent logic):
    
    POST /sam3/infer
    JSON body:
    {
      "text_prompt": "...",
      "image_b64": "...",   # or "image_url": "https://..."
    }
    
    Returns:
    {
      "orig_img_h": int,
      "orig_img_w": int,
      "pred_boxes": [[x, y, w, h], ...],  # normalized [0, 1]
      "pred_masks": ["rle_string", ...],
      "pred_scores": [float, ...]
    }
    """
    # Basic validation
    if "text_prompt" not in body:
        return {"status": "error", "message": "Missing 'text_prompt' in request body."}
    
    text_prompt = body["text_prompt"]
    
    # Get image bytes from either image_b64 or image_url
    if "image_b64" in body:
        try:
            image_bytes = base64.b64decode(body["image_b64"])
        except Exception:
            return {"status": "error", "message": "Invalid base64 in 'image_b64'."}
    elif "image_url" in body:
        import requests
        try:
            resp = requests.get(body["image_url"])
            resp.raise_for_status()
            image_bytes = resp.content
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to download 'image_url': {e}",
            }
    else:
        return {
            "status": "error",
            "message": "Provide either 'image_b64' or 'image_url' in the request body.",
        }
    
    # Call the GPU-backed model for SAM3 inference only
    model_instance = SAM3Model()
    try:
        from PIL import Image
        import io
        import torch
        from sam3.model.box_ops import box_xyxy_to_xywh
        from sam3.train.masks_ops import rle_encode
        
        # Load image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        orig_img_w, orig_img_h = image.size
        
        # Get processor from the model instance (access via remote method)
        # We need to create a method to get the processor, or call a remote method
        # For now, let's create a helper method on SAM3Model
        result = model_instance.sam3_infer_only.remote(
            image_bytes=image_bytes,
            text_prompt=text_prompt,
        )
        return result
        
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc(),
        }


# ------------------------------------------------------------------------------
# HTTP endpoint: /sam3/segment (Full agent with LLM)
# ------------------------------------------------------------------------------


@app.function(timeout=600, image=image)
@fastapi_endpoint(method="POST")
def sam3_segment(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    HTTP endpoint for SAM3 Agent - LLM Provider Agnostic
    
    This endpoint accepts any LLM configuration via the request body.
    No hardcoded providers - works with any OpenAI-compatible API.

    POST /sam3/segment
    JSON body:
    {
      "prompt": "...",                    # Required: text prompt for segmentation
      "image_url": "https://...",         # OR "image_b64": "..." (one required)
      "llm_config": {                     # Required: complete LLM configuration (provider-agnostic)
        "base_url": "https://api.openai.com/v1",  # Any OpenAI-compatible endpoint
        "model": "gpt-4o",                 # Any model name
        "api_key": "sk-...",              # API key (can be empty for some backends)
        "name": "openai-gpt4o",           # Optional: for output files
        "max_tokens": 4096                 # Optional: default 4096
      },
      "debug": true                       # Optional: get visualization
    }
    
    Example with OpenAI:
    {
      "prompt": "segment all objects",
      "image_b64": "...",
      "llm_config": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
        "api_key": "sk-your-key-here"
      }
    }
    
    Example with vLLM:
    {
      "prompt": "segment all objects",
      "image_b64": "...",
      "llm_config": {
        "base_url": "http://localhost:8001/v1",
        "model": "Qwen/Qwen3-VL-8B-Thinking",
        "api_key": "",
        "name": "vllm-local"
      }
    }
    
    Example with Anthropic (if OpenAI-compatible):
    {
      "prompt": "segment all objects",
      "image_b64": "...",
      "llm_config": {
        "base_url": "https://api.anthropic.com/v1",
        "model": "claude-3-opus",
        "api_key": "sk-ant-..."
      }
    }
    """
    # Basic validation
    if "prompt" not in body:
        return {"status": "error", "message": "Missing 'prompt' in request body."}

    if "llm_config" not in body:
        return {"status": "error", "message": "Missing 'llm_config' in request body. Provide complete LLM configuration with 'base_url', 'model', and 'api_key'."}

    prompt = body["prompt"]
    debug = bool(body.get("debug", False))
    llm_config = body["llm_config"]

    # Get image bytes from either image_b64 or image_url
    if "image_b64" in body:
        try:
            image_bytes = base64.b64decode(body["image_b64"])
        except Exception:
            return {"status": "error", "message": "Invalid base64 in 'image_b64'."}
    elif "image_url" in body:
        import requests

        try:
            resp = requests.get(body["image_url"])
            resp.raise_for_status()
            image_bytes = resp.content
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to download 'image_url': {e}",
            }
    else:
        return {
            "status": "error",
            "message": "Provide either 'image_b64' or 'image_url' in the request body.",
        }

    # Call the GPU-backed model
    model = SAM3Model()
    result = model.infer.remote(
        image_bytes=image_bytes,
        prompt=prompt,
        llm_config=llm_config,
        debug=debug,
    )
    return result


# ------------------------------------------------------------------------------
# Optional: local quick test via `modal run modal_agent.py::local_test`
# ------------------------------------------------------------------------------

@app.local_entrypoint()
def local_test():
    """
    Quick sanity check: runs SAM3 on a dummy red image, no HTTP involved.

    Run:
      modal run modal_agent.py::local_test
    """
    from PIL import Image
    from io import BytesIO

    img = Image.new("RGB", (128, 128), color="red")
    buf = BytesIO()
    img.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    prompt = "find the red region"
    llm_config = {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
        "api_key": os.environ.get("OPENAI_API_KEY", ""),
        "name": "openai-gpt4o",
    }
    model = SAM3Model()
    result = model.infer.remote(
        image_bytes=image_bytes,
        prompt=prompt,
        llm_config=llm_config,
        debug=True,
    )

    print(json.dumps(result, indent=2)[:2000])
