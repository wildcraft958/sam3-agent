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
    scaledown_window=3600,  # Keep container alive for 1 hour after last request
    min_containers=1,  # Keep at least 1 container always running (always loaded)
    # Required secrets - SAM3 is a gated repository on HuggingFace:
    #   - "hf-token" containing key HF_TOKEN (REQUIRED - SAM3 model is gated)
    # To add secret: modal secret create hf-token HF_TOKEN=<your-token>
    # Note: LLM configuration is passed in API requests, no LLM secrets needed
    secrets=[
        modal.Secret.from_name("huggingface-secret"),  # REQUIRED - SAM3 is a gated repo
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

        # HF_TOKEN is REQUIRED - SAM3 is a gated repository
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError(
                "HF_TOKEN environment variable is required but not set. "
                "SAM3 is a gated repository on HuggingFace. "
                "Please create the Modal secret: modal secret create hf-token HF_TOKEN=<your-token>"
            )
        
        # Authenticate with HuggingFace before attempting to load model
        try:
            login(token=hf_token)
            print("✓ Authenticated with HuggingFace")
        except Exception as e:
            raise ValueError(
                f"Failed to authenticate with HuggingFace: {e}. "
                "Please verify your HF_TOKEN is valid and has access to facebook/sam3 repository."
            ) from e

        print("Loading SAM3 model...")
        bpe_path = "/root/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
        self.model = build_sam3_image_model(bpe_path=bpe_path)
        # Use lower confidence threshold (0.25) to get more results, can be adjusted per request
        self.processor = Sam3Processor(self.model, confidence_threshold=0.4)
        print(f"SAM3 model loaded successfully with confidence threshold: {self.processor.confidence_threshold}")

    @modal.method()
    def sam3_infer_only(
        self,
        image_bytes: bytes,
        text_prompt: str,
        confidence_threshold: float = None,
    ) -> Dict[str, Any]:
        """
        Pure SAM3 inference only (no LLM/agent).
        Returns the same format as sam3_inference function.
        
        Args:
            image_bytes: Raw image bytes
            text_prompt: Text prompt for segmentation
            confidence_threshold: Optional confidence threshold (0.0-1.0). 
                               If None, uses processor's default (0.4)
        """
        from PIL import Image
        import io
        import torch
        from sam3.model.box_ops import box_xyxy_to_xywh
        from sam3.train.masks_ops import rle_encode
        
        # Set confidence threshold if provided
        if confidence_threshold is not None:
            if not 0.0 <= confidence_threshold <= 1.0:
                return {
                    "status": "error",
                    "message": f"confidence_threshold must be between 0.0 and 1.0, got {confidence_threshold}"
                }
            self.processor.confidence_threshold = confidence_threshold
            print(f"✓ Using confidence threshold: {confidence_threshold}")
        
        # Load image from bytes and convert to RGB if needed
        image = Image.open(io.BytesIO(image_bytes))
        # Convert RGBA/LA/P to RGB (SAM3 expects RGB)
        if image.mode in ('RGBA', 'LA', 'P'):
            # Create white background for transparency
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'P':
                image = image.convert('RGBA')
            rgb_image.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
            image = rgb_image
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        orig_img_w, orig_img_h = image.size
        print(f"✓ Image loaded: {orig_img_w}x{orig_img_h}, mode: {image.mode}")
        
        # Run SAM3 inference
        inference_state = self.processor.set_image(image)
        inference_state = self.processor.set_text_prompt(
            state=inference_state, prompt=text_prompt
        )
        
        # Check if we have any predictions
        num_boxes = inference_state["boxes"].shape[0] if len(inference_state["boxes"].shape) > 0 else 0
        num_masks = inference_state["masks"].shape[0] if len(inference_state["masks"].shape) > 0 else 0
        num_scores = len(inference_state["scores"]) if inference_state["scores"] is not None else 0
        
        print(f"✓ Inference complete: {num_boxes} boxes, {num_masks} masks, {num_scores} scores")
        print(f"  Confidence threshold: {self.processor.confidence_threshold}")
        if num_scores > 0:
            print(f"  Score range: {inference_state['scores'].min().item():.3f} - {inference_state['scores'].max().item():.3f}")
        
        # Handle empty results
        if num_boxes == 0 or num_masks == 0:
            print("⚠ Warning: No predictions found. This could be due to:")
            print("  1. Confidence threshold too high (current: {})".format(self.processor.confidence_threshold))
            print("  2. Text prompt not matching any objects in image")
            print("  3. Image quality or format issues")
            return {
                "status": "success",
                "orig_img_h": orig_img_h,
                "orig_img_w": orig_img_w,
                "pred_boxes": [],
                "pred_masks": [],
                "pred_scores": [],
                "warning": "No objects detected. Try lowering confidence threshold or using a different prompt.",
            }
        
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
        
        # Handle mask encoding - check if masks are empty
        if inference_state["masks"].numel() == 0:
            pred_masks = []
        else:
            pred_masks = rle_encode(inference_state["masks"].squeeze(1))
            # Preserve full RLE structure (counts + size) instead of extracting only counts
            # This ensures the JSON is self-contained and the mask structure is preserved
            pred_masks = [
                {"counts": m["counts"], "size": m["size"]} 
                for m in pred_masks
            ]
        
        # Create initial outputs (same format as sam3_inference)
        outputs = {
            "orig_img_h": orig_img_h,
            "orig_img_w": orig_img_w,
            "pred_boxes": pred_boxes_xywh,
            "pred_masks": pred_masks,
            "pred_scores": inference_state["scores"].tolist() if inference_state["scores"] is not None else [],
        }
        
        # Apply post-processing: remove overlapping masks (same as working sam3_inference path)
        try:
            from sam3.agent.helpers.mask_overlap_removal import remove_overlapping_masks
            outputs = remove_overlapping_masks(outputs)
            print(f"✓ Applied mask overlap removal")
        except Exception as e:
            print(f"⚠ Warning: Could not apply mask overlap removal: {e}")
        
        # Reorder by scores (highest to lowest) - same as working path
        if outputs["pred_scores"] and len(outputs["pred_scores"]) > 0:
            score_indices = sorted(
                range(len(outputs["pred_scores"])),
                key=lambda i: outputs["pred_scores"][i],
                reverse=True,
            )
            outputs["pred_scores"] = [outputs["pred_scores"][i] for i in score_indices]
            outputs["pred_boxes"] = [outputs["pred_boxes"][i] for i in score_indices]
            outputs["pred_masks"] = [outputs["pred_masks"][i] for i in score_indices]
            print(f"✓ Reordered predictions by score")
        
        # Filter invalid masks (too short RLE) - same as working path
        valid_masks = []
        valid_boxes = []
        valid_scores = []
        for i, rle in enumerate(outputs["pred_masks"]):
            # Handle both old format (string) and new format (dict with counts/size)
            rle_counts = rle if isinstance(rle, str) else rle.get("counts", "")
            if len(str(rle_counts)) > 4:  # Valid mask should have more than 4 characters
                valid_masks.append(rle)
                valid_boxes.append(outputs["pred_boxes"][i])
                valid_scores.append(outputs["pred_scores"][i])
        
        outputs["pred_masks"] = valid_masks
        outputs["pred_boxes"] = valid_boxes
        outputs["pred_scores"] = valid_scores
        
        # Add status field
        outputs["status"] = "success"
        
        print(f"✓ Returning {len(outputs['pred_boxes'])} valid predictions (after filtering)")
        return outputs

    @modal.method()
    def infer(
        self,
        image_bytes: bytes,
        prompt: str,
        llm_config: Dict[str, Any],
        debug: bool = False,
        confidence_threshold: float = None,
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
            confidence_threshold: Optional confidence threshold (0.0-1.0). If None, uses processor's default (0.4)
        
        Returns:
            Dict with status, regions, summary, and optional debug visualization
        """
        try:
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

        # Set confidence threshold if provided
        if confidence_threshold is not None:
            if not 0.0 <= confidence_threshold <= 1.0:
                return {
                    "status": "error",
                    "message": f"confidence_threshold must be between 0.0 and 1.0, got {confidence_threshold}"
                }
            self.processor.confidence_threshold = confidence_threshold
            print(f"✓ Using confidence threshold: {confidence_threshold}")

        # Cap max_tokens to 4096 to allow longer reasoning outputs for 32B Thinking model
        # Note: llm_config is already validated, so max_tokens is guaranteed to exist
        requested_max_tokens = llm_config["max_tokens"]
        safe_max_tokens = min(requested_max_tokens, 4096)
        
        send_generate_request = partial(
            send_generate_request_orig,
            server_url=llm_config["base_url"],
            model=llm_config["model"],
            api_key=llm_config["api_key"],
            max_tokens=safe_max_tokens,
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
                # Call agent_inference directly to get data from return values
                from sam3.agent.agent_core import agent_inference
                
                agent_history, final_output_dict, rendered_final_output = agent_inference(
                    image_path,
                    prompt,
                    send_generate_request=send_generate_request,
                    call_sam_service=call_sam_service,
                    debug=debug,
                    output_dir=output_dir,
                    max_generations=7,  # Limit LLM API calls to 7
                )

                if not final_output_dict:
                    return {
                        "status": "error",
                        "message": "No output generated by SAM3.",
                    }

                # Debug visualization image - convert PIL Image to base64
                debug_image_b64 = None
                if debug and rendered_final_output:
                    import io
                    buffer = io.BytesIO()
                    rendered_final_output.save(buffer, format="PNG")
                    debug_image_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")

                # Extract data from final_output_dict (which contains the SAM3 results)
                # The final_output_dict should have pred_boxes, pred_masks, pred_scores, etc.
                raw_json = final_output_dict.copy()
                
                # Remove file paths that aren't needed in response
                raw_json.pop("original_image_path", None)
                raw_json.pop("output_image_path", None)
                raw_json.pop("text_prompt", None)
                raw_json.pop("image_path", None)

                # Try to normalize "regions" field if present, otherwise construct from pred_boxes/pred_masks
                regions = (
                    raw_json.get("regions")
                    or raw_json.get("objects")
                    or raw_json.get("instances")
                    or []
                )
                
                # If no regions found, construct from pred_boxes and pred_masks
                if not regions and "pred_boxes" in raw_json and "pred_masks" in raw_json:
                    pred_boxes = raw_json.get("pred_boxes", [])
                    pred_masks = raw_json.get("pred_masks", [])
                    pred_scores = raw_json.get("pred_scores", [])
                    regions = [
                        {
                            "bbox": box,
                            "mask": mask,
                            "score": pred_scores[i] if i < len(pred_scores) else None,
                        }
                        for i, (box, mask) in enumerate(zip(pred_boxes, pred_masks))
                    ]

                summary = (
                    f"SAM3 returned {len(regions)} regions for prompt: {prompt}"
                )

                return {
                    "status": "success",
                    "summary": summary,
                    "regions": regions,
                    "debug_image_b64": debug_image_b64,
                    "raw_sam3_json": raw_json,
                    "agent_history": agent_history,  # Include agent history for debugging
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
      "confidence_threshold": 0.5  # Optional: confidence threshold (0.0-1.0, default: 0.4)
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
    confidence_threshold = body.get("confidence_threshold")
    
    # Get image bytes from either image_b64 or image_url
    if "image_b64" in body:
        try:
            image_bytes = base64.b64decode(body["image_b64"])
            print(f"✓ Decoded image from base64 ({len(image_bytes)} bytes)")
        except Exception as e:
            return {"status": "error", "message": f"Invalid base64 in 'image_b64': {e}."}
    elif "image_url" in body:
        import requests
        try:
            print(f"📥 Downloading image from URL: {body['image_url']}")
            resp = requests.get(body["image_url"], timeout=30)
            resp.raise_for_status()
            image_bytes = resp.content
            print(f"✓ Downloaded image ({len(image_bytes)} bytes)")
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
    # Use class reference directly to ensure persistent container reuse
    try:
        print(f"📞 Calling sam3_infer_only with prompt: '{text_prompt}'")
        result = SAM3Model().sam3_infer_only.remote(
            image_bytes=image_bytes,
            text_prompt=text_prompt,
            confidence_threshold=confidence_threshold,
        )
        print(f"✓ sam3_infer_only returned {len(result.get('pred_boxes', []))} predictions")
        return result
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback_str = traceback.format_exc()
        print(f"❌ Error in sam3_infer: {error_msg}")
        print(traceback_str)
        return {
            "status": "error",
            "message": error_msg,
            "traceback": traceback_str,
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
    confidence_threshold = body.get("confidence_threshold")

    # Get image bytes from either image_b64 or image_url
    if "image_b64" in body:
        try:
            image_bytes = base64.b64decode(body["image_b64"])
        except Exception:
            return {"status": "error", "message": "Invalid base64 in 'image_b64'."}
    elif "image_url" in body:
        import requests

        try:
            resp = requests.get(body["image_url"], timeout=30)
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
    # Use class reference directly to ensure persistent container reuse
    result = SAM3Model().infer.remote(
        image_bytes=image_bytes,
        prompt=prompt,
        llm_config=llm_config,
        debug=debug,
        confidence_threshold=confidence_threshold,
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
