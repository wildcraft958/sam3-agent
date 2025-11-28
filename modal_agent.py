# modal_agent.py
#
# SAM3 microservice on Modal.
#
# HTTP API (after `modal deploy modal_agent.py`):
#
#   POST /sam3/segment
#   Body (JSON):
#   {
#     "prompt": "segment all ships",
#     "image_url": "https://example.com/image.jpg",   # or "image_b64": "..."
#     "debug": true,
#     "llm_profile": "openai-gpt4o"                  # optional, default
#   }
#
#   Response (JSON):
#   {
#     "status": "success",
#     "summary": "...",
#     "regions": [...],
#     "debug_image_b64": "...",      # only if debug=true
#     "raw_sam3_json": {...},
#     "llm_profile": "openai-gpt4o"
#   }

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
# LLM profiles: define which LLM backends SAM3 can talk to.
# You can add/remove profiles here as needed.
# ------------------------------------------------------------------------------

LLM_PROFILES: Dict[str, Dict[str, str]] = {
    "openai-gpt4o": {
        "provider": "openai-compatible",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
        "api_key_env": "OPENAI_API_KEY",
    },
    # vLLM server profile - point to your vLLM server
    # Example: vllm serve Qwen/Qwen3-VL-8B-Thinking --port 8001
    "vllm-local": {
        "provider": "openai-compatible",
        "base_url": "http://localhost:8001/v1",  # Update with your vLLM server URL
        "model": "Qwen/Qwen3-VL-8B-Thinking",  # Update with your model name
        "api_key_env": None,  # vLLM typically doesn't require API key for local
    },
    # vLLM on Modal - if you deploy vLLM as a separate Modal app
    "vllm-modal": {
        "provider": "openai-compatible",
        "base_url": "https://your-vllm-app.modal.run/v1",  # Your Modal vLLM endpoint
        "model": "Qwen/Qwen3-VL-8B-Thinking",  # Your model name
        "api_key_env": None,  # Or set if your Modal vLLM requires auth
    },
    # Example profile for your own OpenAI-compatible LLM (e.g. vLLM on Modal)
    "modal-myllm-example": {
        "provider": "openai-compatible",
        "base_url": "https://your-modal-llm-url.modal.run/v1",
        "model": "my-llm-13b",
        "api_key_env": "MODAL_LLM_API_KEY",
    },
}


def get_llm_config(profile_name: str) -> Dict[str, str]:
    """Resolve an LLM profile name to a concrete config with base_url, model, api_key."""
    if profile_name not in LLM_PROFILES:
        raise ValueError(f"Unknown llm_profile={profile_name}")

    conf = LLM_PROFILES[profile_name]
    api_key_env = conf.get("api_key_env")
    api_key = os.environ.get(api_key_env, "") if api_key_env else ""

    if api_key_env and not api_key:
        raise ValueError(
            f"LLM profile '{profile_name}' requires env var {api_key_env}, but it's not set. "
            f"Please create a Modal secret: modal secret create {conf.get('api_key_env', '').lower().replace('_', '-')} {api_key_env}=<your-key>"
        )

    return {
        "provider": conf["provider"],
        "base_url": conf["base_url"],
        "model": conf["model"],
        "api_key": api_key,
    }


# ------------------------------------------------------------------------------
# GPU-backed SAM3 model class
# ------------------------------------------------------------------------------

@app.cls(
    gpu="A100",
    timeout=600,
    image=image,
    # Required secrets:
    #   - "huggingface-secret" containing key HF_TOKEN (required - SAM3 is a gated repo)
    #   - "openai-api-key" containing key OPENAI_API_KEY (optional - required for openai-gpt4o profile)
    # To add secrets: modal secret create <name> <KEY>=<value>
    secrets=[
        modal.Secret.from_name("huggingface-secret"),  # Required - SAM3 is a gated repo
        # modal.Secret.from_name("openai-api-key"),  # Optional - required for openai-gpt4o profile
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

        # HF_TOKEN is required for gated repo access
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError(
                "HF_TOKEN environment variable is required but not set. "
                "Please ensure the 'huggingface-secret' Modal secret is configured with HF_TOKEN=<your-token>"
            )
        
        # Login to HuggingFace with the token
        login(token=hf_token)
        print("✓ Authenticated with HuggingFace")

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
        llm_profile: str = "openai-gpt4o",
        debug: bool = False,
    ) -> Dict[str, Any]:
        """
        Core GPU inference method.
        - image_bytes: raw bytes of the input image
        - prompt: natural language query
        - llm_profile: which backend LLM to use (see LLM_PROFILES)
        - debug: whether to return a visualization image (base64)
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

        # Resolve which LLM backend to use
        try:
            llm_config = get_llm_config(llm_profile)
        except Exception as e:
            return {"status": "error", "message": str(e)}

        send_generate_request = partial(
            send_generate_request_orig,
            server_url=llm_config["base_url"],
            model=llm_config["model"],
            api_key=llm_config["api_key"],
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
                    "llm_profile": llm_profile,
                }

            except Exception as e:
                import traceback

                return {
                    "status": "error",
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                    "llm_profile": llm_profile,
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
    HTTP endpoint:

    POST /sam3/segment
    JSON body:
    {
      "prompt": "...",
      "image_url": "https://...",   # or "image_b64": "..."
      "debug": true,
      "llm_profile": "openai-gpt4o"
    }
    """
    # Basic validation
    if "prompt" not in body:
        return {"status": "error", "message": "Missing 'prompt' in request body."}

    prompt = body["prompt"]
    debug = bool(body.get("debug", False))
    llm_profile = body.get("llm_profile", "openai-gpt4o")

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
        llm_profile=llm_profile,
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
    model = SAM3Model()
    result = model.infer.remote(
        image_bytes=image_bytes,
        prompt=prompt,
        llm_profile="openai-gpt4o",
        debug=True,
    )

    print(json.dumps(result, indent=2)[:2000])
