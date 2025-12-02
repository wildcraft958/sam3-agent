# vllm_modal_deploy.py
#
# Deploy Qwen3-VL-32B-Thinking model with vLLM on Modal
#
# This script deploys a vLLM server with Qwen3-VL-32B-Thinking model,
# using Modal volumes to cache model weights for fast loading.
#
# Usage:
#   # First, download model to volume (one-time):
#   modal run vllm_modal_deploy.py::download_model_to_volume
#
#   # Then deploy the server:
#   modal deploy vllm_modal_deploy.py
#
# After deployment, the endpoint will be available at:
#   https://your-username--qwen3-vl-vllm-server.modal.run/v1
#
# Use this endpoint in SAM3 agent requests:
#   {
#     "llm_config": {
#       "base_url": "https://your-username--qwen3-vl-vllm-server.modal.run/v1",
#       "model": "Qwen/Qwen3-VL-32B-Thinking",
#       "api_key": "",
#       "name": "qwen3-vl-32b-modal"
#     }
#   }

import os
import subprocess
from pathlib import Path

import modal

# ------------------------------------------------------------------------------
# Modal app + image
# ------------------------------------------------------------------------------

app = modal.App("qwen3-vl-vllm-server")

# Create volume for model weights (~64GB for bfloat16)
# This will cache the model weights to avoid re-downloading on each deployment
MODEL_VOLUME = modal.Volume.from_name("qwen3-vl-32b-weights", create_if_missing=True)

# Model configuration
MODEL_ID = "Qwen/Qwen3-VL-32B-Thinking"
HF_CACHE_DIR = "/root/.cache/huggingface"  # Standard HuggingFace cache location

# GPU Configuration
# Available GPUs in Modal: "B200", "H200", "H100", "A100", "L40S", "A10", "L4", "T4"
# For single GPU: "H100" or "B200"
# For multiple GPUs: "H100:2", "H100:4", "A100:2", etc. (up to 8 GPUs)
# Note: B200 and H200 may not be available in all regions. If unavailable, use H100 or A100.
GPU_TYPE = "H100"  # Change to "B200", "H200", "A100", etc. as needed
NUM_GPUS = 2  # Number of GPUs (1-8). Set to 1 for single GPU, 2+ for tensor parallelism

# Build GPU string for Modal
if NUM_GPUS == 1:
    GPU_SPEC = GPU_TYPE
else:
    GPU_SPEC = f"{GPU_TYPE}:{NUM_GPUS}"

# Build custom image with vLLM and vision model support
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "git",
        "wget",
        "curl",
        "build-essential",
    )
    .pip_install(
        "vllm>=0.6.0",
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "accelerate>=0.30.0",
        "pillow>=10.0.0",
        "qwen-vl-utils>=0.0.14",  # Required for Qwen3-VL
        "fastapi>=0.100.0",
        "httpx>=0.24.0",
    )
    .env({"HF_HOME": HF_CACHE_DIR})
)

# ------------------------------------------------------------------------------
# vLLM Server Deployment
# ------------------------------------------------------------------------------

@app.function(
    gpu=GPU_SPEC,  # Use string format: "H100", "H100:2", "A100:4", etc.
    image=image,
    volumes={HF_CACHE_DIR: MODEL_VOLUME},  # Mount volume at HuggingFace cache location
    secrets=[
        modal.Secret.from_name("huggingface-secret"),  # For HF_TOKEN if model is gated
    ],
    timeout=3600,  # 1 hour timeout for model loading
    scaledown_window=300,  # Keep container alive for 5 minutes
)
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
def vllm_server():
    """
    Deploy vLLM server with Qwen3-VL-32B-Thinking model.
    
    This function loads the model directly and creates a FastAPI app
    that provides an OpenAI-compatible API, following the working pattern.
    
    GPU Configuration:
    - Single GPU: Set NUM_GPUS=1, GPU_TYPE="H100" (or "B200", "H200", "A100")
    - Multiple GPUs: Set NUM_GPUS=2+ for tensor parallelism
    """
    import base64
    import io
    import fastapi
    import httpx
    from fastapi.responses import JSONResponse
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    
    print(f"🚀 Initializing vLLM with {MODEL_ID}...")
    print(f"GPU Configuration: {GPU_SPEC} ({NUM_GPUS} GPU(s))")
    print(f"HuggingFace cache directory: {HF_CACHE_DIR}")
    
    # Check if model is already cached in volume
    model_cache_path = Path(HF_CACHE_DIR) / "hub"
    
    # Check multiple possible cache locations
    alt_paths = [
        model_cache_path / MODEL_ID.replace("/", "--"),
        model_cache_path / "models--" + MODEL_ID.replace("/", "--"),
        Path(HF_CACHE_DIR) / MODEL_ID.replace("/", "--"),
    ]
    
    model_found = False
    for check_path in alt_paths:
        if check_path.exists():
            # Check if it has model files
            try:
                safetensors = list(check_path.rglob("*.safetensors"))
                if safetensors:
                    print(f"✓ Model found in volume cache at {check_path}")
                    print(f"  Found {len(safetensors)} weight files")
                    model_found = True
                    break
            except:
                pass
    
    if not model_found:
        print(f"⚠ Model not found in expected cache locations")
        print(f"  Will download on first load (may take 20-30 minutes)")
    
    # Set HuggingFace environment variables
    os.environ["HF_HOME"] = HF_CACHE_DIR
    os.environ["TRANSFORMERS_CACHE"] = str(model_cache_path)
    os.environ["HF_HUB_CACHE"] = str(model_cache_path)
    
    # Build vLLM LLM initialization parameters
    llm_kwargs = {
        "model": MODEL_ID,
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.7,
        "max_model_len": 8192,
        "limit_mm_per_prompt": {"image": 1},  # For vision models
        "enforce_eager": True,  # More stable for vision models
        "max_num_seqs": 4,
    }
    
    # Add tensor parallelism for multiple GPUs
    if NUM_GPUS > 1:
        llm_kwargs["tensor_parallel_size"] = NUM_GPUS
        print(f"✓ Tensor parallelism enabled: {NUM_GPUS} GPUs")
    
    # Try to enable flash attention
    try:
        import flash_attn
        llm_kwargs["enable_flash_attn"] = True
        print("✓ Flash attention enabled")
    except ImportError:
        print("⚠ Flash attention not available, using default attention")
    
    # Load the model - this happens during container startup
    print("Loading model into GPU memory (this may take 5-15 minutes)...")
    llm = LLM(**llm_kwargs)
    print("✅ Model loaded successfully!")
    
    # Load processor for chat template formatting
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    print("✅ Processor loaded!")
    
    print(f"✅ Engine ready! Serving as: {MODEL_ID}")
    
    # Create FastAPI app
    app = fastapi.FastAPI(title="Qwen3-VL vLLM Server")
    
    http_client = httpx.AsyncClient(timeout=30.0)
    
    @app.on_event("shutdown")
    async def shutdown_http_client():
        await http_client.aclose()
    
    async def extract_messages_and_image(messages):
        """Extract messages in OpenAI format and image as PIL Image."""
        processed_messages = []
        image = None
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Handle string content (text-only)
            if isinstance(content, str):
                processed_messages.append({
                    "role": role,
                    "content": content
                })
                continue
            
            # Handle list content (multimodal)
            if isinstance(content, list):
                new_content = []
                for item in content:
                    if item.get("type") == "text":
                        new_content.append(item)
                    elif item.get("type") == "image_url" and image is None:
                        # Extract image
                        image_url = item.get("image_url", {}).get("url", "")
                        if not image_url:
                            continue
                        
                        if image_url.startswith("data:"):
                            # Base64 image
                            _, b64_data = image_url.split(",", 1)
                            image_bytes = base64.b64decode(b64_data)
                            image = Image.open(io.BytesIO(image_bytes))
                        else:
                            # URL image
                            resp = await http_client.get(image_url)
                            resp.raise_for_status()
                            image = Image.open(io.BytesIO(resp.content))
                        
                        # Keep image_url in content for processor
                        new_content.append(item)
                
                processed_messages.append({
                    "role": role,
                    "content": new_content
                })
        
        return processed_messages, image
    
    @app.get("/health")
    async def health():
        return {"status": "ok", "model": MODEL_ID}
    
    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": MODEL_ID,
                    "object": "model",
                    "created": 0,
                    "owned_by": "vllm"
                }
            ]
        }
    
    @app.post("/v1/chat/completions")
    async def chat_completions(request: fastapi.Request):
        try:
            req = await request.json()
            messages = req.get("messages", [])
            model = req.get("model", MODEL_ID)
            max_tokens = req.get("max_tokens", req.get("max_completion_tokens", 512))  # Support both
            temperature = req.get("temperature", 0.7)
            
            print(f"📥 Received request: model={model}, max_tokens={max_tokens}, messages={len(messages)}")
            
            # Extract messages and image
            processed_messages, image = await extract_messages_and_image(messages)
            
            if image is None:
                print("⚠️ Warning: No image found in request, but this is a vision model")
            
            # Format messages using processor's chat template
            try:
                formatted_prompt = processor.apply_chat_template(
                    processed_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                # Fallback: format manually
                formatted_prompt = ""
                for msg in processed_messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        formatted_prompt += f"{role}: {content}\n"
                    elif isinstance(content, list):
                        for item in content:
                            if item.get("type") == "text":
                                formatted_prompt += f"{role}: {item.get('text', '')}\n"
                            elif item.get("type") == "image_url":
                                formatted_prompt += "<image>\n"
                print(f"⚠️ Using fallback formatting: {e}")
            
            # Create sampling params
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
            )
            
            # Generate with vLLM
            print(f"🚀 Generating with vLLM...")
            if image:
                outputs = llm.generate(
                    [{"prompt": formatted_prompt, "multi_modal_data": {"image": image}}],
                    sampling_params=sampling_params,
                )
            else:
                outputs = llm.generate(
                    [formatted_prompt],
                    sampling_params=sampling_params,
                )
            
            if not outputs or len(outputs) == 0:
                print("❌ No outputs generated")
                return JSONResponse(
                    {"error": "No outputs generated from model"},
                    status_code=500,
                )
            
            generated_text = outputs[0].outputs[0].text
            
            if generated_text is None:
                print("❌ Generated text is None")
                return JSONResponse(
                    {"error": "Generated text is None"},
                    status_code=500,
                )
            
            print(f"✅ Generated {len(generated_text)} characters")
            
            # Estimate tokens (rough approximation)
            prompt_text = ""
            for msg in processed_messages:
                content = msg.get("content", "")
                if isinstance(content, str):
                    prompt_text += content + " "
                elif isinstance(content, list):
                    for item in content:
                        if item.get("type") == "text":
                            prompt_text += item.get("text", "") + " "
            
            prompt_tokens = len(prompt_text.split()) if prompt_text else 0
            completion_tokens = len(generated_text.split())
            
            return JSONResponse({
                "id": "chatcmpl-modal",
                "object": "chat.completion",
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": generated_text,
                    },
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            })
        except Exception as e:
            import traceback
            error_msg = str(e)
            error_trace = traceback.format_exc()
            print(f"❌ Error in chat_completions: {error_msg}")
            print(f"   Traceback: {error_trace}")
            return JSONResponse(
                {"error": error_msg, "traceback": error_trace},
                status_code=500,
            )
    
    return app


# ------------------------------------------------------------------------------
# Helper function to download model to volume (run once)
# ------------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="A100",  # Use A100 for download (or any available GPU - doesn't need to match server GPU)
    volumes={HF_CACHE_DIR: MODEL_VOLUME},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=7200,  # 2 hours for large model download
)
def download_model_to_volume():
    """
    Download model weights to volume (run once to pre-populate cache).
    
    This function downloads the model weights to the Modal volume,
    which will persist across deployments for fast loading.
    
    Usage:
        modal run vllm_modal_deploy.py::download_model_to_volume
    
    Note: This is a one-time operation. After running, the model
    will be cached in the volume and subsequent deployments will
    load much faster.
    """
    from huggingface_hub import snapshot_download
    import os
    
    print(f"Downloading {MODEL_ID} to volume...")
    print(f"Cache directory: {HF_CACHE_DIR}")
    print("This may take 20-30 minutes for a 32B model...")
    
    # Set HuggingFace environment variables
    os.environ["HF_HOME"] = HF_CACHE_DIR
    os.environ["TRANSFORMERS_CACHE"] = str(Path(HF_CACHE_DIR) / "hub")
    os.environ["HF_HUB_CACHE"] = str(Path(HF_CACHE_DIR) / "hub")
    
    # Download model (this will save to the mounted volume)
    print("Starting model download...")
    snapshot_download(
        MODEL_ID,
        cache_dir=Path(HF_CACHE_DIR) / "hub",
        local_files_only=False,
        force_download=True,  # Resume if interrupted
    )
    
    # Commit volume to persist the downloaded weights
    print("Committing model weights to volume...")
    MODEL_VOLUME.commit()
    
    # Check volume contents
    try:
        volume_files = MODEL_VOLUME.listdir("/", recursive=True)
        print(f"✓ Model downloaded and cached in volume")
        print(f"  Files cached: {len(volume_files)} files")
        
        # Estimate size from file entries
        total_size = 0
        for item in volume_files[:10]:  # Sample first 10 files
            if hasattr(item, 'size') and item.size:
                total_size += item.size
        if total_size > 0:
            print(f"  Estimated size: ~{total_size / (1024**3):.1f} GB (sampled)")
    except Exception as e:
        print(f"✓ Model downloaded and cached in volume")
        print(f"  (Could not list volume contents: {e})")

