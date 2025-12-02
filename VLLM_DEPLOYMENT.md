# Qwen3-VL-32B-Thinking vLLM Deployment Guide

This guide explains how to deploy the Qwen3-VL-32B-Thinking model using vLLM on Modal with persistent volume caching for fast model loading.

## Overview

The deployment uses:
- **Model**: Qwen/Qwen3-VL-32B-Thinking (32B parameter vision-language model)
- **Framework**: vLLM (OpenAI-compatible API server)
- **GPU**: NVIDIA B200 (192GB) or H200 (141GB)
- **Storage**: Modal Volume for persistent model weight caching

## Prerequisites

1. **Modal Account**: Sign up at [modal.com](https://modal.com) and install the CLI:
   ```bash
   pip install modal
   modal setup
   ```

2. **HuggingFace Token** (if model is gated):
   ```bash
   modal secret create huggingface-secret HF_TOKEN=<your-huggingface-token>
   ```

3. **Compute Requirements**:
   - **Single GPU**: H100 (80GB), A100 (80GB), or B200/H200 if available
   - **Multiple GPUs**: 2x A100 (40GB), 2x H100, or 4x A100 for better performance
   - Volume: ~64GB for model weights (bfloat16)
   
   **Note**: B200 and H200 may not be available in all regions. H100 and A100 are widely available alternatives.

## Deployment Steps

### Step 1: Download Model to Volume (One-Time)

Pre-populate the volume with model weights to avoid downloading on every deployment:

```bash
modal run vllm_modal_deploy.py::download_model_to_volume
```

This will:
- Download the 32B model weights (~64GB in bfloat16)
- Cache them in a Modal volume named `qwen3-vl-32b-weights`
- Take approximately 20-30 minutes depending on network speed

**Note**: This is a one-time operation. After this, model loading will be much faster.

### Step 2: Deploy vLLM Server

Deploy the vLLM server:

```bash
modal deploy vllm_modal_deploy.py
```

This will:
- Start the vLLM OpenAI-compatible API server
- Load the model from the volume cache (fast) or download if not cached
- Expose the endpoint at: `https://<your-username>--qwen3-vl-vllm-server.modal.run/v1`

### Step 3: Verify Deployment

Test the deployment:

```bash
# Health check
curl https://<your-username>--qwen3-vl-vllm-server.modal.run/health

# List models
curl https://<your-username>--qwen3-vl-vllm-server.modal.run/v1/models
```

## Configuration

### GPU Selection

Modal uses string-based GPU specification. Available GPUs:
- **B200**: 192GB (may not be available in all regions)
- **H200**: 141GB (may not be available in all regions)
- **H100**: 80GB (widely available)
- **A100**: 40GB or 80GB (widely available)
- **L40S**: 48GB
- **A10**: 24GB
- **L4**: 24GB
- **T4**: 16GB

To configure GPU, edit `vllm_modal_deploy.py`:

```python
# Single GPU configuration
GPU_TYPE = "H100"  # or "B200", "H200", "A100", etc.
NUM_GPUS = 1

# Multiple GPUs (tensor parallelism)
GPU_TYPE = "H100"
NUM_GPUS = 2  # or 4, 8 (up to 8 GPUs per container)
```

**Note**: If B200 or H200 are not available in your region, use H100 or A100. For a 32B model:
- **Single H100 (80GB)**: May work with lower memory utilization
- **Single A100 (80GB)**: Similar to H100
- **2x A100 (40GB)**: Good option with tensor parallelism
- **2x H100**: Best performance for 32B model

### Model Parameters

Adjust vLLM server parameters in the `vllm_server()` function:

- `--gpu-memory-utilization`: GPU memory usage (default: 0.9 = 90%)
- `--max-model-len`: Maximum context length (default: 8192)
- `--dtype`: Model precision (default: bfloat16)

### Volume Management

The volume is automatically created and mounted. To check volume status:

```python
import modal
vol = modal.Volume.from_name("qwen3-vl-32b-weights")
print(f"Volume size: {vol.size_gb} GB")
```

## Integration with SAM3 Agent

Use the deployed endpoint in your SAM3 agent requests:

```json
{
  "llm_config": {
    "base_url": "https://<your-username>--qwen3-vl-vllm-server.modal.run/v1",
    "model": "Qwen/Qwen3-VL-32B-Thinking",
    "api_key": "",
    "name": "qwen3-vl-32b-modal",
    "max_tokens": 2048
  }
}
```

### Frontend Configuration

The SAM3 frontend already supports OpenAI-compatible APIs. Simply configure the LLM settings in the frontend:

1. **Open the frontend** (deployed on Vercel or running locally)
2. **In the LLM Configuration section**, enter:
   - **Base URL**: `https://<your-username>--qwen3-vl-vllm-server.modal.run/v1`
   - **Model**: `Qwen/Qwen3-VL-32B-Thinking`
   - **API Key**: Leave empty (no authentication required for Modal endpoints)
   - **Name** (optional): `qwen3-vl-32b-modal`
   - **Max Tokens**: Set as needed (default: 4096, recommended: 2048-4096)

3. **Upload an image** and enter a segmentation prompt
4. **Click "Run Segmentation"** - the SAM3 agent will use your Modal vLLM endpoint

**Note**: The SAM3 agent is LLM-agnostic and works with any OpenAI-compatible API. No code changes are required - just configure the endpoint in the frontend.

## API Usage

The deployed server provides an OpenAI-compatible API:

### Chat Completions

```python
import requests

response = requests.post(
    "https://<your-username>--qwen3-vl-vllm-server.modal.run/v1/chat/completions",
    json={
        "model": "Qwen/Qwen3-VL-32B-Thinking",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://example.com/image.jpg"
                    },
                    {
                        "type": "text",
                        "text": "Describe this image."
                    }
                ]
            }
        ],
        "max_tokens": 512
    }
)

print(response.json())
```

### Using OpenAI Python Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://<your-username>--qwen3-vl-vllm-server.modal.run/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-VL-32B-Thinking",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "https://example.com/image.jpg"},
                {"type": "text", "text": "What's in this image?"}
            ]
        }
    ],
    max_tokens=512
)

print(response.choices[0].message.content)
```

## Performance Considerations

### Cold Start
- **With volume cache**: ~2-5 minutes (model loading from volume)
- **Without cache**: ~20-30 minutes (model download + loading)

### Inference Speed
- **First request**: May be slower due to model initialization
- **Subsequent requests**: Faster, GPU memory already allocated
- **Concurrent requests**: Up to 10 concurrent requests supported

### Cost Optimization
- **Container idle timeout**: 5 minutes (container stays alive after last request)
- **Volume storage**: Persistent, no additional cost for storage
- **GPU usage**: Billed per second when container is active

## Troubleshooting

### Model Not Loading

1. **Check volume**: Ensure model was downloaded to volume
   ```bash
   modal run vllm_modal_deploy.py::download_model_to_volume
   ```

2. **Check logs**: View Modal logs for errors
   ```bash
   modal app logs qwen3-vl-vllm-server
   ```

3. **Verify GPU**: Ensure B200/H200 is available in your Modal account

### Out of Memory Errors

- Reduce `--gpu-memory-utilization` (e.g., from 0.9 to 0.85)
- Reduce `--max-model-len` (e.g., from 8192 to 4096)
- Use quantization if supported (FP8 variant)

### Slow Inference

- Check GPU utilization in Modal dashboard
- Verify model is loaded from volume (not downloading)
- Consider increasing `container_idle_timeout` to keep container warm

## Monitoring

View deployment status and logs:

```bash
# View app status
modal app list

# View logs
modal app logs qwen3-vl-vllm-server

# View function logs
modal function logs qwen3-vl-vllm-server::vllm_server
```

## Cleanup

To remove the deployment:

```bash
modal app stop qwen3-vl-vllm-server
```

To delete the volume (frees storage but requires re-download):

```python
import modal
vol = modal.Volume.from_name("qwen3-vl-32b-weights")
vol.delete()  # WARNING: This deletes all cached model weights
```

## References

- [vLLM Documentation](https://docs.vllm.ai/)
- [Qwen3-VL Usage Guide](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-VL.html)
- [Modal Documentation](https://modal.com/docs)
- [Qwen3-VL Model Card](https://huggingface.co/Qwen/Qwen3-VL-32B-Thinking)

