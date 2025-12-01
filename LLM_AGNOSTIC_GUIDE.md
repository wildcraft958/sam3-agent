# SAM3 Agent Microservice - LLM Provider Agnostic

This microservice provides SAM3 agent functionality with **complete LLM provider flexibility**. All LLM configuration is passed in API requests - no hardcoded providers or required LLM secrets.

## Key Features

✅ **No Hardcoded LLM Providers** - Works with any OpenAI-compatible API  
✅ **No LLM Secrets Required** - All LLM config passed in requests  
✅ **Provider Agnostic** - OpenAI, Anthropic, vLLM, custom APIs - all supported  
✅ **Optional HF Token** - Only needed if SAM3 model is gated  

## API Endpoint

**POST** `https://your-username--sam3-agent-sam3-segment.modal.run`

## Request Format

```json
{
  "prompt": "segment all objects",
  "image_b64": "base64-encoded-image",  // OR "image_url": "https://..."
  "llm_config": {
    "base_url": "https://api.openai.com/v1",  // Any OpenAI-compatible endpoint
    "model": "gpt-4o",                         // Any model name
    "api_key": "sk-your-key-here",            // API key (can be empty for some backends)
    "name": "openai-gpt4o",                   // Optional: for output files
    "max_tokens": 4096                         // Optional: default 4096
  },
  "debug": true                               // Optional: get visualization
}
```

## Supported LLM Providers

### OpenAI
```json
{
  "llm_config": {
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-4o",
    "api_key": "sk-your-openai-key"
  }
}
```

### Anthropic (if OpenAI-compatible)
```json
{
  "llm_config": {
    "base_url": "https://api.anthropic.com/v1",
    "model": "claude-3-opus",
    "api_key": "sk-ant-your-key"
  }
}
```

### vLLM Local Server
```json
{
  "llm_config": {
    "base_url": "http://localhost:8001/v1",
    "model": "Qwen/Qwen3-VL-8B-Thinking",
    "api_key": "",
    "name": "vllm-local"
  }
}
```

### vLLM on Modal
```json
{
  "llm_config": {
    "base_url": "https://your-vllm-app.modal.run/v1",
    "model": "your-model-name",
    "api_key": "",
    "name": "vllm-modal"
  }
}
```

### Custom API
```json
{
  "llm_config": {
    "base_url": "https://your-custom-api.com/v1",
    "model": "your-model",
    "api_key": "your-key",
    "max_tokens": 8192
  }
}
```

## Deployment

### Step 1: Deploy (No Secrets Required for LLM)

```bash
modal deploy modal_agent.py
```

The deployment will succeed without any LLM-related secrets. Only add `hf-token` secret if SAM3 model is gated.

### Step 2: Optional - Add HF Token (Only if Model is Gated)

```bash
modal secret create hf-token HF_TOKEN=hf_your-token-here
```

Then uncomment the secret in `modal_agent.py` line 151.

### Step 3: Use the API

All LLM configuration is passed in the request - no secrets needed!

## Example Usage

```python
import requests
import base64

# Encode image
with open("image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

# Make request with any LLM provider
response = requests.post(
    "https://your-endpoint.modal.run",
    json={
        "prompt": "segment all objects",
        "image_b64": image_b64,
        "llm_config": {
            "base_url": "https://api.openai.com/v1",  # Or any other provider
            "model": "gpt-4o",                         # Or any other model
            "api_key": "sk-your-key-here"             # Pass directly in request
        }
    }
)

result = response.json()
```

## Benefits

1. **No Provider Lock-in** - Switch between OpenAI, Anthropic, vLLM, etc. without code changes
2. **Direct API Keys** - Pass keys in requests (no Modal secrets needed for LLM)
3. **Easy Testing** - Test different LLMs by just changing the request
4. **Cost Control** - Use different providers per request based on cost/performance needs
5. **Flexibility** - Support any OpenAI-compatible API endpoint

## Architecture

```
Request → API Endpoint → SAM3 Agent
                          ↓
                    LLM Config (from request)
                          ↓
                    Any OpenAI-compatible API
```

- **SAM3 Model**: Loaded once at container startup (GPU-backed)
- **LLM Calls**: Made per-request using config from request body
- **No Hardcoding**: Everything configurable via API

## Response Format

```json
{
  "status": "success",
  "summary": "SAM3 returned 3 regions for prompt: segment all objects",
  "regions": [...],
  "debug_image_b64": "...",  // Only if debug=true
  "raw_sam3_json": {...},
  "llm_config": {
    "name": "openai-gpt4o",
    "model": "gpt-4o",
    "base_url": "https://api.openai.com/v1"
  }
}
```

## Notes

- All LLM configuration comes from the request - no hardcoded values
- API keys are passed in requests (not stored as secrets)
- Works with any OpenAI-compatible API endpoint
- HF token is optional (only if SAM3 model is gated)
- Microservice is truly provider-agnostic

