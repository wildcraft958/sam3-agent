# SAM3 Agent API Examples

Complete examples showing how to use the SAM3 Agent API with flexible LLM configuration.

## API Endpoint

**POST** `https://your-username--sam3-agent-sam3-segment.modal.run`

## Request Format

```json
{
  "prompt": "segment all objects",
  "image_b64": "base64-encoded-image-string",  // OR "image_url": "https://..."
  "llm_config": {
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-4o",
    "api_key": "sk-your-key-here",
    "name": "openai-gpt4o",      // Optional: for output files
    "max_tokens": 4096            // Optional: default 4096
  },
  "debug": true                   // Optional: get visualization
}
```

## Example 1: OpenAI GPT-4o

```python
import requests
import base64

# Encode image
with open("image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

# Make request
response = requests.post(
    "https://your-username--sam3-agent-sam3-segment.modal.run",
    json={
        "prompt": "segment all objects",
        "image_b64": image_b64,
        "llm_config": {
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4o",
            "api_key": "sk-your-openai-api-key-here",
            "name": "openai-gpt4o"
        },
        "debug": True
    },
    timeout=600
)

result = response.json()
print(result)
```

## Example 2: vLLM Local Server

```python
import requests
import base64

with open("image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

response = requests.post(
    "https://your-username--sam3-agent-sam3-segment.modal.run",
    json={
        "prompt": "the leftmost child wearing blue vest",
        "image_b64": image_b64,
        "llm_config": {
            "base_url": "http://localhost:8001/v1",
            "model": "Qwen/Qwen3-VL-8B-Thinking",
            "api_key": "",  # vLLM typically doesn't need API key
            "name": "vllm-local"
        },
        "debug": True
    },
    timeout=600
)

result = response.json()
```

## Example 3: Custom OpenAI-Compatible API

```python
import requests
import base64

with open("image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

response = requests.post(
    "https://your-username--sam3-agent-sam3-segment.modal.run",
    json={
        "prompt": "segment all cars",
        "image_b64": image_b64,
        "llm_config": {
            "base_url": "https://your-custom-api.com/v1",
            "model": "your-model-name",
            "api_key": "your-api-key",
            "name": "custom-llm",
            "max_tokens": 8192
        }
    }
)

result = response.json()
```

## Example 4: Using Image URL

```python
import requests

response = requests.post(
    "https://your-username--sam3-agent-sam3-segment.modal.run",
    json={
        "prompt": "find all people",
        "image_url": "https://example.com/image.jpg",
        "llm_config": {
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4o",
            "api_key": "sk-your-key"
        }
    }
)

result = response.json()
```

## Example 5: cURL

```bash
# Encode image
IMAGE_B64=$(base64 -w 0 image.jpg)

# Make request
curl -X POST https://your-username--sam3-agent-sam3-segment.modal.run \
  -H "Content-Type: application/json" \
  -d "{
    \"prompt\": \"segment all objects\",
    \"image_b64\": \"$IMAGE_B64\",
    \"llm_config\": {
      \"base_url\": \"https://api.openai.com/v1\",
      \"model\": \"gpt-4o\",
      \"api_key\": \"sk-your-openai-key-here\",
      \"name\": \"openai-gpt4o\"
    },
    \"debug\": true
  }"
```

## Response Format

### Success Response

```json
{
  "status": "success",
  "summary": "SAM3 returned 3 regions for prompt: segment all objects",
  "regions": [
    {
      "bbox": [100, 200, 150, 180],
      "mask": {...},
      "score": 0.95
    }
  ],
  "debug_image_b64": "base64-encoded-visualization",  // Only if debug=true
  "raw_sam3_json": {...},
  "llm_config": {
    "name": "openai-gpt4o",
    "model": "gpt-4o",
    "base_url": "https://api.openai.com/v1"
  }
}
```

### Error Response

```json
{
  "status": "error",
  "message": "Invalid llm_config: missing 'base_url'",
  "traceback": "...",  // Only in debug mode
  "llm_config": {
    "name": "unknown",
    "model": "unknown"
  }
}
```

## LLM Config Fields

### Required Fields

- **base_url**: API endpoint URL (e.g., `"https://api.openai.com/v1"`)
- **model**: Model name/identifier (e.g., `"gpt-4o"`)
- **api_key**: API key (can be empty string for some backends)

### Optional Fields

- **name**: Name for output files (defaults to model name)
- **provider**: Provider type (defaults to `"openai-compatible"`)
- **max_tokens**: Maximum tokens (defaults to `4096`)

## Common Configurations

### OpenAI
```json
{
  "base_url": "https://api.openai.com/v1",
  "model": "gpt-4o",
  "api_key": "sk-..."
}
```

### OpenAI-Compatible (Anthropic, etc.)
```json
{
  "base_url": "https://api.anthropic.com/v1",
  "model": "claude-3-opus",
  "api_key": "sk-ant-..."
}
```

### vLLM Local
```json
{
  "base_url": "http://localhost:8001/v1",
  "model": "Qwen/Qwen3-VL-8B-Thinking",
  "api_key": ""
}
```

### Custom Endpoint
```json
{
  "base_url": "https://your-api.com/v1",
  "model": "your-model",
  "api_key": "your-key",
  "max_tokens": 8192
}
```

## Notes

- All LLM configuration is passed in the request - no hardcoded profiles
- API key can be passed directly (no need for Modal secrets for LLM)
- Supports any OpenAI-compatible API endpoint
- `max_tokens` is optional and defaults to 4096
- `name` field is used for output file naming


