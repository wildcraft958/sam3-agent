# Changes Summary - Flexible LLM Configuration

## ✅ What Changed

### Removed Hardcoded LLM Profiles
- ❌ Removed `LLM_PROFILES` dictionary
- ❌ Removed `get_llm_config()` function
- ✅ All LLM configuration now passed via API request

### New API Format

**Before (hardcoded profiles):**
```json
{
  "prompt": "segment objects",
  "image_b64": "...",
  "llm_profile": "openai-gpt4o"  // Limited to predefined profiles
}
```

**After (fully flexible):**
```json
{
  "prompt": "segment objects",
  "image_b64": "...",
  "llm_config": {                 // Complete config in request
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-4o",
    "api_key": "sk-...",
    "name": "openai-gpt4o",       // Optional
    "max_tokens": 4096            // Optional
  }
}
```

## 📝 API Changes

### Endpoint: POST `/sam3/segment`

**Required Fields:**
- `prompt`: Text prompt for segmentation
- `image_b64` OR `image_url`: Image input
- `llm_config`: Complete LLM configuration object

**llm_config Required Fields:**
- `base_url`: API endpoint URL
- `model`: Model name/identifier
- `api_key`: API key (can be empty string for some backends)

**llm_config Optional Fields:**
- `name`: Name for output files (defaults to model name)
- `provider`: Provider type (defaults to "openai-compatible")
- `max_tokens`: Maximum tokens (defaults to 4096)

## 🔧 Code Changes

1. **`validate_llm_config()`** - New function to validate and normalize LLM config
2. **`infer()` method** - Now accepts `llm_config` dict instead of `llm_profile` string
3. **`sam3_segment()` endpoint** - Requires `llm_config` in request body

## 📚 Updated Files

- ✅ `modal_agent.py` - Refactored to accept flexible LLM config
- ✅ `example_usage.py` - Updated with new API format
- ✅ `API_EXAMPLES.md` - Complete examples for different LLM backends

## 🚀 Benefits

1. **No Hardcoding** - Use any OpenAI-compatible API
2. **Direct API Keys** - Pass keys in request (no Modal secrets needed for LLM)
3. **Full Control** - Configure base_url, model, tokens per request
4. **Easy Testing** - Switch between different LLMs without code changes

## 📖 Quick Example

```python
import requests
import base64

with open("image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

response = requests.post(
    "https://your-endpoint.modal.run",
    json={
        "prompt": "segment all objects",
        "image_b64": image_b64,
        "llm_config": {
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4o",
            "api_key": "sk-your-key-here"
        }
    }
)

print(response.json())
```

## ⚠️ Breaking Changes

- `llm_profile` parameter removed - must use `llm_config` object
- API key must be provided in request (not from Modal secrets for LLM)
- All requests must include complete `llm_config`

## 🔄 Migration Guide

**Old code:**
```python
{
    "llm_profile": "openai-gpt4o"
}
```

**New code:**
```python
{
    "llm_config": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
        "api_key": "sk-your-key"
    }
}
```

