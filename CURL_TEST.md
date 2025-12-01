# cURL Test Commands for SAM3 Agent

## Endpoint URL
```
https://aryan-don357--sam3-agent-sam3-segment.modal.run
```

## Prerequisites

1. **Get your OpenAI API key:**
   - Visit: https://platform.openai.com/api-keys
   - Create a new key or use existing one
   - Format: `sk-...`

2. **Prepare a test image:**
   - Use any image file (JPG, PNG, etc.)
   - Or use a public image URL

## Test 1: Basic Request with Image URL

```bash
curl -X POST https://aryan-don357--sam3-agent-sam3-segment.modal.run \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "segment all objects",
    "image_url": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4",
    "llm_config": {
      "base_url": "https://api.openai.com/v1",
      "model": "gpt-4o",
      "api_key": "sk-YOUR-OPENAI-API-KEY-HERE",
      "name": "openai-gpt4o"
    },
    "debug": true
  }'
```

## Test 2: Request with Base64 Encoded Image

### Step 1: Encode image to base64
```bash
# Linux/Mac
IMAGE_B64=$(base64 -w 0 assets/images/test_image.jpg)

# Or if image doesn't exist, create a test image first:
python3 -c "
from PIL import Image
import os
os.makedirs('assets/images', exist_ok=True)
img = Image.new('RGB', (100, 100), color='red')
img.save('assets/images/test_image.jpg')
print('Test image created')
"
IMAGE_B64=$(base64 -w 0 assets/images/test_image.jpg)
```

### Step 2: Make the request
```bash
curl -X POST https://aryan-don357--sam3-agent-sam3-segment.modal.run \
  -H "Content-Type: application/json" \
  -d "{
    \"prompt\": \"segment all objects\",
    \"image_b64\": \"$IMAGE_B64\",
    \"llm_config\": {
      \"base_url\": \"https://api.openai.com/v1\",
      \"model\": \"gpt-4o\",
      \"api_key\": \"sk-YOUR-OPENAI-API-KEY-HERE\",
      \"name\": \"openai-gpt4o\",
      \"max_tokens\": 4096
    },
    \"debug\": true
  }"
```

## Test 3: One-Liner with Environment Variable

```bash
# Set your API key as environment variable
export OPENAI_API_KEY="sk-YOUR-OPENAI-API-KEY-HERE"

# Encode image
IMAGE_B64=$(base64 -w 0 assets/images/test_image.jpg)

# Make request
curl -X POST https://aryan-don357--sam3-agent-sam3-segment.modal.run \
  -H "Content-Type: application/json" \
  -d "{
    \"prompt\": \"find and segment all red objects\",
    \"image_b64\": \"$IMAGE_B64\",
    \"llm_config\": {
      \"base_url\": \"https://api.openai.com/v1\",
      \"model\": \"gpt-4o\",
      \"api_key\": \"$OPENAI_API_KEY\",
      \"name\": \"openai-gpt4o\"
    },
    \"debug\": true
  }" | jq '.'
```

## Test 4: Save Response to File

```bash
export OPENAI_API_KEY="sk-YOUR-OPENAI-API-KEY-HERE"
IMAGE_B64=$(base64 -w 0 assets/images/test_image.jpg)

curl -X POST https://aryan-don357--sam3-agent-sam3-segment.modal.run \
  -H "Content-Type: application/json" \
  -d "{
    \"prompt\": \"segment all objects\",
    \"image_b64\": \"$IMAGE_B64\",
    \"llm_config\": {
      \"base_url\": \"https://api.openai.com/v1\",
      \"model\": \"gpt-4o\",
      \"api_key\": \"$OPENAI_API_KEY\",
      \"name\": \"openai-gpt4o\"
    },
    \"debug\": true
  }" \
  -o response.json

# View response
cat response.json | jq '.'

# Extract and save debug image
cat response.json | jq -r '.debug_image_b64' | base64 -d > debug_output.png
```

## Test 5: Minimal Request (No Debug)

```bash
export OPENAI_API_KEY="sk-YOUR-OPENAI-API-KEY-HERE"

curl -X POST https://aryan-don357--sam3-agent-sam3-segment.modal.run \
  -H "Content-Type: application/json" \
  -d "{
    \"prompt\": \"segment all objects\",
    \"image_url\": \"https://images.unsplash.com/photo-1506905925346-21bda4d32df4\",
    \"llm_config\": {
      \"base_url\": \"https://api.openai.com/v1\",
      \"model\": \"gpt-4o\",
      \"api_key\": \"$OPENAI_API_KEY\"
    }
  }"
```

## Test 6: Using a JSON File

### Create request file: `request.json`
```json
{
  "prompt": "segment all objects in the image",
  "image_url": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4",
  "llm_config": {
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-4o",
    "api_key": "sk-YOUR-OPENAI-API-KEY-HERE",
    "name": "openai-gpt4o",
    "max_tokens": 4096
  },
  "debug": true
}
```

### Make request
```bash
curl -X POST https://aryan-don357--sam3-agent-sam3-segment.modal.run \
  -H "Content-Type: application/json" \
  -d @request.json
```

## Test 7: Verbose Output (Debug Request)

```bash
export OPENAI_API_KEY="sk-YOUR-OPENAI-API-KEY-HERE"

curl -v -X POST https://aryan-don357--sam3-agent-sam3-segment.modal.run \
  -H "Content-Type: application/json" \
  -d "{
    \"prompt\": \"segment all objects\",
    \"image_url\": \"https://images.unsplash.com/photo-1506905925346-21bda4d32df4\",
    \"llm_config\": {
      \"base_url\": \"https://api.openai.com/v1\",
      \"model\": \"gpt-4o\",
      \"api_key\": \"$OPENAI_API_KEY\"
    },
    \"debug\": true
  }" 2>&1 | tee curl_output.log
```

## Expected Response Format

### Success Response
```json
{
  "status": "success",
  "summary": "SAM3 returned 3 regions for prompt: segment all objects",
  "regions": [
    {
      "bbox": [0.1, 0.2, 0.3, 0.4],
      "mask": {
        "counts": "1 2 3 4...",
        "size": [1024, 768]
      },
      "score": 0.95
    }
  ],
  "debug_image_b64": "base64-encoded-image...",
  "raw_sam3_json": {
    "orig_img_h": 1024,
    "orig_img_w": 768,
    "pred_boxes": [[...], [...]],
    "pred_masks": [
      {"counts": "...", "size": [1024, 768]}
    ],
    "pred_scores": [0.95, 0.87]
  },
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
  "message": "Missing 'llm_config' in request body. Provide complete LLM configuration with 'base_url', 'model', and 'api_key'."
}
```

## Troubleshooting

### Error: "Missing 'llm_config'"
- ✅ Ensure `llm_config` is included in request body
- ✅ Check JSON syntax is valid

### Error: "Invalid base64"
- ✅ Ensure base64 string is properly encoded
- ✅ Use `base64 -w 0` to avoid line breaks

### Error: "Failed to download 'image_url'"
- ✅ Check URL is accessible
- ✅ Use `image_b64` instead

### Error: Timeout
- ✅ First request takes 30-60 seconds (cold start)
- ✅ Wait and retry

### Error: "Invalid API key"
- ✅ Check OpenAI API key is correct
- ✅ Ensure key has sufficient credits

## Quick Test Script

Save as `test_curl.sh`:

```bash
#!/bin/bash

# Configuration
ENDPOINT_URL="https://aryan-don357--sam3-agent-sam3-segment.modal.run"
OPENAI_API_KEY="${OPENAI_API_KEY:-sk-YOUR-OPENAI-API-KEY-HERE}"
IMAGE_URL="https://images.unsplash.com/photo-1506905925346-21bda4d32df4"
PROMPT="segment all objects"

# Make request
echo "🚀 Testing SAM3 Agent endpoint..."
echo "📝 Prompt: $PROMPT"
echo "🖼️  Image: $IMAGE_URL"
echo ""

curl -X POST "$ENDPOINT_URL" \
  -H "Content-Type: application/json" \
  -d "{
    \"prompt\": \"$PROMPT\",
    \"image_url\": \"$IMAGE_URL\",
    \"llm_config\": {
      \"base_url\": \"https://api.openai.com/v1\",
      \"model\": \"gpt-4o\",
      \"api_key\": \"$OPENAI_API_KEY\",
      \"name\": \"openai-gpt4o\"
    },
    \"debug\": true
  }" | jq '.'

echo ""
echo "✅ Test complete!"
```

Make executable and run:
```bash
chmod +x test_curl.sh
./test_curl.sh
```

## Notes

- **First Request:** Takes 30-60 seconds (cold start - model loading)
- **Subsequent Requests:** 10-30 seconds (warm container)
- **Timeout:** 600 seconds maximum
- **Rate Limits:** No built-in rate limiting (Modal handles scaling)
- **Costs:** OpenAI API costs apply per request

---

**Ready to test!** Replace `sk-YOUR-OPENAI-API-KEY-HERE` with your actual OpenAI API key.

