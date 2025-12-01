# SAM3 Agent Deployment Guide - Step by Step with OpenAI

This guide walks you through deploying and testing the SAM3 Agent on Modal with OpenAI API keys.

## Prerequisites

1. **Modal Account**: Sign up at https://modal.com
2. **OpenAI API Key**: Get from https://platform.openai.com/api-keys
3. **HuggingFace Token** (optional): For private models, get from https://huggingface.co/settings/tokens

## Step 1: Set Modal Token Credentials

```bash
# Set your Modal token (you should have already done this)
modal token set --token-id <your-token-id> --token-secret <your-token-secret>
```

Verify it works:
```bash
modal app list
```

## Step 2: Create Required Secrets in Modal

### 2a. Create OpenAI API Key Secret (Required for LLM)

```bash
modal secret create openai-api-key OPENAI_API_KEY=sk-your-openai-api-key-here
```

Replace `sk-your-openai-api-key-here` with your actual OpenAI API key.

**To get your OpenAI API key:**
1. Go to https://platform.openai.com/api-keys
2. Click "Create new secret key"
3. Copy the key (it starts with `sk-`)
4. Use it in the command above

### 2b. Create HuggingFace Token Secret (Optional but Recommended)

If SAM3 models are gated, you'll need a HuggingFace token:

```bash
modal secret create hf-token HF_TOKEN=hf_your-huggingface-token-here
```

**To get your HuggingFace token:**
1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Copy the token (it starts with `hf_`)
4. Use it in the command above

### 2c. Verify Secrets Created

```bash
modal secret list
```

You should see:
- `openai-api-key` (required)
- `hf-token` (optional)

## Step 3: Update modal_agent.py to Use Secrets

Make sure your `modal_agent.py` has the secrets configured. Check lines 154-158:

```python
secrets=[
    modal.Secret.from_name("hf-token"),  # Optional - uncomment if you have it
    modal.Secret.from_name("openai-api-key"),  # Required for OpenAI
],
```

If you don't have HF token, comment it out:
```python
secrets=[
    # modal.Secret.from_name("hf-token"),  # Optional
    modal.Secret.from_name("openai-api-key"),  # Required
],
```

## Step 4: Deploy the Application

```bash
cd /home/bakasur/sam3
modal deploy modal_agent.py
```

This will:
- Build the Docker image with all dependencies
- Upload your SAM3 code and assets
- Deploy the endpoint

**Expected output:**
```
✓ Created objects.
├── 🔨 Created function SAM3Model.*.
└── 🔨 Created web function sam3_segment => 
    https://your-username--sam3-agent-sam3-segment.modal.run
```

**Note the endpoint URL** - you'll need it for testing!

## Step 5: Test the Deployment

### 5a. Quick Test with test_deployment.py

```bash
python test_deployment.py --endpoint-url https://your-username--sam3-agent-sam3-segment.modal.run
```

### 5b. Full Test Suite with test_endpoints.py

```bash
python test_endpoints.py --endpoint-url https://your-username--sam3-agent-sam3-segment.modal.run
```

### 5c. Manual Test with curl

```bash
# First, encode an image to base64
IMAGE_B64=$(base64 -w 0 assets/images/test_image.jpg)

# Make the request
curl -X POST https://your-username--sam3-agent-sam3-segment.modal.run \
  -H "Content-Type: application/json" \
  -d "{
    \"prompt\": \"segment all objects\",
    \"image_b64\": \"$IMAGE_B64\",
    \"debug\": true,
    \"llm_profile\": \"openai-gpt4o\"
  }"
```

### 5d. Python Example

Create a file `example_usage.py`:

```python
import requests
import base64
import json

# Your endpoint URL
ENDPOINT_URL = "https://your-username--sam3-agent-sam3-segment.modal.run"

# Load and encode image
with open("assets/images/test_image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

# Make request
response = requests.post(
    ENDPOINT_URL,
    json={
        "prompt": "segment all objects",
        "image_b64": image_b64,
        "debug": True,
        "llm_profile": "openai-gpt4o"
    },
    timeout=600
)

result = response.json()
print(json.dumps(result, indent=2))

if result.get("status") == "success":
    print(f"\n✓ Success! Found {len(result.get('regions', []))} regions")
    if result.get("debug_image_b64"):
        # Save debug image
        with open("output_debug.png", "wb") as f:
            f.write(base64.b64decode(result["debug_image_b64"]))
        print("✓ Saved debug image to output_debug.png")
else:
    print(f"✗ Error: {result.get('message')}")
```

Run it:
```bash
python example_usage.py
```

## Step 6: Using the Endpoint

### Request Format

**POST** `https://your-username--sam3-agent-sam3-segment.modal.run`

**Headers:**
```
Content-Type: application/json
```

**Body (JSON):**
```json
{
  "prompt": "segment all ships",
  "image_url": "https://example.com/image.jpg",  // OR
  "image_b64": "base64-encoded-image-string",    // Use one of these
  "debug": true,                                 // Optional: get visualization
  "llm_profile": "openai-gpt4o"                 // Optional: default is openai-gpt4o
}
```

### Response Format

**Success:**
```json
{
  "status": "success",
  "summary": "SAM3 returned 3 regions for prompt: segment all ships",
  "regions": [
    {
      "bbox": [x, y, w, h],
      "mask": {...},
      "score": 0.95
    }
  ],
  "debug_image_b64": "base64-encoded-image",  // Only if debug=true
  "raw_sam3_json": {...},
  "llm_profile": "openai-gpt4o"
}
```

**Error:**
```json
{
  "status": "error",
  "message": "Error description",
  "traceback": "...",  // Only in debug mode
  "llm_profile": "openai-gpt4o"
}
```

## Troubleshooting

### Error: "Missing OPENAI_API_KEY"

**Solution:** Create the secret:
```bash
modal secret create openai-api-key OPENAI_API_KEY=sk-your-key-here
```

Then redeploy:
```bash
modal deploy modal_agent.py
```

### Error: "Failed to download image_url"

**Solution:** 
- Check the URL is accessible
- Try using `image_b64` instead
- Ensure the URL returns an image (not HTML)

### Error: "Unknown llm_profile"

**Solution:** Use one of the supported profiles:
- `openai-gpt4o` (default)
- `vllm-local` (if you have vLLM server)
- `vllm-modal` (if you deployed vLLM on Modal)

### Error: Timeout

**Solution:**
- First request takes longer (model loading)
- Wait 2-3 minutes for first request
- Subsequent requests are faster

### Check Deployment Status

```bash
modal app list
modal app logs sam3-agent
```

## Example Prompts

Try these prompts to test different capabilities:

```python
# Simple object detection
"segment all objects"

# Complex queries (like notebook example)
"the leftmost child wearing blue vest"

# Specific objects
"find all cars"
"segment all people"
"detect all animals"

# Spatial relationships
"the largest object"
"the object on the right"
"objects in the center"
```

## Cost Estimation

- **Modal**: Pay per GPU hour (A100 ~$2-3/hour)
- **OpenAI**: Pay per API call (GPT-4o ~$0.01-0.05 per request)
- **First request**: ~2-3 minutes (model loading)
- **Subsequent requests**: ~10-30 seconds

## Next Steps

1. **Add more LLM profiles**: Edit `LLM_PROFILES` in `modal_agent.py`
2. **Use vLLM**: Deploy vLLM separately and add profile
3. **Batch processing**: Modify endpoint to accept multiple images
4. **Custom domains**: Add custom domain in Modal dashboard

## Support

- Modal Docs: https://modal.com/docs
- SAM3 Repo: https://github.com/facebookresearch/sam3
- Issues: Check Modal dashboard logs for detailed errors

