# Quick Start Guide - SAM3 Agent with OpenAI

## 🚀 5-Minute Setup

### Step 1: Create OpenAI API Key Secret

```bash
modal secret create openai-api-key OPENAI_API_KEY=sk-your-actual-openai-key-here
```

**Get your OpenAI key:** https://platform.openai.com/api-keys

### Step 2: Update modal_agent.py

Uncomment the OpenAI secret in `modal_agent.py` (line ~157):

```python
secrets=[
    # modal.Secret.from_name("hf-token"),  # Optional
    modal.Secret.from_name("openai-api-key"),  # Uncomment this line
],
```

### Step 3: Deploy

```bash
modal deploy modal_agent.py
```

Copy the endpoint URL from the output (looks like `https://username--sam3-agent-sam3-segment.modal.run`)

### Step 4: Test

```bash
# Quick test
python test_deployment.py --endpoint-url <your-endpoint-url>

# Or use the example script
python example_usage.py --endpoint-url <your-endpoint-url>
```

## 📝 Example Request

```python
import requests
import base64

# Encode your image
with open("assets/images/test_image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

# Make request
response = requests.post(
    "https://your-username--sam3-agent-sam3-segment.modal.run",
    json={
        "prompt": "segment all objects",
        "image_b64": image_b64,
        "debug": True,
        "llm_profile": "openai-gpt4o"
    }
)

print(response.json())
```

## ✅ That's it!

See `DEPLOYMENT_GUIDE.md` for detailed troubleshooting and advanced usage.

