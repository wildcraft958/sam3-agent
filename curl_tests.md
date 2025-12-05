# SAM3 Modal Endpoint - Curl Test Commands

## Your Deployment URLs

SAM3 Endpoints (workspace: `srinjoy59`):
- Count: `https://srinjoy59--sam3-agent-pyramidal-sam3-count.modal.run`
- Area: `https://srinjoy59--sam3-agent-pyramidal-sam3-area.modal.run`
- Segment: `https://srinjoy59--sam3-agent-pyramidal-sam3-segment.modal.run`

vLLM Endpoint:
- `https://srinjoy59--qwen3-vl-vllm-server-30b-vllm-server.modal.run/v1`
- Model: `Qwen/Qwen3-VL-30B-A3B-Instruct`

---

## 1. Count Objects (`/sam3/count`)

### Basic count with image URL:
```bash
curl -X POST "https://srinjoy59--sam3-agent-pyramidal-sam3-count.modal.run" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "trees",
    "image_url": "https://example.com/aerial-image.jpg",
    "confidence_threshold": 0.5
  }'
```

### Count with custom pyramidal config:
```bash
curl -X POST "https://srinjoy59--sam3-agent-pyramidal-sam3-count.modal.run" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "cars",
    "image_url": "https://example.com/parking-lot.jpg",
    "confidence_threshold": 0.4,
    "pyramidal_config": {
      "tile_size": 512,
      "overlap_ratio": 0.15,
      "scales": [1.0, 0.5],
      "batch_size": 16,
      "iou_threshold": 0.5
    }
  }'
```

### Count with base64 image:
```bash
# First encode your image to base64
IMAGE_B64=$(base64 -w0 /path/to/image.jpg)

curl -X POST "https://srinjoy59--sam3-agent-pyramidal-sam3-count.modal.run" \
  -H "Content-Type: application/json" \
  -d "{
    \"prompt\": \"buildings\",
    \"image_b64\": \"$IMAGE_B64\",
    \"confidence_threshold\": 0.5
  }"
```

### Expected Response:
```json
{
  "status": "success",
  "count": 47,
  "object_type": "tree",
  "confidence_summary": {
    "high": 35,
    "medium": 10,
    "low": 2
  },
  "detections": [
    {
      "box": [100.5, 200.3, 150.2, 280.1],
      "mask_rle": {"counts": "...", "size": [1024, 1024]},
      "score": 0.92,
      "scale": 1.0,
      "box_area_pixels": 3948
    }
  ],
  "pyramidal_stats": {
    "scales": [1.0, 0.5],
    "tile_size": 512,
    "total_tiles": 12,
    "successful_tiles": 12
  },
  "orig_img_w": 1024,
  "orig_img_h": 1024
}
```

---

## 2. Calculate Areas (`/sam3/area`)

### Basic area calculation:
```bash
curl -X POST "https://srinjoy59--sam3-agent-pyramidal-sam3-area.modal.run" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "solar panels",
    "image_url": "https://example.com/rooftop.jpg",
    "confidence_threshold": 0.5
  }'
```

### Area with Ground Sample Distance (real-world area):
```bash
curl -X POST "https://srinjoy59--sam3-agent-pyramidal-sam3-area.modal.run" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "solar panels",
    "image_url": "https://example.com/satellite-image.jpg",
    "gsd": 0.5,
    "confidence_threshold": 0.4,
    "pyramidal_config": {
      "tile_size": 512,
      "overlap_ratio": 0.15,
      "scales": [1.0, 0.5],
      "batch_size": 16,
      "iou_threshold": 0.5
    }
  }'
```

### Expected Response:
```json
{
  "status": "success",
  "object_count": 12,
  "total_pixel_area": 125000,
  "total_real_area_m2": 31250.0,
  "gsd": 0.5,
  "coverage_percentage": 12.5,
  "individual_areas": [
    {
      "id": 1,
      "pixel_area": 5000,
      "real_area_m2": 1250.0,
      "score": 0.95,
      "box": [100.0, 200.0, 200.0, 300.0]
    }
  ],
  "pyramidal_stats": {
    "scales": [1.0, 0.5],
    "tile_size": 512,
    "total_tiles": 8
  },
  "orig_img_w": 1000,
  "orig_img_h": 1000
}
```

---

## 3. Full Segmentation with LLM (`/sam3/segment`)

### With Qwen3-VL-30B (vLLM deployment):
```bash
curl -X POST "https://srinjoy59--sam3-agent-pyramidal-sam3-segment.modal.run" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "segment all ships in the harbor",
    "image_url": "https://example.com/harbor.jpg",
    "llm_config": {
      "base_url": "https://srinjoy59--qwen3-vl-vllm-server-30b-vllm-server.modal.run/v1",
      "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
      "api_key": "",
      "name": "qwen3-vl-30b"
    },
    "debug": true,
    "confidence_threshold": 0.4
  }'
```

### With OpenAI (alternative):
```bash
curl -X POST "https://srinjoy59--sam3-agent-pyramidal-sam3-segment.modal.run" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "segment all buildings",
    "image_url": "https://example.com/city.jpg",
    "llm_config": {
      "base_url": "https://api.openai.com/v1",
      "model": "gpt-4o",
      "api_key": "sk-your-openai-api-key",
      "name": "openai-gpt4o",
      "max_tokens": 4096
    },
    "debug": false
  }'
```

### Expected Response:
```json
{
  "status": "success",
  "summary": "SAM3 returned 5 regions for prompt: segment all ships",
  "regions": [
    {
      "bbox": [0.15, 0.25, 0.12, 0.08],
      "mask": {"counts": "...", "size": [1024, 1024]},
      "score": 0.89
    }
  ],
  "debug_image_b64": "iVBORw0KGgo...",
  "raw_sam3_json": {
    "pred_boxes": [[0.15, 0.25, 0.12, 0.08]],
    "pred_masks": [{"counts": "...", "size": [1024, 1024]}],
    "pred_scores": [0.89],
    "orig_img_h": 1024,
    "orig_img_w": 1024
  },
  "agent_history": [...],
  "llm_config": {
    "name": "openai-gpt4o",
    "model": "gpt-4o",
    "base_url": "https://api.openai.com/v1"
  }
}
```

---

## One-liner Examples

### Quick count test:
```bash
curl -s -X POST "https://srinjoy59--sam3-agent-pyramidal-sam3-count.modal.run" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"people","image_url":"https://images.unsplash.com/photo-1517486808906-6ca8b3f04846?w=800"}' | jq '.count'
```

### Quick area test:
```bash
curl -s -X POST "https://srinjoy59--sam3-agent-pyramidal-sam3-area.modal.run" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"cars","image_url":"https://images.unsplash.com/photo-1506521781263-d8422e82f27a?w=800"}' | jq '.object_count, .total_pixel_area'
```

### Quick segment test (with Qwen3-VL):
```bash
curl -s -X POST "https://srinjoy59--sam3-agent-pyramidal-sam3-segment.modal.run" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt":"segment the main subject",
    "image_url":"https://images.unsplash.com/photo-1517486808906-6ca8b3f04846?w=800",
    "llm_config":{
      "base_url":"https://srinjoy59--qwen3-vl-vllm-server-30b-vllm-server.modal.run/v1",
      "model":"Qwen/Qwen3-VL-30B-A3B-Instruct",
      "api_key":"",
      "name":"qwen3-vl-30b"
    }
  }' | jq '.status, .summary'
```

---

## Troubleshooting

### Check deployment status:
```bash
modal app list
```

### View SAM3 logs:
```bash
modal app logs sam3-agent-pyramidal
```

### View vLLM logs:
```bash
modal app logs qwen3-vl-vllm-server-30b
```

### Common errors:
- **"Missing 'prompt'"**: Add the `prompt` field to your JSON body
- **"Missing 'llm_config'"**: Only `/sam3/segment` requires this; `/sam3/count` and `/sam3/area` don't
- **"Failed to download 'image_url'"**: Check that the image URL is publicly accessible
- **"Invalid base64"**: Ensure your base64 encoding is correct (use `base64 -w0` on Linux)
- **Timeout errors**: Large images may take longer; the endpoints have 900s timeout

