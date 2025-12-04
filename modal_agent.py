# modal_agent.py
#
# SAM3 Agent Microservice on Modal - LLM Provider Agnostic
#
# This microservice provides SAM3 agent functionality with complete LLM provider
# flexibility. All LLM configuration is passed in API requests - no hardcoded
# providers or required LLM secrets.
#
# HTTP API (after `modal deploy modal_agent.py`):
#
#   POST /sam3/segment
#   Body (JSON):
#   {
#     "prompt": "segment all ships",
#     "image_url": "https://example.com/image.jpg",   # or "image_b64": "..."
#     "llm_config": {                                 # Required: complete LLM config
#       "base_url": "https://api.openai.com/v1",      # Any OpenAI-compatible API
#       "model": "gpt-4o",                            # Any model name
#       "api_key": "sk-...",                          # API key (can be empty for some backends)
#       "name": "openai-gpt4o",                       # Optional: for output files
#       "max_tokens": 4096                            # Optional: default 4096
#     },
#     "debug": true                                   # Optional: get visualization
#   }
#
#   Response (JSON):
#   {
#     "status": "success",
#     "summary": "...",
#     "regions": [...],
#     "debug_image_b64": "...",      # only if debug=true
#     "raw_sam3_json": {...},
#     "llm_config": {...}
#   }
#
# Supports any OpenAI-compatible LLM provider:
#   - OpenAI (GPT-4o, GPT-4, etc.)
#   - Anthropic (Claude)
#   - vLLM servers
#   - Custom OpenAI-compatible APIs

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
# LLM configuration helpers
# ------------------------------------------------------------------------------

def validate_llm_config(llm_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize LLM config from request.
    
    Required fields:
    - base_url: API endpoint URL
    - model: Model name/identifier
    - api_key: API key (can be empty for some backends)
    
    Optional fields:
    - name: Name for output files (defaults to model name)
    - provider: Provider type (defaults to "openai-compatible")
    - max_tokens: Maximum tokens (defaults to 4096)
    """
    if not isinstance(llm_config, dict):
        raise ValueError("llm_config must be a dictionary")
    
    # Required fields
    if "base_url" not in llm_config:
        raise ValueError("llm_config must include 'base_url'")
    if "model" not in llm_config:
        raise ValueError("llm_config must include 'model'")
    
    # Set defaults
    normalized = {
        "base_url": str(llm_config["base_url"]),
        "model": str(llm_config["model"]),
        "api_key": str(llm_config.get("api_key", "")),
        "name": llm_config.get("name", llm_config["model"]),  # Default to model name
        "provider": llm_config.get("provider", "openai-compatible"),
        "max_tokens": int(llm_config.get("max_tokens", 4096)),
    }
    
    return normalized


# ------------------------------------------------------------------------------
# GPU-backed SAM3 model class
# ------------------------------------------------------------------------------

@app.cls(
    gpu="A100",
    timeout=600,
    image=image,
    scaledown_window=3600,  # Keep container alive for 1 hour after last request
    min_containers=1,  # Keep at least 1 container always running (always loaded)
    # Required secrets - SAM3 is a gated repository on HuggingFace:
    #   - "hf-token" containing key HF_TOKEN (REQUIRED - SAM3 model is gated)
    # To add secret: modal secret create hf-token HF_TOKEN=<your-token>
    # Note: LLM configuration is passed in API requests, no LLM secrets needed
    secrets=[
        modal.Secret.from_name("huggingface-secret"),  # REQUIRED - SAM3 is a gated repo
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

        # HF_TOKEN is REQUIRED - SAM3 is a gated repository
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError(
                "HF_TOKEN environment variable is required but not set. "
                "SAM3 is a gated repository on HuggingFace. "
                "Please create the Modal secret: modal secret create hf-token HF_TOKEN=<your-token>"
            )
        
        # Authenticate with HuggingFace before attempting to load model
        try:
            login(token=hf_token)
            print("✓ Authenticated with HuggingFace")
        except Exception as e:
            raise ValueError(
                f"Failed to authenticate with HuggingFace: {e}. "
                "Please verify your HF_TOKEN is valid and has access to facebook/sam3 repository."
            ) from e

        print("Loading SAM3 model...")
        bpe_path = "/root/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
        self.model = build_sam3_image_model(bpe_path=bpe_path)
        # Use lower confidence threshold (0.25) to get more results, can be adjusted per request
        self.processor = Sam3Processor(self.model, confidence_threshold=0.4)
        print(f"SAM3 model loaded successfully with confidence threshold: {self.processor.confidence_threshold}")

    @modal.method()
    def sam3_infer_only(
        self,
        image_bytes: bytes,
        text_prompt: str,
        confidence_threshold: float = None,
        pyramidal_config: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        SAM3 inference using pyramidal batch processing.
        Returns the same format as original sam3_inference function.
        
        Args:
            image_bytes: Raw image bytes
            text_prompt: Text prompt for segmentation
            confidence_threshold: Optional confidence threshold (0.0-1.0)
            pyramidal_config: Optional pyramidal configuration dict
        """
        # Set confidence threshold
        if confidence_threshold is not None:
            if not 0.0 <= confidence_threshold <= 1.0:
                return {
                    "status": "error",
                    "message": f"confidence_threshold must be between 0.0 and 1.0, got {confidence_threshold}"
                }
            self.processor.confidence_threshold = confidence_threshold
            print(f"✓ Using confidence threshold: {confidence_threshold}")
        
        # Set default pyramidal config
        config = {
            "tile_size": 512,
            "overlap_ratio": 0.15,
            "scales": [1.0, 0.5],
            "batch_size": 16,
            "iou_threshold": 0.5,
        }
        if pyramidal_config:
            config.update(pyramidal_config)
        
        # Run pyramidal inference
        result = self.sam3_pyramidal_infer(
            image_bytes=image_bytes,
            text_prompt=text_prompt,
            tile_size=config["tile_size"],
            overlap_ratio=config["overlap_ratio"],
            scales=config["scales"],
            iou_threshold=config["iou_threshold"],
            confidence_threshold=self.processor.confidence_threshold,
            batch_size=config["batch_size"],
        )
        
        if result["status"] != "success":
            return result
        
        # Convert to expected output format (normalized boxes in xywh)
        orig_w = result["orig_img_w"]
        orig_h = result["orig_img_h"]
        detections = result["detections"]
        
        pred_boxes = []
        pred_masks = []
        pred_scores = []
        
        for det in detections:
            box = det["box"]  # [x1, y1, x2, y2] in pixels
            x1, y1, x2, y2 = box
            # Normalize to [0, 1] and convert to xywh (center x, center y, width, height)
            cx = ((x1 + x2) / 2) / orig_w
            cy = ((y1 + y2) / 2) / orig_h
            w = (x2 - x1) / orig_w
            h = (y2 - y1) / orig_h
            pred_boxes.append([cx, cy, w, h])
            pred_masks.append(det["mask_rle"])
            pred_scores.append(det["score"])
        
        return {
            "status": "success",
            "orig_img_h": orig_h,
            "orig_img_w": orig_w,
            "pred_boxes": pred_boxes,
            "pred_masks": pred_masks,
            "pred_scores": pred_scores,
            "pyramidal_stats": result.get("pyramidal_stats", {}),
        }

    # --------------------------------------------------------------------------
    # Pyramidal Tiling Helper Methods
    # --------------------------------------------------------------------------

    def _create_tiles(self, image, tile_size: int, overlap_ratio: float):
        """
        Generate overlapping tiles from PIL image.
        
        Args:
            image: PIL Image
            tile_size: Size of each tile (e.g., 512)
            overlap_ratio: Overlap between tiles (e.g., 0.15)
            
        Returns:
            List of (tile_image, (offset_x, offset_y))
        """
        img_width, img_height = image.size
        stride = int(tile_size * (1 - overlap_ratio))
        tiles = []
        
        # If image is smaller than tile size, return as single tile
        if img_width <= tile_size and img_height <= tile_size:
            return [(image, (0, 0))]
        
        for y in range(0, img_height, stride):
            for x in range(0, img_width, stride):
                x_end = min(x + tile_size, img_width)
                y_end = min(y + tile_size, img_height)
                x_start = max(0, x_end - tile_size)
                y_start = max(0, y_end - tile_size)
                
                tile = image.crop((x_start, y_start, x_end, y_end))
                tiles.append((tile, (x_start, y_start)))
                
                if x_end >= img_width:
                    break
            if y_end >= img_height:
                break
        
        return tiles

    def _transform_box_to_original(self, box, tile_offset, scale: float, orig_size):
        """
        Transform box from tile coordinates to original image space.
        
        Args:
            box: Box coordinates [x1, y1, x2, y2] in tile space
            tile_offset: (offset_x, offset_y) of tile in scaled image
            scale: Scale factor used for this pyramid level
            orig_size: (width, height) of original image
            
        Returns:
            Transformed box in original image coordinates
        """
        import numpy as np
        import torch
        
        offset_x, offset_y = tile_offset
        orig_w, orig_h = orig_size
        
        # Convert tensor to numpy if needed
        if torch.is_tensor(box):
            box = box.cpu().numpy()
        
        box = np.array(box).copy()
        
        # Add offset (tile position in scaled image)
        box[0] += offset_x
        box[1] += offset_y
        box[2] += offset_x
        box[3] += offset_y
        
        # Scale back to original resolution
        box = box / scale
        
        # Clip to image bounds
        box[0] = max(0, min(box[0], orig_w))
        box[1] = max(0, min(box[1], orig_h))
        box[2] = max(0, min(box[2], orig_w))
        box[3] = max(0, min(box[3], orig_h))
        
        return box

    def _calculate_iou(self, box1, box2) -> float:
        """
        Calculate Intersection over Union between two boxes.
        
        Args:
            box1, box2: Boxes in format [x1, y1, x2, y2]
            
        Returns:
            IoU value (0-1)
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0

    def _apply_nms(self, detections, iou_threshold: float):
        """
        Apply Non-Maximum Suppression to remove duplicate detections.
        Prefers higher scores, then finer scales.
        
        Args:
            detections: List of detection dicts with 'box', 'score', 'scale', 'mask'
            iou_threshold: IoU threshold for suppression
            
        Returns:
            Filtered list of detections
        """
        if not detections:
            return []
        
        # Sort by score (descending), then by scale (ascending = finer first)
        detections = sorted(
            detections, 
            key=lambda d: (-d['score'], d.get('scale', 1.0))
        )
        
        keep = []
        while detections:
            current = detections.pop(0)
            keep.append(current)
            detections = [
                d for d in detections
                if self._calculate_iou(current['box'], d['box']) < iou_threshold
            ]
        
        return keep

    def _extract_tile_backbone(self, backbone_out, tile_idx: int, batch_size: int):
        """
        Extract single tile's backbone features from batched output.
        
        Args:
            backbone_out: Batched backbone output dictionary
            tile_idx: Index of the tile to extract
            batch_size: Total batch size
            
        Returns:
            Dictionary with single-tile backbone features
        """
        import torch
        
        extracted = {}
        
        for key, value in backbone_out.items():
            if isinstance(value, torch.Tensor):
                # Extract single tile from batch dimension
                if value.shape[0] == batch_size:
                    extracted[key] = value[tile_idx:tile_idx+1]
                else:
                    extracted[key] = value
            elif isinstance(value, dict):
                # Recursively extract from nested dicts
                extracted[key] = self._extract_tile_backbone(value, tile_idx, batch_size)
            elif isinstance(value, list) and len(value) == batch_size:
                # Handle lists that match batch size
                extracted[key] = [value[tile_idx]]
            else:
                extracted[key] = value
        
        return extracted

    @modal.method()
    def sam3_pyramidal_infer(
        self,
        image_bytes: bytes,
        text_prompt: str,
        tile_size: int = 512,
        overlap_ratio: float = 0.15,
        scales: list = None,
        iou_threshold: float = 0.5,
        confidence_threshold: float = 0.5,
        batch_size: int = 16,
    ) -> Dict[str, Any]:
        """
        Pyramidal batch inference with text encoding cache.
        
        Optimizations:
        1. Text encoded ONCE via self.processor.model.backbone.forward_text()
        2. Images batch-encoded via self.processor.set_image_batch()
        3. Cached text features injected into each tile state
        4. GPU-accelerated NMS via torchvision.ops.batched_nms
        
        Args:
            image_bytes: Raw image bytes
            text_prompt: Text prompt for segmentation
            tile_size: Size of each tile (default: 512)
            overlap_ratio: Overlap between tiles (default: 0.15)
            scales: List of scales for pyramid (default: [1.0, 0.5])
            iou_threshold: IoU threshold for NMS (default: 0.5)
            confidence_threshold: Minimum confidence threshold (default: 0.5)
            batch_size: Batch size for processing tiles (default: 16)
            
        Returns:
            Dict with detections, count, and processing stats
        """
        from PIL import Image
        import io
        import torch
        import numpy as np
        from sam3.train.masks_ops import rle_encode
        
        # Set defaults
        scales = sorted(scales or [1.0, 0.5], reverse=True)
        
        # Set processor confidence threshold
        original_threshold = self.processor.confidence_threshold
        self.processor.confidence_threshold = confidence_threshold
        
        # Load image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        orig_w, orig_h = image.size
        print(f"✓ Image loaded: {orig_w}x{orig_h}")
        
        # ================================================================
        # OPTIMIZATION 1: Encode text ONCE
        # ================================================================
        text_outputs = self.processor.model.backbone.forward_text(
            [text_prompt],
            device=self.processor.device
        )
        print(f"✓ Text encoded once (cached for all tiles)")
        
        all_detections = []
        total_tiles = 0
        stats = {
            "scales": scales,
            "tile_size": tile_size,
            "overlap_ratio": overlap_ratio,
            "tiles_per_scale": {},
        }
        
        for scale in scales:
            # Scale image
            if scale != 1.0:
                scaled_w = int(orig_w * scale)
                scaled_h = int(orig_h * scale)
                scaled_image = image.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)
            else:
                scaled_image = image
            
            # Generate tiles for this scale
            tiles_with_offsets = self._create_tiles(scaled_image, tile_size, overlap_ratio)
            stats["tiles_per_scale"][str(scale)] = len(tiles_with_offsets)
            total_tiles += len(tiles_with_offsets)
            
            print(f"  Scale {scale}: {len(tiles_with_offsets)} tiles")
            
            # Process tiles in batches
            for batch_start in range(0, len(tiles_with_offsets), batch_size):
                batch_end = min(batch_start + batch_size, len(tiles_with_offsets))
                batch = tiles_with_offsets[batch_start:batch_end]
                
                tile_images = [t[0] for t in batch]
                tile_offsets = [t[1] for t in batch]
                
                # ================================================================
                # OPTIMIZATION 2: Batch encode images
                # ================================================================
                batch_state = self.processor.set_image_batch(tile_images)
                actual_batch_size = len(tile_images)
                
                for i in range(len(tile_images)):
                    try:
                        # Extract per-tile backbone
                        tile_state = {
                            'original_height': batch_state['original_heights'][i],
                            'original_width': batch_state['original_widths'][i],
                            'backbone_out': self._extract_tile_backbone(
                                batch_state['backbone_out'], i, actual_batch_size
                            )
                        }
                        
                        # ================================================================
                        # OPTIMIZATION 1 (cont): Inject cached text features
                        # ================================================================
                        tile_state['backbone_out'].update({
                            'language_features': text_outputs['language_features'],
                            'language_mask': text_outputs['language_mask'],
                            'language_embeds': text_outputs['language_embeds'],
                        })
                        
                        # Initialize geometric prompt
                        if 'geometric_prompt' not in tile_state:
                            tile_state['geometric_prompt'] = self.processor.model._get_dummy_prompt()
                        
                        # Run grounding (text already embedded)
                        tile_state = self.processor._forward_grounding(tile_state)
                        
                        # Collect detections
                        if 'boxes' in tile_state and len(tile_state['boxes']) > 0:
                            boxes = tile_state['boxes'].cpu().numpy()
                            masks = tile_state['masks'].cpu()
                            scores = tile_state['scores'].cpu().numpy()
                            
                            for j in range(len(boxes)):
                                if scores[j] >= confidence_threshold:
                                    # Transform to original coords
                                    orig_box = self._transform_box_to_original(
                                        boxes[j], tile_offsets[i], scale, (orig_w, orig_h)
                                    )
                                    
                                    # Validate box is not degenerate after clipping
                                    if orig_box[2] <= orig_box[0] or orig_box[3] <= orig_box[1]:
                                        continue  # Skip degenerate boxes
                                    
                                    # Encode mask as RLE (at tile resolution)
                                    mask_binary = masks[j].squeeze().numpy() > 0.5
                                    mask_rle = rle_encode(torch.tensor(mask_binary).unsqueeze(0))[0]
                                    
                                    # Calculate box area in original image pixels
                                    box_area_pixels = int((orig_box[2] - orig_box[0]) * (orig_box[3] - orig_box[1]))
                                    
                                    all_detections.append({
                                        'box': orig_box.tolist(),
                                        'mask_rle': mask_rle,
                                        'score': float(scores[j]),
                                        'scale': scale,
                                        'box_area_pixels': box_area_pixels,  # For accurate area calculation
                                    })
                    except Exception as e:
                        print(f"⚠ Error processing tile {i} in batch: {e}")
                        continue
                
                # Free batch memory
                del batch_state
                if batch_start % (batch_size * 2) == 0:
                    torch.cuda.empty_cache()
        
        stats["total_tiles"] = total_tiles
        print(f"✓ Processed {total_tiles} tiles across {len(scales)} scales")
        print(f"  Raw detections: {len(all_detections)}")
        
        # Apply NMS
        final_detections = self._apply_nms(all_detections, iou_threshold)
        print(f"✓ After NMS: {len(final_detections)} detections")
        
        # Restore original confidence threshold
        self.processor.confidence_threshold = original_threshold
        
        return {
            "status": "success",
            "count": len(final_detections),
            "detections": final_detections,
            "orig_img_w": orig_w,
            "orig_img_h": orig_h,
            "pyramidal_stats": stats,
        }

    @modal.method()
    def sam3_count(
        self,
        image_bytes: bytes,
        text_prompt: str,
        confidence_threshold: float = 0.5,
        pyramidal_config: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Count objects using pyramidal SAM3 segmentation.
        
        Args:
            image_bytes: Raw image bytes
            text_prompt: Text prompt describing objects to count
            confidence_threshold: Minimum confidence threshold (default: 0.5)
            pyramidal_config: Optional pyramidal configuration:
                - tile_size: Size of each tile (default: 512)
                - overlap_ratio: Overlap between tiles (default: 0.15)
                - scales: List of scales (default: [1.0, 0.5])
                - batch_size: Batch size (default: 16)
                - iou_threshold: NMS IoU threshold (default: 0.5)
        
        Returns:
            Dict with count, confidence summary, detections, and processing stats
        """
        # Set default pyramidal config
        config = {
            "tile_size": 512,
            "overlap_ratio": 0.15,
            "scales": [1.0, 0.5],
            "batch_size": 16,
            "iou_threshold": 0.5,
        }
        if pyramidal_config:
            config.update(pyramidal_config)
        
        # Run pyramidal inference
        result = self.sam3_pyramidal_infer(
            image_bytes=image_bytes,
            text_prompt=text_prompt,
            tile_size=config["tile_size"],
            overlap_ratio=config["overlap_ratio"],
            scales=config["scales"],
            iou_threshold=config["iou_threshold"],
            confidence_threshold=confidence_threshold,
            batch_size=config["batch_size"],
        )
        
        if result["status"] != "success":
            return result
        
        # Calculate confidence breakdown
        detections = result["detections"]
        high_conf = sum(1 for d in detections if d["score"] >= 0.8)
        medium_conf = sum(1 for d in detections if 0.5 <= d["score"] < 0.8)
        low_conf = sum(1 for d in detections if d["score"] < 0.5)
        
        confidence_summary = {
            "high": high_conf,      # >= 0.8
            "medium": medium_conf,  # 0.5 - 0.8
            "low": low_conf,        # < 0.5
        }
        
        # Extract object type from prompt (simple heuristic)
        object_type = text_prompt.strip().lower()
        # Try to extract key noun (last word typically)
        words = object_type.split()
        if words:
            object_type = words[-1].rstrip('s')  # Remove trailing 's'
        
        return {
            "status": "success",
            "count": result["count"],
            "object_type": object_type,
            "confidence_summary": confidence_summary,
            "detections": detections,
            "pyramidal_stats": result["pyramidal_stats"],
            "orig_img_w": result["orig_img_w"],
            "orig_img_h": result["orig_img_h"],
        }

    @modal.method()
    def sam3_area(
        self,
        image_bytes: bytes,
        text_prompt: str,
        gsd: float = None,
        confidence_threshold: float = 0.5,
        pyramidal_config: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Calculate areas of detected objects using Pyramidal SAM3 segmentation.
        
        Args:
            image_bytes: Raw image bytes
            text_prompt: Text prompt describing objects to measure
            gsd: Ground Sample Distance in meters/pixel (optional)
                 If provided, calculates real-world area in square meters
            confidence_threshold: Minimum confidence threshold (default: 0.5)
            pyramidal_config: Optional pyramidal configuration:
                - tile_size: Size of each tile (default: 512)
                - overlap_ratio: Overlap between tiles (default: 0.15)
                - scales: List of scales (default: [1.0, 0.5])
                - batch_size: Batch size (default: 16)
                - iou_threshold: NMS IoU threshold (default: 0.5)
        
        Returns:
            Dict with object count, total area, coverage percentage, and per-object areas
        """
        import numpy as np
        from pycocotools import mask as mask_utils
        
        # Set default pyramidal config
        config = {
            "tile_size": 512,
            "overlap_ratio": 0.15,
            "scales": [1.0, 0.5],
            "batch_size": 16,
            "iou_threshold": 0.5,
        }
        if pyramidal_config:
            config.update(pyramidal_config)
        
        # Run pyramidal inference
        result = self.sam3_pyramidal_infer(
            image_bytes=image_bytes,
            text_prompt=text_prompt,
            tile_size=config["tile_size"],
            overlap_ratio=config["overlap_ratio"],
            scales=config["scales"],
            iou_threshold=config["iou_threshold"],
            confidence_threshold=confidence_threshold,
            batch_size=config["batch_size"],
        )
        
        if result["status"] != "success":
            return result
        
        detections = result["detections"]
        orig_w = result["orig_img_w"]
        orig_h = result["orig_img_h"]
        total_image_pixels = orig_w * orig_h
        
        # Calculate areas for each detection
        individual_areas = []
        total_pixel_area = 0
        
        for idx, det in enumerate(detections):
            mask_rle = det.get("mask_rle")
            if not mask_rle:
                continue
            
            try:
                # Decode RLE mask using pycocotools
                # Ensure mask_rle has the right format for pycocotools
                if isinstance(mask_rle, dict):
                    rle_for_decode = mask_rle
                else:
                    # If it's not a dict, skip
                    continue
                
                # Calculate pixel area from RLE
                pixel_area = int(mask_utils.area(rle_for_decode))
                total_pixel_area += pixel_area
                
                area_info = {
                    "id": idx + 1,
                    "pixel_area": pixel_area,
                    "score": det["score"],
                    "box": det["box"],
                }
                
                # Calculate real area if GSD is provided
                if gsd is not None and gsd > 0:
                    real_area_m2 = pixel_area * (gsd ** 2)
                    area_info["real_area_m2"] = round(real_area_m2, 4)
                
                individual_areas.append(area_info)
                
            except Exception as e:
                print(f"⚠ Error calculating area for detection {idx}: {e}")
                continue
        
        # Calculate coverage percentage
        coverage_percentage = (total_pixel_area / total_image_pixels * 100) if total_image_pixels > 0 else 0
        
        # Prepare response
        response = {
            "status": "success",
            "object_count": len(individual_areas),
            "total_pixel_area": total_pixel_area,
            "coverage_percentage": round(coverage_percentage, 4),
            "individual_areas": individual_areas,
            "orig_img_w": orig_w,
            "orig_img_h": orig_h,
            "pyramidal_stats": result["pyramidal_stats"],
        }
        
        # Add real-world area if GSD provided
        if gsd is not None and gsd > 0:
            total_real_area_m2 = total_pixel_area * (gsd ** 2)
            response["total_real_area_m2"] = round(total_real_area_m2, 4)
            response["gsd"] = gsd
        
        return response

    def call_sam_service_pyramidal(
        self,
        image_path: str,
        text_prompt: str,
        output_folder_path: str = "sam3_output",
        pyramidal_config: Dict[str, Any] = None,
    ) -> str:
        """
        Pyramidal version of call_sam_service - drop-in replacement.
        
        Uses batch pyramidal SAM3 instead of raw SAM3, then formats output
        to match the expected format for agent_inference.
        """
        import os as _os
        import json as _json
        
        # Set default pyramidal config
        config = {
            "tile_size": 512,
            "overlap_ratio": 0.15,
            "scales": [1.0, 0.5],
            "batch_size": 16,
            "iou_threshold": 0.5,
        }
        if pyramidal_config:
            config.update(pyramidal_config)
        
        print(f"📞 Loading image '{image_path}' for pyramidal segmentation with prompt '{text_prompt}'...")
        
        # Load image bytes
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        
        # Run pyramidal inference
        result = self.sam3_pyramidal_infer(
            image_bytes=image_bytes,
            text_prompt=text_prompt,
            tile_size=config["tile_size"],
            overlap_ratio=config["overlap_ratio"],
            scales=config["scales"],
            iou_threshold=config["iou_threshold"],
            confidence_threshold=self.processor.confidence_threshold,
            batch_size=config["batch_size"],
        )
        
        if result["status"] != "success":
            # Return empty results on error
            outputs = {
                "orig_img_h": 0,
                "orig_img_w": 0,
                "pred_boxes": [],
                "pred_masks": [],
                "pred_scores": [],
            }
        else:
            orig_w = result["orig_img_w"]
            orig_h = result["orig_img_h"]
            detections = result["detections"]
            
            # Convert detections to expected format (normalized boxes in xywh)
            pred_boxes = []
            pred_masks = []
            pred_scores = []
            
            for det in detections:
                box = det["box"]  # [x1, y1, x2, y2] in pixels
                # Normalize to [0, 1] and convert to xywh
                x1, y1, x2, y2 = box
                cx = ((x1 + x2) / 2) / orig_w
                cy = ((y1 + y2) / 2) / orig_h
                w = (x2 - x1) / orig_w
                h = (y2 - y1) / orig_h
                pred_boxes.append([cx, cy, w, h])
                
                # Mask is already in RLE format
                pred_masks.append(det["mask_rle"])
                pred_scores.append(det["score"])
            
            outputs = {
                "orig_img_h": orig_h,
                "orig_img_w": orig_w,
                "pred_boxes": pred_boxes,
                "pred_masks": pred_masks,
                "pred_scores": pred_scores,
                "pyramidal_stats": result.get("pyramidal_stats", {}),
            }
        
        # Save to JSON (same as original call_sam_service)
        text_prompt_for_save = text_prompt.replace("/", "_")
        image_basename = _os.path.basename(image_path)
        image_basename_no_ext = _os.path.splitext(image_basename)[0]
        safe_dir_name = image_basename_no_ext.replace("/", "_").replace("\\", "_")
        
        output_dir = _os.path.join(output_folder_path, safe_dir_name, text_prompt_for_save)
        _os.makedirs(output_dir, exist_ok=True)
        
        json_path = _os.path.join(output_dir, "sam3_output.json")
        with open(json_path, "w") as f:
            _json.dump(outputs, f, indent=2)
        
        print(f"✓ Pyramidal SAM3 found {len(outputs['pred_boxes'])} objects")
        
        return json_path

    @modal.method()
    def infer(
        self,
        image_bytes: bytes,
        prompt: str,
        llm_config: Dict[str, Any],
        debug: bool = False,
        confidence_threshold: float = None,
    ) -> Dict[str, Any]:
        """
        Core GPU inference method - LLM provider agnostic.
        
        This method accepts any LLM configuration and uses it to call the LLM API.
        No hardcoded provider assumptions - works with any OpenAI-compatible API.
        
        Args:
            image_bytes: raw bytes of the input image
            prompt: natural language query for segmentation
            llm_config: Complete LLM configuration dict (provider-agnostic):
                - base_url: API endpoint URL (required) - any OpenAI-compatible endpoint
                - model: Model name/identifier (required) - any model name
                - api_key: API key (required, can be empty string for backends without auth)
            - name: Name for output files (optional, defaults to model name)
            - max_tokens: Maximum tokens (optional, defaults to 4096)
            debug: whether to return a visualization image (base64)
            confidence_threshold: Optional confidence threshold (0.0-1.0). If None, uses processor's default (0.4)
        
        Returns:
            Dict with status, regions, summary, and optional debug visualization
        """
        try:
            from sam3.agent.client_llm import (
                send_generate_request as send_generate_request_orig,
            )
        except ImportError as e:
            return {
                "status": "error",
                "message": f"Failed to import SAM3 agent modules: {e}",
            }

        # Validate and normalize LLM config
        try:
            llm_config = validate_llm_config(llm_config)
        except Exception as e:
            return {"status": "error", "message": f"Invalid llm_config: {str(e)}"}

        # Set confidence threshold if provided
        if confidence_threshold is not None:
            if not 0.0 <= confidence_threshold <= 1.0:
                return {
                    "status": "error",
                    "message": f"confidence_threshold must be between 0.0 and 1.0, got {confidence_threshold}"
                }
            self.processor.confidence_threshold = confidence_threshold
            print(f"✓ Using confidence threshold: {confidence_threshold}")

        # Cap max_tokens to 4096 to allow longer reasoning outputs for 32B Thinking model
        requested_max_tokens = llm_config["max_tokens"]
        safe_max_tokens = min(requested_max_tokens, 4096)
        
        send_generate_request = partial(
            send_generate_request_orig,
            server_url=llm_config["base_url"],
            model=llm_config["model"],
            api_key=llm_config["api_key"],
            max_tokens=safe_max_tokens,
        )

        # Use pyramidal batch SAM3 instead of raw SAM3
        call_sam_service = partial(
            self.call_sam_service_pyramidal,
            pyramidal_config={
                "tile_size": 512,
                "overlap_ratio": 0.15,
                "scales": [1.0, 0.5],
                "batch_size": 16,
                "iou_threshold": 0.5,
            },
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, "input_image.jpg")
            with open(image_path, "wb") as f:
                f.write(image_bytes)

            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(output_dir, exist_ok=True)

            try:
                # Call agent_inference directly to get data from return values
                from sam3.agent.agent_core import agent_inference
                
                agent_history, final_output_dict, rendered_final_output = agent_inference(
                    image_path,
                    prompt,
                    send_generate_request=send_generate_request,
                    call_sam_service=call_sam_service,
                    debug=debug,
                    output_dir=output_dir,
                    max_generations=15,  # Limit LLM API calls to 7
                )

                if not final_output_dict:
                    return {
                        "status": "error",
                        "message": "No output generated by SAM3.",
                    }

                # Debug visualization image - convert PIL Image to base64
                debug_image_b64 = None
                if debug and rendered_final_output:
                    import io
                    buffer = io.BytesIO()
                    rendered_final_output.save(buffer, format="PNG")
                    debug_image_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")

                # Extract data from final_output_dict (which contains the SAM3 results)
                # The final_output_dict should have pred_boxes, pred_masks, pred_scores, etc.
                raw_json = final_output_dict.copy()
                
                # Remove file paths that aren't needed in response
                raw_json.pop("original_image_path", None)
                raw_json.pop("output_image_path", None)
                raw_json.pop("text_prompt", None)
                raw_json.pop("image_path", None)

                # Try to normalize "regions" field if present, otherwise construct from pred_boxes/pred_masks
                regions = (
                    raw_json.get("regions")
                    or raw_json.get("objects")
                    or raw_json.get("instances")
                    or []
                )
                
                # If no regions found, construct from pred_boxes and pred_masks
                if not regions and "pred_boxes" in raw_json and "pred_masks" in raw_json:
                    pred_boxes = raw_json.get("pred_boxes", [])
                    pred_masks = raw_json.get("pred_masks", [])
                    pred_scores = raw_json.get("pred_scores", [])
                    regions = [
                        {
                            "bbox": box,
                            "mask": mask,
                            "score": pred_scores[i] if i < len(pred_scores) else None,
                        }
                        for i, (box, mask) in enumerate(zip(pred_boxes, pred_masks))
                    ]

                summary = (
                    f"SAM3 returned {len(regions)} regions for prompt: {prompt}"
                )

                return {
                    "status": "success",
                    "summary": summary,
                    "regions": regions,
                    "debug_image_b64": debug_image_b64,
                    "raw_sam3_json": raw_json,
                    "agent_history": agent_history,  # Include agent history for debugging
                    "llm_config": {
                        "name": llm_config["name"],
                        "model": llm_config["model"],
                        "base_url": llm_config["base_url"],
                    },
                }

            except Exception as e:
                import traceback

                return {
                    "status": "error",
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                    "llm_config": {
                        "name": llm_config.get("name", "unknown"),
                        "model": llm_config.get("model", "unknown"),
                    },
                }


# ------------------------------------------------------------------------------
# HTTP endpoint: /sam3/infer (SAM3-only inference, no LLM/agent)
# ------------------------------------------------------------------------------

from modal import fastapi_endpoint  # lightweight JSON endpoint decorator


@app.function(timeout=900, image=image)
@fastapi_endpoint(method="POST")
def sam3_infer(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    HTTP endpoint for Pyramidal SAM3 inference (no LLM/agent logic):
    
    POST /sam3/infer
    JSON body:
    {
      "text_prompt": "...",
      "image_b64": "...",   # or "image_url": "https://..."
      "confidence_threshold": 0.5,  # Optional: confidence threshold (0.0-1.0)
      "pyramidal_config": {         # Optional: pyramidal configuration
        "tile_size": 512,
        "overlap_ratio": 0.15,
        "scales": [1.0, 0.5],
        "batch_size": 16,
        "iou_threshold": 0.5
      }
    }
    
    Returns:
    {
      "orig_img_h": int,
      "orig_img_w": int,
      "pred_boxes": [[x, y, w, h], ...],  # normalized [0, 1]
      "pred_masks": ["rle_dict", ...],
      "pred_scores": [float, ...],
      "pyramidal_stats": {...}
    }
    """
    # Basic validation
    if "text_prompt" not in body:
        return {"status": "error", "message": "Missing 'text_prompt' in request body."}
    
    text_prompt = body["text_prompt"]
    confidence_threshold = body.get("confidence_threshold")
    pyramidal_config = body.get("pyramidal_config")
    
    # Get image bytes from either image_b64 or image_url
    if "image_b64" in body:
        try:
            image_bytes = base64.b64decode(body["image_b64"])
            print(f"✓ Decoded image from base64 ({len(image_bytes)} bytes)")
        except Exception as e:
            return {"status": "error", "message": f"Invalid base64 in 'image_b64': {e}."}
    elif "image_url" in body:
        import requests
        try:
            print(f"📥 Downloading image from URL: {body['image_url']}")
            resp = requests.get(body["image_url"], timeout=30)
            resp.raise_for_status()
            image_bytes = resp.content
            print(f"✓ Downloaded image ({len(image_bytes)} bytes)")
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
    
    # Call the GPU-backed pyramidal SAM3 inference
    try:
        print(f"📞 Calling sam3_infer_only (pyramidal) with prompt: '{text_prompt}'")
        result = SAM3Model().sam3_infer_only.remote(
            image_bytes=image_bytes,
            text_prompt=text_prompt,
            confidence_threshold=confidence_threshold,
            pyramidal_config=pyramidal_config,
        )
        print(f"✓ sam3_infer_only returned {len(result.get('pred_boxes', []))} predictions")
        return result
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback_str = traceback.format_exc()
        print(f"❌ Error in sam3_infer: {error_msg}")
        print(traceback_str)
        return {
            "status": "error",
            "message": error_msg,
            "traceback": traceback_str,
        }


# ------------------------------------------------------------------------------
# HTTP endpoint: /sam3/count (Pyramidal SAM3 Counting)
# ------------------------------------------------------------------------------


@app.function(timeout=900, image=image)
@fastapi_endpoint(method="POST")
def sam3_count(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    HTTP endpoint for counting objects using Pyramidal SAM3 segmentation.
    
    POST /sam3/count
    JSON body:
    {
      "prompt": "trees",                   # Required: what objects to count
      "image_b64": "...",                  # OR "image_url": "https://..."
      "confidence_threshold": 0.5,         # Optional: min confidence (default: 0.5)
      "pyramidal_config": {                # Optional: pyramidal configuration
        "tile_size": 512,
        "overlap_ratio": 0.15,
        "scales": [1.0, 0.5],
        "batch_size": 16,
        "iou_threshold": 0.5
      }
    }
    
    Returns:
    {
      "status": "success",
      "count": 47,
      "object_type": "tree",
      "confidence_summary": {
        "high": 35,      // >= 0.8
        "medium": 10,    // 0.5 - 0.8
        "low": 2         // < 0.5
      },
      "detections": [...],
      "pyramidal_stats": {...}
    }
    """
    # Basic validation
    if "prompt" not in body:
        return {"status": "error", "message": "Missing 'prompt' in request body."}
    
    text_prompt = body["prompt"]
    confidence_threshold = body.get("confidence_threshold", 0.5)
    pyramidal_config = body.get("pyramidal_config")
    
    # Get image bytes from either image_b64 or image_url
    if "image_b64" in body:
        try:
            image_bytes = base64.b64decode(body["image_b64"])
            print(f"✓ Decoded image from base64 ({len(image_bytes)} bytes)")
        except Exception as e:
            return {"status": "error", "message": f"Invalid base64 in 'image_b64': {e}."}
    elif "image_url" in body:
        import requests
        try:
            print(f"📥 Downloading image from URL: {body['image_url']}")
            resp = requests.get(body["image_url"], timeout=30)
            resp.raise_for_status()
            image_bytes = resp.content
            print(f"✓ Downloaded image ({len(image_bytes)} bytes)")
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
    
    # Call the GPU-backed counting method
    try:
        print(f"📞 Calling sam3_count with prompt: '{text_prompt}'")
        result = SAM3Model().sam3_count.remote(
            image_bytes=image_bytes,
            text_prompt=text_prompt,
            confidence_threshold=confidence_threshold,
            pyramidal_config=pyramidal_config,
        )
        print(f"✓ sam3_count returned count: {result.get('count', 0)}")
        return result
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback_str = traceback.format_exc()
        print(f"❌ Error in sam3_count: {error_msg}")
        print(traceback_str)
        return {
            "status": "error",
            "message": error_msg,
            "traceback": traceback_str,
        }


# ------------------------------------------------------------------------------
# HTTP endpoint: /sam3/area (Pyramidal SAM3 Area Calculation)
# ------------------------------------------------------------------------------


@app.function(timeout=900, image=image)
@fastapi_endpoint(method="POST")
def sam3_area(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    HTTP endpoint for calculating object areas using Pyramidal SAM3 segmentation.
    
    POST /sam3/area
    JSON body:
    {
      "prompt": "solar panels",            # Required: what objects to measure
      "image_b64": "...",                  # OR "image_url": "https://..."
      "gsd": 0.5,                          # Optional: Ground Sample Distance (m/pixel)
      "confidence_threshold": 0.5,         # Optional: min confidence (default: 0.5)
      "pyramidal_config": {                # Optional: pyramidal configuration
        "tile_size": 512,
        "overlap_ratio": 0.15,
        "scales": [1.0, 0.5],
        "batch_size": 16,
        "iou_threshold": 0.5
      }
    }
    
    Returns:
    {
      "status": "success",
      "object_count": 12,
      "total_pixel_area": 125000,
      "total_real_area_m2": 31250.0,       // Only if gsd provided
      "coverage_percentage": 12.5,
      "individual_areas": [
        {"id": 1, "pixel_area": 5000, "real_area_m2": 1250.0, "score": 0.9, "box": [...]},
        ...
      ],
      "pyramidal_stats": {...}
    }
    """
    # Basic validation
    if "prompt" not in body:
        return {"status": "error", "message": "Missing 'prompt' in request body."}
    
    text_prompt = body["prompt"]
    gsd = body.get("gsd")
    confidence_threshold = body.get("confidence_threshold", 0.5)
    pyramidal_config = body.get("pyramidal_config")
    
    # Get image bytes from either image_b64 or image_url
    if "image_b64" in body:
        try:
            image_bytes = base64.b64decode(body["image_b64"])
            print(f"✓ Decoded image from base64 ({len(image_bytes)} bytes)")
        except Exception as e:
            return {"status": "error", "message": f"Invalid base64 in 'image_b64': {e}."}
    elif "image_url" in body:
        import requests
        try:
            print(f"📥 Downloading image from URL: {body['image_url']}")
            resp = requests.get(body["image_url"], timeout=30)
            resp.raise_for_status()
            image_bytes = resp.content
            print(f"✓ Downloaded image ({len(image_bytes)} bytes)")
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
    
    # Call the GPU-backed area calculation method
    try:
        print(f"📞 Calling sam3_area with prompt: '{text_prompt}'")
        result = SAM3Model().sam3_area.remote(
            image_bytes=image_bytes,
            text_prompt=text_prompt,
            gsd=gsd,
            confidence_threshold=confidence_threshold,
            pyramidal_config=pyramidal_config,
        )
        print(f"✓ sam3_area returned {result.get('object_count', 0)} objects, total area: {result.get('total_pixel_area', 0)} pixels")
        return result
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback_str = traceback.format_exc()
        print(f"❌ Error in sam3_area: {error_msg}")
        print(traceback_str)
        return {
            "status": "error",
            "message": error_msg,
            "traceback": traceback_str,
        }


# ------------------------------------------------------------------------------
# HTTP endpoint: /sam3/segment (Full agent with LLM + Pyramidal SAM3)
# ------------------------------------------------------------------------------


@app.function(timeout=900, image=image)
@fastapi_endpoint(method="POST")
def sam3_segment(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    HTTP endpoint for SAM3 Agent with Pyramidal Batch Processing - LLM Provider Agnostic
    
    Uses VLM for prompt refinement + Pyramidal Batch SAM3 for segmentation.
    This endpoint accepts any LLM configuration via the request body.
    No hardcoded providers - works with any OpenAI-compatible API.

    POST /sam3/segment
    JSON body:
    {
      "prompt": "...",                    # Required: text prompt for segmentation
      "image_url": "https://...",         # OR "image_b64": "..." (one required)
      "llm_config": {                     # Required: complete LLM configuration (provider-agnostic)
        "base_url": "https://api.openai.com/v1",  # Any OpenAI-compatible endpoint
        "model": "gpt-4o",                 # Any model name
        "api_key": "sk-...",              # API key (can be empty for some backends)
        "name": "openai-gpt4o",           # Optional: for output files
        "max_tokens": 4096                 # Optional: default 4096
      },
      "debug": true                       # Optional: get visualization
    }
    
    Example with OpenAI:
    {
      "prompt": "segment all objects",
      "image_b64": "...",
      "llm_config": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
        "api_key": "sk-your-key-here"
      }
    }
    
    Example with vLLM:
    {
      "prompt": "segment all objects",
      "image_b64": "...",
      "llm_config": {
        "base_url": "http://localhost:8001/v1",
        "model": "Qwen/Qwen3-VL-8B-Thinking",
        "api_key": "",
        "name": "vllm-local"
      }
    }
    
    Example with Anthropic (if OpenAI-compatible):
    {
      "prompt": "segment all objects",
      "image_b64": "...",
      "llm_config": {
        "base_url": "https://api.anthropic.com/v1",
        "model": "claude-3-opus",
        "api_key": "sk-ant-..."
      }
    }
    """
    # Basic validation
    if "prompt" not in body:
        return {"status": "error", "message": "Missing 'prompt' in request body."}

    if "llm_config" not in body:
        return {"status": "error", "message": "Missing 'llm_config' in request body. Provide complete LLM configuration with 'base_url', 'model', and 'api_key'."}

    prompt = body["prompt"]
    debug = bool(body.get("debug", False))
    llm_config = body["llm_config"]
    confidence_threshold = body.get("confidence_threshold")

    # Get image bytes from either image_b64 or image_url
    if "image_b64" in body:
        try:
            image_bytes = base64.b64decode(body["image_b64"])
        except Exception:
            return {"status": "error", "message": "Invalid base64 in 'image_b64'."}
    elif "image_url" in body:
        import requests

        try:
            resp = requests.get(body["image_url"], timeout=30)
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
    # Use class reference directly to ensure persistent container reuse
    result = SAM3Model().infer.remote(
        image_bytes=image_bytes,
        prompt=prompt,
        llm_config=llm_config,
        debug=debug,
        confidence_threshold=confidence_threshold,
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
    llm_config = {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
        "api_key": os.environ.get("OPENAI_API_KEY", ""),
        "name": "openai-gpt4o",
    }
    model = SAM3Model()
    result = model.infer.remote(
        image_bytes=image_bytes,
        prompt=prompt,
        llm_config=llm_config,
        debug=True,
    )

    print(json.dumps(result, indent=2)[:2000])
