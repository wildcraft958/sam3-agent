"""
Visual Grounding Pipeline with Oriented Bounding Boxes - Modal vLLM Integration
================================================================================
A comprehensive pipeline that:
1. Uses Qwen3-VL-30B via Modal vLLM API to refine natural language queries
2. Employs SAM3 for object detection/segmentation
3. Applies intelligent box filtering and selection
4. Generates oriented bounding boxes (OBBs)


"""
%uv pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
!git clone https://github.com/facebookresearch/sam3.git
%cd sam3
%uv pip install -e .
%uv pip install --upgrade transformers
%uv pip install opencv-python

%uv pip install einops decord pycocotools
import os
import json
import re
import cv2
import numpy as np
import torch
import requests
import base64
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from PIL import Image, ImageDraw
from tqdm import tqdm
from io import BytesIO

# SAM3 imports
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Detection:
    """Container for a single detection with mask and metadata"""
    mask: np.ndarray
    box: np.ndarray  # [x1, y1, x2, y2]
    score: float
    scale: float = 1.0
    tile_offset: Tuple[int, int] = (0, 0)


@dataclass
class RefinedQuery:
    """Container for refined query information"""
    object_classes: List[str]
    detailed_query: str
    original_query: str


@dataclass
class GroundingResult:
    """Complete grounding result for one image"""
    image_id: str
    original_query: str
    refined_queries: List[RefinedQuery]
    detected_boxes: List[np.ndarray]
    box_scores: List[float]
    selected_boxes: List[np.ndarray]
    obbs: List[Optional[List[float]]]  # [cx, cy, w, h, angle]
    masks: List[np.ndarray]
    success: bool
    error: Optional[str] = None


# ============================================================================
# CONFIGURATION
# ============================================================================

class PipelineConfig:
    """Configuration for the visual grounding pipeline"""
    
    # Modal vLLM API settings
    MODAL_BASE_URL = "https://your-username--qwen3-vl-vllm-server-30b.modal.run/v1"  # Replace with your Modal URL
    VLM_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    API_KEY = ""  # vLLM doesn't require API key
    MAX_TOKENS = 4096
    TEMPERATURE = 0.7
    REQUEST_TIMEOUT = 600  # seconds
    
    # SAM3 settings
    SAM3_DEVICE = 0
    
    # Processing settings
    MAX_REFINEMENT_ATTEMPTS = 2
    MIN_DETECTIONS_REQUIRED = 1
    TILE_SIZE = 512
    OVERLAP_RATIO = 0.15
    PYRAMID_SCALES = [1.0, 0.5, 0.25]
    
    # Filtering thresholds
    CONFIDENCE_THRESHOLD = 0.3
    IOU_MERGE_THRESHOLD = 0.5
    IOU_SIMILARITY_THRESHOLD = 0.8
    NMS_IOU_THRESHOLD = 0.6
    
    # OBB settings
    MIN_CONTOUR_AREA = 10
    MIN_OBB_SIZE = 5
    CROP_PADDING_RATIO = 0.2
    SAM3_SCORE_THRESHOLD = 0.2


# ============================================================================
# QUERY REFINEMENT MODULE WITH MODAL vLLM API
# ============================================================================

class QueryRefiner:
    """Handles query refinement using Qwen VLM via Modal vLLM API"""
    
    def __init__(
        self,
        base_url: str = PipelineConfig.MODAL_BASE_URL,
        model: str = PipelineConfig.VLM_MODEL,
        api_key: str = PipelineConfig.API_KEY,
        max_tokens: int = PipelineConfig.MAX_TOKENS,
        temperature: float = PipelineConfig.TEMPERATURE,
        timeout: int = PipelineConfig.REQUEST_TIMEOUT
    ):
        print(f"Initializing VLM API client...")
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.chat_endpoint = f"{self.base_url}/chat/completions"
        
        # Test connection
        try:
            health_url = base_url.replace('/v1', '/health')
            response = requests.get(health_url, timeout=10)
            if response.status_code == 200:
                print(f"✓ VLM API connected: {response.json()}")
            else:
                print(f"⚠ VLM API health check returned status {response.status_code}")
        except Exception as e:
            print(f"⚠ VLM API health check failed: {e}")
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def _call_vlm_api(self, image: Image.Image, prompt: str) -> Optional[str]:
        """Call the Modal vLLM API with image and prompt"""
        image_b64 = self._image_to_base64(image)
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        try:
            response = requests.post(
                self.chat_endpoint,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Extract response text from OpenAI-compatible format
            if "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    return choice["message"]["content"]
            
            print(f"⚠ Unexpected API response format: {result}")
            return None
            
        except requests.exceptions.Timeout:
            print(f"⚠ API request timed out after {self.timeout}s")
            return None
        except requests.exceptions.RequestException as e:
            print(f"⚠ API request failed: {e}")
            return None
        except Exception as e:
            print(f"⚠ Unexpected error calling API: {e}")
            return None
    
    def _extract_json_from_response(self, response_text: str) -> Optional[Dict]:
        """Extract and parse JSON from model response"""
        # Remove markdown code blocks
        text = re.sub(r'```json\s*', '', response_text)
        text = re.sub(r'```\s*', '', text)
        text = text.strip()
        
        # Try to find JSON object
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                return None
        return None
    
    def refine_query(self, image: Image.Image, original_query: str) -> Optional[RefinedQuery]:
        """
        Refine a natural language query into structured object classes and detailed description.
        
        Args:
            image: PIL Image
            original_query: Original natural language description
            
        Returns:
            RefinedQuery object or None if refinement fails
        """
        prompt = f"""You are a vision-language expert for object grounding and segmentation.

Given an image and a natural-language description, produce 2 outputs:

1. **object_class**: A list of synonyms/variations for the main object category.
   - Include color and attributes IF AND ONLY IF mentioned in the query
   - Be specific about what class of object is referenced
   Example: if keyword is "vehicle", output ['vehicle', 'car', 'automobile', 'car']

2. **object_class_query**: Detailed information about the object's location and attributes:
   - Global position (top-left, bottom-right, center, etc.)
   - Local position relative to other objects as seen in the image
   - Color, size, and other attributes
   - If multiple instances exist, specify which one (leftmost, rightmost, etc.)

CRITICAL RULES:
- Do NOT hallucinate - only describe what you see
- If relative position is mentioned, include both relative AND global position
- Output MUST be valid JSON with these 2 fields

Input description: {original_query}

Output (strict JSON only, no markdown):
{{
  "object_class": ["primary term", "term with attributes"],
  "object_class_query": "detailed description of location and attributes"
}}"""

        response_text = self._call_vlm_api(image, prompt)
        
        if response_text is None:
            print(f"⚠ No response from VLM API")
            return None
        
        try:
            parsed = self._extract_json_from_response(response_text)
            
            if parsed and "object_class" in parsed and "object_class_query" in parsed:
                # Ensure object_class is a list
                obj_classes = parsed["object_class"]
                if isinstance(obj_classes, str):
                    obj_classes = [obj_classes]
                print(f"  Refined query: {parsed['object_class_query']}")
                return RefinedQuery(
                    object_classes=obj_classes,
                    detailed_query=parsed["object_class_query"],
                    original_query=original_query
                )
            else:
                print(f"⚠ Invalid JSON structure in VLM response")
                return None
                
        except Exception as e:
            print(f"⚠ Query refinement error: {e}")
            return None
    
    def select_best_boxes(
        self,
        image: Image.Image,
        boxes: List[np.ndarray],
        query: str,
        crop_images: Optional[List[Image.Image]] = None
    ) -> List[int]:
        """
        Use VLM to select boxes that best match the query.
        
        Args:
            image: Original PIL Image
            boxes: List of [x1, y1, x2, y2] boxes
            query: Query description
            crop_images: Optional list of cropped images for each box
            
        Returns:
            List of indices of selected boxes
        """
        if len(boxes) == 0:
            return []
        
        img_width, img_height = image.size
        
        # Create detailed box descriptions
        boxes_descriptions = []
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            h_pos = "left" if center_x < img_width/3 else ("right" if center_x > 2*img_width/3 else "center")
            v_pos = "top" if center_y < img_height/3 else ("bottom" if center_y > 2*img_height/3 else "middle")
            
            box_desc = f"{i+1}. Box at ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}) - {v_pos}-{h_pos}"
            boxes_descriptions.append(box_desc)
        
        boxes_str = "\n".join(boxes_descriptions)
        
        prompt = f"""You are a visual grounding expert. Select ALL bounding boxes that match the query description.

INSTRUCTIONS:
1. Examine the image and identify all objects matching the description
2. Consider location information (top/middle/bottom, left/center/right, upper-left, lower right)
3. The query may describe multiple objects - select all that match. But try to select the least number of boxes according to the query
4. Reply with box numbers as a comma-separated list

Query: "{query}"

Available boxes:
{boxes_str}

Reply with ONLY the box numbers (e.g., "1, 3, 5" or "2"):"""

        response = self._call_vlm_api(image, prompt)
        
        if response is None:
            print(f"⚠ Box selection failed. Defaulting to first box")
            return [0]
        
        try:
            # Parse box numbers
            numbers = re.findall(r'\d+', response.strip())
            selected_indices = []
            for num_str in numbers:
                idx = int(num_str)
                if 1 <= idx <= len(boxes):
                    selected_indices.append(idx - 1)  # Convert to 0-indexed
            
            return selected_indices if selected_indices else [0]  # Default to first box
            
        except Exception as e:
            print(f"⚠ Box selection error: {e}. Defaulting to first box")
            return [0]


# ============================================================================
# PYRAMIDAL SAM3 DETECTOR
# ============================================================================

class PyramidalSAM3:
    """Pyramidal tiling segmentation wrapper for SAM3"""
    
    def __init__(
        self,
        model,
        processor,
        tile_size: int = PipelineConfig.TILE_SIZE,
        overlap_ratio: float = PipelineConfig.OVERLAP_RATIO,
        scales: List[float] = PipelineConfig.PYRAMID_SCALES,
        iou_threshold: float = PipelineConfig.IOU_MERGE_THRESHOLD,
        conf_threshold: float = PipelineConfig.CONFIDENCE_THRESHOLD,
        verbose: bool = False
    ):
        self.model = model
        self.processor = processor
        self.tile_size = tile_size
        self.overlap_ratio = overlap_ratio
        self.scales = sorted(scales, reverse=True)
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.verbose = verbose
    
    def create_tiles(self, image: Image.Image) -> List[Tuple[Image.Image, Tuple[int, int]]]:
        """Generate overlapping tiles from an image"""
        img_width, img_height = image.size
        stride = int(self.tile_size * (1 - self.overlap_ratio))
        tiles = []
        
        if img_width <= self.tile_size and img_height <= self.tile_size:
            return [(image, (0, 0))]
        
        for y in range(0, img_height, stride):
            for x in range(0, img_width, stride):
                x_end = min(x + self.tile_size, img_width)
                y_end = min(y + self.tile_size, img_height)
                x_start = max(0, x_end - self.tile_size)
                y_start = max(0, y_end - self.tile_size)
                
                tile = image.crop((x_start, y_start, x_end, y_end))
                tiles.append((tile, (x_start, y_start)))
                
                if x_end >= img_width:
                    break
            if y_end >= img_height:
                break
        
        return tiles
    
    def transform_detection(
        self,
        mask,
        box,
        score,
        tile_offset: Tuple[int, int],
        scale: float,
        orig_size: Tuple[int, int]
    ) -> Detection:
        """Transform detection from tile coordinates to original image coordinates"""
        offset_x, offset_y = tile_offset
        
        # Convert to numpy if tensor
        if torch.is_tensor(box):
            box = box.cpu().numpy()
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy()
        if torch.is_tensor(score):
            score = float(score.cpu().numpy())
        
        # Transform box coordinates
        box_scaled = box.copy()
        box_scaled[0] += offset_x
        box_scaled[1] += offset_y
        box_scaled[2] += offset_x
        box_scaled[3] += offset_y
        
        box_orig = box_scaled / scale
        
        # Clip to image bounds
        box_orig[0] = max(0, min(box_orig[0], orig_size[0]))
        box_orig[1] = max(0, min(box_orig[1], orig_size[1]))
        box_orig[2] = max(0, min(box_orig[2], orig_size[0]))
        box_orig[3] = max(0, min(box_orig[3], orig_size[1]))
        
        return Detection(
            mask=mask,
            box=box_orig,
            score=score,
            scale=scale,
            tile_offset=tile_offset
        )
    
    def calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate Intersection over Union between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def merge_overlapping_detections(self, detections: List[Detection]) -> List[Detection]:
        """Merge overlapping detections"""
        if len(detections) == 0:
            return []
        
        detections = sorted(detections, key=lambda d: -d.score)
        merged = []
        used = [False] * len(detections)
        
        for i, det_i in enumerate(detections):
            if used[i]:
                continue
            
            group = [det_i]
            for j, det_j in enumerate(detections[i+1:], start=i+1):
                if not used[j] and self.calculate_iou(det_i.box, det_j.box) > self.iou_threshold:
                    group.append(det_j)
                    used[j] = True
            
            # Merge group
            if len(group) == 1:
                merged.append(det_i)
            else:
                boxes = np.array([d.box for d in group])
                merged_box = np.mean(boxes, axis=0)
                merged_score = np.mean([d.score for d in group])
                best_det = max(group, key=lambda d: d.score)
                
                merged.append(Detection(
                    mask=best_det.mask,
                    box=merged_box,
                    score=merged_score,
                    scale=best_det.scale,
                    tile_offset=best_det.tile_offset
                ))
            
            used[i] = True
        
        return merged
    
    def segment_with_tiling(self, image: Image.Image, prompt: str) -> Dict:
        """Perform pyramidal tiling segmentation"""
        orig_size = image.size
        all_detections = []
        
        if self.verbose:
            print(f"  Processing {len(self.scales)} pyramid levels...")
        
        for scale_idx, scale in enumerate(self.scales):
            if scale != 1.0:
                new_size = (int(orig_size[0] * scale), int(orig_size[1] * scale))
                scaled_image = image.resize(new_size, Image.Resampling.LANCZOS)
            else:
                scaled_image = image
            
            tiles = self.create_tiles(scaled_image)
            
            for tile_idx, (tile, tile_offset) in enumerate(tiles):
                try:
                    inference_state = self.processor.set_image(tile)
                    output = self.processor.set_text_prompt(
                        state=inference_state,
                        prompt=prompt
                    )
                    
                    masks = output.get("masks", [])
                    boxes = output.get("boxes", [])
                    scores = output.get("scores", [])
                    
                    for mask, box, score in zip(masks, boxes, scores):
                        if score >= self.conf_threshold:
                            detection = self.transform_detection(
                                mask, box, score, tile_offset, scale, orig_size
                            )
                            all_detections.append(detection)
                
                except Exception as e:
                    if self.verbose:
                        print(f"    Tile {tile_idx} error: {e}")
                    continue
        
        final_detections = self.merge_overlapping_detections(all_detections)
        
        return {
            'detections': final_detections,
            'masks': [d.mask for d in final_detections],
            'boxes': [d.box for d in final_detections],
            'scores': [d.score for d in final_detections]
        }


# ============================================================================
# POST-PROCESSING MODULE
# ============================================================================

class DetectionPostProcessor:
    """Handles NMS and box filtering"""
    
    @staticmethod
    def apply_nms(
        boxes: List[np.ndarray],
        scores: List[float],
        iou_threshold: float = PipelineConfig.NMS_IOU_THRESHOLD
    ) -> Tuple[List[np.ndarray], List[float]]:
        """Apply Non-Maximum Suppression"""
        if len(boxes) == 0:
            return [], []
        
        boxes_array = np.array(boxes)
        scores_array = np.array(scores)
        
        # Sort by score (descending)
        sorted_indices = np.argsort(scores_array)[::-1]
        
        keep = []
        suppress = np.zeros(len(boxes), dtype=bool)
        
        for i in sorted_indices:
            if suppress[i]:
                continue
            
            keep.append(i)
            current_box = boxes_array[i]
            
            # Suppress overlapping boxes
            for j in range(i + 1, len(boxes)):
                if suppress[j]:
                    continue
                
                iou = PyramidalSAM3(None, None).calculate_iou(current_box, boxes_array[j])
                if iou > iou_threshold:
                    suppress[j] = True
        
        filtered_boxes = [boxes[i] for i in keep]
        filtered_scores = [scores[i] for i in keep]
        
        return filtered_boxes, filtered_scores
    
    @staticmethod
    def remove_similar_boxes(
        boxes: List[np.ndarray],
        iou_threshold: float = PipelineConfig.IOU_SIMILARITY_THRESHOLD
    ) -> List[np.ndarray]:
        """Remove highly similar boxes"""
        if len(boxes) <= 1:
            return boxes
        
        boxes_array = np.array(boxes)
        areas = (boxes_array[:, 2] - boxes_array[:, 0]) * (boxes_array[:, 3] - boxes_array[:, 1])
        sorted_indices = np.argsort(areas)[::-1]
        
        keep = []
        suppress = np.zeros(len(boxes), dtype=bool)
        
        for i in sorted_indices:
            if suppress[i]:
                continue
            
            keep.append(i)
            current_box = boxes_array[i]
            
            for j in range(i + 1, len(boxes)):
                if suppress[j]:
                    continue
                
                iou = PyramidalSAM3(None, None).calculate_iou(current_box, boxes_array[j])
                if iou > iou_threshold:
                    suppress[j] = True
        
        return [boxes[i] for i in sorted(keep)]


# ============================================================================
# OBB GENERATION MODULE
# ============================================================================

class OBBGenerator:
    """Generates oriented bounding boxes from detections"""
    
    def __init__(self, processor):
        self.processor = processor
    
    def validate_box(
        self,
        box: List[float],
        img_width: int,
        img_height: int
    ) -> Optional[List[int]]:
        """Validate and fix box coordinates"""
        xmin, ymin, xmax, ymax = box
        
        xmin = max(0, int(xmin))
        ymin = max(0, int(ymin))
        xmax = min(img_width - 1, int(xmax))
        ymax = min(img_height - 1, int(ymax))
        
        # Check for valid box
        if xmax <= xmin or ymax <= ymin:
            return None
        
        # Check minimum size
        if (xmax - xmin) < 10 or (ymax - ymin) < 10:
            return None
        
        return [xmin, ymin, xmax, ymax]
    
    def generate_obbs(
        self,
        image: Image.Image,
        boxes: List[np.ndarray],
        debug: bool = False
    ) -> Tuple[List[Optional[List[float]]], List[np.ndarray]]:
        """
        Generate OBBs from bounding boxes using SAM3 masks.
        
        Args:
            image: PIL Image
            boxes: List of [x1, y1, x2, y2] boxes
            debug: Print debug information
            
        Returns:
            Tuple of (OBB list, mask list)
            OBB format: [cx, cy, w, h, angle]
        """
        cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        H, W = cv_img.shape[:2]
        
        obb_list = []
        mask_list = []
        
        for box_idx, box in enumerate(boxes):
            valid_box = self.validate_box(box, W, H)
            if valid_box is None:
                if debug:
                    print(f"    Box {box_idx+1}: Invalid")
                obb_list.append(None)
                mask_list.append(np.zeros((H, W), dtype=np.uint8))
                continue
            
            xmin, ymin, xmax, ymax = valid_box
            box_width = xmax - xmin
            box_height = ymax - ymin
            
            try:
                # Crop with padding
                padding_x = max(30, int(box_width * PipelineConfig.CROP_PADDING_RATIO))
                padding_y = max(30, int(box_height * PipelineConfig.CROP_PADDING_RATIO))
                
                crop_xmin = max(0, xmin - padding_x)
                crop_ymin = max(0, ymin - padding_y)
                crop_xmax = min(W, xmax + padding_x)
                crop_ymax = min(H, ymax + padding_y)
                
                crop_img = image.crop((crop_xmin, crop_ymin, crop_xmax, crop_ymax))
                crop_w, crop_h = crop_img.size
                
                if crop_w < 10 or crop_h < 10:
                    if debug:
                        print(f"    Box {box_idx+1}: Crop too small")
                    obb_list.append(None)
                    mask_list.append(np.zeros((H, W), dtype=np.uint8))
                    continue
                
                # Set image and get mask using box prompt
                inference_state = self.processor.set_image(crop_img)
                
                # Convert to normalized box format
                rel_xmin = xmin - crop_xmin
                rel_ymin = ymin - crop_ymin
                rel_xmax = xmax - crop_xmin
                rel_ymax = ymax - crop_ymin
                
                center_x = (rel_xmin + rel_xmax) / 2.0 / crop_w
                center_y = (rel_ymin + rel_ymax) / 2.0 / crop_h
                width = (rel_xmax - rel_xmin) / crop_w
                height = (rel_ymax - rel_ymin) / crop_h
                
                # Clip to valid range
                center_x = max(0.0, min(1.0, center_x))
                center_y = max(0.0, min(1.0, center_y))
                width = max(0.01, min(1.0, width))
                height = max(0.01, min(1.0, height))
                
                box_cxcywh = [center_x, center_y, width, height]
                
                # Get mask using geometric prompt
                output = self.processor.add_geometric_prompt(
                    box=box_cxcywh,
                    label=True,
                    state=inference_state
                )
                
                masks = output.get("masks", [])
                scores = output.get("scores", [])
                
                # Convert scores
                if torch.is_tensor(scores):
                    scores = scores.cpu().numpy().tolist()
                elif len(scores) > 0:
                    scores = [float(s.cpu().numpy()) if torch.is_tensor(s) else float(s) for s in scores]
                
                if len(masks) == 0 or len(scores) == 0:
                    if debug:
                        print(f"    Box {box_idx+1}: No masks")
                    obb_list.append(None)
                    mask_list.append(np.zeros((H, W), dtype=np.uint8))
                    continue
                
                # Filter by score
                scores_array = np.array(scores)
                valid_indices = scores_array >= PipelineConfig.SAM3_SCORE_THRESHOLD
                
                if not valid_indices.any():
                    if debug:
                        print(f"    Box {box_idx+1}: No valid masks")
                    obb_list.append(None)
                    mask_list.append(np.zeros((H, W), dtype=np.uint8))
                    continue
                
    
                # Get best mask (continuing from line that was cut off)
                best_idx = np.argmax(scores_array[valid_indices])
                valid_indices_list = np.where(valid_indices)[0]
                best_mask = masks[valid_indices_list[best_idx]]
                
                # Convert mask to numpy
                if torch.is_tensor(best_mask):
                    best_mask = best_mask.cpu().numpy()
                elif isinstance(best_mask, (list, tuple)):
                    if torch.is_tensor(best_mask[0]):
                        best_mask = best_mask[0].cpu().numpy()
                    else:
                        best_mask = np.array(best_mask[0])
                
                # Handle mask dimensions
                if best_mask.ndim == 3:
                    if best_mask.shape[0] == 1:
                        best_mask = best_mask[0]
                    elif best_mask.shape[-1] == 1:
                        best_mask = best_mask[:, :, 0]
                    else:
                        best_mask = best_mask[0]
                
                # Resize mask if needed
                if best_mask.shape != (crop_h, crop_w):
                    best_mask = cv2.resize(
                        best_mask.astype(np.float32),
                        (crop_w, crop_h),
                        interpolation=cv2.INTER_LINEAR
                    )
                
                # Binarize
                best_mask_binary = (best_mask > 0.5).astype(np.uint8)
                
                if best_mask_binary.sum() == 0:
                    if debug:
                        print(f"    Box {box_idx+1}: Empty mask")
                    obb_list.append(None)
                    mask_list.append(np.zeros((H, W), dtype=np.uint8))
                    continue
                
                # Find contours
                cnts, _ = cv2.findContours(
                    best_mask_binary,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                
                if len(cnts) == 0:
                    if debug:
                        print(f"    Box {box_idx+1}: No contours")
                    obb_list.append(None)
                    mask_list.append(np.zeros((H, W), dtype=np.uint8))
                    continue
                
                # Get largest contour
                cnt = max(cnts, key=cv2.contourArea)
                
                if cv2.contourArea(cnt) < PipelineConfig.MIN_CONTOUR_AREA:
                    if debug:
                        print(f"    Box {box_idx+1}: Contour too small")
                    obb_list.append(None)
                    mask_list.append(np.zeros((H, W), dtype=np.uint8))
                    continue
                
                # Shift contour to full image coordinates
                cnt_shifted = cnt.copy()
                cnt_shifted[:, 0, 0] += crop_xmin
                cnt_shifted[:, 0, 1] += crop_ymin
                
                # Compute OBB
                rect = cv2.minAreaRect(cnt_shifted)
                (cx, cy), (w, h), angle = rect
                
                # Normalize angle
                if w < h:
                    angle = angle + 90
                    w, h = h, w
                
                angle = angle % 180
                
                # Validate OBB size
                if w < PipelineConfig.MIN_OBB_SIZE or h < PipelineConfig.MIN_OBB_SIZE:
                    if debug:
                        print(f"    Box {box_idx+1}: OBB too small")
                    obb_list.append(None)
                    mask_list.append(np.zeros((H, W), dtype=np.uint8))
                    continue
                
                obb_list.append([float(cx), float(cy), float(w), float(h), float(angle)])
                
                # Create full-size mask
                full_mask = np.zeros((H, W), dtype=np.uint8)
                clean_crop = np.zeros((crop_h, crop_w), dtype=np.uint8)
                cv2.drawContours(clean_crop, [cnt], -1, 1, -1)
                full_mask[crop_ymin:crop_ymax, crop_xmin:crop_xmax] = clean_crop
                mask_list.append(full_mask)
                
                if debug:
                    print(f"    Box {box_idx+1}: ✓ OBB generated")
            
            except Exception as e:
                if debug:
                    print(f"    Box {box_idx+1}: Error - {e}")
                obb_list.append(None)
                mask_list.append(np.zeros((H, W), dtype=np.uint8))
        
        return obb_list, mask_list


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class VisualGroundingPipeline:
    """Complete visual grounding pipeline with OBB generation and Modal vLLM API"""
    
    def __init__(
        self,
        modal_base_url: str = PipelineConfig.MODAL_BASE_URL,
        vlm_model: str = PipelineConfig.VLM_MODEL,
        sam3_device: int = PipelineConfig.SAM3_DEVICE,
        verbose: bool = True
    ):
        """
        Initialize the pipeline.
        
        Args:
            modal_base_url: Modal vLLM API endpoint URL
            vlm_model: Qwen model name
            sam3_device: CUDA device for SAM3
            verbose: Print detailed progress
        """
        self.verbose = verbose
        
        # Initialize components
        if self.verbose:
            print("="*60)
            print("Initializing Visual Grounding Pipeline with Modal vLLM")
            print("="*60)
        
        # Load VLM via Modal API
        self.query_refiner = QueryRefiner(
            base_url=modal_base_url,
            model=vlm_model
        )
        
        # Load SAM3
        if self.verbose:
            print("Loading SAM3 model...")
        self.sam3_model = build_sam3_image_model()
        self.sam3_processor = Sam3Processor(self.sam3_model)
        if self.verbose:
            print("✓ SAM3 model loaded")
        
        # Initialize pyramidal SAM3
        self.pyramidal_sam = PyramidalSAM3(
            self.sam3_model,
            self.sam3_processor,
            verbose=self.verbose
        )
        
        # Initialize post-processor
        self.post_processor = DetectionPostProcessor()
        
        # Initialize OBB generator
        self.obb_generator = OBBGenerator(self.sam3_processor)
        
        if self.verbose:
            print("✓ Pipeline initialized\n")
    
    def process_single_image(
        self,
        image: Image.Image,
        query: str,
        image_id: str = "unknown"
    ) -> GroundingResult:
        """
        Process a single image with the complete pipeline.
        
        Args:
            image: PIL Image
            query: Natural language query
            image_id: Identifier for the image
            
        Returns:
            GroundingResult with all detections and OBBs
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Processing: {image_id}")
            print(f"Query: {query[:80]}...")
            print(f"{'='*60}")
        
        all_refined_queries = []
        all_boxes = []
        all_scores = []
        
        # Refinement loop (up to MAX_REFINEMENT_ATTEMPTS)
        for attempt in range(PipelineConfig.MAX_REFINEMENT_ATTEMPTS):
            if self.verbose:
                print(f"\n--- Refinement Attempt {attempt + 1} ---")
            
            # Step 1: Refine query using Modal vLLM
            refined_query = self.query_refiner.refine_query(image, query)
            
            if refined_query is None:
                if self.verbose:
                    print("⚠ Query refinement failed, using fallback")
                refined_query = RefinedQuery(
                    object_classes=["object"],
                    detailed_query=query,
                    original_query=query
                )
            
            all_refined_queries.append(refined_query)
            
            if self.verbose:
                print(f"Object classes: {refined_query.object_classes}")
                print(f"Detailed query: {refined_query.detailed_query[:80]}...")
            
            # Step 2: Run SAM3 for each object class
            attempt_boxes = []
            attempt_scores = []
            
            for obj_class in refined_query.object_classes:
                if self.verbose:
                    print(f"\n  Searching for: '{obj_class}'")
                
                try:
                    output = self.pyramidal_sam.segment_with_tiling(image, obj_class)
                    
                    boxes = output.get("boxes", [])
                    scores = output.get("scores", [])
                    
                    if len(boxes) > 0:
                        attempt_boxes.extend(boxes)
                        attempt_scores.extend(scores)
                        
                        if self.verbose:
                            print(f"  ✓ Found {len(boxes)} detections")
                    else:
                        if self.verbose:
                            print(f"  ⊘ No detections")
                
                except Exception as e:
                    if self.verbose:
                        print(f"  ⚠ Error: {e}")
            
            # Add to cumulative results
            all_boxes.extend(attempt_boxes)
            all_scores.extend(attempt_scores)
            
            if self.verbose:
                print(f"\nTotal detections so far: {len(all_boxes)}")
            
            # Check if we have enough detections
            if len(all_boxes) >= PipelineConfig.MIN_DETECTIONS_REQUIRED:
                if self.verbose:
                    print(f"✓ Sufficient detections ({len(all_boxes)} >= {PipelineConfig.MIN_DETECTIONS_REQUIRED})")
                break
            else:
                if self.verbose:
                    print(f"⚠ Insufficient detections ({len(all_boxes)} < {PipelineConfig.MIN_DETECTIONS_REQUIRED})")
                    if attempt < PipelineConfig.MAX_REFINEMENT_ATTEMPTS - 1:
                        print("  Attempting another refinement...")
        
        # Step 3: Post-processing
        if self.verbose:
            print(f"\n--- Post-Processing ---")
            print(f"Raw detections: {len(all_boxes)}")
        
        if len(all_boxes) == 0:
            if self.verbose:
                print("⊘ No detections found")
            return GroundingResult(
                image_id=image_id,
                original_query=query,
                refined_queries=all_refined_queries,
                detected_boxes=[],
                box_scores=[],
                selected_boxes=[],
                obbs=[],
                masks=[],
                success=False,
                error="No detections found"
            )
        
        # Apply NMS
        filtered_boxes, filtered_scores = self.post_processor.apply_nms(
            all_boxes, all_scores
        )
        
        if self.verbose:
            print(f"After NMS: {len(filtered_boxes)}")
        
        # Remove similar boxes
        unique_boxes = self.post_processor.remove_similar_boxes(filtered_boxes)
        
        if self.verbose:
            print(f"After similarity filter: {len(unique_boxes)}")
        
        # Step 4: VLM selection using Modal API
        if self.verbose:
            print(f"\n--- VLM Box Selection (via Modal API) ---")
        
        selected_indices = self.query_refiner.select_best_boxes(
            image, unique_boxes, query
        )
        
        selected_boxes = [unique_boxes[i] for i in selected_indices if i < len(unique_boxes)]
        
        if self.verbose:
            print(f"Selected {len(selected_boxes)} boxes: {selected_indices}")
        
        # Step 5: Generate OBBs
        if self.verbose:
            print(f"\n--- OBB Generation ---")
        
        obbs, masks = self.obb_generator.generate_obbs(
            image, selected_boxes, debug=self.verbose
        )
        
        valid_obbs = sum(1 for obb in obbs if obb is not None)
        if self.verbose:
            print(f"Generated {valid_obbs}/{len(obbs)} valid OBBs")
        
        return GroundingResult(
            image_id=image_id,
            original_query=query,
            refined_queries=all_refined_queries,
            detected_boxes=all_boxes,
            box_scores=all_scores,
            selected_boxes=selected_boxes,
            obbs=obbs,
            masks=masks,
            success=len(selected_boxes) > 0
        )
    
    def process_batch(
        self,
        image_paths: List[str],
        queries: List[str],
        image_ids: Optional[List[str]] = None
    ) -> List[GroundingResult]:
        """
        Process a batch of images.
        
        Args:
            image_paths: List of image file paths
            queries: List of queries (one per image)
            image_ids: Optional list of image identifiers
            
        Returns:
            List of GroundingResult objects
        """
        if image_ids is None:
            image_ids = [f"image_{i}" for i in range(len(image_paths))]
        
        results = []
        
        for img_path, query, img_id in tqdm(
            zip(image_paths, queries, image_ids),
            total=len(image_paths),
            desc="Processing batch"
        ):
            try:
                image = Image.open(img_path).convert("RGB")
                result = self.process_single_image(image, query, img_id)
                results.append(result)
            except Exception as e:
                print(f"\n⚠ Error processing {img_id}: {e}")
                results.append(GroundingResult(
                    image_id=img_id,
                    original_query=query,
                    refined_queries=[],
                    detected_boxes=[],
                    box_scores=[],
                    selected_boxes=[],
                    obbs=[],
                    masks=[],
                    success=False,
                    error=str(e)
                ))
        
        return results


# ============================================================================
# VISUALIZATION UTILITIES
# ============================================================================

def draw_obb(image: np.ndarray, obb: List[float], color, thickness: int = 2, label: str = None):
    """Draw an oriented bounding box on the image"""
    cx, cy, w, h, angle = obb
    
    theta = np.radians(angle)
    dw, dh = w/2, h/2
    corners = np.array([[-dw, -dh], [dw, -dh], [dw, dh], [-dw, dh]])
    
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated = corners @ R.T
    rotated[:, 0] += cx
    rotated[:, 1] += cy
    
    pts = rotated.astype(np.int32)
    cv2.drawContours(image, [pts], -1, color, thickness)
    
    if label:
        cv2.putText(image, label, (int(cx) - 20, int(cy) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def visualize_result(result: GroundingResult, image: Image.Image, output_path: str):
    """Create visualization of grounding result"""
    cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    vis = cv_img.copy()
    
    # Draw selected boxes
    for i, box in enumerate(result.selected_boxes):
        x1, y1, x2, y2 = [int(coord) for coord in box]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f"Box {i+1}", (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw OBBs
    for i, obb in enumerate(result.obbs):
        if obb is not None:
            draw_obb(vis, obb, (255, 0, 0), thickness=2, label=f"OBB{i+1}")
    
    # Save
    cv2.imwrite(output_path, vis)


def save_results_json(results: List[GroundingResult], output_path: str):
    """Save results to JSON file"""
    output_data = []
    
    for result in results:
        output_data.append({
            "image_id": result.image_id,
            "original_query": result.original_query,
            "refined_queries": [
                {
                    "object_classes": rq.object_classes,
                    "detailed_query": rq.detailed_query
                }
                for rq in result.refined_queries
            ],
            "num_detected": len(result.detected_boxes),
            "num_selected": len(result.selected_boxes),
            "selected_boxes": [box.tolist() for box in result.selected_boxes],
            "obbs": result.obbs,
            "success": result.success,
            "error": result.error
        })
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)


def bbox_to_corners(bbox: List[float]) -> List[float]:
    """
    Convert axis-aligned bbox to 8-point corner format (clockwise from top-left)
    
    Args:
        bbox: [x1, y1, x2, y2] where (x1,y1) is top-left, (x2,y2) is bottom-right
        
    Returns:
        [x1, y1, x2, y2, x3, y3, x4, y4] clockwise from top-left
        Corners: top-left, top-right, bottom-right, bottom-left
    """
    x1, y1, x2, y2 = bbox
    
    # Corners in clockwise order from top-left
    corners = [
        x1, y1,  # top-left
        x2, y1,  # top-right
        x2, y2,  # bottom-right
        x1, y2   # bottom-left
    ]
    
    return corners


def obb_to_corners(obb: List[float]) -> List[float]:
    """
    Convert OBB to 8-point corner format (clockwise from top-left equivalent)
    
    Args:
        obb: [cx, cy, w, h, angle] where angle is in degrees
        
    Returns:
        [x1, y1, x2, y2, x3, y3, x4, y4] clockwise from top-left equivalent
        Corners follow the rotation, starting from what would be top-left before rotation
    """
    cx, cy, w, h, angle = obb
    
    # Convert angle to radians
    theta = np.radians(angle)
    
    # Half dimensions
    dw, dh = w / 2, h / 2
    
    # Corner points in local coordinates (clockwise from top-left)
    local_corners = np.array([
        [-dw, -dh],  # top-left
        [dw, -dh],   # top-right
        [dw, dh],    # bottom-right
        [-dw, dh]    # bottom-left
    ])
    
    # Rotation matrix
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    
    # Rotate and translate to global coordinates
    rotated = local_corners @ R.T
    rotated[:, 0] += cx
    rotated[:, 1] += cy
    
    # Flatten to 8-point format
    corners = rotated.flatten().tolist()
    
    return corners


def extract_all_corners(result) -> Dict:
    """
    Extract HBB and OBB corners from a GroundingResult
    
    Args:
        result: GroundingResult object from pipeline
        
    Returns:
        Dictionary with HBB and OBB corners for all selected boxes
    """
    output = {
        'image_id': result.image_id,
        'query': result.original_query,
        'detections': []
    }
    
    # Process each selected box
    for i, (bbox, obb) in enumerate(zip(result.selected_boxes, result.obbs)):
        detection = {
            'detection_id': i,
            'hbb_xyxy': bbox.tolist(),  # Original [x1, y1, x2, y2]
            'hbb_corners': bbox_to_corners(bbox.tolist()),  # [x1,y1,x2,y2,x3,y3,x4,y4]
        }
        
        # Add OBB if valid
        if obb is not None:
            detection['obb_cxcywha'] = obb  # Original [cx, cy, w, h, angle]
            detection['obb_corners'] = obb_to_corners(obb)  # [x1,y1,x2,y2,x3,y3,x4,y4]
        else:
            detection['obb_cxcywha'] = None
            detection['obb_corners'] = None
        
        output['detections'].append(detection)
    
    return output


def format_corners_for_dota(corners: List[float], difficulty: int = 0) -> str:
    """
    Format corners in DOTA dataset format (for saving to text files)
    
    Args:
        corners: [x1, y1, x2, y2, x3, y3, x4, y4]
        difficulty: 0 for easy, 1 for difficult
        
    Returns:
        String in DOTA format: "x1 y1 x2 y2 x3 y3 x4 y4 category difficulty"
    """
    x1, y1, x2, y2, x3, y3, x4, y4 = corners
    return f"{x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f} {x3:.1f} {y3:.1f} {x4:.1f} {y4:.1f} object {difficulty}"


def save_corners_to_json(results: List, output_path: str):
    """
    Save all HBB and OBB corners to JSON file
    
    Args:
        results: List of GroundingResult objects
        output_path: Path to save JSON file
    """
    all_outputs = []
    
    for result in results:
        corners_data = extract_all_corners(result)
        all_outputs.append(corners_data)
    
    with open(output_path, 'w') as f:
        json.dump(all_outputs, f, indent=2)
    
    print(f"✓ Corners saved to {output_path}")


def save_corners_to_dota_format(results: List, output_dir: str, category: str = "object"):
    """
    Save OBB corners in DOTA dataset format (one .txt file per image)
    
    Args:
        results: List of GroundingResult objects
        output_dir: Directory to save text files
        category: Object category name
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for result in results:
        if not result.success:
            continue
        
        txt_path = os.path.join(output_dir, f"{result.image_id}.txt")
        
        with open(txt_path, 'w') as f:
            for i, (bbox, obb) in enumerate(zip(result.selected_boxes, result.obbs)):
                if obb is not None:
                    corners = obb_to_corners(obb)
                    line = format_corners_for_dota(corners, difficulty=0)
                    # Replace "object" with actual category if available
                    line = line.replace("object", category)
                    f.write(line + '\n')
    
    print(f"✓ DOTA format files saved to {output_dir}")


def print_corners_summary(result):
    """
    Print a formatted summary of HBB and OBB corners for a single result
    
    Args:
        result: GroundingResult object
    """
    print(f"\n{'='*80}")
    print(f"Image: {result.image_id}")
    print(f"Query: {result.original_query}")
    print(f"{'='*80}")
    
    if not result.success or len(result.selected_boxes) == 0:
        print("No detections found.")
        return
    
    for i, (bbox, obb) in enumerate(zip(result.selected_boxes, result.obbs)):
        print(f"\n--- Detection {i+1} ---")
        
        # HBB
        hbb_corners = bbox_to_corners(bbox.tolist())
        print(f"HBB (Horizontal Bounding Box):")
        print(f"  Format [x1,y1,x2,y2]:     {bbox.tolist()}")
        print(f"  8-point corners:")
        print(f"    Top-Left:     ({hbb_corners[0]:.2f}, {hbb_corners[1]:.2f})")
        print(f"    Top-Right:    ({hbb_corners[2]:.2f}, {hbb_corners[3]:.2f})")
        print(f"    Bottom-Right: ({hbb_corners[4]:.2f}, {hbb_corners[5]:.2f})")
        print(f"    Bottom-Left:  ({hbb_corners[6]:.2f}, {hbb_corners[7]:.2f})")
        
        # OBB
        if obb is not None:
            obb_corners = obb_to_corners(obb)
            print(f"\nOBB (Oriented Bounding Box):")
            print(f"  Format [cx,cy,w,h,angle]: {obb}")
            print(f"  8-point corners:")
            print(f"    Corner 1: ({obb_corners[0]:.2f}, {obb_corners[1]:.2f})")
            print(f"    Corner 2: ({obb_corners[2]:.2f}, {obb_corners[3]:.2f})")
            print(f"    Corner 3: ({obb_corners[4]:.2f}, {obb_corners[5]:.2f})")
            print(f"    Corner 4: ({obb_corners[6]:.2f}, {obb_corners[7]:.2f})")
        else:
            print(f"\nOBB: Not generated")

