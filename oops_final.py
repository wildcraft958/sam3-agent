"""
Object-Oriented Satellite Image Area Calculation System

INPUTS:
-------
1. image_path (str): Path to satellite/aerial image file
2. query (str): Natural language query (e.g., "Calculate area of storage tanks")
3. gsd (float): Ground Sample Distance in meters/pixel (image resolution)
4. vlm_endpoint (str): API endpoint for Vision Language Model
5. model_name (str): VLM model identifier

OUTPUTS:
--------
1. SegmentationResults: Contains detected objects with masks, boxes, scores
2. AreaCalculationResults: Area metrics in m² and km²
3. CoverageStatistics: Percentage cover analysis
4. Visualizations: PNG images with annotated detections
5. JSON Report: Complete results in structured format

PIPELINE FLOW:
--------------
Image → Query Refinement (VLM) → Multi-scale Segmentation (SAM3) → 
Mask Validation (VLM) → Area Calculation → Results + Visualizations
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import re
import requests
import base64
from io import BytesIO
from pathlib import Path
from abc import ABC, abstractmethod


# ============================================================================
#                           DATA MODELS
# ============================================================================

@dataclass
class Detection:
    """Single detection result"""
    mask: np.ndarray
    box: np.ndarray  # [x1, y1, x2, y2]
    score: float
    scale: float
    tile_offset: Tuple[int, int]


@dataclass
class ObjectMetrics:
    """Metrics for a single detected object"""
    object_id: int
    box: List[float]
    area_m2: float
    area_km2: float
    percentage_of_image: float
    confidence: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class AreaCalculationResults:
    """Complete results from area calculation pipeline"""
    query: str
    keywords: List[str]
    gsd: float
    image_dimensions: Tuple[int, int]
    total_image_area_m2: float
    total_image_area_km2: float
    total_detected_area_m2: float
    total_detected_area_km2: float
    percentage_cover: float
    num_objects: int
    objects: List[ObjectMetrics]
    detections: List[Detection]
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict"""
        return {
            'query': self.query,
            'keywords': self.keywords,
            'gsd': self.gsd,
            'image_dimensions': self.image_dimensions,
            'total_image_area_m2': self.total_image_area_m2,
            'total_image_area_km2': self.total_image_area_km2,
            'total_detected_area_m2': self.total_detected_area_m2,
            'total_detected_area_km2': self.total_detected_area_km2,
            'percentage_cover': self.percentage_cover,
            'num_objects': self.num_objects,
            'objects': [obj.to_dict() for obj in self.objects]
        }


@dataclass
class CoverageStatistics:
    """Percentage cover statistics"""
    total_image_area_m2: float
    total_image_area_km2: float
    detected_area_m2: float
    detected_area_km2: float
    percentage_cover: float
    object_percentages: List[Dict]
    num_objects: int


# ============================================================================
#                           INTERFACES
# ============================================================================

class VisionLanguageModel(ABC):
    """Abstract base class for VLM interfaces"""
    
    @abstractmethod
    def query(self, image: Image.Image, prompt: str, 
              max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Query the VLM with an image and prompt"""
        pass


class SegmentationModel(ABC):
    """Abstract base class for segmentation models"""
    
    @abstractmethod
    def segment(self, image: Image.Image, prompt: str) -> Dict:
        """Segment image based on text prompt"""
        pass


# ============================================================================
#                           VLM IMPLEMENTATION
# ============================================================================

class QwenVLMInterface(VisionLanguageModel):
    """
    vLLM-based interface for Qwen3-VL model via Modal deployment.
    
    INPUTS:
        - base_url: vLLM server base URL (e.g., "https://user--qwen3-vl-vllm-server-30b.modal.run/v1")
        - model_name: Model identifier
        - timeout: Request timeout in seconds
    
    OUTPUTS:
        - query(): Returns text response from VLM
    """
    
    def __init__(self, 
                 base_url: str = "https://aryan-don357--qwen3-vl-vllm-server-30b-vllm-server.modal.run/v1",
                 model_name: str = "Qwen/Qwen3-VL-30B-A3B-Instruct", 
                 timeout: int = 120):
        self.base_url = base_url.rstrip('/')
        self.endpoint = f"{self.base_url}/chat/completions"
        self.model_name = model_name
        self.timeout = timeout
        
        # Verify server is accessible
        self._check_health()
    
    def _check_health(self):
        """Check if vLLM server is healthy"""
        try:
            health_url = self.base_url.replace('/v1', '/health')
            response = requests.get(health_url, timeout=10)
            if response.status_code == 200:
                print(f"✅ vLLM server healthy: {response.json()}")
            else:
                print(f"⚠️  vLLM server returned status {response.status_code}")
        except Exception as e:
            print(f"⚠️  Could not reach vLLM health endpoint: {e}")
    
    @staticmethod
    def encode_image_to_base64(image: Image.Image) -> str:
        """Encode PIL Image to base64 string"""
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def query(self, image: Image.Image, prompt: str, 
              max_tokens: int = 512, temperature: float = 0.7) -> Optional[str]:
        """
        Query the vLLM API with an image and text prompt.
        
        INPUT:
            - image: PIL Image object
            - prompt: Text prompt/question
            - max_tokens: Maximum response length (capped at 8192 by server)
            - temperature: Sampling temperature (0-1)
            
        OUTPUT:
            - Generated text response or None on error
        """
        image_base64 = self.encode_image_to_base64(image)
        
        # OpenAI-compatible message format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                    }
                ]
            }
        ]
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": min(max_tokens, 8192),  # Server caps at 8192
            "temperature": temperature
        }
        
        try:
            response = requests.post(
                self.endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                print(f"❌ vLLM API Error: {response.status_code}")
                print(f"Response: {response.text}")
                return None
        
        except requests.exceptions.Timeout:
            print(f"❌ Request timeout after {self.timeout}s")
            return None
        except Exception as e:
            print(f"❌ API Exception: {e}")
            return None



# ============================================================================
#                           SEGMENTATION ENGINE
# ============================================================================

class PyramidalSAM3(SegmentationModel):
    """
    Multi-scale pyramidal segmentation using SAM3.
    
    INPUTS:
        - image: PIL Image
        - prompt: Text description of objects to segment
        - tile_size: Tile size for processing large images
        - scales: List of scale factors for pyramid
        
    OUTPUTS:
        - Dictionary with detections, masks, boxes, scores
    """
    
    def __init__(self, model, processor, 
                 tile_size: int = 512,
                 overlap_ratio: float = 0.15,
                 scales: List[float] = [1.0, 0.5, 0.25],
                 iou_threshold: float = 0.5,
                 conf_threshold: float = 0.5):
        self.model = model
        self.processor = processor
        self.tile_size = tile_size
        self.overlap_ratio = overlap_ratio
        self.scales = sorted(scales, reverse=True)
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
    
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
    
    def transform_detection(self, mask, box, score, tile_offset, 
                          scale, orig_size) -> Detection:
        """Transform detection from tile coordinates to original image coordinates"""
        offset_x, offset_y = tile_offset
        
        if torch.is_tensor(box):
            box = box.cpu().numpy()
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy()
        if torch.is_tensor(score):
            score = float(score.cpu().numpy())
        
        box_scaled = box.copy()
        box_scaled[0] += offset_x
        box_scaled[1] += offset_y
        box_scaled[2] += offset_x
        box_scaled[3] += offset_y
        
        box_orig = box_scaled / scale
        
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
    
    def non_max_suppression(self, detections: List[Detection]) -> List[Detection]:
        """Apply Non-Maximum Suppression to remove duplicate detections"""
        if len(detections) == 0:
            return []
        
        detections = sorted(detections, key=lambda d: (-d.score, d.scale))
        keep = []
        
        while detections:
            current = detections.pop(0)
            keep.append(current)
            detections = [
                d for d in detections
                if self.calculate_iou(current.box, d.box) < self.iou_threshold
            ]
        
        return keep
    
    def segment(self, image: Image.Image, prompt: str) -> Dict:
        """
        Perform multi-scale pyramidal segmentation.
        
        INPUT:
            - image: PIL Image
            - prompt: Text description of objects
            
        OUTPUT:
            - Dictionary with detections, masks, boxes, scores
        """
        orig_size = image.size
        all_detections = []
        
        print(f"Processing {len(self.scales)} pyramid levels...")
        
        for scale_idx, scale in enumerate(self.scales):
            print(f"\nLevel {scale_idx + 1}/{len(self.scales)}: Scale {scale}")
            
            if scale != 1.0:
                new_size = (int(orig_size[0] * scale), int(orig_size[1] * scale))
                scaled_image = image.resize(new_size, Image.Resampling.LANCZOS)
            else:
                scaled_image = image
            
            tiles = self.create_tiles(scaled_image)
            print(f"  Generated {len(tiles)} tiles")
            
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
                    
                    if tile_idx % 10 == 0:
                        print(f"  Processed {tile_idx + 1}/{len(tiles)} tiles", end='\r')
                
                except Exception as e:
                    print(f"  Error processing tile {tile_idx}: {e}")
                    continue
            
            print(f"  Completed: {len(tiles)} tiles processed")
        
        print(f"\nTotal detections before NMS: {len(all_detections)}")
        final_detections = self.non_max_suppression(all_detections)
        print(f"Final detections after NMS: {len(final_detections)}")
        
        return {
            'detections': final_detections,
            'masks': [d.mask for d in final_detections],
            'boxes': [d.box for d in final_detections],
            'scores': [d.score for d in final_detections]
        }


# ============================================================================
#                           QUERY PROCESSOR
# ============================================================================

class QueryProcessor:
    """
    Processes natural language queries to extract segmentation keywords.
    
    INPUT:
        - image: PIL Image
        - user_query: Natural language query
        
    OUTPUT:
        - List of segmentation keywords
    """
    
    def __init__(self, vlm: VisionLanguageModel):
        self.vlm = vlm
    
    def refine_query_to_keywords(self, image: Image.Image, 
                                 user_query: str) -> List[str]:
        """
        Use VLM to refine user query into segmentation keywords.
        
        INPUT:
            - image: PIL Image
            - user_query: Natural language query
            
        OUTPUT:
            - List of 2-4 keywords for segmentation
        """
        prompt = f"""Analyze this satellite/aerial image and the user's query to extract segmentation keywords.

User Query: "{user_query}"

Your task:
1. Identify the PRIMARY object(s) the user wants to measure area for
2. Generate 2-4 SHORT keywords/phrases (2-4 words each) that describe these objects visually
3. Focus on: color, shape, texture, context (e.g., "white circular tanks", "rectangular buildings")
4. DO NOT include numbers or counts
5. DO NOT answer the query, just extract visual descriptors

Output format (JSON):
{{
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "primary_object": "main object type",
  "reasoning": "brief explanation"
}}

Now analyze the image and query."""
        
        try:
            response = self.vlm.query(image, prompt, max_tokens=256, temperature=0.7)
            
            if response is None:
                print("VLM returned None, using fallback")
                return [user_query]
            
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                keywords = result.get('keywords', [])
                primary_object = result.get('primary_object', 'object')
                reasoning = result.get('reasoning', '')
                
                print(f"\n=== Query Refinement ===")
                print(f"Primary Object: {primary_object}")
                print(f"Keywords: {keywords}")
                print(f"Reasoning: {reasoning}")
                print(f"========================\n")
                
                return keywords
            else:
                print(f"Could not parse JSON from response: {response}")
                return [user_query]
                
        except Exception as e:
            print(f"Error in query refinement: {e}")
            return [user_query]


# ============================================================================
#                           MASK VALIDATOR
# ============================================================================

class MaskValidator:
    """
    Validates detected masks against original query using VLM.
    
    INPUT:
        - image: PIL Image
        - mask: Binary mask array
        - box: Bounding box [x1, y1, x2, y2]
        - query: Original user query
        
    OUTPUT:
        - Tuple of (is_valid, confidence, reasoning)
    """
    
    def __init__(self, vlm: VisionLanguageModel):
        self.vlm = vlm
    
    def validate(self, image: Image.Image, mask: np.ndarray, 
                box: np.ndarray, original_query: str) -> Tuple[bool, float, str]:
        """
        Validate if a detected mask matches the original query.
        
        INPUT:
            - image: Full PIL Image
            - mask: Binary mask
            - box: Bounding box [x1, y1, x2, y2]
            - original_query: User's original query
            
        OUTPUT:
            - (is_valid, confidence, reasoning)
        """
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.width, x2), min(image.height, y2)
        
        if x2 <= x1 or y2 <= y1:
            return False, 0.0, "Invalid bounding box"
        
        cropped = image.crop((x1, y1, x2, y2))
        
        prompt = f"""This is a detected object from a larger satellite/aerial image.

Original Query: "{original_query}"

Your task: Determine if this detected object matches what the user is looking for.

Answer in JSON format:
{{
  "is_valid": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}}

Consider:
- Does the object type match the query?
- Is it a complete object or partial/edge artifact?
- Does it have the expected visual characteristics?

Be lenient but fair."""
        
        try:
            response = self.vlm.query(cropped, prompt, max_tokens=128, temperature=0.3)
            
            if response is None:
                return True, 0.5, "Validation uncertain (API error)"
            
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                is_valid = result.get('is_valid', False)
                confidence = float(result.get('confidence', 0.5))
                reasoning = result.get('reasoning', 'No reasoning provided')
                
                return is_valid, confidence, reasoning
            else:
                return True, 0.5, "Validation uncertain (parse error)"
                
        except Exception as e:
            print(f"Error in validation: {e}")
            return True, 0.5, "Validation uncertain (exception)"


# ============================================================================
#                           AREA CALCULATOR
# ============================================================================

class AreaCalculator:
    """
    Calculates areas from binary masks using GSD.
    
    INPUT:
        - mask: Binary mask array
        - gsd: Ground Sample Distance (meters/pixel)
        - image_shape: (height, width) of original image
        
    OUTPUT:
        - Area in square meters
    """
    
    @staticmethod
    def calculate_mask_area(mask: np.ndarray, gsd: float, 
                           image_shape: Tuple[int, int]) -> float:
        """
        Calculate area of a binary mask in square meters.
        
        INPUT:
            - mask: Binary mask (can be multi-dimensional)
            - gsd: Ground Sample Distance (m/pixel)
            - image_shape: (height, width) of full image
            
        OUTPUT:
            - Area in square meters
        """
        if mask.ndim > 2:
            mask = mask.squeeze()
        
        if mask.ndim == 3:
            mask = mask[:, :, 0] if mask.shape[2] == 1 else mask.max(axis=2)
        
        if mask.shape[:2] != image_shape:
            from scipy.ndimage import zoom
            zoom_factors = (
                image_shape[0] / mask.shape[0],
                image_shape[1] / mask.shape[1]
            )
            mask = zoom(mask, zoom_factors, order=0) > 0.5
        
        pixel_count = np.sum(mask > 0)
        area_m2 = pixel_count * (gsd ** 2)
        
        return area_m2


# ============================================================================
#                           VISUALIZATION ENGINE
# ============================================================================

class VisualizationEngine:
    """
    Creates visualizations of segmentation results.
    
    INPUTS:
        - image_path: Path to original image
        - results: AreaCalculationResults object
        - save_path: Optional path to save visualization
        
    OUTPUTS:
        - Matplotlib figure saved to disk and/or displayed
    """
    
    @staticmethod
    def visualize_results(image_path: str, results: AreaCalculationResults, 
                         save_path: Optional[str] = None):
        """Visualize detection results with masks and bounding boxes"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        ax.imshow(image_np)
        
        if results.num_objects > 0:
            colors = plt.cm.rainbow(np.linspace(0, 1, results.num_objects))
            
            for idx, detection in enumerate(results.detections):
                mask = detection.mask
                if mask.ndim > 2:
                    mask = mask.squeeze()
                
                if mask.shape[:2] != image_np.shape[:2]:
                    from scipy.ndimage import zoom
                    zoom_factors = (
                        image_np.shape[0] / mask.shape[0],
                        image_np.shape[1] / mask.shape[1]
                    )
                    mask = zoom(mask, zoom_factors, order=0) > 0.5
                
                color = colors[idx]
                mask_colored = np.zeros_like(image_np, dtype=np.float32)
                mask_colored[mask > 0] = np.array(color[:3]) * 255
                
                blended = image_np.astype(np.float32)
                mask_indices = mask > 0
                blended[mask_indices] = (0.6 * image_np[mask_indices] + 
                                        0.4 * mask_colored[mask_indices])
                ax.imshow(blended.astype(np.uint8))
                
                x1, y1, x2, y2 = detection.box
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                    fill=False, color=color, linewidth=2)
                ax.add_patch(rect)
                
                obj = results.objects[idx]
                label = f"#{obj.object_id}: {obj.area_m2:.1f}m²"
                ax.text(x1, y1-10, label, color='white', fontsize=10,
                       bbox=dict(facecolor=color, alpha=0.7))
        
        ax.set_title(f"Detection Results: {results.num_objects} objects, " + 
                    f"Total: {results.total_detected_area_m2:.2f} m²",
                    fontsize=14)
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f" Saved to: {save_path}")
        
        plt.show()
    
    @staticmethod
    def visualize_percentage_cover(image_path: str, results: AreaCalculationResults,
                                   save_path: Optional[str] = None):
        """Create visualization showing percentage cover"""
        fig = plt.figure(figsize=(12, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])
        
        ax1 = fig.add_subplot(gs[0])
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        ax1.imshow(image_np)
        
        if results.num_objects > 0:
            colors = plt.cm.rainbow(np.linspace(0, 1, results.num_objects))
            mask_overlay = np.zeros_like(image_np, dtype=np.float32)
            
            for idx, detection in enumerate(results.detections):
                mask = detection.mask
                
                if mask.ndim > 2:
                    mask = mask.squeeze()
                
                if mask.shape[:2] != image_np.shape[:2]:
                    from scipy.ndimage import zoom
                    zoom_factors = (
                        image_np.shape[0] / mask.shape[0],
                        image_np.shape[1] / mask.shape[1]
                    )
                    mask = zoom(mask, zoom_factors, order=0) > 0.5
                
                color = colors[idx]
                mask_colored = np.zeros_like(image_np, dtype=np.float32)
                mask_colored[mask > 0] = np.array(color[:3]) * 255
                mask_overlay = np.maximum(mask_overlay, mask_colored)
            
            image_with_masks = image_np.astype(np.float32)
            mask_indices = mask_overlay.sum(axis=2) > 0
            image_with_masks[mask_indices] = (0.4 * image_np[mask_indices] + 
                                             0.6 * mask_overlay[mask_indices])
            ax1.imshow(image_with_masks.astype(np.uint8))
        
        ax1.set_title(f"Detected Objects: {results.num_objects}", fontsize=12)
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[1])
        percentage_cover = results.percentage_cover
        remaining = 100 - percentage_cover
        
        colors_pie = ['#ff6b6b', '#e0e0e0']
        explode = (0.05, 0)
        
        ax2.pie([percentage_cover, remaining], 
                labels=[f'Detected\n{percentage_cover:.2f}%', 
                       f'Background\n{remaining:.2f}%'],
                colors=colors_pie,
                autopct='',
                startangle=90,
                explode=explode,
                textprops={'fontsize': 10})
        ax2.set_title('Area Coverage', fontsize=12, pad=10)
        
        fig.suptitle(f'Percentage Cover Analysis\n' + 
                    f'Query: {results.query}\n' + 
                    f'Total: {results.total_detected_area_m2:.2f} m² ({percentage_cover:.2f}% of image)',
                    fontsize=13, y=1.02)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f" Saved to: {save_path}")
        
        plt.show()


# ============================================================================
#                           MAIN PIPELINE
# ============================================================================

class SatelliteImageAreaPipeline:
    """
    Main pipeline for satellite image area calculation.
    
    INPUTS (via process_image):
        - image_path: Path to satellite/aerial image
        - query: Natural language query
        - gsd: Ground Sample Distance (meters/pixel)
        - validate_masks: Whether to use VLM validation
        - validation_threshold: Confidence threshold
        
    OUTPUTS:
        - AreaCalculationResults object with complete metrics
        - Visualizations (if requested)
        - JSON report (if requested)
    """
    
    def __init__(self, 
                 vlm: VisionLanguageModel,
                 segmentation_model: SegmentationModel,
                 device: str = 'cuda'):
        """
        Initialize the pipeline.
        
        INPUTS:
            - vlm: Vision Language Model interface
            - segmentation_model: Segmentation model (PyramidalSAM3)
            - device: 'cuda' or 'cpu'
        """
        self.vlm = vlm
        self.segmentation_model = segmentation_model
        self.device = device
        
        self.query_processor = QueryProcessor(vlm)
        self.mask_validator = MaskValidator(vlm)
        self.area_calculator = AreaCalculator()
        self.visualizer = VisualizationEngine()
    
    def process_image(self, 
                     image_path: str, 
                     query: str, 
                     gsd: float,
                     validate_masks: bool = True,
                     validation_threshold: float = 0.6) -> AreaCalculationResults:
        
        """
        Main processing pipeline.
        
        INPUTS:
            - image_path: Path to image file
            - query: Natural language query (e.g., "Calculate area of tanks")
            - gsd: Ground Sample Distance in meters/pixel
            - validate_masks: Use VLM to validate detections
            - validation_threshold: Minimum confidence for validation
"""

        # Load image
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        img_height, img_width = image_np.shape[:2]
        
        print(f" Loaded image: {image.size}")
        print(f" Dimensions: {img_width}x{img_height}")
        print(f" GSD: {gsd} m/pixel")
        print(f" Query: {query}\n")
        
        # Calculate total image area
        total_image_pixels = img_width * img_height
        total_image_area_m2 = total_image_pixels * (gsd ** 2)
        total_image_area_km2 = total_image_area_m2 / 1_000_000
        
        print(f" Total Image Area: {total_image_area_m2:.2f} m² ({total_image_area_km2:.6f} km²)\n")
        
        # Step 1: Refine query to keywords
        print(" Step 1: Query Refinement")
        keywords = self.query_processor.refine_query_to_keywords(image, query)
        
        # Step 2: Segment with each keyword
        print("\n Step 2: Multi-scale Segmentation")
        all_results = []
        for keyword in keywords:
            print(f"\n--- Keyword: '{keyword}' ---")
            results = self.segmentation_model.segment(image, keyword)
            
            if len(results['boxes']) > 0:
                all_results.append({
                    'keyword': keyword,
                    'results': results
                })
        
        if not all_results:
            print("\n  No objects detected!")
            return self._create_empty_results(
                query, keywords, gsd, img_width, img_height,
                total_image_area_m2, total_image_area_km2
            )
        
        # Step 3: Combine and deduplicate
        print("\n Step 3: Combining Results")
        combined_detections = self._combine_detections(all_results)
        
        # Step 4: Validate masks
        validated_objects = combined_detections
        if validate_masks:
            print(f"\n Step 4: Validating {len(combined_detections)} detections")
            validated_objects = self._validate_detections(
                image, combined_detections, query, validation_threshold
            )
        
        # Step 5: Calculate areas
        print("\n Step 5: Calculating Areas")
        return self._calculate_areas(
            query, keywords, gsd, img_width, img_height,
            total_image_area_m2, total_image_area_km2,
            validated_objects, (img_height, img_width)
        )
    
    def _create_empty_results(self, query, keywords, gsd, width, height,
                             total_area_m2, total_area_km2) -> AreaCalculationResults:
        """Create empty results when no objects detected"""
        return AreaCalculationResults(
            query=query,
            keywords=keywords,
            gsd=gsd,
            image_dimensions=(width, height),
            total_image_area_m2=total_area_m2,
            total_image_area_km2=total_area_km2,
            total_detected_area_m2=0,
            total_detected_area_km2=0,
            percentage_cover=0,
            num_objects=0,
            objects=[],
            detections=[]
        )
    
    def _combine_detections(self, all_results: List[Dict]) -> List[Detection]:
        """Combine and deduplicate detections across keywords"""
        combined = []
        
        for res in all_results:
            for det in res['results']['detections']:
                is_duplicate = False
                for existing in combined:
                    iou = self.segmentation_model.calculate_iou(det.box, existing.box)
                    if iou > 0.5:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    combined.append(det)
        
        print(f"Combined: {len(combined)} unique detections")
        return combined
    
    def _validate_detections(self, image: Image.Image, detections: List[Detection],
                            query: str, threshold: float) -> List[Detection]:
        """Validate detections using VLM"""
        validated = []
        
        for idx, detection in enumerate(detections):
            is_valid, confidence, reasoning = self.mask_validator.validate(
                image, detection.mask, detection.box, query
            )
            
            print(f"  Detection {idx+1}: Valid={is_valid}, Conf={confidence:.2f}, {reasoning}")
            
            if is_valid and confidence >= threshold:
                validated.append(detection)
        
        print(f" Validated: {len(validated)}/{len(detections)}")
        return validated
    
    def _calculate_areas(self, query, keywords, gsd, width, height,
                        total_image_area_m2, total_image_area_km2,
                        detections, image_shape) -> AreaCalculationResults:
        """Calculate areas for all validated detections"""
        total_detected_area_m2 = 0
        object_metrics = []
        
        for idx, detection in enumerate(detections):
            area_m2 = self.area_calculator.calculate_mask_area(
                detection.mask, gsd, image_shape
            )
            
            total_detected_area_m2 += area_m2
            percentage_of_image = (area_m2 / total_image_area_m2) * 100
            
            obj_metric = ObjectMetrics(
                object_id=idx + 1,
                box=detection.box.tolist(),
                area_m2=area_m2,
                area_km2=area_m2 / 1_000_000,
                percentage_of_image=percentage_of_image,
                confidence=detection.score
            )
            object_metrics.append(obj_metric)
            
            print(f"  Object {idx+1}: {area_m2:.2f} m² ({percentage_of_image:.2f}% of image)")
        
        total_detected_area_km2 = total_detected_area_m2 / 1_000_000
        percentage_cover = (total_detected_area_m2 / total_image_area_m2) * 100
        
        print(f"\n{'='*60}")
        print(f" FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Total Image: {total_image_area_m2:.2f} m² ({total_image_area_km2:.6f} km²)")
        print(f"Objects: {len(detections)}")
        print(f"Detected Area: {total_detected_area_m2:.2f} m² ({total_detected_area_km2:.6f} km²)")
        print(f"Coverage: {percentage_cover:.2f}%")
        print(f"{'='*60}\n")
        
        return AreaCalculationResults(
            query=query,
            keywords=keywords,
            gsd=gsd,
            image_dimensions=(width, height),
            total_image_area_m2=total_image_area_m2,
            total_image_area_km2=total_image_area_km2,
            total_detected_area_m2=total_detected_area_m2,
            total_detected_area_km2=total_detected_area_km2,
            percentage_cover=percentage_cover,
            num_objects=len(detections),
            objects=object_metrics,
            detections=detections
        )
    
    def calculate_coverage_statistics(self, results: AreaCalculationResults) -> CoverageStatistics:
        """
        Calculate detailed coverage statistics.
        
        INPUT:
            - results: AreaCalculationResults object
            
        OUTPUT:
            - CoverageStatistics object with detailed metrics
        """
        object_percentages = []
        for obj in results.objects:
            object_percentages.append({
                'object_id': obj.object_id,
                'area_m2': obj.area_m2,
                'percentage_of_total': obj.percentage_of_image
            })
        
        stats = CoverageStatistics(
            total_image_area_m2=results.total_image_area_m2,
            total_image_area_km2=results.total_image_area_km2,
            detected_area_m2=results.total_detected_area_m2,
            detected_area_km2=results.total_detected_area_km2,
            percentage_cover=results.percentage_cover,
            object_percentages=object_percentages,
            num_objects=results.num_objects
        )
        
        print(f"\n{'='*60}")
        print(f"📈 COVERAGE STATISTICS")
        print(f"{'='*60}")
        print(f"Total Image: {stats.total_image_area_m2:.2f} m²")
        print(f"Detected: {stats.detected_area_m2:.2f} m²")
        print(f"Coverage: {stats.percentage_cover:.2f}%")
        print(f"Objects: {stats.num_objects}")
        print(f"{'='*60}\n")
        
        return stats
    
    def save_results_to_json(self, results: AreaCalculationResults, 
                            output_path: str):
        """
        Save results to JSON file.
        
        INPUTS:
            - results: AreaCalculationResults object
            - output_path: Path to save JSON file
            
        OUTPUT:
            - JSON file written to disk
        """
        with open(output_path, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)
        print(f" Results saved to: {output_path}")
    
    def visualize(self, image_path: str, results: AreaCalculationResults,
                 output_dir: Optional[str] = None):
        """
        Generate all visualizations.
        
        INPUTS:
            - image_path: Path to original image
            - results: AreaCalculationResults object
            - output_dir: Optional directory to save visualizations
            
        OUTPUTS:
            - Detection visualization (PNG)
            - Coverage visualization (PNG)
        """
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            results_path = f"{output_dir}/detections.png"
            coverage_path = f"{output_dir}/coverage.png"
        else:
            results_path = None
            coverage_path = None
        
        print("\n Generating Visualizations...")
        
        self.visualizer.visualize_results(
            image_path, results, save_path=results_path
        )
        
        self.visualizer.visualize_percentage_cover(
            image_path, results, save_path=coverage_path
        )

