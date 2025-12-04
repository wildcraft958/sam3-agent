"""
OOP-Based Visual Counting System
=================================

SYSTEM OVERVIEW:
- Pyramidal tiling for multi-scale object detection
- VLM-based prompt refinement and verification
- Automatic retry with query rephrasing
- Comprehensive evaluation pipeline

ARCHITECTURE COMPONENTS:
1. PyramidalSAM3: Multi-scale image segmentation
2. VLMInterface: Vision-Language Model API wrapper
3. PromptRefiner: Query-to-visual-prompt conversion
4. DetectionVerifier: Post-detection validation
5. CountingPipeline: End-to-end counting workflow
6. EvaluationManager: Batch evaluation and metrics
"""

import torch
import numpy as np
import pandas as pd
from PIL import Image
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import requests
import base64
import json
import re
from io import BytesIO
from tqdm import tqdm
from abc import ABC, abstractmethod


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class Detection:
    """Single object detection result.
    
    Attributes:
        mask: Binary segmentation mask (numpy array)
        box: Bounding box [x1, y1, x2, y2]
        score: Confidence score (0-1)
        scale: Pyramid scale used for detection
        tile_offset: (x, y) tile position in scaled image
    """
    mask: np.ndarray
    box: np.ndarray
    score: float
    scale: float
    tile_offset: Tuple[int, int]


@dataclass
class VerificationResult:
    """Result of VLM verification for a detection.
    
    Attributes:
        is_valid: Whether detection passed verification
        box: Bounding box that was verified
        score: Detection confidence score
        verification_response: Raw VLM response text
    """
    is_valid: bool
    box: np.ndarray
    score: float
    verification_response: str = ""


@dataclass
class CountingResult:
    """Complete result from counting pipeline.
    
    Attributes:
        count: Final verified object count
        visual_prompt: Final prompt used for detection
        detections: List of verified Detection objects
        verification_info: Detailed verification statistics
        execution_time: Total processing time in seconds
    """
    count: int
    visual_prompt: str
    detections: List[Detection]
    verification_info: Dict[str, Any]
    execution_time: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'count': self.count,
            'visual_prompt': self.visual_prompt,
            'num_detections': len(self.detections),
            'verification_info': self.verification_info,
            'execution_time': self.execution_time
        }


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for counting accuracy.
    
    Attributes:
        total_samples: Total number of evaluated samples
        correct_predictions: Number of correct counts
        accuracy: Percentage accuracy (0-100)
        total_verified: Total verified detections across all samples
        total_rejected: Total rejected detections
        rejection_rate: Percentage of detections rejected (0-100)
        avg_attempts: Average retry attempts per sample
    """
    total_samples: int = 0
    correct_predictions: int = 0
    accuracy: float = 0.0
    total_verified: int = 0
    total_rejected: int = 0
    rejection_rate: float = 0.0
    avg_attempts: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'total_samples': self.total_samples,
            'correct_predictions': self.correct_predictions,
            'accuracy': self.accuracy,
            'verification_stats': {
                'total_verified': self.total_verified,
                'total_rejected': self.total_rejected,
                'rejection_rate': self.rejection_rate,
                'avg_attempts': self.avg_attempts
            }
        }


# ============================================================================
# ABSTRACT BASE CLASSES
# ============================================================================

class BaseSegmentationModel(ABC):
    """Abstract base class for segmentation models."""
    
    @abstractmethod
    def segment(self, image: Image.Image, prompt: str) -> Dict[str, Any]:
        """Segment objects in image based on text prompt.
        
        Args:
            image: PIL Image
            prompt: Text description of objects to segment
            
        Returns:
            Dictionary with 'masks', 'boxes', 'scores'
        """
        pass


class BaseVLMInterface(ABC):
    """Abstract base class for Vision-Language Model interfaces."""
    
    @abstractmethod
    def query(self, image: Image.Image, prompt: str, **kwargs) -> Optional[str]:
        """Query VLM with image and text prompt.
        
        Args:
            image: PIL Image
            prompt: Text query
            **kwargs: Additional parameters
            
        Returns:
            Generated text response or None if error
        """
        pass


# ============================================================================
# PYRAMIDAL SEGMENTATION
# ============================================================================

class PyramidalSAM3(BaseSegmentationModel):
    """Multi-scale pyramidal tiling wrapper for SAM3.
    
    INPUT:
        - image: PIL.Image (RGB)
        - prompt: str (e.g., "car", "white plane")
        
    OUTPUT:
        - Dictionary with:
            'detections': List[Detection]
            'masks': List[np.ndarray]
            'boxes': List[np.ndarray]
            'scores': List[float]
    
    PARAMETERS:
        - tile_size: Size of each tile (default 512x512)
        - overlap_ratio: Overlap between tiles (default 0.15)
        - scales: Pyramid scales [1.0, 0.5, 0.25]
        - iou_threshold: IoU threshold for NMS (default 0.5)
        - conf_threshold: Min confidence to keep (default 0.5)
    """
    
    def __init__(
        self,
        model: Any,
        processor: Any,
        tile_size: int = 512,
        overlap_ratio: float = 0.15,
        scales: List[float] = None,
        iou_threshold: float = 0.5,
        conf_threshold: float = 0.5
    ):
        self.model = model
        self.processor = processor
        self.tile_size = tile_size
        self.overlap_ratio = overlap_ratio
        self.scales = sorted(scales or [1.0, 0.5, 0.25], reverse=True)
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
    
    def segment(self, image: Image.Image, prompt: str) -> Dict[str, Any]:
        """Main segmentation method (implements abstract method)."""
        return self.segment_with_tiling(image, prompt)
    
    def create_tiles(self, image: Image.Image) -> List[Tuple[Image.Image, Tuple[int, int]]]:
        """Generate overlapping tiles from image.
        
        Returns:
            List of (tile_image, (offset_x, offset_y))
        """
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
        mask: np.ndarray,
        box: np.ndarray,
        score: float,
        tile_offset: Tuple[int, int],
        scale: float,
        orig_size: Tuple[int, int]
    ) -> Detection:
        """Transform detection from tile coordinates to original image space."""
        offset_x, offset_y = tile_offset
        
        # Convert tensors to numpy
        if torch.is_tensor(box):
            box = box.cpu().numpy()
        if torch.is_tensor(mask):
            mask = mask.cpu().numpy()
        if torch.is_tensor(score):
            score = float(score.cpu().numpy())
        
        # Apply transformations
        box_scaled = box.copy()
        box_scaled[0] += offset_x
        box_scaled[1] += offset_y
        box_scaled[2] += offset_x
        box_scaled[3] += offset_y
        
        box_orig = box_scaled / scale
        
        # Clip to bounds
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
        """Calculate Intersection over Union between two boxes."""
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
        """Apply NMS to remove duplicates. Prefers higher resolution detections."""
        if not detections:
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
    
    def segment_with_tiling(self, image: Image.Image, prompt: str) -> Dict[str, Any]:
        """Perform pyramidal tiling segmentation."""
        orig_size = image.size
        all_detections = []
        
        print(f"Processing {len(self.scales)} pyramid levels...")
        
        for scale_idx, scale in enumerate(self.scales):
            print(f"\nLevel {scale_idx + 1}/{len(self.scales)}: Scale {scale}")
            
            # Scale image
            if scale != 1.0:
                new_size = (int(orig_size[0] * scale), int(orig_size[1] * scale))
                scaled_image = image.resize(new_size, Image.Resampling.LANCZOS)
            else:
                scaled_image = image
            
            # Generate and process tiles
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
        
        # Apply NMS
        final_detections = self.non_max_suppression(all_detections)
        print(f"Final detections after NMS: {len(final_detections)}")
        
        return {
            'detections': final_detections,
            'masks': [d.mask for d in final_detections],
            'boxes': [d.box for d in final_detections],
            'scores': [d.score for d in final_detections]
        }


# ============================================================================
# VLM INTERFACE
# ============================================================================

class QwenVLMInterface(BaseVLMInterface):
    """Qwen3-VL API interface for vision-language tasks.
    
    INPUT:
        - image: PIL.Image
        - prompt: str
        - system_prompt: str (optional)
        - max_tokens: int
        - temperature: float
        
    OUTPUT:
        - str: Generated text response
        - None: If API call fails
    """
    
    def __init__(
        self,
        endpoint: str,
        model_name: str = "Qwen/Qwen3-VL-4B-Instruct",
        timeout: int = 120
    ):
        self.endpoint = endpoint
        self.model_name = model_name
        self.timeout = timeout
    
    @staticmethod
    def encode_image_to_base64(image: Image.Image) -> str:
        """Encode PIL Image to base64 string."""
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def query(
        self,
        image: Image.Image,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> Optional[str]:
        """Query Qwen VLM with image and text."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        image_base64 = self.encode_image_to_base64(image)
        
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                }
            ]
        })
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            response = requests.post(
                self.endpoint,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                print(f"❌ API Error: {response.status_code} - {response.text}")
                return None
        
        except Exception as e:
            print(f"❌ API Exception: {e}")
            return None


# ============================================================================
# PROMPT REFINEMENT
# ============================================================================

class PromptRefiner:
    """Refines counting queries into visual detection prompts using VLM.
    
    INPUT:
        - image: PIL.Image
        - user_query: str (e.g., "How many cars on the bridge?")
        
    OUTPUT:
        - str: Refined visual prompt (e.g., "car")
    """
    
    REFINEMENT_TEMPLATE = """You are a visual descriptor for object segmentation on satellite images. You ONLY describe what objects look like, you NEVER count or answer questions. NEVER INCLUDE NUMBERS IN ANY FORM.

YOUR TASK:
Extract the OBJECT NAME from the user query and describe it for visual detection.
- Output: 1-3 words maximum
- Format: [object name] or [attribute + object name]
- Use singular form

USER QUERY: "{query}"

CORE RULES:
1. NEVER include numbers (no "2", "two", "four", etc.)
2. NEVER count objects or answer questions
3. NEVER change the target object class
4. Stay focused on the EXACT object class mentioned in the query

ATTRIBUTE GUIDELINES:
Include 1-2 visual attributes ONLY when objects are large/clear enough.

SPECIAL CASES:
- Storage tanks → "circular white tank"
- Harbors → "dock"
- Vehicles → "car" or "vehicle"
- Sports facilities → "rectangular basketball court"
- Planes → "white plane" or "plane"

OUTPUT ONLY THE OBJECT DESCRIPTION (1-3 words):"""
    
    REPHRASE_TEMPLATE = """You are a visual descriptor for object segmentation. Generate alternative keywords (max 2-3 words).

Original question: "{query}"
Previous visual prompt that found nothing: "{previous_prompt}"

USE CLOSELY RELATED SYNONYMS OF THE PREVIOUS PROMPT.

Generate an ALTERNATIVE visual keyword or synonym (2-3 words).
Consider:
- Different color variations
- Size descriptions (small, large, tiny)
- Shape alternatives
- Context clues from the image

DO NOT include numbers. Output ONLY the alternative object description.

Alternative description:"""
    
    def __init__(self, vlm_interface: BaseVLMInterface):
        self.vlm = vlm_interface
    
    def refine(self, image: Image.Image, user_query: str) -> str:
        """Refine user query into visual prompt."""
        prompt = self.REFINEMENT_TEMPLATE.format(query=user_query)
        
        refined = self.vlm.query(
            image=image,
            prompt=prompt,
            max_tokens=20,
            temperature=0.7
        )
        
        if refined is None:
            return ""
        
        # Clean response
        refined = refined.strip().lower()
        refined = re.sub(r'\d+', '', refined).strip()
        
        return refined
    
    def rephrase(
        self,
        image: Image.Image,
        original_query: str,
        previous_prompt: str
    ) -> str:
        """Generate alternative prompt when no detections found."""
        prompt = self.REPHRASE_TEMPLATE.format(
            query=original_query,
            previous_prompt=previous_prompt
        )
        
        rephrased = self.vlm.query(
            image=image,
            prompt=prompt,
            max_tokens=20,
            temperature=0.7
        )
        
        if rephrased is None:
            return previous_prompt
        
        rephrased = rephrased.strip().lower()
        rephrased = re.sub(r'\d+', '', rephrased).strip()
        
        return rephrased


# ============================================================================
# DETECTION VERIFICATION
# ============================================================================

class DetectionVerifier:
    """Verifies detections match the query using VLM.
    
    INPUT:
        - image: PIL.Image
        - box: List[float] [x1, y1, x2, y2]
        - query: str (original user query)
        - visual_prompt: str (detection prompt used)
        
    OUTPUT:
        - VerificationResult object with is_valid boolean
    """
    
    VERIFICATION_TEMPLATE = """You are a visual verification assistant. 
Answer ONLY with 'yes' or 'no'. DEFAULT 'yes'

TASK:
1. Look only at the region inside the bounding box.
2. Check if the object CLEARLY matches the USER QUERY or VISUAL PROMPT.

USER QUERY: {query}
VISUAL PROMPT: {visual_prompt}

ACCEPT (answer 'yes') ONLY IF:
- The object type clearly matches the query/prompt
- The object is reasonably visible and identifiable
- Visual attributes match when specified
- Location constraints in query are met
- Region contains primarily the target object

REJECT (answer 'no') IF:
- Object type is different from requested
- Region contains multiple different object types
- Location constraints not met

Respond with ONLY 'yes' or 'no'."""
    
    def __init__(self, vlm_interface: BaseVLMInterface):
        self.vlm = vlm_interface
    
    def verify(
        self,
        image: Image.Image,
        box: List[float],
        query: str,
        visual_prompt: str
    ) -> VerificationResult:
        """Verify if detection matches query."""
        try:
            prompt = self.VERIFICATION_TEMPLATE.format(
                query=query,
                visual_prompt=visual_prompt
            )
            
            response = self.vlm.query(
                image=image,
                prompt=prompt,
                max_tokens=4,
                temperature=0.3
            )
            
            if response is None:
                return VerificationResult(
                    is_valid=False,
                    box=np.array(box),
                    score=0.0,
                    verification_response=""
                )
            
            is_valid = "yes" in response.strip().lower()
            
            return VerificationResult(
                is_valid=is_valid,
                box=np.array(box),
                score=1.0 if is_valid else 0.0,
                verification_response=response
            )
        
        except Exception as e:
            print(f"Verification error: {e}")
            return VerificationResult(
                is_valid=False,
                box=np.array(box),
                score=0.0,
                verification_response=""
            )
    
    def verify_batch(
        self,
        image: Image.Image,
        boxes: List[np.ndarray],
        scores: List[float],
        query: str,
        visual_prompt: str
    ) -> List[VerificationResult]:
        """Verify multiple detections."""
        results = []
        
        for box, score in zip(boxes, scores):
            result = self.verify(image, box.tolist(), query, visual_prompt)
            result.score = score  # Preserve original detection score
            results.append(result)
        
        return results


# ============================================================================
# COUNTING PIPELINE
# ============================================================================

class CountingPipeline:
    """End-to-end counting pipeline with verification and retry.
    
    INPUT:
        - image: PIL.Image
        - query: str (e.g., "How many planes are on the tarmac?")
        - max_retries: int (default 2)
        
    OUTPUT:
        - CountingResult object containing:
            * count: int
            * visual_prompt: str
            * detections: List[Detection]
            * verification_info: Dict
            * execution_time: float
    """
    
    def __init__(
        self,
        segmentation_model: BaseSegmentationModel,
        prompt_refiner: PromptRefiner,
        verifier: DetectionVerifier
    ):
        self.segmentation = segmentation_model
        self.refiner = prompt_refiner
        self.verifier = verifier
    
    def count(
        self,
        image: Image.Image,
        query: str,
        max_retries: int = 2
    ) -> CountingResult:
        """Execute counting pipeline with retry logic."""
        import time
        start_time = time.time()
        
        verification_info = {
            'attempts': [],
            'verified_boxes': [],
            'rejected_boxes': []
        }
        
        # Initial prompt refinement
        visual_prompt = self.refiner.refine(image, query)
        attempt_num = 0
        
        while attempt_num <= max_retries:
            print(f"Attempt {attempt_num + 1}: Using prompt '{visual_prompt}'")
            
            try:
                # Segment objects
                results = self.segmentation.segment(image, visual_prompt)
                
                boxes = results.get("boxes", [])
                scores = results.get("scores", [])
                
                verification_info['attempts'].append({
                    'attempt': attempt_num + 1,
                    'prompt': visual_prompt,
                    'initial_detections': len(boxes)
                })
                
                # If no detections and retries left, rephrase
                if len(boxes) == 0 and attempt_num < max_retries:
                    print("  No detections. Rephrasing...")
                    visual_prompt = self.refiner.rephrase(
                        image, query, visual_prompt
                    )
                    attempt_num += 1
                    continue
                
                # Verify detections
                if len(boxes) > 0:
                    print(f"  Verifying {len(boxes)} detections...")
                    
                    verification_results = self.verifier.verify_batch(
                        image, boxes, scores, query, visual_prompt
                    )
                    
                    verified_detections = []
                    
                    for idx, (detection, ver_result) in enumerate(
                        zip(results['detections'], verification_results)
                    ):
                        if ver_result.is_valid:
                            verified_detections.append(detection)
                            verification_info['verified_boxes'].append({
                                'box': ver_result.box.tolist(),
                                'score': ver_result.score
                            })
                        else:
                            verification_info['rejected_boxes'].append({
                                'box': ver_result.box.tolist(),
                                'score': ver_result.score
                            })
                    
                    print(f"  Verified: {len(verified_detections)}/{len(boxes)}")
                    
                    execution_time = time.time() - start_time
                    
                    return CountingResult(
                        count=len(verified_detections),
                        visual_prompt=visual_prompt,
                        detections=verified_detections,
                        verification_info=verification_info,
                        execution_time=execution_time
                    )
                
                # No detections and no retries left
                execution_time = time.time() - start_time
                return CountingResult(
                    count=0,
                    visual_prompt=visual_prompt,
                    detections=[],
                    verification_info=verification_info,
                    execution_time=execution_time
                )
            
            except Exception as e:
                print(f"Error in attempt {attempt_num + 1}: {e}")
                attempt_num += 1
                if attempt_num > max_retries:
                    execution_time = time.time() - start_time
                    return CountingResult(
                        count=0,
                        visual_prompt=visual_prompt,
                        detections=[],
                        verification_info=verification_info,
                        execution_time=execution_time
                    )
        
        execution_time = time.time() - start_time
        return CountingResult(
            count=0,
            visual_prompt=visual_prompt,
            detections=[],
            verification_info=verification_info,
            execution_time=execution_time
        )


# ============================================================================
# VISUALIZATION
# ============================================================================

class ResultVisualizer:
    """Visualize counting results with verified/rejected boxes.
    
    INPUT:
        - image: PIL.Image
        - counting_result: CountingResult
        - query: str
        - ground_truth: int (optional)
        - save_path: str
        
    OUTPUT:
        - Saves visualization to save_path
    """
    
    @staticmethod
    def visualize(
        image: Image.Image,
        counting_result: CountingResult,
        query: str,
        ground_truth: Optional[int] = None,
        save_path: str = "result.png"
    ):
        """Create and save visualization."""
        fig, ax = plt.subplots(1, figsize=(14, 14))
        ax.imshow(image)
        
        verification_info = counting_result.verification_info
        
        # Draw rejected boxes (red, dashed)
        for rejected in verification_info['rejected_boxes']:
            box = rejected['box']
            score = rejected['score']
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            
            rect = patches.Rectangle(
                (x1, y1), w, h,
                linewidth=2,
                edgecolor='red',
                facecolor='none',
                linestyle='--'
            )
            ax.add_patch(rect)
            ax.text(
                x1, y1 - 5, f'✗ {score:.2f}',
                color='white',
                fontsize=8,
                weight='bold',
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.7)
            )
        
        # Draw verified boxes (green)
        for verified in verification_info['verified_boxes']:
            box = verified['box']
            score = verified['score']
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            
            rect = patches.Rectangle(
                (x1, y1), w, h,
                linewidth=2,
                edgecolor='green',
                facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(
                x1, y1 - 5, f'✓ {score:.2f}',
                color='white',
                fontsize=8,
                weight='bold',
                bbox=dict(boxstyle='round', facecolor='green', alpha=0.7)
            )
        
        # Build title (continuation)
        title = f'{correct} Query: {query}\n'
        
        for attempt in verification_info['attempts']:
            title += f"Attempt {attempt['attempt']}: \"{attempt['prompt']}\" → {attempt['initial_detections']} detections\n"
        
        title += f'Final: {len(verification_info["verified_boxes"])} verified, '
        title += f'{len(verification_info["rejected_boxes"])} rejected\n'
        title += f'Predicted: {counting_result.count}'
        if ground_truth is not None:
            title += f' | Ground Truth: {ground_truth}'
        
        ax.set_title(title, fontsize=9, pad=10)
        ax.axis('off')
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"Visualization saved to: {save_path}")


# ============================================================================
# EVALUATION MANAGER
# ============================================================================

class EvaluationManager:
    """Batch evaluation on datasets with metrics tracking.
    
    INPUT:
        - parquet_path: str (path to dataset)
        - images_dir: str (directory with images)
        - output_dir: str (where to save results)
        - max_samples: int (limit number of samples)
        
    OUTPUT:
        - results_df: pd.DataFrame with per-sample results
        - metrics: EvaluationMetrics object
        - Saves: CSV results, JSON summary, visualizations
    """
    
    def __init__(
        self,
        counting_pipeline: CountingPipeline,
        visualizer: ResultVisualizer
    ):
        self.pipeline = counting_pipeline
        self.visualizer = visualizer
    
    @staticmethod
    def extract_number_from_answer(answer_text: str) -> Optional[int]:
        """Extract numerical value from answer text."""
        numbers = re.findall(r'\d+', str(answer_text))
        if numbers:
            return int(numbers[0])
        return None
    
    def evaluate_dataset(
        self,
        parquet_path: str,
        images_dir: str,
        output_dir: str = "evaluation_results",
        max_samples: int = 100,
        max_retries: int = 2
    ) -> Tuple[pd.DataFrame, EvaluationMetrics]:
        """Run evaluation on dataset."""
        import pyarrow.parquet as pq
        
        # Load dataset
        table = pq.read_table(parquet_path)
        df = table.to_pandas()
        
        # Filter numerical questions
        numerical_df = df[df['type'] == 'numerical'].reset_index(drop=True)
        numerical_df = numerical_df.head(max_samples)
        
        print(f"Evaluating {len(numerical_df)} samples")
        
        # Setup output directories
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        vis_path = output_path / "visualizations"
        vis_path.mkdir(exist_ok=True)
        
        images_base = Path(images_dir)
        
        # Tracking variables
        results = []
        metrics = EvaluationMetrics()
        
        # Process each sample
        for idx, row in tqdm(numerical_df.iterrows(), total=len(numerical_df)):
            image_file = row['image_file']
            question = row['question']
            answer = row['answer']
            
            gt_number = self.extract_number_from_answer(answer)
            if gt_number is None:
                continue
            
            image_path = images_base / image_file
            if not image_path.exists():
                continue
            
            try:
                image = Image.open(image_path).convert("RGB")
            except:
                continue
            
            # Run counting pipeline
            try:
                counting_result = self.pipeline.count(
                    image, question, max_retries
                )
                
                predicted_count = counting_result.count
                num_attempts = len(counting_result.verification_info['attempts'])
                verified_count = len(counting_result.verification_info['verified_boxes'])
                rejected_count = len(counting_result.verification_info['rejected_boxes'])
                
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue
            
            # Check correctness
            is_correct = (predicted_count == gt_number)
            if is_correct:
                metrics.correct_predictions += 1
            
            metrics.total_samples += 1
            metrics.total_verified += verified_count
            metrics.total_rejected += rejected_count
            
            # Save visualization
            vis_filename = f"{idx:04d}_{image_file.replace('/', '_')}"
            vis_save_path = vis_path / vis_filename
            
            self.visualizer.visualize(
                image, counting_result, question,
                gt_number, str(vis_save_path)
            )
            
            # Store result
            results.append({
                'index': idx,
                'image_file': image_file,
                'question': question,
                'ground_truth': gt_number,
                'predicted': predicted_count,
                'correct': is_correct,
                'num_attempts': num_attempts,
                'verified_boxes': verified_count,
                'rejected_boxes': rejected_count,
                'visual_prompt': counting_result.visual_prompt,
                'execution_time': counting_result.execution_time
            })
            
            if (idx + 1) % 10 == 0:
                temp_acc = (metrics.correct_predictions / metrics.total_samples) * 100
                print(f"\nProgress: {idx + 1}/{max_samples} | Accuracy: {temp_acc:.2f}%")
        
        # Calculate final metrics
        if metrics.total_samples > 0:
            metrics.accuracy = (metrics.correct_predictions / metrics.total_samples) * 100
            metrics.avg_attempts = sum(r['num_attempts'] for r in results) / len(results)
        
        if (metrics.total_verified + metrics.total_rejected) > 0:
            metrics.rejection_rate = (metrics.total_rejected / 
                                      (metrics.total_verified + metrics.total_rejected)) * 100
        
        # Print final results
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        print(f"Total samples: {metrics.total_samples}")
        print(f"Correct predictions: {metrics.correct_predictions}")
        print(f"Accuracy: {metrics.accuracy:.2f}%")
        print(f"\nVerification Statistics:")
        print(f"  Verified boxes: {metrics.total_verified}")
        print(f"  Rejected boxes: {metrics.total_rejected}")
        print(f"  Rejection rate: {metrics.rejection_rate:.2f}%")
        print(f"  Avg attempts: {metrics.avg_attempts:.2f}")
        print("="*70)
        
        # Save results
        results_df = pd.DataFrame(results)
        results_csv_path = output_path / "results.csv"
        results_df.to_csv(results_csv_path, index=False)
        
        summary_path = output_path / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)
        
        print(f"\nResults saved to: {results_csv_path}")
        print(f"Summary saved to: {summary_path}")
        
        return results_df, metrics


# ============================================================================
# SINGLE IMAGE TESTER
# ============================================================================

class SingleImageTester:
    """Test pipeline on individual images.
    
    INPUT:
        - image_path: str
        - query: str
        - ground_truth: Optional[int]
        - output_dir: str
        
    OUTPUT:
        - Dictionary with counting_result and paths
        - Saves visualization
    """
    
    def __init__(
        self,
        counting_pipeline: CountingPipeline,
        visualizer: ResultVisualizer
    ):
        self.pipeline = counting_pipeline
        self.visualizer = visualizer
    
    def test(
        self,
        image_path: str,
        query: str,
        ground_truth: Optional[int] = None,
        output_dir: str = "single_test",
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """Test on single image."""
        print("="*80)
        print("SINGLE IMAGE TEST")
        print("="*80)
        print(f"Image: {image_path}")
        print(f"Query: {query}")
        print("="*80)
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        print(f"✓ Image loaded: {image.size}")
        
        # Run counting
        counting_result = self.pipeline.count(image, query, max_retries)
        
        print(f"\n📊 Results:")
        print(f"   Visual Prompt: {counting_result.visual_prompt}")
        print(f"   Count: {counting_result.count}")
        print(f"   Attempts: {len(counting_result.verification_info['attempts'])}")
        print(f"   Verified: {len(counting_result.verification_info['verified_boxes'])}")
        print(f"   Rejected: {len(counting_result.verification_info['rejected_boxes'])}")
        print(f"   Time: {counting_result.execution_time:.2f}s")
        
        if ground_truth is not None:
            correct = "✓" if counting_result.count == ground_truth else "✗"
            print(f"   Ground Truth: {ground_truth} {correct}")
        
        # Save visualization
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        vis_filename = Path(image_path).stem + "_result.png"
        vis_save_path = output_path / vis_filename
        
        self.visualizer.visualize(
            image, counting_result, query,
            ground_truth, str(vis_save_path)
        )
        
        print("="*80)
        
        return {
            'counting_result': counting_result,
            'visualization_path': str(vis_save_path)
        }


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

"""
EXAMPLE 1: Basic Setup and Single Image Test
============================================

# Initialize components
vlm = QwenVLMInterface(
    endpoint="https://maximuspookus--qwen3-vl-vllm-server-4b-vllm-server.modal.run/v1/chat/completions",
    model_name="Qwen/Qwen3-VL-4B-Instruct"
)

pyramidal_sam = PyramidalSAM3(
    model=your_sam_model,
    processor=your_sam_processor,
    tile_size=512,
    overlap_ratio=0.15,
    scales=[1.0, 0.5, 0.25],
    iou_threshold=0.5,
    conf_threshold=0.5
)

refiner = PromptRefiner(vlm)
verifier = DetectionVerifier(vlm)
pipeline = CountingPipeline(pyramidal_sam, refiner, verifier)

# Test single image
tester = SingleImageTester(pipeline, ResultVisualizer())
result = tester.test(
    image_path="airport.jpg",
    query="How many planes are on the tarmac?",
    ground_truth=12,
    max_retries=2
)

print(f"Predicted: {result['counting_result'].count}")


EXAMPLE 2: Batch Evaluation
============================

# Setup evaluation manager
evaluator = EvaluationManager(pipeline, ResultVisualizer())

# Run evaluation on dataset
results_df, metrics = evaluator.evaluate_dataset(
    parquet_path="dataset.parquet",
    images_dir="images/",
    output_dir="evaluation_output",
    max_samples=100,
    max_retries=2
)

print(f"Accuracy: {metrics.accuracy:.2f}%")
print(f"Rejection rate: {metrics.rejection_rate:.2f}%")


EXAMPLE 3: Custom Segmentation Model
=====================================

class CustomSegmentationModel(BaseSegmentationModel):
    def __init__(self, model):
        self.model = model
    
    def segment(self, image: Image.Image, prompt: str) -> Dict[str, Any]:
        # Your custom segmentation logic
        masks, boxes, scores = self.model.predict(image, prompt)
        return {
            'masks': masks,
            'boxes': boxes,
            'scores': scores,
            'detections': []  # Create Detection objects if needed
        }

# Use custom model in pipeline
custom_model = CustomSegmentationModel(your_model)
pipeline = CountingPipeline(custom_model, refiner, verifier)


EXAMPLE 4: Direct Component Usage
==================================

# Use individual components
image = Image.open("test.jpg")
query = "How many cars?"

# 1. Refine prompt
visual_prompt = refiner.refine(image, query)
print(f"Visual prompt: {visual_prompt}")

# 2. Segment
results = pyramidal_sam.segment(image, visual_prompt)
print(f"Found {len(results['boxes'])} detections")

# 3. Verify
verification_results = verifier.verify_batch(
    image, 
    results['boxes'], 
    results['scores'],
    query, 
    visual_prompt
)

verified_count = sum(1 for v in verification_results if v.is_valid)
print(f"Verified: {verified_count}")


INPUT/OUTPUT SUMMARY
====================

1. PyramidalSAM3.segment()
   INPUT: image (PIL.Image), prompt (str)
   OUTPUT: {
       'detections': List[Detection],
       'masks': List[np.ndarray],
       'boxes': List[np.ndarray],
       'scores': List[float]
   }

2. PromptRefiner.refine()
   INPUT: image (PIL.Image), user_query (str)
   OUTPUT: visual_prompt (str)

3. PromptRefiner.rephrase()
   INPUT: image, original_query (str), previous_prompt (str)
   OUTPUT: alternative_prompt (str)

4. DetectionVerifier.verify()
   INPUT: image, box (List[float]), query (str), visual_prompt (str)
   OUTPUT: VerificationResult (is_valid, box, score, response)

5. CountingPipeline.count()
   INPUT: image (PIL.Image), query (str), max_retries (int)
   OUTPUT: CountingResult (count, visual_prompt, detections, verification_info, execution_time)

6. EvaluationManager.evaluate_dataset()
   INPUT: parquet_path (str), images_dir (str), output_dir (str), max_samples (int)
   OUTPUT: (results_df: pd.DataFrame, metrics: EvaluationMetrics)

7. SingleImageTester.test()
   INPUT: image_path (str), query (str), ground_truth (Optional[int])
   OUTPUT: {
       'counting_result': CountingResult,
       'visualization_path': str
   }
"""