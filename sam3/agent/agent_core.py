# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import copy
import json
import os

import cv2
from PIL import Image

from .client_llm import send_generate_request
from .client_sam3 import call_sam_service
from .viz import visualize
# from .helpers.spatial_filtering import filter_masks_by_spatial_position  # Disabled - spatial filtering done in LLM thinking
from .helpers.region_segmentation import segment_in_region, transform_masks_to_global
from .helpers.attribute_filtering import filter_masks_by_attributes
# from .helpers.relative_spatial import filter_masks_by_relative_position  # Disabled - relative filtering done in LLM thinking
from .helpers.pyramidal_tiling import (
    create_pyramidal_tiles,
    transform_mask_coordinates,
    merge_tile_results,
)
import re
import base64
import requests
import numpy as np
from io import BytesIO
import torch


def parse_tool_calls_from_text(text: str) -> list:
    """
    Parse tool calls from text response when not using OpenAI function calling API.
    
    Supports multiple formats:
    1. JSON blocks: {"name": "function_name", "arguments": {...}}
    2. Function call syntax: function_name(arg1="value1", arg2="value2")
    3. XML-like tags: <function_call>{"name": "...", "arguments": {...}}</function_call>
    
    Returns list of tool call dicts in format:
    [{"id": "...", "function": {"name": "...", "arguments": "..."}}]
    """
    tool_calls = []
    
    # Known function names (matching active tools in TOOLS list)
    known_functions = [
        "segment_phrase", "examine_each_mask", "select_masks_and_return", 
        "report_no_mask", 
        # "filter_masks_by_spatial_position",  # Disabled - spatial filtering done in LLM thinking
        # "segment_phrase_in_region", 
        "filter_masks_by_attributes",
        "segment_with_lisat",  # LISAT segmentation fallback
        # "filter_masks_by_relative_position"  # Disabled - relative filtering done in LLM thinking
        # "segment_phrase_with_tiling"  # Commented out - not currently active
    ]
    
    # Pattern 1: JSON object with name/arguments structure
    # Match patterns like: {"name": "segment_phrase", "arguments": {"text_prompt": "car"}}
    json_pattern = r'\{[^{}]*"name"\s*:\s*"([^"]+)"[^{}]*"arguments"\s*:\s*(\{[^{}]*\})[^{}]*\}'
    for match in re.finditer(json_pattern, text, re.DOTALL):
        func_name = match.group(1)
        args_str = match.group(2)
        if func_name in known_functions:
            tool_calls.append({
                "id": f"call_{len(tool_calls)}",
                "type": "function",  # Required by OpenAI/vLLM format
                "function": {
                    "name": func_name,
                    "arguments": args_str
                }
            })
    
    if tool_calls:
        return tool_calls
    
    # Pattern 2: Function call syntax like segment_phrase(text_prompt="car")
    for func_name in known_functions:
        # Match function_name followed by parentheses with arguments
        pattern = rf'{func_name}\s*\(\s*([^)]*)\s*\)'
        for match in re.finditer(pattern, text):
            args_text = match.group(1).strip()
            
            # Parse arguments into dict
            args_dict = {}
            if args_text:
                # Handle keyword arguments like: text_prompt="car", other_arg=123
                kwarg_pattern = r'(\w+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|(\[[^\]]*\])|(\{[^}]*\})|([^,\s]+))'
                for kwmatch in re.finditer(kwarg_pattern, args_text):
                    key = kwmatch.group(1)
                    # Get the first non-None value from the capture groups
                    value = kwmatch.group(2) or kwmatch.group(3) or kwmatch.group(4) or kwmatch.group(5) or kwmatch.group(6)
                    if value:
                        # Try to parse as JSON for arrays/objects/numbers
                        try:
                            args_dict[key] = json.loads(value)
                        except (json.JSONDecodeError, TypeError):
                            args_dict[key] = value
            
            tool_calls.append({
                "id": f"call_{len(tool_calls)}",
                "type": "function",  # Required by OpenAI/vLLM format
                "function": {
                    "name": func_name,
                    "arguments": json.dumps(args_dict)
                }
            })
    
    if tool_calls:
        return tool_calls
    
    # Pattern 3: Look for function names followed by JSON arguments
    # e.g., segment_phrase {"text_prompt": "car"}
    for func_name in known_functions:
        pattern = rf'{func_name}\s*(\{{[^{{}}]*\}})'
        for match in re.finditer(pattern, text):
            args_str = match.group(1)
            try:
                # Validate it's valid JSON
                json.loads(args_str)
                tool_calls.append({
                    "id": f"call_{len(tool_calls)}",
                    "type": "function",  # Required by OpenAI/vLLM format
                    "function": {
                        "name": func_name,
                        "arguments": args_str
                    }
                })
            except json.JSONDecodeError:
                pass
    
    if tool_calls:
        return tool_calls
    
    # Pattern 4: Look for simple mentions of functions with text_prompt in quotes nearby
    # This is a fallback for less structured outputs
    if "segment_phrase" in text:
        # Look for text_prompt patterns
        prompt_patterns = [
            r'"text_prompt"\s*:\s*"([^"]+)"',
            r"'text_prompt'\s*:\s*'([^']+)'",
            r'text_prompt\s*=\s*"([^"]+)"',
            r"text_prompt\s*=\s*'([^']+)'",
        ]
        for pattern in prompt_patterns:
            match = re.search(pattern, text)
            if match:
                text_prompt = match.group(1)
                tool_calls.append({
                    "id": "call_0",
                    "type": "function",  # Required by OpenAI/vLLM format
                    "function": {
                        "name": "segment_phrase",
                        "arguments": json.dumps({"text_prompt": text_prompt})
                    }
                })
                return tool_calls
    
    # Pattern 5: Check for other tools without arguments
    for func_name in ["examine_each_mask", "report_no_mask"]:
        if func_name in text and func_name not in [tc["function"]["name"] for tc in tool_calls]:
            # These tools don't require arguments
            tool_calls.append({
                "id": f"call_{len(tool_calls)}",
                "type": "function",  # Required by OpenAI/vLLM format
                "function": {
                    "name": func_name,
                    "arguments": "{}"
                }
            })
            return tool_calls
    
    # Pattern 6: select_masks_and_return with final_answer_masks
    if "select_masks_and_return" in text:
        # Look for array of integers
        masks_pattern = r'"final_answer_masks"\s*:\s*\[([^\]]+)\]'
        match = re.search(masks_pattern, text)
        if match:
            masks_str = match.group(1)
            try:
                masks = json.loads(f"[{masks_str}]")
                tool_calls.append({
                    "id": "call_0",
                    "type": "function",  # Required by OpenAI/vLLM format
                    "function": {
                        "name": "select_masks_and_return",
                        "arguments": json.dumps({"final_answer_masks": masks})
                    }
                })
                return tool_calls
            except json.JSONDecodeError:
                pass
        
        # Alternative: look for array directly after function name
        alt_pattern = r'select_masks_and_return[^[]*\[([^\]]+)\]'
        match = re.search(alt_pattern, text)
        if match:
            masks_str = match.group(1)
            try:
                masks = json.loads(f"[{masks_str}]")
                tool_calls.append({
                    "id": "call_0",
                    "type": "function",  # Required by OpenAI/vLLM format
                    "function": {
                        "name": "select_masks_and_return",
                        "arguments": json.dumps({"final_answer_masks": masks})
                    }
                })
                return tool_calls
            except json.JSONDecodeError:
                pass
    
    return tool_calls


# OpenAI function calling format for Qwen3
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "segment_phrase",
            "description": "Use SAM3 (Segment Anything Model 3) to segment all instances of a feature by generating segmentation mask(s). Works best with simple noun phrases (1-3 words). All previously generated mask(s) will be deleted when called.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text_prompt": {
                        "type": "string",
                        "description": "A simple noun phrase (1-3 words). Examples: building, road, water, tree, vehicle, ship, field, roof, pool, runway"
                    }
                },
                "required": ["text_prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "examine_each_mask",
            "description": "Use this tool when the segment_phrase tool generates multiple small or overlapping mask(s), making it difficult to distinguish the correct mask(s). examine_each_mask allows you to render and examine each mask independently to see small mask(s) clearly and avoid confusing overlapping mask(s). (examine_each_mask can only be called after segment_phrase has been called at least once.)",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "select_masks_and_return",
            "description": "Call this tool to select a subset of or all of the mask(s) rendered on the most recent image as your final output. When calling select_masks_and_return, you cannot select any mask(s) generated by previous rounds other than the most recent round in your 'final_answer_masks'. You can only use mask(s) from the most recent image in your message history. (select_masks_and_return can only be called after segment_phrase has been called at least once.)",
            "parameters": {
                "type": "object",
                "properties": {
                    "final_answer_masks": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "An array of integers representing the selected mask(s) you want to choose as your final output, e.g., [1, 4, 5]"
                    }
                },
                "required": ["final_answer_masks"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "report_no_mask",
            "description": "Call this tool when you are absolutely sure that there are no object(s) in the image that match or answer the initial user input query.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    # DISABLED: Spatial filtering is now done in LLM thinking process, not as a tool
    # The LLM reasons about mask positions and directly selects appropriate masks
    # using select_masks_and_return instead of calling filter_masks_by_spatial_position
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "filter_masks_by_spatial_position",
    #         "description": "Filter existing masks by their spatial position in the image.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "spatial_criteria": {
    #                     "type": "string",
    #                     "description": "Spatial position criteria"
    #                 }
    #             },
    #             "required": ["spatial_criteria"]
    #         }
    #     }
    # },
    # DISABLED: segment_phrase_in_region - Commented out as per user request
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "segment_phrase_in_region",
    #         "description": "Segment a phrase within a specific bounding box region. Useful when you know the approximate location or want to focus on a specific area.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "text_prompt": {
    #                     "type": "string",
    #                     "description": "A short and simple noun phrase"
    #                 },
    #                 "bbox": {
    #                     "type": "array",
    #                     "items": {"type": "number"},
    #                     "description": "Bounding box [x_min, y_min, x_max, y_max] in normalized coordinates (0-1)",
    #                     "minItems": 4,
    #                     "maxItems": 4
    #                 },
    #                 "use_normalized": {
    #                     "type": "boolean",
    #                     "description": "Whether bbox is in normalized coordinates (default: true)",
    #                     "default": True
    #                 }
    #             },
    #             "required": ["text_prompt", "bbox"]
    #         }
    #     }
    # },
    {
        "type": "function",
        "function": {
            "name": "filter_masks_by_attributes",
            "description": "Filter existing masks by visual attributes like color, size, or shape. Use after segment_phrase to narrow down results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "color": {
                        "type": "string",
                        "description": "Filter by dominant color (e.g., 'red', 'blue', 'white', 'black')"
                    },
                    # "min_size_ratio": {
                    #     "type": "number",
                    #     "description": "Minimum size as ratio of image (0-1)"
                    # },
                    # "max_size_ratio": {
                    #     "type": "number",
                    #     "description": "Maximum size as ratio of image (0-1)"
                    # },
                    # "aspect_ratio_range": {
                    #     "type": "array",
                    #     "items": {"type": "number"},
                    #     "description": "Min and max aspect ratio [min, max]",
                    #     "minItems": 2,
                    #     "maxItems": 2
                    # }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "segment_with_lisat",
            "description": "Use LISAT (Large Language Instructed Segmentation Assistant) as an alternative segmentation method. This tool uses a different model that may succeed when SAM3 fails. Should be used when segment_phrase has been attempted multiple times without success.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text_prompt": {
                        "type": "string",
                        "description": "A descriptive prompt for what to segment. Can be more detailed than SAM3 prompts, including attributes like color, size, location."
                    }
                },
                "required": ["text_prompt"]
            }
        }
    },
    # DISABLED: Relative position filtering is now done in LLM thinking process
    # The LLM reasons about mask positions relative to other objects/masks
    # and directly selects appropriate masks using select_masks_and_return
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "filter_masks_by_relative_position",
    #         "description": "Filter masks based on their position relative to other masks.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "relationship": {"type": "string", "description": "Spatial relationship"},
    #                 "reference_mask_indices": {"type": "array", "items": {"type": "integer"}}
    #             },
    #             "required": ["relationship", "reference_mask_indices"]
    #         }
    #     }
    # },
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "segment_phrase_with_tiling",
    #         "description": "Segment a phrase using pyramidal tiling for better detection of small objects or handling large images. Splits image into overlapping tiles, segments each, and merges results intelligently.",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "text_prompt": {
    #                     "type": "string",
    #                     "description": "A short and simple noun phrase"
    #                 },
    #                 "tile_size": {
    #                     "type": "integer",
    #                     "description": "Size of each tile (default: 1024)",
    #                     "default": 1024
    #                 },
    #                 "overlap_ratio": {
    #                     "type": "number",
    #                     "description": "Overlap ratio between tiles (0.0-0.5, default: 0.2)",
    #                     "default": 0.2
    #                 },
    #                 "min_tile_size": {
    #                     "type": "integer",
    #                     "description": "Minimum tile size for recursive splitting (default: 512)",
    #                     "default": 512
    #                 }
    #             },
    #             "required": ["text_prompt"]
    #         }
    #     }
    # }
]


def save_debug_messages(messages_list, debug, debug_folder_path, debug_jsonl_path):
    """Save messages to debug jsonl file if debug is enabled"""
    if debug and debug_jsonl_path:
        # Ensure the debug directory exists before writing
        os.makedirs(debug_folder_path, exist_ok=True)
        with open(debug_jsonl_path, "w") as f:
            for msg in messages_list:
                f.write(json.dumps(msg, indent=4) + "\n")


def cleanup_debug_files(debug, debug_folder_path, debug_jsonl_path):
    """Clean up debug files when function successfully returns"""
    if debug and debug_folder_path:
        try:
            if os.path.exists(debug_jsonl_path):
                os.remove(debug_jsonl_path)
            if os.path.exists(debug_folder_path):
                os.rmdir(debug_folder_path)
        except Exception as e:
            print(f"Warning: Could not clean up debug files: {e}")


def count_images(messages):
    """Count the total number of images present in the messages history."""
    total = 0
    for message in messages:
        # Check if message has content (should be a list)
        if "content" in message and isinstance(message["content"], list):
            # Iterate through each content item
            for content_item in message["content"]:
                # Check if content item is a dict with type "image"
                if (
                    isinstance(content_item, dict)
                    and content_item.get("type") == "image"
                ):
                    total += 1
    return total


def call_lisat_segmentation(image_path, text_prompt, modal_url, max_new_tokens=256):
    """
    Call LISAT segmentation service via Modal endpoint.
    
    Args:
        image_path: Path to the image file
        text_prompt: Text prompt for segmentation
        modal_url: Modal deployment URL
        max_new_tokens: Maximum tokens for response
        
    Returns:
        Dict with segmentation results in SAM3 format
    """
    try:
        # Read and encode image to base64
        with open(image_path, "rb") as img_file:
            image_bytes = img_file.read()
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Prepare request
        payload = {
            "prompt": text_prompt,
            "image_base64": image_b64,
            "max_new_tokens": max_new_tokens
        }
        
        print(f"🔍 Calling LISAT API with prompt: {text_prompt}")
        
        # Call LISAT API
        response = requests.post(
            modal_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=120  # 2 minute timeout
        )
        
        if response.status_code != 200:
            raise ValueError(f"LISAT API returned status {response.status_code}: {response.text}")
        
        result = response.json()
        print(f"✓ LISAT API returned successfully")
        
        # New LISAT output format: {'text', 'has_seg', 'mask_base64', 'mask_shape'}
        if not result.get("has_seg", False) or "mask_base64" not in result:
            print("⚠ LISAT returned no segmentation")
            return None
        
        # Decode mask from base64
        mask_b64 = result["mask_base64"]
        mask_bytes = base64.b64decode(mask_b64)
        mask_array = np.frombuffer(mask_bytes, dtype=np.uint8)
        mask_img = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE)
        
        if mask_img is None:
            print("❌ Failed to decode mask image")
            return None
        
        # Extract bounding boxes from mask using cv2
        # Find all unique non-zero values in the mask (each represents a different object)
        unique_values = np.unique(mask_img)
        unique_values = unique_values[unique_values > 0]  # Remove background (0)
        
        if len(unique_values) == 0:
            print("⚠ No objects found in mask")
            return None
        
        # Load original image to get dimensions
        original_image = Image.open(image_path)
        orig_img_w, orig_img_h = original_image.size
        
        pred_boxes = []
        pred_masks = []
        pred_scores = []
        
        # Process each unique object in the mask
        for obj_id in unique_values:
            # Create binary mask for this object
            binary_mask = (mask_img == obj_id).astype(np.uint8) * 255
            
            # Find contours
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) == 0:
                continue
            
            # Get bounding box from the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Normalize coordinates to [0, 1]
            x_norm = x / orig_img_w
            y_norm = y / orig_img_h
            w_norm = w / orig_img_w
            h_norm = h / orig_img_h
            
            # Store in XYWH format (normalized)
            pred_boxes.append([x_norm, y_norm, w_norm, h_norm])
            
            # Convert mask to RLE format (SAM3 format)
            # For simplicity, store as binary array; SAM3 visualizer can handle it
            from sam3.train.masks_ops import rle_encode
            mask_tensor = torch.from_numpy(binary_mask).unsqueeze(0)
            rle_mask = rle_encode(mask_tensor)
            pred_masks.append(rle_mask[0])  # Get first (and only) mask
            
            # Assign score of 1.0 (high confidence)
            pred_scores.append(1.0)
        
        if len(pred_boxes) == 0:
            print("⚠ No valid bounding boxes extracted")
            return None
        
        print(f"✓ Extracted {len(pred_boxes)} masks from LISAT output")
        
        # Convert LISAT format to SAM3 format
        sam3_format = {
            "original_image_path": image_path,
            "orig_img_h": orig_img_h,
            "orig_img_w": orig_img_w,
            "pred_boxes": pred_boxes,
            "pred_masks": pred_masks,
            "pred_scores": pred_scores,
        }
        
        return sam3_format
        
    except requests.exceptions.Timeout:
        print("❌ LISAT API call timed out")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ LISAT API call failed: {e}")
        return None
    except Exception as e:
        print(f"❌ Error in LISAT segmentation: {e}")
        import traceback
        traceback.print_exc()
        return None


def _prune_messages_for_next_round(
    messages_list,
    used_text_prompts,
    latest_sam3_text_prompt,
    img_path,
    initial_text_prompt,
):
    """Return a new messages list that contains only:
    1) messages[:2] (with optional warning text added to the second message's content)
    2) the latest assistant message (and everything after it) that contains a segment_phrase tool call
    """
    # Warn if messages list is getting large, but don't crash - pruning will handle it
    if len(messages_list) >= 10:
        print(f"Warning: messages_list has {len(messages_list)} messages before pruning")

    # Part 1: always keep the first two message JSONs
    part1 = copy.deepcopy(messages_list[:2])

    # Part 2: search backwards for the latest assistant message containing a segment_phrase tool call
    part2_start_idx = None
    for idx in range(len(messages_list) - 1, 1, -1):
        msg = messages_list[idx]
        # We only consider assistant messages
        if msg.get("role") != "assistant":
            continue
        # Check for tool_calls field (structured format)
        if "tool_calls" in msg and msg["tool_calls"]:
            for tool_call in msg["tool_calls"]:
                function_name = tool_call.get("function", {}).get("name", "")
                if function_name == "segment_phrase":
                    part2_start_idx = idx
                    break
        # Fallback: check for <tool> tags in content (backward compatibility)
        elif "content" in msg and isinstance(msg["content"], list):
            for content in msg["content"]:
                if (
                    isinstance(content, dict)
                    and content.get("type") == "text"
                    and "<tool>" in content.get("text", "")
                    and "segment_phrase" in content.get("text", "")
                ):
                    part2_start_idx = idx
                    break
        if part2_start_idx is not None:
            break

    part2 = messages_list[part2_start_idx:] if part2_start_idx is not None else []

    # Part 3: decide whether to add warning text to the second message in part1
    previously_used = (
        [p for p in used_text_prompts if p != latest_sam3_text_prompt]
        if latest_sam3_text_prompt
        else list(used_text_prompts)
    )
    if part2 and len(previously_used) > 0:
        warning_text = f'Note that we have previously called the segment_phrase tool with each "text_prompt" in this list: {list(previously_used)}, but none of the generated results were satisfactory. So make sure that you do not use any of these phrases as the "text_prompt" to call the segment_phrase tool again.'
        # Replace the second message entirely to keep exactly 2 content items
        part1[1] = {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {
                    "type": "text",
                    "text": f"The above image is the raw input image. The initial user input query is: '{initial_text_prompt}'."
                    + " "
                    + warning_text,
                },
            ],
        }
        assert len(part1[1]["content"]) == 2

    # Build the new messages list: part1 (with optional warning), then part2
    new_messages = list(part1)
    new_messages.extend(part2)
    return new_messages


def agent_inference(
    img_path: str,
    initial_text_prompt: str,
    debug: bool = False,
    send_generate_request=send_generate_request,
    call_sam_service=call_sam_service,
    max_generations: int = 100,
    output_dir="../../sam3_agent_out",
):
    """
    Given a text prompt and an image, this tool will perform all aspects of agentic problem solving,
    while saving sam3 and MLLM outputs to their respective directories.

    Args:
        img_path: Path to the input image
        initial_text_prompt: Initial text prompt from the user
        debug: Whether to enable debug mode
        max_generations: Maximum number of send_generate_request calls allowed (default: 100)
    """
    # setup dir
    sam_output_dir = os.path.join(output_dir, "sam_out")
    error_save_dir = os.path.join(output_dir, "none_out")
    debug_save_dir = os.path.join(output_dir, "agent_debug_out")
    os.makedirs(sam_output_dir, exist_ok=True)
    os.makedirs(error_save_dir, exist_ok=True)
    os.makedirs(debug_save_dir, exist_ok=True)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    MLLM_SYSTEM_PROMPT_PATH = os.path.join(
        current_dir, "system_prompts/system_prompt_remote_sensing.txt"
    )
    ITERATIVE_CHECKING_SYSTEM_PROMPT_PATH = os.path.join(
        current_dir, "system_prompts/system_prompt_iterative_checking_remote_sensing.txt"
    )
    # init variables
    PATH_TO_LATEST_OUTPUT_JSON = ""
    LATEST_SAM3_TEXT_PROMPT = ""
    USED_TEXT_PROMPTS = (
        set()
    )  # Track all previously used text prompts for segment_phrase
    generation_count = 0  # Counter for number of send_generate_request calls
    
    # Track tool call history for fallback logic
    recent_tool_calls = []  # Track last 3 tool calls
    sam3_segmentation_count = 0  # Count SAM3 segmentation attempts
    lisat_attempts = 0  # Count LISAT attempts
    lisat_used_prompts = set()  # Track LISAT prompts to avoid repeats
    LISAT_MODAL_URL = "https://srinjoy59--lisat-seg-api-infer.modal.run"

    # debug setup
    debug_folder_path = None
    debug_jsonl_path = None
    if debug:
        debug_folder_path = os.path.join(
            debug_save_dir, f"{img_path.rsplit('/', 1)[-1].rsplit('.', 1)[0]}"
        )
        debug_jsonl_path = os.path.join(debug_folder_path, "debug_history.json")
        os.makedirs(debug_folder_path, exist_ok=True)

    # The helper functions are now defined outside the agent_inference function
    with open(MLLM_SYSTEM_PROMPT_PATH, "r") as f:
        system_prompt = f.read().strip()
    with open(ITERATIVE_CHECKING_SYSTEM_PROMPT_PATH, "r") as f:
        iterative_checking_system_prompt = f.read().strip()

    # Construct the initial message list
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {
                    "type": "text",
                    "text": f"The above image is the raw input image. The initial user input query is: '{initial_text_prompt}'.",
                },
            ],
        },
    ]
    print(f"> Text prompt: {initial_text_prompt}")
    print(f"> Image path: {img_path}")

    print("\n\n")
    print("-" * 30 + f" Round {str(generation_count + 1)}" + "-" * 30)
    print("\n\n")
    # Check before first call
    if generation_count >= max_generations:
        raise ValueError(
            f"Exceeded maximum number of allowed generation requests ({max_generations})"
        )
    generation_count += 1
    generated_response = send_generate_request(messages, tools=TOOLS)
    print(f"\n>>> MLLM Response [start]\n{generated_response}\n<<< MLLM Response [end]\n")
    
    # Handle structured tool calls or plain text response
    while generated_response is not None:
        save_debug_messages(messages, debug, debug_folder_path, debug_jsonl_path)
        
        # Initialize tool_calls_list to avoid NameError
        tool_calls_list = []
        generated_text = ""
        
        # Comprehensive type checking for all possible response types
        if generated_response is None:
            print("❌ Error: generated_response is None")
            print(f"   This indicates the LLM API returned no response")
            break
        
        elif isinstance(generated_response, dict):
            # Check if response contains structured tool calls
            if "tool_calls" in generated_response:
                tool_calls_list = generated_response.get("tool_calls", [])
                generated_text = generated_response.get("content", "")
                
                # Validate tool_calls_list is actually a list
                if not isinstance(tool_calls_list, list):
                    print(f"❌ Error: tool_calls is not a list, got {type(tool_calls_list)}")
                    print(f"   Response: {str(generated_response)[:500]}")
                    break
                
                if not tool_calls_list:
                    print("⚠️ Warning: tool_calls list is empty")
                    break
                    
                print(f"✅ Received {len(tool_calls_list)} tool call(s) in structured format")
            else:
                # Dict response without tool_calls - might be an error response
                print(f"⚠️ Warning: Received dict response without 'tool_calls' field")
                print(f"   Response keys: {list(generated_response.keys())}")
                print(f"   Response: {str(generated_response)[:500]}")
                break
                
        elif isinstance(generated_response, list):
            # Handle unexpected list response
            print(f"❌ Error: Received list response instead of dict/string")
            print(f"   Response type: {type(generated_response)}")
            print(f"   List length: {len(generated_response)}")
            print(f"   First item type: {type(generated_response[0]) if generated_response else 'N/A'}")
            print(f"   Response preview: {str(generated_response)[:500]}")
            print(f"   Expected: dict with 'tool_calls' field or string")
            break
            
        elif isinstance(generated_response, str):
            # Text response - parse tool calls from text
            generated_text = generated_response
            print(f"📝 Received text response, parsing for tool calls...")
            print(f"   Response length: {len(generated_text)}")
            
            # Parse tool calls from text response
            tool_calls_list = parse_tool_calls_from_text(generated_text)
            
            if tool_calls_list:
                print(f"✅ Parsed {len(tool_calls_list)} tool call(s) from text")
            else:
                print(f"⚠️ No tool calls found in text response")
                print(f"   Response preview: {generated_text[:500]}")
                break
            
        else:
            # Unexpected response type
            print(f"❌ Error: Unexpected response type: {type(generated_response)}")
            print(f"   Response value: {str(generated_response)[:500]}")
            print(f"   Expected: dict with 'tool_calls' field, string, or None")
            break
        
        # Validate tool_calls_list before processing
        if not tool_calls_list:
            print("⚠️ Warning: No tool calls to process")
            break
        
        # Validate tool call structure before processing
        if not isinstance(tool_calls_list, list):
            print(f"❌ Error: tool_calls_list is not a list, got {type(tool_calls_list)}")
            break
        
        # Process the first tool call (we handle one at a time)
        tool_call_data = tool_calls_list[0]
        
        # Validate tool_call_data structure
        if not isinstance(tool_call_data, dict):
            print(f"❌ Error: tool_call_data is not a dict, got {type(tool_call_data)}")
            print(f"   Tool call data: {tool_call_data}")
            break
        
        # Extract and validate tool_call_id
        tool_call_id = tool_call_data.get("id")
        if not tool_call_id:
            tool_call_id = f"call_{generation_count}_{hash(str(tool_call_data)) % 10000}"
            print(f"⚠️ Warning: tool_call missing 'id' field, generated: {tool_call_id}")
        
        # Extract and validate function data
        function_data = tool_call_data.get("function", {})
        if not isinstance(function_data, dict):
            print(f"❌ Error: tool_call 'function' field is not a dict, got {type(function_data)}")
            print(f"   Tool call data: {tool_call_data}")
            break
        
        # Extract and validate function name
        function_name = function_data.get("name")
        if not function_name or not isinstance(function_name, str):
            print(f"❌ Error: tool_call missing or invalid 'function.name' field")
            print(f"   Function data: {function_data}")
            print(f"   Tool call data: {tool_call_data}")
            break
        
        # Extract and validate function arguments
        function_arguments_str = function_data.get("arguments", "{}")
        if not isinstance(function_arguments_str, str):
            print(f"⚠️ Warning: function arguments is not a string, converting: {type(function_arguments_str)}")
            try:
                function_arguments_str = json.dumps(function_arguments_str)
            except (TypeError, ValueError) as e:
                print(f"❌ Error: Cannot convert arguments to JSON string: {e}")
                print(f"   Arguments: {function_arguments_str}")
                break
        
        # Parse function arguments with better error handling
        try:
            function_arguments = json.loads(function_arguments_str)
            if not isinstance(function_arguments, dict):
                print(f"⚠️ Warning: Parsed arguments is not a dict, got {type(function_arguments)}")
                print(f"   Arguments: {function_arguments}")
                # Try to wrap it in a dict or use empty dict
                function_arguments = {"value": function_arguments} if function_arguments else {}
        except json.JSONDecodeError as e:
            print(f"❌ Error: Invalid JSON in tool call arguments")
            print(f"   Arguments string: {function_arguments_str[:200]}")
            print(f"   JSON error: {e}")
            print(f"   Tool call: {function_name}")
            raise ValueError(f"Invalid JSON in tool call arguments for '{function_name}': {function_arguments_str[:100]}..., error: {e}")
        
        # Build tool_call dict in expected format
        tool_call = {
            "name": function_name,
            "parameters": function_arguments,
        }
        
        # Track tool call history (keep last 3)
        recent_tool_calls.append(function_name)
        if len(recent_tool_calls) > 3:
            recent_tool_calls.pop(0)
        
        # Count SAM3 segmentation attempts (only segment_phrase now - others disabled)
        if function_name == "segment_phrase":  # Only count segment_phrase
            sam3_segmentation_count += 1
            print(f"📊 SAM3 segmentation attempt #{sam3_segmentation_count}")
        
        # Count LISAT segmentation attempts
        if function_name == "segment_with_lisat":
            lisat_attempts += 1
            print(f"📊 LISAT segmentation attempt #{lisat_attempts}")
        
        if PATH_TO_LATEST_OUTPUT_JSON == "":
            # The first tool call must be segment_phrase, segment_with_lisat, or report_no_mask
            # Note: segment_phrase_in_region and segment_phrase_with_tiling have been disabled
            assert (
                tool_call["name"] == "segment_phrase"
                # or tool_call["name"] == "segment_phrase_with_tiling"  # DISABLED
                # or tool_call["name"] == "segment_phrase_in_region"  # DISABLED
                or tool_call["name"] == "segment_with_lisat"
                or tool_call["name"] == "report_no_mask"
            ), f"First tool call must be segment_phrase, segment_with_lisat, or report_no_mask, got {tool_call['name']}"

        if tool_call["name"] == "segment_phrase":
            print("🔍 Calling segment_phrase tool...")
            assert list(tool_call["parameters"].keys()) == ["text_prompt"], f"Expected ['text_prompt'], got {list(tool_call['parameters'].keys())}"

            # Check if this text_prompt has been used before
            current_text_prompt = tool_call["parameters"]["text_prompt"]
            if current_text_prompt in USED_TEXT_PROMPTS:
                print(
                    f"❌ Text prompt '{current_text_prompt}' has been used before. Requesting a different prompt."
                )
                duplicate_prompt_message = f"You have previously used '{current_text_prompt}' as your text_prompt to call the segment_phrase tool. You may not use it again. Please call the segment_phrase tool again with a different, perhaps more general, or more creative simple noun phrase prompt, while adhering to all the rules stated in the system prompt. You must also never use any of the following text_prompt(s): {str(list(USED_TEXT_PROMPTS))}."
                # Append assistant message with tool call
                messages.append({
                    "role": "assistant",
                    "content": generated_text if generated_text else None,
                    "tool_calls": [tool_call_data],
                })
                messages.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": duplicate_prompt_message}],
                    }
                )
            else:
                # Add the text_prompt to the set of used prompts
                USED_TEXT_PROMPTS.add(current_text_prompt)
                LATEST_SAM3_TEXT_PROMPT = current_text_prompt
                PATH_TO_LATEST_OUTPUT_JSON = call_sam_service(
                    image_path=img_path,
                    text_prompt=current_text_prompt,
                    output_folder_path=sam_output_dir,
                )
                # Check if file exists before reading
                if not os.path.exists(PATH_TO_LATEST_OUTPUT_JSON):
                    raise FileNotFoundError(
                        f"SAM3 output file not found: {PATH_TO_LATEST_OUTPUT_JSON}. "
                        f"This may be due to a path construction issue with temporary directories."
                    )
                sam3_outputs = json.load(open(PATH_TO_LATEST_OUTPUT_JSON, "r"))
                sam3_output_image_path = sam3_outputs.get("output_image_path", "")
                num_masks = len(sam3_outputs.get("pred_boxes", []))

                # Append assistant message with tool call
                messages.append({
                    "role": "assistant",
                    "content": generated_text if generated_text else None,
                    "tool_calls": [tool_call_data],
                })
                
                if num_masks == 0:
                    print("❌ No masks generated by SAM3, reporting no mask to Qwen.")
                    
                    # Deterministic count tracking - inject into message
                    attempt_status = f"\n\n📊 ATTEMPT TRACKING:\n- SAM3 attempts: {sam3_segmentation_count}/3\n- LISAT attempts: {lisat_attempts}/3\n- Total attempts: {sam3_segmentation_count + lisat_attempts}/6"
                    
                    # Check if we should suggest LISAT as alternative
                    if sam3_segmentation_count >= 3 and lisat_attempts == 0:
                        sam3_output_text_message = f"The segment_phrase tool did not generate any masks for the text_prompt '{current_text_prompt}'.{attempt_status}\n\n⚠️ You have exhausted all 3 SAM3 attempts. It's time to try a different approach. Please use the segment_with_lisat tool as an alternative segmentation method. LISAT uses a different model that may succeed where SAM3 has failed. Include relevant attributes from the original query ('{initial_text_prompt}') such as color, size, or location in your prompt to LISAT."
                    else:
                        sam3_output_text_message = f"The segment_phrase tool did not generate any masks for the text_prompt '{current_text_prompt}'.{attempt_status}\n\nNow, please call the segment_phrase tool again with a different, perhaps more general, or more creative simple noun phrase text_prompt, while adhering to all the rules stated in the system prompt. Please be reminded that the original user query was '{initial_text_prompt}'."
                    
                    # Append tool result with role: "tool"
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": json.dumps({"message": sam3_output_text_message, "num_masks": 0}),
                    })
                else:
                    # Deterministic count tracking - inject into message
                    attempt_status = f"\n\n📊 ATTEMPT TRACKING:\n- SAM3 attempts: {sam3_segmentation_count}/3\n- LISAT attempts: {lisat_attempts}/3\n- Total attempts: {sam3_segmentation_count + lisat_attempts}/6"
                    
                    sam3_output_text_message = rf"The segment_phrase tool generated {num_masks} available masks.{attempt_status}\n\nAll {num_masks} available masks are rendered in this image below, now you must analyze the {num_masks} available mask(s) carefully, compare them against the raw input image and the original user query, and determine your next action. Please be reminded that the original user query was '{initial_text_prompt}'."
                    # Per Qwen3 docs: Add tool result with role: "tool"
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": json.dumps({
                            "message": sam3_output_text_message,
                            "num_masks": num_masks,
                            "output_image_path": sam3_output_image_path,
                        }),
                    })
                    # For vision models: Add user message with image so model can visually analyze masks
                    # Note: Tool result contains the message, user message only provides the image for visual analysis
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Please analyze the masks shown in the image below."},
                            {"type": "image", "image": sam3_output_image_path},
                        ],
                    })
                print("\n\n>>> sam3_output_text_message:\n", sam3_output_text_message)

        # DISABLED: segment_phrase_in_region handler - Commented out as per user request
        # elif tool_call["name"] == "segment_phrase_in_region":
        #     print("🔍 Calling segment_phrase_in_region tool...")
        #     assert "text_prompt" in tool_call["parameters"] and "bbox" in tool_call["parameters"], \
        #         f"Expected ['text_prompt', 'bbox'], got {list(tool_call['parameters'].keys())}"
            
            current_text_prompt = tool_call["parameters"]["text_prompt"]
            bbox = tool_call["parameters"]["bbox"]
            use_normalized = tool_call["parameters"].get("use_normalized", True)
            
            # Validate bbox
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError(f"bbox must be a list of 4 numbers [x_min, y_min, x_max, y_max], got {bbox}")
            
            if use_normalized:
                if not all(0 <= coord <= 1 for coord in bbox):
                    raise ValueError(f"Normalized bbox coordinates must be in [0, 1], got {bbox}")
                if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    raise ValueError(f"Invalid bbox: x_max <= x_min or y_max <= y_min, got {bbox}")
            else:
                if any(coord < 0 for coord in bbox):
                    raise ValueError(f"Pixel bbox coordinates must be non-negative, got {bbox}")
                if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    raise ValueError(f"Invalid bbox: x_max <= x_min or y_max <= y_min, got {bbox}")
            
            # Check if this text_prompt has been used before
            if current_text_prompt in USED_TEXT_PROMPTS:
                print(
                    f"❌ Text prompt '{current_text_prompt}' has been used before. Requesting a different prompt."
                )
                duplicate_prompt_message = f"You have previously used '{current_text_prompt}' as your text_prompt to call the segment_phrase tool. You may not use it again. Please call the segment_phrase or segment_phrase_in_region tool again with a different, perhaps more general, or more creative simple noun phrase prompt, while adhering to all the rules stated in the system prompt. You must also never use any of the following text_prompt(s): {str(list(USED_TEXT_PROMPTS))}."
                messages.append({
                    "role": "assistant",
                    "content": generated_text if generated_text else None,
                    "tool_calls": [tool_call_data],
                })
                messages.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": duplicate_prompt_message}],
                    }
                )
            else:
                # Add the text_prompt to the set of used prompts
                USED_TEXT_PROMPTS.add(current_text_prompt)
                LATEST_SAM3_TEXT_PROMPT = current_text_prompt
                
                # Load original image and crop to region
                original_image = Image.open(img_path)
                cropped_image, region_bbox = segment_in_region(original_image, bbox, use_normalized)
                
                # Save cropped image temporarily
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False, dir=sam_output_dir) as tmp_file:
                    cropped_image_path = tmp_file.name
                    cropped_image.save(cropped_image_path)
                
                try:
                    # Call SAM3 on cropped region
                    region_output_json = call_sam_service(
                        image_path=cropped_image_path,
                        text_prompt=current_text_prompt,
                        output_folder_path=sam_output_dir,
                    )
                    
                    # Load region results
                    region_outputs = json.load(open(region_output_json, "r"))
                    
                    # Transform masks back to global coordinates
                    orig_img_h = int(region_outputs["orig_img_h"])
                    orig_img_w = int(region_outputs["orig_img_w"])
                    # Note: region_outputs has dimensions of cropped image, we need original image dimensions
                    original_image = Image.open(img_path)
                    orig_img_h_full, orig_img_w_full = original_image.size[1], original_image.size[0]
                    
                    transformed_boxes, transformed_masks = transform_masks_to_global(
                        region_outputs["pred_boxes"],
                        region_outputs["pred_masks"],
                        region_bbox,
                        orig_img_h_full,
                        orig_img_w_full,
                    )
                    
                    # Create outputs in global coordinates
                    sam3_outputs = {
                        "original_image_path": img_path,
                        "orig_img_h": orig_img_h_full,
                        "orig_img_w": orig_img_w_full,
                        "pred_boxes": transformed_boxes,
                        "pred_masks": transformed_masks,
                        "pred_scores": region_outputs["pred_scores"],
                    }
                    
                    # Save outputs
                    text_prompt_for_save_path = (
                        current_text_prompt.replace("/", "_") if "/" in current_text_prompt else current_text_prompt
                    )
                    image_basename = os.path.basename(img_path)
                    image_basename_no_ext = os.path.splitext(image_basename)[0]
                    safe_dir_name = image_basename_no_ext.replace("/", "_").replace("\\", "_")
                    if not safe_dir_name or safe_dir_name.startswith("-"):
                        safe_dir_name = "image_" + safe_dir_name if safe_dir_name else "image"
                    
                    os.makedirs(
                        os.path.join(sam_output_dir, safe_dir_name), exist_ok=True
                    )
                    PATH_TO_LATEST_OUTPUT_JSON = os.path.join(
                        sam_output_dir,
                        safe_dir_name,
                        rf"{text_prompt_for_save_path}_region.json",
                    )
                    sam3_output_image_path = os.path.join(
                        sam_output_dir,
                        safe_dir_name,
                        rf"{text_prompt_for_save_path}_region.png",
                    )
                    
                    sam3_outputs["output_image_path"] = sam3_output_image_path
                    json.dump(sam3_outputs, open(PATH_TO_LATEST_OUTPUT_JSON, "w"), indent=4)
                    
                    # Visualize (using globally imported visualize)
                    viz_image = visualize(sam3_outputs)
                    viz_image.save(sam3_output_image_path)
                    
                    num_masks = len(sam3_outputs.get("pred_boxes", []))
                    
                    # Append assistant message with tool call
                    messages.append({
                        "role": "assistant",
                        "content": generated_text if generated_text else None,
                        "tool_calls": [tool_call_data],
                    })
                    
                    if num_masks == 0:
                        print("❌ No masks generated by SAM3 in region, reporting no mask to Qwen.")
                        
                        # Check if we should suggest LISAT as alternative
                        if sam3_segmentation_count >= 3 and lisat_attempts == 0:
                            sam3_output_text_message = f"The segment_phrase_in_region tool did not generate any masks for the text_prompt '{current_text_prompt}' in the specified region. You have now attempted SAM3 segmentation {sam3_segmentation_count} times without success. It's time to try a different approach. Please use the segment_with_lisat tool as an alternative segmentation method. LISAT uses a different model that may succeed where SAM3 has failed. Include relevant attributes from the original query ('{initial_text_prompt}') such as color, size, or location in your prompt to LISAT."
                        else:
                            sam3_output_text_message = f"The segment_phrase_in_region tool did not generate any masks for the text_prompt '{current_text_prompt}' in the specified region. Now, please call the segment_phrase or segment_phrase_in_region tool again with a different, perhaps more general, or more creative simple noun phrase text_prompt, while adhering to all the rules stated in the system prompt. Please be reminded that the original user query was '{initial_text_prompt}'."
                        
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": json.dumps({"message": sam3_output_text_message, "num_masks": 0}),
                        })
                    else:
                        sam3_output_text_message = rf"The segment_phrase_in_region tool generated {num_masks} available masks in the specified region. All {num_masks} available masks are rendered in this image below, now you must analyze the {num_masks} available mask(s) carefully, compare them against the raw input image and the original user query, and determine your next action. Please be reminded that the original user query was '{initial_text_prompt}'."
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": json.dumps({
                                "message": sam3_output_text_message,
                                "num_masks": num_masks,
                                "output_image_path": sam3_output_image_path,
                            }),
                        })
                        messages.append({
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Please analyze the masks shown in the image below."},
                                {"type": "image", "image": sam3_output_image_path},
                            ],
                        })
                    print("\n\n>>> sam3_output_text_message:\n", sam3_output_text_message)
                    
                finally:
                    # Clean up temporary cropped image
                    try:
                        os.unlink(cropped_image_path)
                    except Exception as e:
                        print(f"Warning: Could not delete temporary cropped image: {e}")

        # DISABLED: segment_phrase_with_tiling handler - Commented out as per user request
        # elif tool_call["name"] == "segment_phrase_with_tiling":
        #     print("🔍 Calling segment_phrase_with_tiling tool...")
        #     assert "text_prompt" in tool_call["parameters"], \
        #         f"Expected ['text_prompt'], got {list(tool_call['parameters'].keys())}"
            
            current_text_prompt = tool_call["parameters"]["text_prompt"]
            tile_size = tool_call["parameters"].get("tile_size", 1024)
            overlap_ratio = tool_call["parameters"].get("overlap_ratio", 0.2)
            min_tile_size = tool_call["parameters"].get("min_tile_size", 512)
            
            # Validate parameters
            if tile_size < 256 or tile_size > 4096:
                raise ValueError(f"tile_size must be in [256, 4096], got {tile_size}")
            if overlap_ratio < 0.0 or overlap_ratio > 0.5:
                raise ValueError(f"overlap_ratio must be in [0.0, 0.5], got {overlap_ratio}")
            if min_tile_size < 128 or min_tile_size > tile_size:
                raise ValueError(f"min_tile_size must be in [128, tile_size], got {min_tile_size}")
            
            # Check if this text_prompt has been used before
            if current_text_prompt in USED_TEXT_PROMPTS:
                print(
                    f"❌ Text prompt '{current_text_prompt}' has been used before. Requesting a different prompt."
                )
                duplicate_prompt_message = f"You have previously used '{current_text_prompt}' as your text_prompt to call the segment_phrase tool. You may not use it again. Please call the segment_phrase or segment_phrase_with_tiling tool again with a different, perhaps more general, or more creative simple noun phrase prompt, while adhering to all the rules stated in the system prompt. You must also never use any of the following text_prompt(s): {str(list(USED_TEXT_PROMPTS))}."
                messages.append({
                    "role": "assistant",
                    "content": generated_text if generated_text else None,
                    "tool_calls": [tool_call_data],
                })
                messages.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": duplicate_prompt_message}],
                    }
                )
            else:
                # Add the text_prompt to the set of used prompts
                USED_TEXT_PROMPTS.add(current_text_prompt)
                LATEST_SAM3_TEXT_PROMPT = current_text_prompt
                
                # Load original image
                original_image = Image.open(img_path)
                orig_img_w, orig_img_h = original_image.size
                
                # Create tiles
                tiles = create_pyramidal_tiles(
                    orig_img_w, orig_img_h, tile_size, overlap_ratio, min_tile_size
                )
                print(f"📐 Created {len(tiles)} tiles for pyramidal tiling")
                
                # Segment each tile
                all_tile_results = []
                import tempfile
                
                for tile_idx, tile_bbox in enumerate(tiles):
                    x_min, y_min, x_max, y_max = tile_bbox
                    print(f"🔍 Processing tile {tile_idx + 1}/{len(tiles)}: ({x_min}, {y_min}, {x_max}, {y_max})")
                    
                    # Crop image to tile
                    tile_image = original_image.crop((x_min, y_min, x_max, y_max))
                    
                    # Save cropped tile temporarily
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False, dir=sam_output_dir) as tmp_file:
                        tile_image_path = tmp_file.name
                        tile_image.save(tile_image_path)
                    
                    try:
                        # Call SAM3 on tile
                        tile_output_json = call_sam_service(
                            image_path=tile_image_path,
                            text_prompt=current_text_prompt,
                            output_folder_path=sam_output_dir,
                        )
                        
                        # Load tile results
                        tile_outputs = json.load(open(tile_output_json, "r"))
                        
                        # Transform masks to global coordinates
                        transformed_boxes = []
                        transformed_masks = []
                        
                        for box, mask in zip(tile_outputs["pred_boxes"], tile_outputs["pred_masks"]):
                            trans_box, trans_mask = transform_mask_coordinates(
                                box, mask, tile_bbox, orig_img_h, orig_img_w
                            )
                            transformed_boxes.append(trans_box)
                            transformed_masks.append(trans_mask)
                        
                        # Store tile results
                        all_tile_results.append({
                            "pred_boxes": transformed_boxes,
                            "pred_masks": transformed_masks,
                            "pred_scores": tile_outputs["pred_scores"],
                            "tile_bbox": tile_bbox,
                        })
                        
                    finally:
                        # Clean up temporary tile image
                        try:
                            os.unlink(tile_image_path)
                        except Exception as e:
                            print(f"Warning: Could not delete temporary tile image: {e}")
                
                # Merge results from all tiles
                print(f"🔗 Merging results from {len(tiles)} tiles...")
                merged_outputs = merge_tile_results(all_tile_results, orig_img_h, orig_img_w)
                
                # Apply overlap removal
                try:
                    from .helpers.mask_overlap_removal import remove_overlapping_masks
                    merged_outputs = remove_overlapping_masks(merged_outputs)
                    print(f"✓ Applied mask overlap removal")
                except Exception as e:
                    print(f"⚠ Warning: Could not apply mask overlap removal: {e}")
                
                # Save outputs
                text_prompt_for_save_path = (
                    current_text_prompt.replace("/", "_") if "/" in current_text_prompt else current_text_prompt
                )
                image_basename = os.path.basename(img_path)
                image_basename_no_ext = os.path.splitext(image_basename)[0]
                safe_dir_name = image_basename_no_ext.replace("/", "_").replace("\\", "_")
                if not safe_dir_name or safe_dir_name.startswith("-"):
                    safe_dir_name = "image_" + safe_dir_name if safe_dir_name else "image"
                
                os.makedirs(
                    os.path.join(sam_output_dir, safe_dir_name), exist_ok=True
                )
                PATH_TO_LATEST_OUTPUT_JSON = os.path.join(
                    sam_output_dir,
                    safe_dir_name,
                    rf"{text_prompt_for_save_path}_tiling.json",
                )
                sam3_output_image_path = os.path.join(
                    sam_output_dir,
                    safe_dir_name,
                    rf"{text_prompt_for_save_path}_tiling.png",
                )
                
                sam3_outputs = {
                    "original_image_path": img_path,
                    "output_image_path": sam3_output_image_path,
                    **merged_outputs,
                }
                
                json.dump(sam3_outputs, open(PATH_TO_LATEST_OUTPUT_JSON, "w"), indent=4)
                
                # Visualize
                viz_image = visualize(sam3_outputs)
                viz_image.save(sam3_output_image_path)
                
                num_masks = len(sam3_outputs.get("pred_boxes", []))
                
                # Append assistant message with tool call
                messages.append({
                    "role": "assistant",
                    "content": generated_text if generated_text else None,
                    "tool_calls": [tool_call_data],
                })
                
                if num_masks == 0:
                    print("❌ No masks generated by SAM3 with tiling, reporting no mask to Qwen.")
                    sam3_output_text_message = f"The segment_phrase_with_tiling tool did not generate any masks for the text_prompt '{current_text_prompt}'. Now, please call the segment_phrase or segment_phrase_with_tiling tool again with a different, perhaps more general, or more creative simple noun phrase text_prompt, while adhering to all the rules stated in the system prompt. Please be reminded that the original user query was '{initial_text_prompt}'."
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": json.dumps({"message": sam3_output_text_message, "num_masks": 0}),
                    })
                else:
                    sam3_output_text_message = rf"The segment_phrase_with_tiling tool generated {num_masks} available masks using pyramidal tiling across {len(tiles)} tiles. All {num_masks} available masks are rendered in this image below, now you must analyze the {num_masks} available mask(s) carefully, compare them against the raw input image and the original user query, and determine your next action. Please be reminded that the original user query was '{initial_text_prompt}'."
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": json.dumps({
                            "message": sam3_output_text_message,
                            "num_masks": num_masks,
                            "output_image_path": sam3_output_image_path,
                            "num_tiles": len(tiles),
                        }),
                    })
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Please analyze the masks shown in the image below."},
                            {"type": "image", "image": sam3_output_image_path},
                        ],
                    })
                print("\n\n>>> sam3_output_text_message:\n", sam3_output_text_message)

        elif tool_call["name"] == "examine_each_mask":
            print("🔍 Calling examine_each_mask tool...")
            assert LATEST_SAM3_TEXT_PROMPT != ""

            # Make sure that the last message is a image
            assert (
                messages[-1]["content"][1]["type"] == "image"
            ), "Second content element should be an image"
            messages.pop()  # Remove the last user message
            # Add simplified replacement message
            simplified_message = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "The segment_phrase tool generated several masks. Now you must analyze the mask(s) carefully, compare them against the raw input image and the original user query, and determine your next action.",
                    }
                ],
            }
            messages.append(simplified_message)

            if not os.path.exists(PATH_TO_LATEST_OUTPUT_JSON):
                raise FileNotFoundError(
                    f"SAM3 output file not found: {PATH_TO_LATEST_OUTPUT_JSON}"
                )
            current_outputs = json.load(open(PATH_TO_LATEST_OUTPUT_JSON, "r"))
            num_masks = len(current_outputs.get("pred_masks", []))
            masks_to_keep = []

            # MLLM check the mask one by one
            for i in range(num_masks):
                print(f"🔍 Checking mask {i+1}/{num_masks}...")
                image_w_mask_i, image_w_zoomed_in_mask_i = visualize(current_outputs, i)

                image_w_zoomed_in_mask_i_path = os.path.join(
                    sam_output_dir, rf"{LATEST_SAM3_TEXT_PROMPT}.png".replace("/", "_")
                ).replace(".png", f"_zoom_in_mask_{i + 1}.png")
                image_w_mask_i_path = os.path.join(
                    sam_output_dir, rf"{LATEST_SAM3_TEXT_PROMPT}.png".replace("/", "_")
                ).replace(".png", f"_selected_mask_{i + 1}.png")
                image_w_zoomed_in_mask_i.save(image_w_zoomed_in_mask_i_path)
                image_w_mask_i.save(image_w_mask_i_path)

                # Keep vision requests within 2-image limit for providers like OpenAI/vLLM
                iterative_checking_messages = [
                    {"role": "system", "content": iterative_checking_system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"The initial user input query is: '{initial_text_prompt}'. The raw image is the background of the mask render below (omitted as a separate attachment to respect the 2-image limit).",
                            },
                            {
                                "type": "text",
                                "text": f"Image with the predicted segmentation mask rendered on it: ",
                            },
                            {"type": "image", "image": image_w_mask_i_path},
                            {
                                "type": "text",
                                "text": f"Zoomed-in view of the selected mask: ",
                            },
                            {"type": "image", "image": image_w_zoomed_in_mask_i_path},
                        ],
                    },
                ]
                # Count iterative checking as an LLM call
                generation_count += 1
                if generation_count > max_generations:
                    raise ValueError(
                        f"Exceeded maximum number of allowed generation requests ({max_generations})"
                    )
                checking_response = send_generate_request(
                    iterative_checking_messages,
                    tools=None  # Iterative checking doesn't use tools
                )

                # Normalize response to string (handle dict/list responses)
                if checking_response is None:
                    raise ValueError(
                        "Generated text is None, which is unexpected. Please check the Qwen server and the input parameters."
                    )
                elif isinstance(checking_response, dict):
                    # Extract content from dict response (might have tool_calls)
                    checking_generated_text = checking_response.get("content", "")
                    if not checking_generated_text:
                        raise ValueError(
                            "Response dict has no 'content' field. This may indicate the model returned tool calls instead of text."
                        )
                elif isinstance(checking_response, list):
                    # Handle unexpected list response
                    print(f"⚠️ Warning: Received list response in iterative checking, converting to string")
                    checking_generated_text = str(checking_response)
                elif not isinstance(checking_response, str):
                    # Convert other types to string
                    checking_generated_text = str(checking_response)
                else:
                    checking_generated_text = checking_response
                
                print(f"Generated text for mask {i+1}: {checking_generated_text}")
                verdict = (
                    checking_generated_text.split("<verdict>")[-1]
                    .split("</verdict>")[0]
                    .strip()
                )
                if "Accept" in verdict:
                    assert not "Reject" in verdict
                    print(f"Mask {i+1} accepted, keeping it in the outputs.")
                    masks_to_keep.append(i)
                elif "Reject" in verdict:
                    assert not "Accept" in verdict
                    print(f"Mask {i+1} rejected, removing it from the outputs.")
                else:
                    raise ValueError(
                        f"Unexpected verdict in generated text: {checking_generated_text}. Expected 'Accept' or 'Reject'."
                    )

            updated_outputs = {
                "original_image_path": current_outputs["original_image_path"],
                "orig_img_h": current_outputs["orig_img_h"],
                "orig_img_w": current_outputs["orig_img_w"],
                "pred_boxes": [current_outputs["pred_boxes"][i] for i in masks_to_keep],
                "pred_scores": [
                    current_outputs["pred_scores"][i] for i in masks_to_keep
                ],
                "pred_masks": [current_outputs["pred_masks"][i] for i in masks_to_keep],
            }

            image_w_check_masks = visualize(updated_outputs)
            image_w_check_masks_path = os.path.join(
                sam_output_dir, rf"{LATEST_SAM3_TEXT_PROMPT}.png"
            ).replace(
                ".png",
                f"_selected_masks_{'-'.join(map(str, [i+1 for i in masks_to_keep]))}.png".replace(
                    "/", "_"
                ),
            )
            image_w_check_masks.save(image_w_check_masks_path)
            # save the updated json outputs and append to message history
            messages.append({
                "role": "assistant",
                "content": generated_text if generated_text else None,
                "tool_calls": [tool_call_data],
            })
            
            if len(masks_to_keep) == 0:
                tool_result_message = f"The original user query was: '{initial_text_prompt}'. The examine_each_mask tool examined and rejected all of the masks generated by the segment_phrase tool. Now, please call the segment_phrase tool again with a different, perhaps more general, or more creative simple noun phrase text_prompt, while adhering to all the rules stated in the system prompt."
                # Per Qwen3 docs: Only add tool result, not duplicate user message
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps({"message": tool_result_message, "masks_to_keep": []}),
                })
            else:
                tool_result_message = f"The original user query was: '{initial_text_prompt}'. After calling the examine_each_mask tool on the available masks, the number of available masks is now {len(masks_to_keep)}. All {len(masks_to_keep)} available masks are rendered in this image below, now you must analyze the {len(masks_to_keep)} available mask(s) carefully, compare them against the raw input image and the original user query, and determine your next action."
                # Per Qwen3 docs: Only add tool result
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps({
                        "message": tool_result_message,
                        "masks_to_keep": masks_to_keep,
                        "output_image_path": image_w_check_masks_path,
                    }),
                })
                # For vision models: Add user message with image so model can visually analyze masks
                # Note: Tool result contains the message, user message only provides the image for visual analysis
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Please analyze the filtered masks shown in the image below."},
                        {"type": "image", "image": image_w_check_masks_path},
                    ],
                })

            # Create a new filename based on the original path to avoid filename length issues
            base_path = PATH_TO_LATEST_OUTPUT_JSON
            # Remove any existing "masks_" suffix to avoid duplication
            if "masks_" in base_path:
                base_path = base_path.split("masks_")[0] + ".json"
            # Create new filename with current masks; use a clearer suffix when empty
            if len(masks_to_keep) == 0:
                PATH_TO_LATEST_OUTPUT_JSON = base_path.replace(
                    ".json", "masks_none.json"
                )
            else:
                PATH_TO_LATEST_OUTPUT_JSON = base_path.replace(
                    ".json", f"masks_{'_'.join(map(str, masks_to_keep))}.json"
                )
            json.dump(updated_outputs, open(PATH_TO_LATEST_OUTPUT_JSON, "w"), indent=4)

        # DISABLED: filter_masks_by_spatial_position - spatial filtering now done in LLM thinking
        # The LLM reasons about mask positions based on their visual location in the image
        # and directly calls select_masks_and_return with the appropriate mask indices.
        # elif tool_call["name"] == "filter_masks_by_spatial_position":
        #     ... (tool handling code removed - see git history if needed)

        elif tool_call["name"] == "filter_masks_by_attributes":
            print("🔍 Calling filter_masks_by_attributes tool...")
            if not os.path.exists(PATH_TO_LATEST_OUTPUT_JSON):
                raise FileNotFoundError(
                    f"SAM3 output file not found: {PATH_TO_LATEST_OUTPUT_JSON}"
                )
            current_outputs = json.load(open(PATH_TO_LATEST_OUTPUT_JSON, "r"))
            
            # Extract filter parameters (all optional)
            color = tool_call["parameters"].get("color")
            min_size_ratio = tool_call["parameters"].get("min_size_ratio")
            max_size_ratio = tool_call["parameters"].get("max_size_ratio")
            aspect_ratio_range = tool_call["parameters"].get("aspect_ratio_range")
            
            # Validate parameters
            if min_size_ratio is not None and (min_size_ratio < 0 or min_size_ratio > 1):
                raise ValueError(f"min_size_ratio must be in [0, 1], got {min_size_ratio}")
            if max_size_ratio is not None and (max_size_ratio < 0 or max_size_ratio > 1):
                raise ValueError(f"max_size_ratio must be in [0, 1], got {max_size_ratio}")
            if aspect_ratio_range is not None:
                if not isinstance(aspect_ratio_range, list) or len(aspect_ratio_range) != 2:
                    raise ValueError(f"aspect_ratio_range must be a list of 2 numbers [min, max], got {aspect_ratio_range}")
                if aspect_ratio_range[0] < 0 or aspect_ratio_range[1] < 0:
                    raise ValueError(f"aspect_ratio_range values must be non-negative, got {aspect_ratio_range}")
                if aspect_ratio_range[0] > aspect_ratio_range[1]:
                    raise ValueError(f"aspect_ratio_range min must be <= max, got {aspect_ratio_range}")
            
            # Check that at least one filter is provided
            if color is None and min_size_ratio is None and max_size_ratio is None and aspect_ratio_range is None:
                raise ValueError("At least one filter parameter (color, min_size_ratio, max_size_ratio, or aspect_ratio_range) must be provided")
            
            # Apply attribute filtering
            filtered_outputs = filter_masks_by_attributes(
                current_outputs,
                color=color,
                min_size_ratio=min_size_ratio,
                max_size_ratio=max_size_ratio,
                aspect_ratio_range=aspect_ratio_range,
            )
            num_kept = len(filtered_outputs["pred_masks"])
            filter_desc = filtered_outputs.get("attribute_filter_description", "")
            
            # Save updated outputs
            image_w_filtered_masks = visualize(filtered_outputs)
            image_w_filtered_masks_path = os.path.join(
                sam_output_dir, rf"{LATEST_SAM3_TEXT_PROMPT}.png"
            ).replace(
                ".png",
                f"_attribute_filter.png".replace("/", "_"),
            )
            image_w_filtered_masks.save(image_w_filtered_masks_path)
            
            # Update PATH_TO_LATEST_OUTPUT_JSON
            base_path = PATH_TO_LATEST_OUTPUT_JSON
            if "masks_" in base_path:
                base_path = base_path.split("masks_")[0] + ".json"
            if num_kept == 0:
                PATH_TO_LATEST_OUTPUT_JSON = base_path.replace(
                    ".json", "masks_none.json"
                )
            else:
                kept_indices = list(range(num_kept))
                PATH_TO_LATEST_OUTPUT_JSON = base_path.replace(
                    ".json", f"masks_{'_'.join(map(str, kept_indices))}.json"
                )
            json.dump(filtered_outputs, open(PATH_TO_LATEST_OUTPUT_JSON, "w"), indent=4)
            
            # Append assistant message with tool call
            messages.append({
                "role": "assistant",
                "content": generated_text if generated_text else None,
                "tool_calls": [tool_call_data],
            })
            
            if num_kept == 0:
                tool_result_message = f"The original user query was: '{initial_text_prompt}'. The filter_masks_by_attributes tool filtered all masks out. {filter_desc} Now, please call the segment_phrase tool again with a different text_prompt, or try different attribute filters."
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps({"message": tool_result_message, "num_masks": 0}),
                })
            else:
                tool_result_message = f"The original user query was: '{initial_text_prompt}'. After calling the filter_masks_by_attributes tool, the number of available masks is now {num_kept}. {filter_desc} All {num_kept} available masks are rendered in this image below, now you must analyze the {num_kept} available mask(s) carefully, compare them against the raw input image and the original user query, and determine your next action."
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps({
                        "message": tool_result_message,
                        "num_masks": num_kept,
                        "output_image_path": image_w_filtered_masks_path,
                    }),
                })
                # For vision models: Add user message with image so model can visually analyze masks
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Please analyze the attribute-filtered masks shown in the image below."},
                        {"type": "image", "image": image_w_filtered_masks_path},
                    ],
                })

        # DISABLED: filter_masks_by_relative_position - relative filtering now done in LLM thinking
        # The LLM reasons about mask positions relative to other objects/features
        # and directly calls select_masks_and_return with the appropriate mask indices.
        # elif tool_call["name"] == "filter_masks_by_relative_position":
        #     ... (tool handling code removed - see git history if needed)

        elif tool_call["name"] == "segment_with_lisat":
            print("🔍 Calling segment_with_lisat tool...")
            assert list(tool_call["parameters"].keys()) == ["text_prompt"], \
                f"Expected ['text_prompt'], got {list(tool_call['parameters'].keys())}"
            
            current_text_prompt = tool_call["parameters"]["text_prompt"]
            
            # Check if this is beyond the 3-attempt limit for LISAT
            if lisat_attempts >= 3:
                print(f"❌ LISAT has been attempted 3 times already. Giving up.")
                messages.append({
                    "role": "assistant",
                    "content": generated_text if generated_text else None,
                    "tool_calls": [tool_call_data],
                })
                
                final_message = f"You have attempted LISAT segmentation 3 times without success after SAM3 also failed. Unfortunately, no segmentation method was able to find the requested objects in the image. Please call report_no_mask to conclude."
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps({"message": final_message, "num_masks": 0}),
                })
            
            # Check if this prompt was already used with LISAT
            elif current_text_prompt in lisat_used_prompts:
                print(f"❌ Text prompt '{current_text_prompt}' has been used with LISAT before.")
                messages.append({
                    "role": "assistant",
                    "content": generated_text if generated_text else None,
                    "tool_calls": [tool_call_data],
                })
                
                duplicate_prompt_message = f"You have previously used '{current_text_prompt}' with LISAT. Please try segment_with_lisat again with a different prompt that includes more specific attributes from the original query '{initial_text_prompt}' such as color, size, shape, or location. You have {3 - lisat_attempts} attempts remaining with LISAT."
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": duplicate_prompt_message}],
                })
            
            else:
                # Track this LISAT attempt
                lisat_attempts += 1
                lisat_used_prompts.add(current_text_prompt)
                print(f"📊 LISAT attempt #{lisat_attempts}/3")
                
                # Call LISAT API
                lisat_outputs = call_lisat_segmentation(
                    image_path=img_path,
                    text_prompt=current_text_prompt,
                    modal_url=LISAT_MODAL_URL,
                )
                
                messages.append({
                    "role": "assistant",
                    "content": generated_text if generated_text else None,
                    "tool_calls": [tool_call_data],
                })
                
                if lisat_outputs is None or len(lisat_outputs.get("pred_masks", [])) == 0:
                    print("❌ No masks generated by LISAT")
                    
                    remaining_attempts = 3 - lisat_attempts
                    if remaining_attempts > 0:
                        lisat_output_text_message = f"The segment_with_lisat tool did not generate any masks for the prompt '{current_text_prompt}'. You have {remaining_attempts} more attempts with LISAT. Try refining your prompt with more specific attributes from the original query '{initial_text_prompt}', such as adding color descriptors, size information, or location details."
                    else:
                        lisat_output_text_message = f"The segment_with_lisat tool did not generate any masks for the prompt '{current_text_prompt}'. You have exhausted all 3 LISAT attempts after SAM3 also failed. Please call report_no_mask to conclude that no matching objects exist in the image."
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": json.dumps({"message": lisat_output_text_message, "num_masks": 0}),
                    })
                else:
                    # Success! Save outputs and visualize
                    num_masks = len(lisat_outputs["pred_masks"])
                    print(f"✓ LISAT generated {num_masks} masks")
                    
                    # Save outputs
                    text_prompt_for_save_path = (
                        current_text_prompt.replace("/", "_") if "/" in current_text_prompt else current_text_prompt
                    )
                    image_basename = os.path.basename(img_path)
                    image_basename_no_ext = os.path.splitext(image_basename)[0]
                    safe_dir_name = image_basename_no_ext.replace("/", "_").replace("\\", "_")
                    if not safe_dir_name or safe_dir_name.startswith("-"):
                        safe_dir_name = "image_" + safe_dir_name if safe_dir_name else "image"
                    
                    os.makedirs(os.path.join(sam_output_dir, safe_dir_name), exist_ok=True)
                    
                    PATH_TO_LATEST_OUTPUT_JSON = os.path.join(
                        sam_output_dir,
                        safe_dir_name,
                        rf"{text_prompt_for_save_path}_lisat.json",
                    )
                    lisat_output_image_path = os.path.join(
                        sam_output_dir,
                        safe_dir_name,
                        rf"{text_prompt_for_save_path}_lisat.png",
                    )
                    
                    lisat_outputs["output_image_path"] = lisat_output_image_path
                    json.dump(lisat_outputs, open(PATH_TO_LATEST_OUTPUT_JSON, "w"), indent=4)
                    
                    # Visualize
                    viz_image = visualize(lisat_outputs)
                    viz_image.save(lisat_output_image_path)
                    
                    lisat_output_text_message = f"The segment_with_lisat tool successfully generated {num_masks} masks for the prompt '{current_text_prompt}'. All {num_masks} available masks are rendered in this image below. Now you must analyze the {num_masks} available mask(s) carefully, compare them against the raw input image and the original user query, and determine your next action. Please be reminded that the original user query was '{initial_text_prompt}'."
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": json.dumps({
                            "message": lisat_output_text_message,
                            "num_masks": num_masks,
                            "output_image_path": lisat_output_image_path,
                        }),
                    })
                    
                    # Add user message with image for visual analysis
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Please analyze the masks shown in the image below."},
                            {"type": "image", "image": lisat_output_image_path},
                        ],
                    })
                    
                    # Update LATEST_SAM3_TEXT_PROMPT to track current prompt
                    LATEST_SAM3_TEXT_PROMPT = current_text_prompt
                
                print("\n\n>>> lisat_output_text_message:\n", lisat_output_text_message if 'lisat_output_text_message' in locals() else "N/A")

        elif tool_call["name"] == "select_masks_and_return":
            print("🔍 Calling select_masks_and_return tool...")
            if not os.path.exists(PATH_TO_LATEST_OUTPUT_JSON):
                raise FileNotFoundError(
                    f"SAM3 output file not found: {PATH_TO_LATEST_OUTPUT_JSON}"
                )
            current_outputs = json.load(open(PATH_TO_LATEST_OUTPUT_JSON, "r"))

            assert list(tool_call["parameters"].keys()) == ["final_answer_masks"]
            masks_to_keep = tool_call["parameters"]["final_answer_masks"]

            # Keep only valid mask indices, remove duplicates, and preserve deterministic ascending order
            available_masks = set(range(1, len(current_outputs["pred_masks"]) + 1))
            masks_to_keep = sorted({i for i in masks_to_keep if i in available_masks})
            # Change this to a update message telling the model to try again along with information about errors made.

            final_outputs = {
                "original_image_path": current_outputs["original_image_path"],
                "orig_img_h": current_outputs["orig_img_h"],
                "orig_img_w": current_outputs["orig_img_w"],
                "pred_boxes": [
                    current_outputs["pred_boxes"][i - 1] for i in masks_to_keep
                ],
                "pred_scores": [
                    current_outputs["pred_scores"][i - 1] for i in masks_to_keep
                ],
                "pred_masks": [
                    current_outputs["pred_masks"][i - 1] for i in masks_to_keep
                ],
            }

            rendered_final_output = visualize(final_outputs)
            messages.append({
                "role": "assistant",
                "content": generated_text if generated_text else None,
                "tool_calls": [tool_call_data],
            })

            # Clean up debug files before successful return
            cleanup_debug_files(debug, debug_folder_path, debug_jsonl_path)
            return messages, final_outputs, rendered_final_output

        elif tool_call["name"] == "report_no_mask":
            print("🔍 Calling report_no_mask tool...")
            height, width = cv2.imread(img_path).shape[:2]
            final_outputs = {
                "original_image_path": img_path,
                "orig_img_h": height,
                "orig_img_w": width,
                "pred_boxes": [],
                "pred_scores": [],
                "pred_masks": [],
            }
            rendered_final_output = Image.open(img_path)
            messages.append({
                "role": "assistant",
                "content": generated_text if generated_text else None,
                "tool_calls": [tool_call_data],
            })
            return messages, final_outputs, rendered_final_output

        else:
            raise ValueError(f"Unknown tool call: {tool_call['name']}")

        # Prune the messages history before the next MLLM generation round according to the 3-part rules.
        # This keeps history compact and ensures the model sees only the allowed parts.
        messages = _prune_messages_for_next_round(
            messages,
            USED_TEXT_PROMPTS,
            LATEST_SAM3_TEXT_PROMPT,
            img_path,
            initial_text_prompt,
        )
        # make sure there can never be more than 2 images in the context
        assert count_images(messages) <= 2
        
        # Continue to next iteration to get next tool call or final response
        generation_count += 1
        if generation_count > max_generations:
            raise ValueError(
                f"Exceeded maximum number of allowed generation requests ({max_generations})"
            )

        print("\n\n")
        print("-" * 30 + f" Round {str(generation_count + 1)}" + "-" * 30)
        print("\n\n")
        generated_response = send_generate_request(messages, tools=TOOLS)
        print(f"\n>>> MLLM Response [start]\n{generated_response}\n<<< MLLM Response [end]\n")

    print("\n\n>>> SAM 3 Agent execution ended.\n\n")

    error_save_path = os.path.join(
        error_save_dir,
        f"{img_path.rsplit('/', 1)[-1].rsplit('.', 1)[0]}_error_history.json",
    )
    with open(error_save_path, "w") as f:
        json.dump(messages, f, indent=4)
    print("Saved messages history that caused error to:", error_save_path)
    raise ValueError(
        rf"Generated text is None, which is unexpected. Please check the Qwen server and the input parameters for image path: {img_path} and initial text prompt: {initial_text_prompt}."
    )
