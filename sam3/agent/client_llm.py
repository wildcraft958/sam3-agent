# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import base64
import os
from typing import Any, Optional

from openai import OpenAI


def get_image_base64_and_mime(image_path):
    """Convert image file to base64 string and get MIME type"""
    try:
        # Get MIME type based on file extension
        ext = os.path.splitext(image_path)[1].lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }
        mime_type = mime_types.get(ext, "image/jpeg")  # Default to JPEG

        # Convert image to base64
        with open(image_path, "rb") as image_file:
            base64_data = base64.b64encode(image_file.read()).decode("utf-8")
            return base64_data, mime_type
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None, None


def send_generate_request(
    messages,
    server_url=None,
    model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    api_key=None,
    max_tokens=2048,  # Reduced default to be safer for models with smaller context windows
    tools=None,
):
    """
    Sends a request to the OpenAI-compatible API endpoint using the OpenAI client library.

    Args:
        server_url (str): The base URL of the server, e.g. "http://127.0.0.1:8000"
        messages (list): A list of message dicts, each containing role and content.
        model (str): The model to use for generation (default: "llama-4")
        api_key (str): API key for authentication (can be empty for some backends)
        max_tokens (int): Maximum number of tokens to generate (default: 2048, reduced for safety with smaller context windows)
        tools (list, optional): List of tool definitions in OpenAI function calling format

    Returns:
        str or dict: The generated response text from the server, or a dict with 'tool_calls' and 'content' 
                    if tool calls are present in the response.
    """
    # Process messages to convert image paths to base64
    processed_messages = []
    for message in messages:
        processed_message = message.copy()
        if message["role"] == "user" and "content" in message:
            processed_content = []
            for c in message["content"]:
                if isinstance(c, dict) and c.get("type") == "image":
                    # Convert image path to base64 format
                    image_path = c["image"]

                    print("image_path", image_path)
                    new_image_path = image_path.replace(
                        "?", "%3F"
                    )  # Escape ? in the path

                    # Read the image file and convert to base64
                    try:
                        base64_image, mime_type = get_image_base64_and_mime(
                            new_image_path
                        )
                        if base64_image is None:
                            print(
                                f"Warning: Could not convert image to base64: {new_image_path}"
                            )
                            continue

                        # Create the proper image_url structure with base64 data
                        processed_content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}",
                                    "detail": "high",
                                },
                            }
                        )

                    except FileNotFoundError:
                        print(f"Warning: Image file not found: {new_image_path}")
                        continue
                    except Exception as e:
                        print(f"Warning: Error processing image {new_image_path}: {e}")
                        continue
                else:
                    processed_content.append(c)

            processed_message["content"] = processed_content
        processed_messages.append(processed_message)

    # Normalize server URL - ensure it ends with /v1 for OpenAI-compatible APIs
    if server_url and not server_url.endswith('/v1'):
        if server_url.endswith('/'):
            server_url = server_url + 'v1'
        else:
            server_url = server_url + '/v1'

    # Create OpenAI client with custom base URL
    client = OpenAI(api_key=api_key or "not-needed", base_url=server_url)

    try:
        print(f"🔍 Calling model {model}...")
        print(f"   Server URL: {server_url}")
        print(f"   Messages: {len(processed_messages)} messages")
        if tools:
            print(f"   Tools: {len(tools)} tool(s) provided")
        
        # Prepare request parameters
        request_params = {
            "model": model,
            "messages": processed_messages,
            "max_tokens": max_tokens,
            "n": 1,
        }
        
        # Add tools if provided
        if tools:
            request_params["tools"] = tools
            request_params["tool_choice"] = "auto"  # Let model decide when to use tools
        
        response = client.chat.completions.create(**request_params)

        # Extract the response
        if response.choices and len(response.choices) > 0:
            message = response.choices[0].message
            
            # Check for tool calls in response
            if hasattr(message, 'tool_calls') and message.tool_calls:
                print(f"✅ Received {len(message.tool_calls)} tool call(s)")
                # Return structured response with tool calls
                tool_calls_list = []
                for tool_call in message.tool_calls:
                    tool_calls_list.append({
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,  # JSON string
                        }
                    })
                
                return {
                    "tool_calls": tool_calls_list,
                    "content": message.content or "",
                }
            
            # Regular text response
            content = message.content
            if content is None:
                print(f"⚠️ Warning: Response content is None")
                print(f"   Full response: {response}")
            return content
        else:
            print(f"❌ Unexpected response format: {response}")
            return None

    except Exception as e:
        import traceback
        print(f"❌ Request failed: {e}")
        print(f"   Traceback: {traceback.format_exc()}")
        # Re-raise to see the actual error in logs
        raise


def send_direct_request(
    llm: Any,
    messages: list[dict[str, Any]],
    sampling_params: Any,
) -> Optional[str]:
    """
    Run inference on a vLLM model instance directly without using a server.

    Args:
        llm: Initialized vLLM LLM instance (passed from external initialization)
        messages: List of message dicts with role and content (OpenAI format)
        sampling_params: vLLM SamplingParams instance (initialized externally)

    Returns:
        str: Generated response text, or None if inference fails
    """
    try:
        # Process messages to handle images (convert to base64 if needed)
        processed_messages = []
        for message in messages:
            processed_message = message.copy()
            if message["role"] == "user" and "content" in message:
                processed_content = []
                for c in message["content"]:
                    if isinstance(c, dict) and c.get("type") == "image":
                        # Convert image path to base64 format
                        image_path = c["image"]
                        new_image_path = image_path.replace("?", "%3F")

                        try:
                            base64_image, mime_type = get_image_base64_and_mime(
                                new_image_path
                            )
                            if base64_image is None:
                                print(
                                    f"Warning: Could not convert image: {new_image_path}"
                                )
                                continue

                            # vLLM expects image_url format
                            processed_content.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime_type};base64,{base64_image}"
                                    },
                                }
                            )
                        except Exception as e:
                            print(
                                f"Warning: Error processing image {new_image_path}: {e}"
                            )
                            continue
                    else:
                        processed_content.append(c)

                processed_message["content"] = processed_content
            processed_messages.append(processed_message)

        print("🔍 Running direct inference with vLLM...")

        # Run inference using vLLM's chat interface
        outputs = llm.chat(
            messages=processed_messages,
            sampling_params=sampling_params,
        )

        # Extract the generated text from the first output
        if outputs and len(outputs) > 0:
            generated_text = outputs[0].outputs[0].text
            return generated_text
        else:
            print(f"Unexpected output format: {outputs}")
            return None

    except Exception as e:
        print(f"Direct inference failed: {e}")
        return None
