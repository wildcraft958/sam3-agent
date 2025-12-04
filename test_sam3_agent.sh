#!/bin/bash
#
# Test script for SAM3 Agent on Modal
#
# Usage:
#   ./test_sam3_agent.sh "your prompt" "image_url"
#   ./test_sam3_agent.sh "segment all buildings" "https://example.com/satellite.jpg"
#
# Or with environment variables:
#   IMAGE_URL="https://..." PROMPT="segment cars" ./test_sam3_agent.sh
#

# ============================================================================
# Configuration - Update these with your endpoints
# ============================================================================

SAM3_ENDPOINT="${SAM3_ENDPOINT:-https://aryan-don357--sam3-agent-sam3-segment.modal.run}"
VLLM_ENDPOINT="${VLLM_ENDPOINT:-https://aryan-don357--qwen3-vl-vllm-server-30b-vllm-server.modal.run/v1}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-32B-Instruct}"
API_KEY="${API_KEY:-dummy}"  # vLLM doesn't require real key

# ============================================================================
# Parse arguments
# ============================================================================

PROMPT="${1:-${PROMPT:-segment all buildings}}"
IMAGE_URL="${2:-${IMAGE_URL:-}}"
DEBUG="${DEBUG:-true}"

# ============================================================================
# Validate inputs
# ============================================================================

if [ -z "$IMAGE_URL" ]; then
    echo "❌ Error: No image URL provided"
    echo ""
    echo "Usage:"
    echo "  $0 \"prompt\" \"image_url\""
    echo ""
    echo "Examples:"
    echo "  $0 \"segment all buildings\" \"https://example.com/satellite.jpg\""
    echo "  $0 \"segment roads\" \"https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/800px-Camponotus_flavomarginatus_ant.jpg\""
    echo ""
    echo "Or use environment variables:"
    echo "  IMAGE_URL=\"https://...\" PROMPT=\"segment cars\" $0"
    exit 1
fi

# ============================================================================
# Display configuration
# ============================================================================

echo "=============================================="
echo "🚀 SAM3 Agent Test"
echo "=============================================="
echo "SAM3 Endpoint: $SAM3_ENDPOINT"
echo "vLLM Endpoint: $VLLM_ENDPOINT"
echo "Model: $MODEL_NAME"
echo "Prompt: $PROMPT"
echo "Image URL: $IMAGE_URL"
echo "Debug: $DEBUG"
echo "=============================================="
echo ""

# ============================================================================
# Make API request
# ============================================================================

echo "📡 Sending request to SAM3 agent..."
echo ""

RESPONSE=$(curl -s -X POST "$SAM3_ENDPOINT" \
  -H "Content-Type: application/json" \
  -d "{
    \"prompt\": \"$PROMPT\",
    \"image_url\": \"$IMAGE_URL\",
    \"llm_config\": {
      \"base_url\": \"$VLLM_ENDPOINT\",
      \"model\": \"$MODEL_NAME\",
      \"api_key\": \"$API_KEY\",
      \"max_tokens\": 4096
    },
    \"debug\": $DEBUG
  }")

# ============================================================================
# Parse and display response
# ============================================================================

echo "📋 Response:"
echo "=============================================="

# Check if response is valid JSON
if echo "$RESPONSE" | python3 -c "import sys, json; json.load(sys.stdin)" 2>/dev/null; then
    # Pretty print JSON
    echo "$RESPONSE" | python3 -m json.tool
    
    # Extract key info
    STATUS=$(echo "$RESPONSE" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('status', 'unknown'))" 2>/dev/null)
    
    if [ "$STATUS" = "success" ]; then
        echo ""
        echo "✅ Success!"
        
        # Extract summary if available
        SUMMARY=$(echo "$RESPONSE" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('summary', ''))" 2>/dev/null)
        if [ -n "$SUMMARY" ]; then
            echo "📝 Summary: $SUMMARY"
        fi
        
        # Save debug image if present
        if [ "$DEBUG" = "true" ]; then
            DEBUG_IMAGE=$(echo "$RESPONSE" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('debug_image_b64', ''))" 2>/dev/null)
            if [ -n "$DEBUG_IMAGE" ] && [ "$DEBUG_IMAGE" != "None" ]; then
                OUTPUT_FILE="sam3_result_$(date +%Y%m%d_%H%M%S).png"
                echo "$DEBUG_IMAGE" | base64 -d > "$OUTPUT_FILE"
                echo "🖼️  Debug image saved to: $OUTPUT_FILE"
            fi
        fi
    else
        echo ""
        echo "❌ Request failed"
        ERROR=$(echo "$RESPONSE" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('error', d.get('message', 'Unknown error')))" 2>/dev/null)
        echo "Error: $ERROR"
    fi
else
    # Not valid JSON, print raw response
    echo "$RESPONSE"
fi

echo ""
echo "=============================================="

