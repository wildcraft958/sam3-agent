#!/bin/bash

# SAM3 Agent cURL Test Script
# Usage: ./test_curl.sh [YOUR_OPENAI_API_KEY]

set -e

# Configuration
ENDPOINT_URL="https://srinjoy59--sam3-agent-sam3-segment.modal.run"
OPENAI_API_KEY="${1:-${OPENAI_API_KEY:-}}"
IMAGE_URL="https://images.unsplash.com/photo-1506905925346-21bda4d32df4"
PROMPT="segment all objects"

# Check API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ Error: OpenAI API key required"
    echo ""
    echo "Usage:"
    echo "  ./test_curl.sh sk-your-openai-api-key"
    echo ""
    echo "Or set environment variable:"
    echo "  export OPENAI_API_KEY=sk-your-openai-api-key"
    echo "  ./test_curl.sh"
    exit 1
fi

# Display configuration
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 SAM3 Agent Endpoint Test"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📍 Endpoint: $ENDPOINT_URL"
echo "📝 Prompt: $PROMPT"
echo "🖼️  Image: $IMAGE_URL"
echo "🔑 API Key: ${OPENAI_API_KEY:0:10}...${OPENAI_API_KEY: -4}"
echo ""
echo "⏳ Sending request (this may take 30-60 seconds for first request)..."
echo ""

# Make request
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$ENDPOINT_URL" \
  -H "Content-Type: application/json" \
  -d "{
    \"prompt\": \"$PROMPT\",
    \"image_url\": \"$IMAGE_URL\",
    \"llm_config\": {
      \"base_url\": \"https://api.openai.com/v1\",
      \"model\": \"gpt-4o\",
      \"api_key\": \"$OPENAI_API_KEY\",
      \"name\": \"openai-gpt4o\"
    },
    \"debug\": true
  }")

# Extract HTTP status code (last line)
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | sed '$d')

# Check HTTP status
if [ "$HTTP_CODE" != "200" ]; then
    echo "❌ HTTP Error: $HTTP_CODE"
    echo ""
    echo "Response:"
    echo "$BODY" | jq '.' 2>/dev/null || echo "$BODY"
    exit 1
fi

# Parse response
STATUS=$(echo "$BODY" | jq -r '.status' 2>/dev/null || echo "unknown")

if [ "$STATUS" = "success" ]; then
    echo "✅ SUCCESS!"
    echo ""
    
    # Extract key information
    SUMMARY=$(echo "$BODY" | jq -r '.summary' 2>/dev/null || echo "N/A")
    REGIONS_COUNT=$(echo "$BODY" | jq '.regions | length' 2>/dev/null || echo "0")
    
    echo "📊 Results:"
    echo "   Summary: $SUMMARY"
    echo "   Regions found: $REGIONS_COUNT"
    echo ""
    
    # Save full response
    echo "$BODY" | jq '.' > response.json
    echo "💾 Full response saved to: response.json"
    
    # Save debug image if available
    DEBUG_IMAGE=$(echo "$BODY" | jq -r '.debug_image_b64' 2>/dev/null)
    if [ "$DEBUG_IMAGE" != "null" ] && [ -n "$DEBUG_IMAGE" ]; then
        echo "$DEBUG_IMAGE" | base64 -d > debug_output.png 2>/dev/null
        echo "🖼️  Debug visualization saved to: debug_output.png"
    fi
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✅ Test completed successfully!"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
else
    echo "❌ ERROR"
    echo ""
    ERROR_MSG=$(echo "$BODY" | jq -r '.message' 2>/dev/null || echo "Unknown error")
    echo "Error message: $ERROR_MSG"
    echo ""
    echo "Full response:"
    echo "$BODY" | jq '.' 2>/dev/null || echo "$BODY"
    exit 1
fi

