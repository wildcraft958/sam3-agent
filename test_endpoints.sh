#!/bin/bash
# =============================================================================
# SAM3 Modal Endpoint Test Script (VLM-Enhanced)
# =============================================================================
# 
# Tests the VLM-enhanced /sam3/count and /sam3/area endpoints
# Requires llm_config with Qwen3-VL-30B deployment
#
# =============================================================================

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Modal workspace
MODAL_WORKSPACE="srinjoy59"

# SAM3 endpoint base URL
SAM3_BASE_URL="https://${MODAL_WORKSPACE}--sam3-agent-pyramidal"

# Qwen3-VL vLLM endpoint for VLM operations
VLLM_BASE_URL="https://srinjoy59--qwen3-vl-vllm-server-30b-vllm-server.modal.run/v1"
VLLM_MODEL="Qwen/Qwen3-VL-30B-A3B-Instruct"

# Test image URL (use any publicly accessible image)
TEST_IMAGE_URL="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/1200px-Camponotus_flavomarginatus_ant.jpg"

# Optional: Ground Sample Distance for area calculation (meters per pixel)
GSD="0.5"

# Confidence threshold (0.0 - 1.0)
CONFIDENCE_THRESHOLD="0.4"

# Max retries for VLM rephrasing
MAX_RETRIES="2"

# -----------------------------------------------------------------------------
# Color output helpers
# -----------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
echo_success() { echo -e "${GREEN}✅ $1${NC}"; }
echo_warn() { echo -e "${YELLOW}⚠️  $1${NC}"; }
echo_error() { echo -e "${RED}❌ $1${NC}"; }

# -----------------------------------------------------------------------------
# Test 1: /sam3/count - Count objects with VLM enhancement
# -----------------------------------------------------------------------------

test_count() {
    local prompt="${1:-ant}"
    local endpoint="${SAM3_BASE_URL}-sam3-count.modal.run"
    
    echo ""
    echo "=============================================="
    echo_info "Testing /sam3/count endpoint (VLM-enhanced)"
    echo "=============================================="
    echo "Endpoint: $endpoint"
    echo "Prompt: $prompt"
    echo "Image URL: $TEST_IMAGE_URL"
    echo "VLM: $VLLM_MODEL"
    echo ""
    
    local response=$(curl -s -X POST "$endpoint" \
        -H "Content-Type: application/json" \
        -d "{
            \"prompt\": \"$prompt\",
            \"image_url\": \"$TEST_IMAGE_URL\",
            \"llm_config\": {
                \"base_url\": \"$VLLM_BASE_URL\",
                \"model\": \"$VLLM_MODEL\",
                \"api_key\": \"\"
            },
            \"confidence_threshold\": $CONFIDENCE_THRESHOLD,
            \"max_retries\": $MAX_RETRIES,
            \"pyramidal_config\": {
                \"tile_size\": 512,
                \"overlap_ratio\": 0.15,
                \"scales\": [1.0, 0.5],
                \"batch_size\": 16,
                \"iou_threshold\": 0.5
            }
        }")
    
    echo "Response:"
    echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"
    
    # Check status
    local status=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null)
    if [ "$status" = "success" ]; then
        local count=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('count',0))" 2>/dev/null)
        local visual_prompt=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('visual_prompt',''))" 2>/dev/null)
        echo_success "Count endpoint returned: $count objects (visual prompt: '$visual_prompt')"
    else
        echo_error "Count endpoint failed"
    fi
}

# -----------------------------------------------------------------------------
# Test 2: /sam3/area - Calculate object areas with VLM enhancement
# -----------------------------------------------------------------------------

test_area() {
    local prompt="${1:-ant}"
    local endpoint="${SAM3_BASE_URL}-sam3-area.modal.run"
    
    echo ""
    echo "=============================================="
    echo_info "Testing /sam3/area endpoint (VLM-enhanced)"
    echo "=============================================="
    echo "Endpoint: $endpoint"
    echo "Prompt: $prompt"
    echo "Image URL: $TEST_IMAGE_URL"
    echo "GSD: ${GSD}"
    echo "VLM: $VLLM_MODEL"
    echo ""
    
    local response=$(curl -s -X POST "$endpoint" \
        -H "Content-Type: application/json" \
        -d "{
            \"prompt\": \"$prompt\",
            \"image_url\": \"$TEST_IMAGE_URL\",
            \"llm_config\": {
                \"base_url\": \"$VLLM_BASE_URL\",
                \"model\": \"$VLLM_MODEL\",
                \"api_key\": \"\"
            },
            \"gsd\": $GSD,
            \"confidence_threshold\": $CONFIDENCE_THRESHOLD,
            \"max_retries\": $MAX_RETRIES,
            \"pyramidal_config\": {
                \"tile_size\": 512,
                \"overlap_ratio\": 0.15,
                \"scales\": [1.0, 0.5],
                \"batch_size\": 16,
                \"iou_threshold\": 0.5
            }
        }")
    
    echo "Response:"
    echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"
    
    # Check status
    local status=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null)
    if [ "$status" = "success" ]; then
        local count=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('object_count',0))" 2>/dev/null)
        local total_area=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('total_real_area_m2',0))" 2>/dev/null)
        local coverage=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('coverage_percentage',0))" 2>/dev/null)
        echo_success "Area endpoint returned: $count objects, ${total_area} m², ${coverage}% coverage"
    else
        echo_error "Area endpoint failed"
    fi
}

# -----------------------------------------------------------------------------
# Test 3: /sam3/segment - Full agent with LLM
# -----------------------------------------------------------------------------

test_segment() {
    local prompt="${1:-segment all visible objects}"
    local endpoint="${SAM3_BASE_URL}-sam3-segment.modal.run"
    
    echo ""
    echo "=============================================="
    echo_info "Testing /sam3/segment endpoint"
    echo "=============================================="
    echo "Endpoint: $endpoint"
    echo "Prompt: $prompt"
    echo "Image URL: $TEST_IMAGE_URL"
    echo "LLM: $VLLM_MODEL"
    echo ""
    
    # Using Qwen3-VL vLLM deployment (no API key needed)
    local response=$(curl -s -X POST "$endpoint" \
        -H "Content-Type: application/json" \
        -d "{
            \"prompt\": \"$prompt\",
            \"image_url\": \"$TEST_IMAGE_URL\",
            \"llm_config\": {
                \"base_url\": \"$VLLM_BASE_URL\",
                \"model\": \"$VLLM_MODEL\",
                \"api_key\": \"\",
                \"name\": \"qwen3-vl-30b\",
                \"max_tokens\": 4096
            },
            \"debug\": false,
            \"confidence_threshold\": $CONFIDENCE_THRESHOLD
        }")
    
    echo "Response (truncated):"
    echo "$response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    # Truncate large fields for display
    if 'debug_image_b64' in data and data['debug_image_b64']:
        data['debug_image_b64'] = data['debug_image_b64'][:50] + '... (truncated)'
    if 'regions' in data and len(data['regions']) > 3:
        data['regions'] = data['regions'][:3] + [{'...': f'{len(data[\"regions\"])-3} more regions'}]
    print(json.dumps(data, indent=2))
except:
    print(sys.stdin.read())
" 2>/dev/null || echo "$response"
    
    # Check status
    local status=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null)
    if [ "$status" = "success" ]; then
        local regions=$(echo "$response" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('regions',[])))" 2>/dev/null)
        echo_success "Segment endpoint returned: $regions regions"
    else
        echo_error "Segment endpoint failed"
    fi
}

# -----------------------------------------------------------------------------
# Main - Run all tests or specific test
# -----------------------------------------------------------------------------

usage() {
    echo "Usage: $0 [command] [prompt]"
    echo ""
    echo "Commands:"
    echo "  count    Test /sam3/count endpoint (VLM-enhanced counting)"
    echo "  area     Test /sam3/area endpoint (VLM-enhanced area calculation)"
    echo "  segment  Test /sam3/segment endpoint (Qwen3-VL + SAM3 agent)"
    echo "  all      Run all tests"
    echo ""
    echo "Examples:"
    echo "  $0 count \"trees\""
    echo "  $0 area \"solar panels\""
    echo "  $0 segment \"segment all buildings\""
    echo "  $0 all"
    echo ""
    echo "Endpoints:"
    echo "  SAM3: ${SAM3_BASE_URL}-sam3-*.modal.run"
    echo "  vLLM: $VLLM_BASE_URL ($VLLM_MODEL)"
}

main() {
    echo "=============================================="                   
    echo "SAM3 Modal Endpoint Test Script (VLM-Enhanced)"
    echo "=============================================="
    echo "Workspace: $MODAL_WORKSPACE"
    echo "vLLM Model: $VLLM_MODEL"
    echo ""
    
    case "${1:-all}" in
        count)
            test_count "${2:-ant}"
            ;;
        area)
            test_area "${2:-ant}"
            ;;
        segment)
            test_segment "${2:-segment the ant}"
            ;;
        all)
            test_count "ant"
            test_area "ant"
            test_segment "segment the ant"
            ;;
        -h|--help|help)
            usage
            ;;
        *)
            echo_error "Unknown command: $1"
            usage
            exit 1
            ;;
    esac
    
    echo ""
    echo "=============================================="
    echo "Tests completed!"
    echo "=============================================="
}

main "$@"

