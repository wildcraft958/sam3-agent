import axios from 'axios';

const API_ENDPOINT = import.meta.env.VITE_API_ENDPOINT || 'https://srinjoy59--sam3-agent-pyramidal-sam3-segment.modal.run';
const COUNT_ENDPOINT = import.meta.env.VITE_COUNT_ENDPOINT || 'https://srinjoy59--sam3-agent-pyramidal-sam3-count.modal.run';

export interface LLMConfig {
  base_url: string;
  model: string;
  api_key: string;
  name?: string;
  max_tokens?: number;
}

export interface SegmentRequest {
  prompt: string;
  image_b64?: string;
  image_url?: string;
  llm_config: LLMConfig;
  debug?: boolean;
  confidence_threshold?: number;
}

export interface CountRequest {
  prompt: string;
  image_b64?: string;
  image_url?: string;
  confidence_threshold?: number;
}

export interface Region {
  bbox?: number[];
  mask?: {
    counts: string | number[];
    size: number[];
  };
  score?: number;
}

export interface SegmentResponse {
  status: 'success' | 'error';
  summary?: string;
  regions?: Region[];
  debug_image_b64?: string;
  raw_sam3_json?: {
    orig_img_h: number;
    orig_img_w: number;
    pred_boxes: number[][];
    pred_masks: Array<{
      counts: string | number[];
      size: number[];
    }>;
    pred_scores: number[];
  };
  llm_config?: {
    name: string;
    model: string;
    base_url: string;
  };
  message?: string;
  traceback?: string;
}

export interface CountResponse {
  status: 'success' | 'error';
  count?: number;
  object_type?: string;
  confidence_summary?: {
    high: number;
    medium: number;
    low: number;
  };
  detections?: Array<{
    box: number[];
    mask_rle: { counts: string | number[]; size: number[] };
    score: number;
    scale: number;
  }>;
  orig_img_h?: number;
  orig_img_w?: number;
  message?: string;
  traceback?: string;
}

export async function segmentImage(
  request: SegmentRequest,
  signal?: AbortSignal
): Promise<SegmentResponse> {
  const endpoint = getApiEndpoint();
  const startTime = Date.now();
  console.log('[API] Starting segmentImage request', {
    endpoint: endpoint,
    prompt: request.prompt?.substring(0, 50) + '...',
    hasImage: !!request.image_b64,
    imageSize: request.image_b64 ? `${Math.round(request.image_b64.length / 1024)}KB` : 'N/A',
  });

  try {
    const response = await axios.post<SegmentResponse>(endpoint, request, {
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 600000, // 10 minutes timeout (matches Modal backend timeout)
      signal, // Add abort signal support
    });
    
    const duration = Date.now() - startTime;
    console.log('[API] segmentImage request succeeded', {
      duration: `${duration}ms`,
      status: response.data.status,
      regionsCount: response.data.regions?.length || 0,
      hasRawData: !!response.data.raw_sam3_json,
    });

    if (response.data.status === 'error') {
      console.error('[API] segmentImage returned error status', {
        message: response.data.message,
        traceback: response.data.traceback,
      });
    }

    return response.data;
  } catch (error) {
    const duration = Date.now() - startTime;
    
    if (axios.isAxiosError(error)) {
      // Check for CORS errors
      const isCorsError = error.code === 'ERR_NETWORK' || 
                         error.message.includes('CORS') ||
                         error.message.includes('cross-origin') ||
                         (!error.response && error.request);
      
      if (isCorsError) {
        console.error('[API] CORS error detected in segmentImage', {
          endpoint: endpoint,
          error: error.message,
          code: error.code,
          duration: `${duration}ms`,
        });
        return {
          status: 'error',
          message: `CORS Error: Cannot connect to ${endpoint}. The server may not allow cross-origin requests, or the endpoint is unreachable. Check browser console for details.`,
        };
      }

      // Handle cancellation
      if (error.code === 'ERR_CANCELED' || error.message === 'canceled') {
        console.log('[API] segmentImage request cancelled by user');
        return {
          status: 'error',
          message: 'Request was cancelled by user',
        };
      }
      
      // Handle timeout specifically
      if (error.code === 'ECONNABORTED' || error.message.includes('timeout')) {
        console.error('[API] segmentImage request timed out', {
          endpoint: endpoint,
          duration: `${duration}ms`,
        });
        return {
          status: 'error',
          message: 'Request timed out. The backend is still processing. Please check Modal logs or try again.',
        };
      }

      // Network errors
      if (error.code === 'ERR_NETWORK' || error.code === 'ECONNREFUSED') {
        console.error('[API] Network error in segmentImage', {
          endpoint: endpoint,
          code: error.code,
          message: error.message,
        });
        return {
          status: 'error',
          message: `Network Error: Cannot reach ${endpoint}. Check your internet connection and ensure the endpoint is accessible.`,
        };
      }

      console.error('[API] segmentImage request failed', {
        endpoint: endpoint,
        status: error.response?.status,
        statusText: error.response?.statusText,
        errorCode: error.code,
        errorMessage: error.message,
        responseData: error.response?.data,
        duration: `${duration}ms`,
      });

      return {
        status: 'error',
        message: error.response?.data?.message || error.message || 'Network error',
      };
    }

    console.error('[API] Unexpected error in segmentImage', {
      error: error instanceof Error ? error.message : String(error),
      errorType: error?.constructor?.name,
    });

    return {
      status: 'error',
      message: error instanceof Error ? error.message : 'Unknown error',
    };
  }
}

export async function countImage(
  request: CountRequest,
  signal?: AbortSignal
): Promise<CountResponse> {
  try {
    const response = await axios.post<CountResponse>(COUNT_ENDPOINT, request, {
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 600000, // 10 minutes timeout (matches Modal backend timeout)
      signal, // Add abort signal support
    });
    
    const duration = Date.now() - startTime;
    console.log('[API] inferImage request succeeded', {
      duration: `${duration}ms`,
      status: response.data.status,
      boxesCount: response.data.pred_boxes?.length || 0,
      masksCount: response.data.pred_masks?.length || 0,
    });

    if (response.data.status === 'error') {
      console.error('[API] inferImage returned error status', {
        message: response.data.message,
        traceback: response.data.traceback,
      });
    }

    return response.data;
  } catch (error) {
    const duration = Date.now() - startTime;
    
    if (axios.isAxiosError(error)) {
      // Check for CORS errors
      const isCorsError = error.code === 'ERR_NETWORK' || 
                         error.message.includes('CORS') ||
                         error.message.includes('cross-origin') ||
                         (!error.response && error.request);
      
      if (isCorsError) {
        console.error('[API] CORS error detected in inferImage', {
          endpoint: endpoint,
          error: error.message,
          code: error.code,
          duration: `${duration}ms`,
        });
        return {
          status: 'error',
          message: `CORS Error: Cannot connect to ${endpoint}. The server may not allow cross-origin requests, or the endpoint is unreachable. Check browser console for details.`,
        };
      }

      // Handle cancellation
      if (error.code === 'ERR_CANCELED' || error.message === 'canceled') {
        console.log('[API] inferImage request cancelled by user');
        return {
          status: 'error',
          message: 'Request was cancelled by user',
        };
      }
      
      // Handle timeout specifically
      if (error.code === 'ECONNABORTED' || error.message.includes('timeout')) {
        console.error('[API] inferImage request timed out', {
          endpoint: endpoint,
          duration: `${duration}ms`,
        });
        return {
          status: 'error',
          message: 'Request timed out. The backend is still processing. Please check Modal logs or try again.',
        };
      }

      // Network errors
      if (error.code === 'ERR_NETWORK' || error.code === 'ECONNREFUSED') {
        console.error('[API] Network error in inferImage', {
          endpoint: endpoint,
          code: error.code,
          message: error.message,
        });
        return {
          status: 'error',
          message: `Network Error: Cannot reach ${endpoint}. Check your internet connection and ensure the endpoint is accessible.`,
        };
      }

      console.error('[API] inferImage request failed', {
        endpoint: endpoint,
        status: error.response?.status,
        statusText: error.response?.statusText,
        errorCode: error.code,
        errorMessage: error.message,
        responseData: error.response?.data,
        duration: `${duration}ms`,
      });

      return {
        status: 'error',
        message: error.response?.data?.message || error.message || 'Network error',
      };
    }

    console.error('[API] Unexpected error in inferImage', {
      error: error instanceof Error ? error.message : String(error),
      errorType: error?.constructor?.name,
    });

    return {
      status: 'error',
      message: error instanceof Error ? error.message : 'Unknown error',
    };
  }
}

export function imageToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const img = new Image();
      img.onload = () => {
        // Create canvas to convert image to RGB format (remove alpha channel if present)
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d');
        
        if (!ctx) {
          reject(new Error('Could not get canvas context'));
          return;
        }
        
        // Fill with white background (in case of transparency)
        ctx.fillStyle = '#FFFFFF';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Draw image (this will convert RGBA to RGB if needed)
        ctx.drawImage(img, 0, 0);
        
        // Convert to base64 JPEG (always RGB, no alpha)
        const base64 = canvas.toDataURL('image/jpeg', 0.95).split(',')[1];
        resolve(base64);
      };
      img.onerror = () => reject(new Error('Failed to load image'));
      img.src = reader.result as string;
    };
    reader.onerror = (error) => reject(error);
    reader.readAsDataURL(file);
  });
}

// Connectivity check result interface
export interface ConnectivityResult {
  endpoint: string;
  reachable: boolean;
  statusCode?: number;
  statusText?: string;
  error?: string;
  errorType?: 'CORS' | 'NETWORK' | 'TIMEOUT' | 'HTTP' | 'UNKNOWN';
  responseTime?: number;
}

// Check if an endpoint is reachable (doesn't require valid request body)
export async function checkEndpointConnectivity(
  endpoint: string,
  timeout: number = 5000
): Promise<ConnectivityResult> {
  const startTime = Date.now();
  console.log('[Connectivity] Checking endpoint:', endpoint);

  try {
    // Try a simple OPTIONS request first (preflight) or HEAD request
    // Many APIs don't support OPTIONS/HEAD, so we'll try a minimal POST
    const testResponse = await axios.post(
      endpoint,
      {}, // Empty body
      {
        headers: {
          'Content-Type': 'application/json',
        },
        timeout: timeout,
        validateStatus: (status) => {
          // Accept any status code - we just want to know if it's reachable
          return status < 600; // Accept all HTTP status codes
        },
      }
    );

    const responseTime = Date.now() - startTime;
    console.log('[Connectivity] Endpoint reachable', {
      endpoint,
      status: testResponse.status,
      responseTime: `${responseTime}ms`,
    });

    return {
      endpoint,
      reachable: true,
      statusCode: testResponse.status,
      statusText: testResponse.statusText,
      responseTime,
    };
  } catch (error) {
    const responseTime = Date.now() - startTime;

    if (axios.isAxiosError(error)) {
      // CORS errors
      if (
        error.code === 'ERR_NETWORK' ||
        (!error.response && error.request) ||
        error.message.includes('CORS') ||
        error.message.includes('cross-origin')
      ) {
        console.error('[Connectivity] CORS error', { endpoint, error: error.message });
        return {
          endpoint,
          reachable: false,
          error: `CORS Error: ${error.message}`,
          errorType: 'CORS',
          responseTime,
        };
      }

      // Timeout
      if (error.code === 'ECONNABORTED' || error.message.includes('timeout')) {
        console.error('[Connectivity] Timeout', { endpoint, responseTime: `${responseTime}ms` });
        return {
          endpoint,
          reachable: false,
          error: `Request timed out after ${timeout}ms`,
          errorType: 'TIMEOUT',
          responseTime,
        };
      }

      // Network errors
      if (error.code === 'ERR_NETWORK' || error.code === 'ECONNREFUSED') {
        console.error('[Connectivity] Network error', {
          endpoint,
          code: error.code,
          error: error.message,
        });
        return {
          endpoint,
          reachable: false,
          error: `Network Error: ${error.message}`,
          errorType: 'NETWORK',
          responseTime,
        };
      }

      // HTTP errors (endpoint is reachable, but returned an error)
      if (error.response) {
        console.log('[Connectivity] Endpoint reachable but returned error', {
          endpoint,
          status: error.response.status,
          statusText: error.response.statusText,
        });
        return {
          endpoint,
          reachable: true, // Endpoint is reachable, just returned an error
          statusCode: error.response.status,
          statusText: error.response.statusText,
          error: error.response.statusText,
          errorType: 'HTTP',
          responseTime,
        };
      }
    }

    console.error('[Connectivity] Unknown error', {
      endpoint,
      error: error instanceof Error ? error.message : String(error),
    });

    return {
      endpoint,
      reachable: false,
      error: error instanceof Error ? error.message : 'Unknown error',
      errorType: 'UNKNOWN',
      responseTime,
    };
  }
}

// Check connectivity of both endpoints
export async function checkAllEndpoints(
  timeout: number = 5000
): Promise<{
  apiEndpoint: ConnectivityResult;
  inferEndpoint: ConnectivityResult;
}> {
  const apiEndpoint = getApiEndpoint();
  const inferEndpoint = getInferEndpoint();

  console.log('[Connectivity] Checking all endpoints');

  const [apiResult, inferResult] = await Promise.all([
    checkEndpointConnectivity(apiEndpoint, timeout),
    checkEndpointConnectivity(inferEndpoint, timeout),
  ]);

  return {
    apiEndpoint: apiResult,
    inferEndpoint: inferResult,
  };
}

