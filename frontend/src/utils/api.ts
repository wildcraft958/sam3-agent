import axios from 'axios';

// Storage keys for runtime endpoint override
const STORAGE_KEY_API_ENDPOINT = 'sam3_api_endpoint';
const STORAGE_KEY_INFER_ENDPOINT = 'sam3_infer_endpoint';

// Default endpoints (build-time)
const DEFAULT_API_ENDPOINT = 'https://srinjoy59--sam3-agent-sam3-segment.modal.run';
const DEFAULT_INFER_ENDPOINT = 'https://srinjoy59--sam3-agent-sam3-infer.modal.run';

// Validate URL format
function isValidUrl(url: string): boolean {
  try {
    const parsed = new URL(url);
    return parsed.protocol === 'http:' || parsed.protocol === 'https:';
  } catch {
    return false;
  }
}

// Get endpoint with fallback: runtime override > env var > default
function getEndpoint(
  envVar: string | undefined,
  storageKey: string,
  defaultValue: string
): string {
  // 1. Check localStorage for runtime override
  if (typeof window !== 'undefined') {
    const stored = localStorage.getItem(storageKey);
    if (stored && stored.trim() && isValidUrl(stored.trim())) {
      return stored.trim();
    }
  }

  // 2. Check environment variable
  if (envVar && envVar.trim() && isValidUrl(envVar.trim())) {
    return envVar.trim();
  }

  // 3. Fall back to default
  return defaultValue;
}

// Helper function to get current endpoint (checks localStorage each time for runtime changes)
function getCurrentEndpoint(
  envVar: string | undefined,
  storageKey: string,
  defaultValue: string
): string {
  // 1. Check localStorage for runtime override (checked each time for dynamic updates)
  if (typeof window !== 'undefined') {
    const stored = localStorage.getItem(storageKey);
    if (stored && stored.trim() && isValidUrl(stored.trim())) {
      return stored.trim();
    }
  }

  // 2. Check environment variable
  if (envVar && envVar.trim() && isValidUrl(envVar.trim())) {
    return envVar.trim();
  }

  // 3. Fall back to default
  return defaultValue;
}

// Initial endpoint values for logging (functions use dynamic lookup)
const INITIAL_API_ENDPOINT = getCurrentEndpoint(
  import.meta.env.VITE_API_ENDPOINT,
  STORAGE_KEY_API_ENDPOINT,
  DEFAULT_API_ENDPOINT
);
const INITIAL_INFER_ENDPOINT = getCurrentEndpoint(
  import.meta.env.VITE_INFER_ENDPOINT,
  STORAGE_KEY_INFER_ENDPOINT,
  DEFAULT_INFER_ENDPOINT
);

// Export functions to get and set endpoints at runtime (always gets current value)
export function getApiEndpoint(): string {
  return getCurrentEndpoint(
    import.meta.env.VITE_API_ENDPOINT,
    STORAGE_KEY_API_ENDPOINT,
    DEFAULT_API_ENDPOINT
  );
}

export function getInferEndpoint(): string {
  return getCurrentEndpoint(
    import.meta.env.VITE_INFER_ENDPOINT,
    STORAGE_KEY_INFER_ENDPOINT,
    DEFAULT_INFER_ENDPOINT
  );
}

export function setApiEndpoint(url: string): { success: boolean; error?: string } {
  if (!isValidUrl(url)) {
    return { success: false, error: 'Invalid URL format. Must start with http:// or https://' };
  }
  if (typeof window !== 'undefined') {
    localStorage.setItem(STORAGE_KEY_API_ENDPOINT, url.trim());
    console.log('[API Config] API endpoint updated to:', url.trim());
    return { success: true };
  }
  return { success: false, error: 'localStorage not available' };
}

export function setInferEndpoint(url: string): { success: boolean; error?: string } {
  if (!isValidUrl(url)) {
    return { success: false, error: 'Invalid URL format. Must start with http:// or https://' };
  }
  if (typeof window !== 'undefined') {
    localStorage.setItem(STORAGE_KEY_INFER_ENDPOINT, url.trim());
    console.log('[API Config] Infer endpoint updated to:', url.trim());
    return { success: true };
  }
  return { success: false, error: 'localStorage not available' };
}

export function resetEndpoints(): void {
  if (typeof window !== 'undefined') {
    localStorage.removeItem(STORAGE_KEY_API_ENDPOINT);
    localStorage.removeItem(STORAGE_KEY_INFER_ENDPOINT);
    console.log('[API Config] Endpoints reset to defaults');
  }
}

export function getEndpointConfig() {
  return {
    apiEndpoint: getApiEndpoint(),
    inferEndpoint: getInferEndpoint(),
    apiEndpointOverride: typeof window !== 'undefined' ? localStorage.getItem(STORAGE_KEY_API_ENDPOINT) : null,
    inferEndpointOverride: typeof window !== 'undefined' ? localStorage.getItem(STORAGE_KEY_INFER_ENDPOINT) : null,
    envApiEndpoint: import.meta.env.VITE_API_ENDPOINT || '(not set)',
    envInferEndpoint: import.meta.env.VITE_INFER_ENDPOINT || '(not set)',
    defaultApiEndpoint: DEFAULT_API_ENDPOINT,
    defaultInferEndpoint: DEFAULT_INFER_ENDPOINT,
  };
}

// Log endpoint configuration on load
console.log('[API Config] Endpoint Configuration:', getEndpointConfig());

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

export interface InferRequest {
  text_prompt: string;
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

export interface InferResponse {
  status: 'success' | 'error';
  orig_img_h?: number;
  orig_img_w?: number;
  pred_boxes?: number[][];
  pred_masks?: Array<{
    counts: string | number[];
    size: number[];
  }>;
  pred_scores?: number[];
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

export async function inferImage(
  request: InferRequest,
  signal?: AbortSignal
): Promise<InferResponse> {
  const endpoint = getInferEndpoint();
  const startTime = Date.now();
  console.log('[API] Starting inferImage request', {
    endpoint: endpoint,
    prompt: request.text_prompt?.substring(0, 50) + '...',
    hasImage: !!request.image_b64,
    imageSize: request.image_b64 ? `${Math.round(request.image_b64.length / 1024)}KB` : 'N/A',
  });

  try {
    const response = await axios.post<InferResponse>(endpoint, request, {
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

