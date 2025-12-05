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
  try {
    const response = await axios.post<SegmentResponse>(API_ENDPOINT, request, {
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 600000, // 10 minutes timeout (matches Modal backend timeout)
      signal, // Add abort signal support
    });
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      // Handle cancellation
      if (error.code === 'ERR_CANCELED' || error.message === 'canceled') {
        return {
          status: 'error',
          message: 'Request was cancelled by user',
        };
      }
      // Handle timeout specifically
      if (error.code === 'ECONNABORTED' || error.message.includes('timeout')) {
        return {
          status: 'error',
          message: 'Request timed out. The backend is still processing. Please check Modal logs or try again.',
        };
      }
      return {
        status: 'error',
        message: error.response?.data?.message || error.message || 'Network error',
      };
    }
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
    return response.data;
  } catch (error) {
    if (axios.isAxiosError(error)) {
      // Handle cancellation
      if (error.code === 'ERR_CANCELED' || error.message === 'canceled') {
        return {
          status: 'error',
          message: 'Request was cancelled by user',
        };
      }
      // Handle timeout specifically
      if (error.code === 'ECONNABORTED' || error.message.includes('timeout')) {
        return {
          status: 'error',
          message: 'Request timed out. The backend is still processing. Please check Modal logs or try again.',
        };
      }
      return {
        status: 'error',
        message: error.response?.data?.message || error.message || 'Network error',
      };
    }
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

