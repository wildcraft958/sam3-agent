import axios from 'axios';

const API_ENDPOINT = import.meta.env.VITE_API_ENDPOINT || 'https://aryan-don357--sam3-agent-sam3-segment.modal.run';
const INFER_ENDPOINT = import.meta.env.VITE_INFER_ENDPOINT || 'https://aryan-don357--sam3-agent-sam3-infer.modal.run';

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
}

export interface InferRequest {
  text_prompt: string;
  image_b64?: string;
  image_url?: string;
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

export async function inferImage(
  request: InferRequest,
  signal?: AbortSignal
): Promise<InferResponse> {
  try {
    const response = await axios.post<InferResponse>(INFER_ENDPOINT, request, {
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

