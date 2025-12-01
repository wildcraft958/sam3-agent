import { decodeRLE, normalizedBboxToPixels } from './maskUtils';

export interface RLEMask {
  counts: string | number[];
  size: number[];
}

/**
 * Generate a color palette for multiple masks
 */
export function generateColors(count: number): string[] {
  const colors: string[] = [];
  const hueStep = 360 / count;

  for (let i = 0; i < count; i++) {
    const hue = (i * hueStep) % 360;
    const color = `hsl(${hue}, 70%, 50%)`;
    colors.push(color);
  }

  return colors;
}

/**
 * Convert HSL color to RGB
 */
function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  s /= 100;
  l /= 100;
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = l - c / 2;
  let r = 0, g = 0, b = 0;

  if (0 <= h && h < 60) {
    r = c; g = x; b = 0;
  } else if (60 <= h && h < 120) {
    r = x; g = c; b = 0;
  } else if (120 <= h && h < 180) {
    r = 0; g = c; b = x;
  } else if (180 <= h && h < 240) {
    r = 0; g = x; b = c;
  } else if (240 <= h && h < 300) {
    r = x; g = 0; b = c;
  } else if (300 <= h && h < 360) {
    r = c; g = 0; b = x;
  }

  r = Math.round((r + m) * 255);
  g = Math.round((g + m) * 255);
  b = Math.round((b + m) * 255);

  return [r, g, b];
}

/**
 * Draw a mask on canvas with semi-transparent overlay
 */
export function drawMask(
  ctx: CanvasRenderingContext2D,
  mask: RLEMask,
  imgWidth: number,
  imgHeight: number,
  color: string,
  alpha: number = 0.3
): void {
  const decodedMask = decodeRLE(mask, imgWidth, imgHeight);
  const imageData = ctx.getImageData(0, 0, imgWidth, imgHeight);
  const data = imageData.data;

  // Parse color
  const rgb = parseColor(color);
  const [r, g, b] = rgb;

  for (let i = 0; i < decodedMask.length; i++) {
    if (decodedMask[i] > 0) {
      const idx = i * 4;
      // Alpha blend
      data[idx] = Math.round(r * alpha + data[idx] * (1 - alpha));
      data[idx + 1] = Math.round(g * alpha + data[idx + 1] * (1 - alpha));
      data[idx + 2] = Math.round(b * alpha + data[idx + 2] * (1 - alpha));
      // Keep original alpha
    }
  }

  ctx.putImageData(imageData, 0, 0);
}

/**
 * Draw a bounding box on canvas
 */
export function drawBbox(
  ctx: CanvasRenderingContext2D,
  bbox: number[],
  imgWidth: number,
  imgHeight: number,
  color: string,
  lineWidth: number = 2
): void {
  const [x1, y1, x2, y2] = normalizedBboxToPixels(bbox, imgWidth, imgHeight);

  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
}

/**
 * Draw a label with score on canvas
 */
export function drawLabel(
  ctx: CanvasRenderingContext2D,
  bbox: number[],
  imgWidth: number,
  imgHeight: number,
  label: string,
  color: string,
  score?: number
): void {
  const [x1, y1] = normalizedBboxToPixels(bbox, imgWidth, imgHeight);
  const text = score !== undefined ? `${label} (${score.toFixed(2)})` : label;

  ctx.fillStyle = color;
  ctx.font = '14px Arial';
  ctx.fillText(text, x1, Math.max(y1 - 5, 15));
}

/**
 * Parse color string to RGB
 */
function parseColor(color: string): [number, number, number] {
  // Handle HSL
  if (color.startsWith('hsl(')) {
    const match = color.match(/hsl\((\d+),\s*(\d+)%,\s*(\d+)%\)/);
    if (match) {
      const h = parseInt(match[1]);
      const s = parseInt(match[2]);
      const l = parseInt(match[3]);
      return hslToRgb(h, s, l);
    }
  }

  // Handle hex
  if (color.startsWith('#')) {
    const hex = color.slice(1);
    const r = parseInt(hex.slice(0, 2), 16);
    const g = parseInt(hex.slice(2, 4), 16);
    const b = parseInt(hex.slice(4, 6), 16);
    return [r, g, b];
  }

  // Handle rgb()
  if (color.startsWith('rgb(')) {
    const match = color.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
    if (match) {
      return [parseInt(match[1]), parseInt(match[2]), parseInt(match[3])];
    }
  }

  // Default to red
  return [255, 0, 0];
}

/**
 * Draw mask contours on canvas
 */
export function drawMaskContour(
  ctx: CanvasRenderingContext2D,
  mask: RLEMask,
  imgWidth: number,
  imgHeight: number,
  color: string,
  lineWidth: number = 2
): void {
  const decodedMask = decodeRLE(mask, imgWidth, imgHeight);
  const imageData = new ImageData(imgWidth, imgHeight);
  
  // Create binary image for contour detection
  for (let i = 0; i < decodedMask.length; i++) {
    const idx = i * 4;
    const value = decodedMask[i] > 0 ? 255 : 0;
    imageData.data[idx] = value;
    imageData.data[idx + 1] = value;
    imageData.data[idx + 2] = value;
    imageData.data[idx + 3] = 255;
  }

  // Draw contours using a simple edge detection
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;

  for (let y = 1; y < imgHeight - 1; y++) {
    for (let x = 1; x < imgWidth - 1; x++) {
      const idx = y * imgWidth + x;
      const current = decodedMask[idx];
      const right = decodedMask[y * imgWidth + (x + 1)];
      const bottom = decodedMask[(y + 1) * imgWidth + x];

      // Draw edge pixels
      if (current !== right || current !== bottom) {
        ctx.fillStyle = color;
        ctx.fillRect(x, y, 1, 1);
      }
    }
  }
}

