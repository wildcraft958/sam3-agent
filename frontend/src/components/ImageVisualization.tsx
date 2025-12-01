import { useEffect, useRef } from 'react';
import { drawMask, drawBbox, drawLabel, generateColors } from '../utils/visualization';
import { RLEMask } from '../utils/maskUtils';

interface Region {
  bbox?: number[];
  mask?: RLEMask;
  score?: number;
}

interface ImageVisualizationProps {
  imageUrl: string | null;
  regions?: Region[];
  rawData?: {
    orig_img_h: number;
    orig_img_w: number;
    pred_boxes: number[][];
    pred_masks: RLEMask[];
    pred_scores: number[];
  };
  showMasks?: boolean;
  showBboxes?: boolean;
  showLabels?: boolean;
}

export default function ImageVisualization({
  imageUrl,
  regions,
  rawData,
  showMasks = true,
  showBboxes = true,
  showLabels = true,
}: ImageVisualizationProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);

  useEffect(() => {
    if (!imageUrl || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = new Image();
    img.crossOrigin = 'anonymous';
    
    img.onload = () => {
      imageRef.current = img;
      canvas.width = img.width;
      canvas.height = img.height;

      // Draw base image
      ctx.drawImage(img, 0, 0);

      // Determine data source
      let masks: RLEMask[] = [];
      let boxes: number[][] = [];
      let scores: number[] = [];

      if (rawData) {
        masks = rawData.pred_masks || [];
        boxes = rawData.pred_boxes || [];
        scores = rawData.pred_scores || [];
      } else if (regions) {
        masks = regions.map(r => r.mask!).filter(m => m);
        boxes = regions.map(r => r.bbox!).filter(b => b);
        scores = regions.map(r => r.score || 0);
      }

      if (masks.length === 0 && boxes.length === 0) {
        return;
      }

      const colors = generateColors(Math.max(masks.length, boxes.length));

      // Draw masks
      if (showMasks) {
        masks.forEach((mask, idx) => {
          if (mask) {
            drawMask(ctx, mask, img.width, img.height, colors[idx], 0.3);
          }
        });
      }

      // Draw bounding boxes
      if (showBboxes) {
        boxes.forEach((box, idx) => {
          if (box) {
            drawBbox(ctx, box, img.width, img.height, colors[idx], 2);
          }
        });
      }

      // Draw labels
      if (showLabels) {
        boxes.forEach((box, idx) => {
          if (box) {
            const score = scores[idx];
            drawLabel(ctx, box, img.width, img.height, `Mask ${idx + 1}`, colors[idx], score);
          }
        });
      }
    };

    img.onerror = () => {
      console.error('Failed to load image');
    };

    img.src = imageUrl;
  }, [imageUrl, regions, rawData, showMasks, showBboxes, showLabels]);

  if (!imageUrl) {
    return (
      <div className="image-visualization-placeholder">
        <p>Upload an image to see visualization</p>
      </div>
    );
  }

  return (
    <div className="image-visualization-container">
      <canvas ref={canvasRef} className="visualization-canvas" />
    </div>
  );
}

