import { useEffect, useRef, useState } from 'react';
import { drawMasksBatch, drawBbox, drawLabel, generateColors } from '../utils/visualization';
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
  maxMasks?: number; // Limit number of masks to render for performance
}

export default function ImageVisualization({
  imageUrl,
  regions,
  rawData,
  showMasks = true,
  showBboxes = true,
  showLabels = true,
  maxMasks = 50, // Default limit to prevent performance issues
}: ImageVisualizationProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const [isRendering, setIsRendering] = useState(false);

  useEffect(() => {
    if (!imageUrl || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = new Image();
    img.crossOrigin = 'anonymous';
    
    img.onload = async () => {
      setIsRendering(true);
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
        setIsRendering(false);
        return;
      }

      // Sort by score (descending) and limit number of masks for performance
      if (masks.length > maxMasks) {
        console.warn(`Limiting masks from ${masks.length} to ${maxMasks} for performance`);
        const indices = scores
          .map((score, idx) => ({ score, idx }))
          .sort((a, b) => b.score - a.score)
          .slice(0, maxMasks)
          .map(item => item.idx);
        
        masks = indices.map(idx => masks[idx]);
        boxes = indices.map(idx => boxes[idx]);
        scores = indices.map(idx => scores[idx]);
      }

      const colors = generateColors(Math.max(masks.length, boxes.length));

      // Use requestAnimationFrame to keep UI responsive
      requestAnimationFrame(() => {
        try {
          // Draw masks using batch processing for better performance
          if (showMasks && masks.length > 0) {
            const validMasks = masks.filter(m => m);
            if (validMasks.length > 0) {
              drawMasksBatch(ctx, validMasks, img.width, img.height, colors, 0.3);
            }
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
                drawLabel(ctx, box, img.width, img.height, `${idx + 1}`, colors[idx], score);
              }
            });
          }
        } catch (error) {
          console.error('Error rendering masks:', error);
        } finally {
          setIsRendering(false);
        }
      });
    };

    img.onerror = () => {
      console.error('Failed to load image');
      setIsRendering(false);
    };

    img.src = imageUrl;
  }, [imageUrl, regions, rawData, showMasks, showBboxes, showLabels, maxMasks]);

  if (!imageUrl) {
    return (
      <div className="image-visualization-placeholder">
        <p>Upload an image to see visualization</p>
      </div>
    );
  }

  return (
    <div className="image-visualization-container">
      {isRendering && (
        <div className="rendering-overlay">
          <div className="spinner"></div>
          <p>Rendering masks...</p>
        </div>
      )}
      <canvas ref={canvasRef} className="visualization-canvas" />
    </div>
  );
}

