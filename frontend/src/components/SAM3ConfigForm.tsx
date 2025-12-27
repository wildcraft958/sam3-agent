import { useState, useEffect } from 'react';
import { SAM3Config, PyramidalConfig } from '../utils/api';

interface SAM3ConfigFormProps {
  onConfigChange: (config: SAM3Config) => void;
  initialConfig?: Partial<SAM3Config>;
}

export default function SAM3ConfigForm({ onConfigChange, initialConfig }: SAM3ConfigFormProps) {
  const [config, setConfig] = useState<SAM3Config>({
    confidence_threshold: initialConfig?.confidence_threshold ?? 0.4,
    max_retries: initialConfig?.max_retries ?? 2,
    include_obb: initialConfig?.include_obb ?? false,
    obb_as_polygon: initialConfig?.obb_as_polygon ?? false,
    pyramidal_config: initialConfig?.pyramidal_config ?? {
      tile_size: 512,
      overlap_ratio: 0.15,
      scales: [1.0],
      batch_size: 16,
      iou_threshold: 0.5
    }
  });

  const [showAdvanced, setShowAdvanced] = useState(false);
  const [scalesStr, setScalesStr] = useState(config.pyramidal_config?.scales?.join(', ') || '1.0');

  useEffect(() => {
    onConfigChange(config);
  }, [config]);

  const handleChange = (field: keyof SAM3Config, value: any) => {
    setConfig(prev => ({ ...prev, [field]: value }));
  };

  const handlePyramidalChange = (field: keyof PyramidalConfig, value: any) => {
    setConfig(prev => ({
      ...prev,
      pyramidal_config: {
        ...prev.pyramidal_config,
        [field]: value
      }
    }));
  };

  const handleScalesChange = (value: string) => {
    setScalesStr(value);
    const scales = value.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n));
    if (scales.length > 0) {
      handlePyramidalChange('scales', scales);
    }
  };

  return (
    <div className="sam3-config-form">
      <h3>SAM3 Configuration</h3>

      <div className="form-group">
        <label htmlFor="confidence_threshold">Confidence Threshold ({config.confidence_threshold})</label>
        <input
          id="confidence_threshold"
          type="range"
          step="0.05"
          min="0"
          max="1"
          value={config.confidence_threshold}
          onChange={(e) => handleChange('confidence_threshold', parseFloat(e.target.value))}
        />
      </div>

      <div className="form-group checkbox-group">
        <label>
          <input
            type="checkbox"
            checked={showAdvanced}
            onChange={(e) => setShowAdvanced(e.target.checked)}
          />
          Show Advanced Settings
        </label>
      </div>

      {showAdvanced && (
        <div className="advanced-settings">
          <h4>Pyramidal Inference (Batching)</h4>
          <div className="form-row">
            <div className="form-group half">
              <label>Batch Size</label>
              <input
                type="number"
                value={config.pyramidal_config?.batch_size}
                onChange={(e) => handlePyramidalChange('batch_size', parseInt(e.target.value) || 16)}
              />
            </div>
            <div className="form-group half">
              <label>Scales (comma sep)</label>
              <input
                type="text"
                value={scalesStr}
                onChange={(e) => handleScalesChange(e.target.value)}
                placeholder="1.0, 0.5"
              />
            </div>
          </div>

          <div className="form-row">
            <div className="form-group half">
              <label>Tile Size (px)</label>
              <input
                type="number"
                value={config.pyramidal_config?.tile_size}
                onChange={(e) => handlePyramidalChange('tile_size', parseInt(e.target.value) || 512)}
              />
            </div>
            <div className="form-group half">
              <label>Overlap Ratio</label>
              <input
                type="number"
                step="0.05"
                min="0"
                max="1"
                value={config.pyramidal_config?.overlap_ratio}
                onChange={(e) => handlePyramidalChange('overlap_ratio', parseFloat(e.target.value) || 0.15)}
              />
            </div>
          </div>

          <h4>Agent Options</h4>
          <div className="form-group">
            <label>Max Retries (Verification)</label>
            <input
              type="number"
              min="0"
              max="5"
              value={config.max_retries}
              onChange={(e) => handleChange('max_retries', parseInt(e.target.value))}
            />
          </div>

          <div className="form-group checkbox-group">
            <label>
              <input
                type="checkbox"
                checked={config.include_obb}
                onChange={(e) => handleChange('include_obb', e.target.checked)}
              />
              Include OBB (Oriented Bounding Boxes)
            </label>
          </div>

          {config.include_obb && (
            <div className="form-group checkbox-group indent">
              <label>
                <input
                  type="checkbox"
                  checked={config.obb_as_polygon}
                  onChange={(e) => handleChange('obb_as_polygon', e.target.checked)}
                />
                OBB as Polygon (Default: [cx, cy, w, h, angle])
              </label>
            </div>
          )}
        </div>
      )}

      <style>{`
        .form-row { display: flex; gap: 10px; }
        .half { flex: 1; }
        .checkbox-group label { display: flex; align-items: center; gap: 8px; cursor: pointer; }
        .advanced-settings { border-top: 1px solid #333; margin-top: 10px; padding-top: 10px; }
        .indent { margin-left: 20px; }
      `}</style>
    </div>
  );
}

