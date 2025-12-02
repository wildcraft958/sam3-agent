import { useState } from 'react';

export interface SAM3Config {
  confidence_threshold: number;
}

interface SAM3ConfigFormProps {
  onConfigChange: (config: SAM3Config) => void;
  initialConfig?: Partial<SAM3Config>;
}

export default function SAM3ConfigForm({ onConfigChange, initialConfig }: SAM3ConfigFormProps) {
  const [config, setConfig] = useState<SAM3Config>({
    confidence_threshold: initialConfig?.confidence_threshold ?? 0.4,
  });

  const handleChange = (field: keyof SAM3Config, value: number) => {
    const newConfig = { ...config, [field]: value };
    setConfig(newConfig);
    onConfigChange(newConfig);
  };

  return (
    <div className="sam3-config-form">
      <h3>SAM3 Configuration</h3>
      
      <div className="form-group">
        <label htmlFor="confidence_threshold">Confidence Threshold</label>
        <input
          id="confidence_threshold"
          type="number"
          step="0.01"
          min="0"
          max="1"
          value={config.confidence_threshold}
          onChange={(e) => {
            const value = parseFloat(e.target.value);
            if (!isNaN(value) && value >= 0 && value <= 1) {
              handleChange('confidence_threshold', value);
            }
          }}
          onBlur={(e) => {
            // On blur, validate and clamp value
            const value = parseFloat(e.target.value);
            if (isNaN(value) || value < 0) {
              handleChange('confidence_threshold', 0);
            } else if (value > 1) {
              handleChange('confidence_threshold', 1);
            }
          }}
        />
        <p className="hint">
          Confidence threshold (0.0-1.0). Lower values return more detections but may include false positives. 
          Default: 0.4
        </p>
      </div>
    </div>
  );
}

