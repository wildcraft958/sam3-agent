import { useState } from 'react';
import { LLMConfig } from '../utils/api';

interface LLMConfigFormProps {
  onConfigChange: (config: LLMConfig) => void;
  initialConfig?: Partial<LLMConfig>;
}

export default function LLMConfigForm({ onConfigChange, initialConfig }: LLMConfigFormProps) {
  const [config, setConfig] = useState<LLMConfig>({
    base_url: initialConfig?.base_url || 'https://api.openai.com/v1',
    model: initialConfig?.model || 'gpt-4o',
    api_key: initialConfig?.api_key || '',
    name: initialConfig?.name || '',
    max_tokens: initialConfig?.max_tokens || 2048,
  });

  const [showAdvanced, setShowAdvanced] = useState(false);

  const handleChange = (field: keyof LLMConfig, value: string | number) => {
    const newConfig = { ...config, [field]: value };
    setConfig(newConfig);
    onConfigChange(newConfig);
  };

  return (
    <div className="llm-config-form">
      <h3>LLM Configuration</h3>
      
      <div className="form-group">
        <label htmlFor="base_url">Base URL</label>
        <input
          id="base_url"
          type="text"
          value={config.base_url}
          onChange={(e) => handleChange('base_url', e.target.value)}
          placeholder="https://api.openai.com/v1"
        />
      </div>

      <div className="form-group">
        <label htmlFor="model">Model</label>
        <input
          id="model"
          type="text"
          value={config.model}
          onChange={(e) => handleChange('model', e.target.value)}
          placeholder="gpt-4o"
        />
      </div>

      <div className="form-group">
        <label htmlFor="api_key">API Key</label>
        <input
          id="api_key"
          type="password"
          value={config.api_key}
          onChange={(e) => handleChange('api_key', e.target.value)}
          placeholder="sk-..."
        />
      </div>

      <button
        type="button"
        className="toggle-advanced"
        onClick={() => setShowAdvanced(!showAdvanced)}
      >
        {showAdvanced ? 'Hide' : 'Show'} Advanced Options
      </button>

      {showAdvanced && (
        <>
          <div className="form-group">
            <label htmlFor="name">Name (Optional)</label>
            <input
              id="name"
              type="text"
              value={config.name || ''}
              onChange={(e) => handleChange('name', e.target.value)}
              placeholder="openai-gpt4o"
            />
          </div>

          <div className="form-group">
            <label htmlFor="max_tokens">Max Tokens</label>
            <input
              id="max_tokens"
              type="number"
              value={config.max_tokens ?? 2048}
              onChange={(e) => {
                const value = e.target.value;
                if (value === '') {
                  // Allow empty input temporarily
                  handleChange('max_tokens', undefined as any);
                } else {
                  const numValue = parseInt(value, 10);
                  if (!isNaN(numValue) && numValue > 0) {
                    handleChange('max_tokens', numValue);
                  }
                }
              }}
              onBlur={(e) => {
                // On blur, if empty or invalid, set to default
                const value = e.target.value;
                if (value === '' || isNaN(parseInt(value, 10)) || parseInt(value, 10) <= 0) {
                  handleChange('max_tokens', 2048);
                }
              }}
              min="1"
              max="32768"
            />
          </div>
        </>
      )}
    </div>
  );
}

