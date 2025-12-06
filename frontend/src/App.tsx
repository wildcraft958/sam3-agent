import { useState, useEffect, useRef } from 'react';
import ImageUpload from './components/ImageUpload';
import LLMConfigForm from './components/LLMConfigForm';
import SAM3ConfigForm, { SAM3Config } from './components/SAM3ConfigForm';
import ImageVisualization from './components/ImageVisualization';
import ResultsPanel from './components/ResultsPanel';
import CommunicationLog from './components/CommunicationLog';
import { segmentImage, countImage, checkHealth, LLMConfig, SegmentResponse } from './utils/api';

function App() {
  const [imageBase64, setImageBase64] = useState<string | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [llmConfig, setLlmConfig] = useState<LLMConfig>({
    base_url: 'https://srinjoy59--qwen3-vl-vllm-server-30b-vllm-server.modal.run/v1',
    model: 'Qwen/Qwen3-VL-30B-A3B-Instruct',
    api_key: '',
    name: 'qwen3-vl-30b-modal',
    max_tokens: 2048,
  });
  const [prompt, setPrompt] = useState<string>('segment all objects');
  const [response, setResponse] = useState<SegmentResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [useInfer, setUseInfer] = useState<boolean>(false);
  const [sam3Config, setSam3Config] = useState<SAM3Config>({
    confidence_threshold: 0.4,
  });
  const [apiHealth, setApiHealth] = useState<'checking' | 'healthy' | 'unhealthy' | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const handleImageSelect = (base64: string, file: File) => {
    setImageBase64(base64);
    setImageFile(file);
    const url = URL.createObjectURL(file);
    setImageUrl(url);
    setResponse(null);
    setError(null);
  };

  const handleConfigChange = (config: LLMConfig) => {
    setLlmConfig(config);
  };

  const handleSAM3ConfigChange = (config: SAM3Config) => {
    setSam3Config(config);
  };

  const handleStop = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
      setLoading(false);
      setError('Request cancelled by user');
      setResponse({
        status: 'error',
        message: 'Request was cancelled by user',
      });
    }
  };

  // Check API health on mount
  useEffect(() => {
    const checkApiHealth = async () => {
      setApiHealth('checking');
      try {
        const health = await checkHealth();
        setApiHealth(health.status === 'ok' ? 'healthy' : 'unhealthy');
      } catch {
        setApiHealth('unhealthy');
      }
    };
    checkApiHealth();
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  const handleSegment = async () => {
    if (!imageBase64) {
      setError('Please upload an image first');
      return;
    }

    // Only require API key for OpenAI and similar providers (not Modal vLLM)
    const isModalVLLM = llmConfig.base_url.includes('modal.run');
    if (!useInfer && !isModalVLLM && !llmConfig.api_key) {
      setError('Please provide an API key');
      return;
    }

    // Cancel any existing request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    // Create new AbortController for this request
    abortControllerRef.current = new AbortController();
    const signal = abortControllerRef.current.signal;

    setLoading(true);
    setError(null);
    setResponse(null);

    try {
      if (useInfer) {
        // Use SAM3 counting endpoint (still requires llm_config for VLM verification)
        // Convert base64 to data URI format
        const imageUrl = imageBase64 ? `data:image/jpeg;base64,${imageBase64}` : undefined;
        const countResponse = await countImage({
          prompt: prompt,
          image_url: imageUrl,
          llm_config: llmConfig,
          confidence_threshold: sam3Config.confidence_threshold,
        }, signal);

        // Check if request was cancelled
        if (signal.aborted) {
          return;
        }

        // Convert count response to segment response format for compatibility
        if (countResponse.status === 'success' && countResponse.detections) {
          const segmentResponse: SegmentResponse = {
            status: 'success',
            summary: `SAM3 found ${countResponse.count || 0} ${countResponse.object_type || 'objects'}`,
            regions: countResponse.detections?.map((det) => ({
              bbox: det.box,
              mask: det.mask_rle,
              score: det.score,
            })) || [],
            raw_sam3_json: {
              orig_img_h: countResponse.orig_img_h || 0,
              orig_img_w: countResponse.orig_img_w || 0,
              pred_boxes: countResponse.detections?.map(d => d.box) || [],
              pred_masks: countResponse.detections?.map(d => d.mask_rle) || [],
              pred_scores: countResponse.detections?.map(d => d.score) || [],
            },
          };
          setResponse(segmentResponse);
        } else {
          setResponse({
            status: 'error',
            message: countResponse.message || 'Counting failed',
          });
        }
      } else {
        // Use full agent with LLM
        // Convert base64 to data URI format
        const imageUrl = imageBase64 ? `data:image/jpeg;base64,${imageBase64}` : undefined;
        const segmentResponse = await segmentImage({
          prompt,
          image_url: imageUrl,
          llm_config: llmConfig,
          debug: true,
          confidence_threshold: sam3Config.confidence_threshold,
        }, signal);

        // Check if request was cancelled
        if (signal.aborted) {
          return;
        }

        setResponse(segmentResponse);
      }
    } catch (err) {
      // Don't set error if request was cancelled
      if (signal.aborted) {
        return;
      }
      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
      setError(errorMessage);
      setResponse({
        status: 'error',
        message: errorMessage,
      });
    } finally {
      // Only reset loading if not cancelled
      if (!signal.aborted) {
        setLoading(false);
        abortControllerRef.current = null;
      }
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h1>SAM3 Agent Visualization</h1>
            <p>Upload an image and visualize segmentation results with masks and bounding boxes</p>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            {apiHealth === 'checking' && (
              <span style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                Checking API...
              </span>
            )}
            {apiHealth === 'healthy' && (
              <span style={{ 
                fontSize: '0.875rem', 
                color: 'var(--success)',
                display: 'flex',
                alignItems: 'center',
                gap: '0.25rem'
              }}>
                <span style={{ 
                  width: '8px', 
                  height: '8px', 
                  borderRadius: '50%', 
                  backgroundColor: 'var(--success)',
                  display: 'inline-block'
                }}></span>
                API Connected
              </span>
            )}
            {apiHealth === 'unhealthy' && (
              <span style={{ 
                fontSize: '0.875rem', 
                color: 'var(--error)',
                display: 'flex',
                alignItems: 'center',
                gap: '0.25rem'
              }}>
                <span style={{ 
                  width: '8px', 
                  height: '8px', 
                  borderRadius: '50%', 
                  backgroundColor: 'var(--error)',
                  display: 'inline-block'
                }}></span>
                API Unavailable
              </span>
            )}
          </div>
        </div>
      </header>

      <div className="app-container">
        <div className="left-panel">
          <div className="panel-section">
            <h2>Image Upload</h2>
            <ImageUpload
              onImageSelect={handleImageSelect}
              currentImage={imageUrl || undefined}
            />
          </div>

          <div className="panel-section">
            <h2>Prompt</h2>
            <div className="form-group">
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Enter segmentation prompt (e.g., 'segment all objects')"
                rows={3}
              />
            </div>
          </div>

          <div className="panel-section">
            <div className="mode-toggle">
              <label>
                <input
                  type="checkbox"
                  checked={useInfer}
                  onChange={(e) => setUseInfer(e.target.checked)}
                />
                Use SAM3 Counting Endpoint
              </label>
              <p className="hint">Check to use SAM3 counting endpoint (optimized for counting tasks, still uses VLM for verification)</p>
            </div>
          </div>

          {!useInfer && (
            <div className="panel-section">
              <LLMConfigForm
                onConfigChange={handleConfigChange}
                initialConfig={llmConfig}
              />
            </div>
          )}

          <div className="panel-section">
            <SAM3ConfigForm
              onConfigChange={handleSAM3ConfigChange}
              initialConfig={sam3Config}
            />
          </div>

          <div className="panel-section">
            <div className="button-group">
              <button
                className="segment-button"
                onClick={handleSegment}
                disabled={loading || !imageBase64}
              >
                {loading ? 'Processing...' : 'Run Segmentation'}
              </button>
              {loading && (
                <button
                  className="stop-button"
                  onClick={handleStop}
                >
                  Stop
                </button>
              )}
            </div>
          </div>

          {error && (
            <div className="error-message">
              <strong>Error:</strong> {error}
            </div>
          )}
        </div>

        <div className="center-panel">
          <div className="panel-section">
            <h2>Visualization</h2>
            <ImageVisualization
              imageUrl={imageUrl}
              regions={response?.regions}
              rawData={response?.raw_sam3_json}
            />
          </div>
        </div>

        <div className="right-panel">
          <div className="panel-section">
            <ResultsPanel response={response} loading={loading} />
          </div>

          <div className="panel-section">
            <CommunicationLog response={response} />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;

