import { useState, useEffect, useRef } from 'react';
import ImageUpload from './components/ImageUpload';
import LLMConfigForm from './components/LLMConfigForm';
import SAM3ConfigForm, { SAM3Config } from './components/SAM3ConfigForm';
import ImageVisualization from './components/ImageVisualization';
import ResultsPanel from './components/ResultsPanel';
import CommunicationLog from './components/CommunicationLog';
import DiagnosticPage from './components/DiagnosticPage';
import { segmentImage, inferImage, LLMConfig, SegmentResponse } from './utils/api';

type ViewMode = 'main' | 'diagnostic';

function App() {
  const [viewMode, setViewMode] = useState<ViewMode>('main');
  const [imageBase64, setImageBase64] = useState<string | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [llmConfig, setLlmConfig] = useState<LLMConfig>({
    base_url: 'https://api.openai.com/v1',
    model: 'gpt-4o',
    api_key: '',
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

    if (!useInfer && !llmConfig.api_key) {
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
        // Use pure SAM3 inference (no LLM)
        const inferResponse = await inferImage({
          text_prompt: prompt,
          image_b64: imageBase64,
          confidence_threshold: sam3Config.confidence_threshold,
        }, signal);

        // Check if request was cancelled
        if (signal.aborted) {
          return;
        }

        // Convert infer response to segment response format for compatibility
        if (inferResponse.status === 'success' && inferResponse.orig_img_h && inferResponse.orig_img_w) {
          const segmentResponse: SegmentResponse = {
            status: 'success',
            summary: `SAM3 found ${inferResponse.pred_boxes?.length || 0} regions`,
            regions: inferResponse.pred_boxes?.map((box, idx) => ({
              bbox: box,
              mask: inferResponse.pred_masks?.[idx],
              score: inferResponse.pred_scores?.[idx],
            })) || [],
            raw_sam3_json: {
              orig_img_h: inferResponse.orig_img_h,
              orig_img_w: inferResponse.orig_img_w,
              pred_boxes: inferResponse.pred_boxes || [],
              pred_masks: inferResponse.pred_masks || [],
              pred_scores: inferResponse.pred_scores || [],
            },
          };
          setResponse(segmentResponse);
        } else {
          setResponse({
            status: 'error',
            message: inferResponse.message || 'Inference failed',
          });
        }
      } else {
        // Use full agent with LLM
        const segmentResponse = await segmentImage({
          prompt,
          image_b64: imageBase64,
          llm_config: llmConfig,
          debug: true,
          confidence_threshold: sam3Config.confidence_threshold,
        }, signal);

        // Check if request was cancelled
        if (signal.aborted) {
          return;
        }

        setResponse(segmentResponse);
        // Also set error state if response has error status
        if (segmentResponse.status === 'error') {
          setError(segmentResponse.message || 'Segmentation failed');
        }
      }
    } catch (err) {
      // Don't set error if request was cancelled
      if (signal.aborted) {
        return;
      }
      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
      console.error('[App] Error in handleSegment:', err);
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

  if (viewMode === 'diagnostic') {
    return (
      <div className="app">
        <header className="app-header">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <h1>SAM3 Agent Diagnostics</h1>
              <p>Diagnostic tools and system configuration</p>
            </div>
            <button
              className="segment-button"
              onClick={() => setViewMode('main')}
              style={{ marginTop: 0 }}
            >
              Back to Main
            </button>
          </div>
        </header>
        <DiagnosticPage />
      </div>
    );
  }

  return (
    <div className="app">
      <header className="app-header">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h1>SAM3 Agent Visualization</h1>
            <p>Upload an image and visualize segmentation results with masks and bounding boxes</p>
          </div>
          <button
            className="segment-button"
            onClick={() => setViewMode('diagnostic')}
            style={{ marginTop: 0 }}
          >
            Diagnostics
          </button>
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
                Use Pure SAM3 (No LLM)
              </label>
              <p className="hint">Check to use SAM3 inference only (faster, no LLM costs)</p>
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

          {(error || response?.status === 'error') && (
            <div className="error-message">
              <strong>Error:</strong>
              <div className="error-details">
                {error && <div className="error-text">{error}</div>}
                {response?.status === 'error' && response?.message && (
                  <div className="error-text">{response.message}</div>
                )}
                {response?.traceback && (
                  <details className="error-traceback">
                    <summary>Technical Details</summary>
                    <pre>{response.traceback}</pre>
                  </details>
                )}
              </div>
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

