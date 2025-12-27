import { useState, useEffect, useRef } from 'react';
import ImageUpload from './components/ImageUpload';
import LLMConfigForm from './components/LLMConfigForm';
import SAM3ConfigForm from './components/SAM3ConfigForm';
import ImageVisualization from './components/ImageVisualization';
import ResultsPanel from './components/ResultsPanel';
import CommunicationLog from './components/CommunicationLog';
import DiagnosticPage from './components/DiagnosticPage';
import { segmentImage, countImage, SegmentResponse } from './utils/api';
import { useStore } from './store';

type ViewMode = 'main' | 'diagnostic';

function App() {
  const [viewMode, setViewMode] = useState<ViewMode>('main');

  // Global State
  const {
    imageBase64, imageUrl, prompt,
    llmConfig, sam3Config, useInfer,
    response, loading, loadingStage, error,
    setImage, setPrompt, setLlmConfig, setSam3Config,
    setUseInfer, setResponse, setLoading, setLoadingStage, setError
  } = useStore();

  const abortControllerRef = useRef<AbortController | null>(null);

  const handleImageSelect = (base64: string, file: File) => {
    const url = URL.createObjectURL(file);
    setImage(base64, file, url);
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
    setLoadingStage('starting');
    setError(null);
    setResponse(null);

    try {
      if (useInfer) {
        // Stage 1: Text encoding (implicit in backend)
        setLoadingStage('encoding_text');

        // Use pure SAM3 counting (no LLM)
        const countResponse = await countImage({
          prompt: prompt,
          image_url: `data:image/jpeg;base64,${imageBase64}`,
          llm_config: llmConfig,
          ...sam3Config // Pass all config including pyramidal_config, include_obb, etc.
        }, signal);

        // Stage 2: Finalizing
        setLoadingStage('finalizing');

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
          const errMsg = countResponse.message || 'Counting failed';
          setError(errMsg);
          setResponse({
            status: 'error',
            message: errMsg,
          });
        }
      } else {
        // Stage 1: Text encoding + Processing
        setLoadingStage('encoding_text');

        // Use full agent with LLM
        const segmentResponse = await segmentImage({
          prompt,
          image_url: `data:image/jpeg;base64,${imageBase64}`,
          llm_config: llmConfig,
          debug: true,
          ...sam3Config // Pass full config
        }, signal);

        // Stage 2: Finalizing
        setLoadingStage('finalizing');

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
        setLoadingStage(null);
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
                Use Pure SAM3 Counting (No LLM)
              </label>
              <p className="hint">Check to use SAM3 counting only (faster, no LLM costs)</p>
            </div>
          </div>

          {!useInfer && (
            <div className="panel-section">
              <LLMConfigForm
                onConfigChange={setLlmConfig}
                initialConfig={llmConfig}
              />
            </div>
          )}

          <div className="panel-section">
            <SAM3ConfigForm
              onConfigChange={setSam3Config}
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
            <ResultsPanel response={response} loading={loading} loadingStage={loadingStage || undefined} />
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
