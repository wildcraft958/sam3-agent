# SAM3 Agent Frontend

A React + Vite frontend application for visualizing SAM3 segmentation results with masks, bounding boxes, and internal communications.

## Features

- 🖼️ **Image Upload**: Drag-and-drop or click to upload images
- 🎨 **Visualization**: Canvas-based rendering of masks and bounding boxes
- 📊 **Results Display**: View segmentation results, regions, and scores
- 🔍 **Internal Data**: Explore raw SAM3 JSON data and communication logs
- ⚙️ **LLM Configuration**: Configure LLM settings (OpenAI, Anthropic, vLLM, etc.)
- 🚀 **Dual Mode**: Full agent mode (with LLM) or pure SAM3 inference mode

## Setup

### Prerequisites

- Node.js 18+ and npm

### Installation

```bash
cd frontend
npm install
```

### Development

```bash
npm run dev
```

The app will be available at `http://localhost:5173`

### Build

```bash
npm run build
```

Output will be in the `dist/` directory.

## Deployment to Vercel

### Option 1: Vercel CLI

1. Install Vercel CLI:
```bash
npm i -g vercel
```

2. Deploy:
```bash
cd frontend
vercel
```

### Option 2: Vercel Dashboard

1. Push your code to GitHub/GitLab/Bitbucket
2. Import the project in Vercel dashboard
3. Vercel will auto-detect Vite and configure build settings
4. The `vercel.json` file is already configured

### Environment Variables (Optional)

You can set these in Vercel dashboard:

- `VITE_API_ENDPOINT`: Custom API endpoint URL (defaults to Modal endpoint)
- `VITE_INFER_ENDPOINT`: Custom inference endpoint URL

## Usage

1. **Upload Image**: Click or drag an image into the upload area
2. **Enter Prompt**: Type your segmentation prompt (e.g., "segment all objects")
3. **Configure LLM** (if using full agent mode):
   - Base URL: API endpoint (e.g., `https://api.openai.com/v1`)
   - Model: Model name (e.g., `gpt-4o`)
   - API Key: Your API key
4. **Run Segmentation**: Click "Run Segmentation"
5. **View Results**: 
   - See visualization with masks and bboxes
   - Check results panel for summary and regions
   - Explore internal data in communication log

## Project Structure

```
frontend/
├── src/
│   ├── components/          # React components
│   │   ├── ImageUpload.tsx
│   │   ├── ImageVisualization.tsx
│   │   ├── ResultsPanel.tsx
│   │   ├── CommunicationLog.tsx
│   │   └── LLMConfigForm.tsx
│   ├── utils/              # Utility functions
│   │   ├── api.ts          # API client
│   │   ├── maskUtils.ts    # RLE decoding
│   │   └── visualization.ts # Canvas drawing
│   ├── App.tsx             # Main app component
│   ├── main.tsx            # Entry point
│   └── index.css           # Global styles
├── package.json
├── vite.config.ts
├── tsconfig.json
├── vercel.json             # Vercel configuration
└── index.html
```

## API Endpoints

The frontend connects to Modal endpoints:

- **Full Agent**: `https://aryan-don357--sam3-agent-sam3-segment.modal.run`
- **Pure SAM3**: `https://aryan-don357--sam3-agent-sam3-infer.modal.run`

These can be overridden with environment variables.

## Technologies

- **React 18**: UI framework
- **TypeScript**: Type safety
- **Vite**: Build tool and dev server
- **Axios**: HTTP client
- **Canvas API**: Image rendering and mask visualization

## License

Same as parent project.

