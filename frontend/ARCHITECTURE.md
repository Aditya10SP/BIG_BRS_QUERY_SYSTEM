# Frontend Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Browser                             │
│                                                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    Frontend Application                    │  │
│  │                                                             │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │  │
│  │  │  index.html  │  │  styles.css  │  │    app.js    │   │  │
│  │  │              │  │              │  │              │   │  │
│  │  │  Structure   │  │   Styling    │  │    Logic     │   │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘   │  │
│  │                                                             │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                    │
│                              │ HTTP/REST                          │
│                              ▼                                    │
└─────────────────────────────────────────────────────────────────┘
                               │
                               │
┌──────────────────────────────┼────────────────────────────────┐
│                              │                                  │
│                    Backend API Server                           │
│                    (http://localhost:8000)                      │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    API Endpoints                          │  │
│  │                                                            │  │
│  │  POST /ingest          Upload documents                   │  │
│  │  GET  /ingest/{id}     Check processing status            │  │
│  │  POST /query           Submit queries                     │  │
│  │                                                            │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Query & Ingestion Pipelines                  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
└──────────────────────────────┼────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Storage Layer                             │
│                                                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Qdrant   │  │  Neo4j   │  │PostgreSQL│  │  Ollama  │       │
│  │ (Vector) │  │ (Graph)  │  │  (Docs)  │  │  (LLM)   │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Component Architecture

### Frontend Layer

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend Components                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Upload Section                          │   │
│  │  • Drag & Drop Area                                  │   │
│  │  • File List with Status                             │   │
│  │  • Progress Tracking                                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                    │
│                          ▼                                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Query Section                           │   │
│  │  • Mode Selector (Auto/Vector/Graph/Hybrid)         │   │
│  │  • Query Input Textarea                              │   │
│  │  • Submit Button                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                    │
│                          ▼                                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Response Section                        │   │
│  │  • Answer Display                                    │   │
│  │  • Citations List                                    │   │
│  │  • Metrics Dashboard                                 │   │
│  │  • Error Messages                                    │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                    │
│                          ▼                                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Status Dashboard                        │   │
│  │  • Total Documents                                   │   │
│  │  • Processing Count                                  │   │
│  │  • Completed Count                                   │   │
│  │  • Failed Count                                      │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

### Upload Flow

```
User Action                Frontend                Backend
    │                         │                       │
    │  Select Files           │                       │
    ├────────────────────────>│                       │
    │                         │                       │
    │                         │  POST /ingest         │
    │                         ├──────────────────────>│
    │                         │                       │
    │                         │  { job_id, status }   │
    │                         │<──────────────────────┤
    │                         │                       │
    │                         │  Poll Status          │
    │                         │  GET /ingest/{id}     │
    │                         ├──────────────────────>│
    │                         │                       │
    │                         │  { status: ... }      │
    │                         │<──────────────────────┤
    │                         │                       │
    │  Status Update          │                       │
    │<────────────────────────┤                       │
    │  (Processing/Complete)  │                       │
    │                         │                       │
```

### Query Flow

```
User Action                Frontend                Backend
    │                         │                       │
    │  Enter Query            │                       │
    ├────────────────────────>│                       │
    │                         │                       │
    │  Select Mode            │                       │
    ├────────────────────────>│                       │
    │                         │                       │
    │  Click Submit           │                       │
    ├────────────────────────>│                       │
    │                         │                       │
    │                         │  POST /query          │
    │                         │  { query_text, mode } │
    │                         ├──────────────────────>│
    │                         │                       │
    │                         │  Processing...        │
    │                         │  (5-40 seconds)       │
    │                         │                       │
    │                         │  { answer, citations, │
    │                         │    metrics, ... }     │
    │                         │<──────────────────────┤
    │                         │                       │
    │  Display Response       │                       │
    │<────────────────────────┤                       │
    │  (Answer + Citations)   │                       │
    │                         │                       │
```

## State Management

```
┌─────────────────────────────────────────────────────────────┐
│                    Application State                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  uploadedFiles: Array<FileObject>                           │
│  ├─ id: string                                               │
│  ├─ file: File                                               │
│  ├─ name: string                                             │
│  ├─ size: string                                             │
│  ├─ status: 'pending' | 'processing' | 'completed' | 'failed'│
│  └─ jobId: string | null                                     │
│                                                               │
│  jobStatuses: Map<jobId, status>                            │
│  └─ Tracks processing status for each upload                │
│                                                               │
│  currentQuery: string                                        │
│  └─ Current query text                                       │
│                                                               │
│  selectedMode: 'AUTO' | 'VECTOR' | 'GRAPH' | 'HYBRID'       │
│  └─ Selected query mode                                      │
│                                                               │
│  lastResponse: QueryResponse | null                         │
│  └─ Most recent query response                               │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Event Handling

```
┌─────────────────────────────────────────────────────────────┐
│                      Event Handlers                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Upload Events:                                              │
│  • click (uploadArea) ──────> Open file picker              │
│  • dragover (uploadArea) ───> Show drag indicator           │
│  • drop (uploadArea) ────────> Process dropped files        │
│  • change (fileInput) ───────> Process selected files       │
│                                                               │
│  Query Events:                                               │
│  • click (queryBtn) ─────────> Submit query                 │
│  • keydown (queryInput) ─────> Ctrl+Enter to submit         │
│  • change (modeRadio) ───────> Update selected mode         │
│                                                               │
│  File Management:                                            │
│  • click (removeBtn) ────────> Remove file from list        │
│                                                               │
│  Status Polling:                                             │
│  • setInterval ──────────────> Poll job status every 5s     │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## API Integration

```
┌─────────────────────────────────────────────────────────────┐
│                    API Communication                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  POST /ingest                                                │
│  ├─ Request: FormData { file: File }                        │
│  ├─ Response: { job_id: string, status: string }            │
│  └─ Purpose: Upload document for processing                 │
│                                                               │
│  GET /ingest/{job_id}                                        │
│  ├─ Request: Path parameter (job_id)                        │
│  ├─ Response: { status: string, message: string }           │
│  └─ Purpose: Check document processing status               │
│                                                               │
│  POST /query                                                 │
│  ├─ Request: {                                               │
│  │    query_text: string,                                   │
│  │    mode?: string,                                        │
│  │    top_k?: number,                                       │
│  │    max_depth?: number                                    │
│  │  }                                                        │
│  ├─ Response: {                                              │
│  │    answer: string,                                       │
│  │    citations: object,                                    │
│  │    faithfulness_score: number,                           │
│  │    retrieval_mode: string,                               │
│  │    metrics: object,                                      │
│  │    warnings: string[]                                    │
│  │  }                                                        │
│  └─ Purpose: Submit query and get AI-generated answer       │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## File Structure

```
frontend/
│
├── index.html              # Main HTML structure
│   ├── Header Section
│   ├── Upload Section
│   ├── Query Section
│   ├── Response Section
│   ├── Status Dashboard
│   └── Footer
│
├── styles.css              # Complete styling
│   ├── CSS Variables
│   ├── Base Styles
│   ├── Component Styles
│   ├── Responsive Design
│   └── Animations
│
├── app.js                  # Application logic
│   ├── Configuration
│   ├── State Management
│   ├── Event Handlers
│   ├── API Integration
│   ├── UI Updates
│   └── Utility Functions
│
├── server.py               # Development server
│   ├── HTTP Server
│   ├── CORS Support
│   └── Static File Serving
│
└── Documentation
    ├── README.md           # Complete guide
    ├── FEATURES.md         # Feature descriptions
    ├── ARCHITECTURE.md     # This file
    └── demo.html           # Demo page
```

## Technology Stack

```
┌─────────────────────────────────────────────────────────────┐
│                    Technology Layers                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Presentation Layer:                                         │
│  ├─ HTML5 (Semantic markup)                                 │
│  ├─ CSS3 (Grid, Flexbox, Variables)                         │
│  └─ Vanilla JavaScript (ES6+)                               │
│                                                               │
│  Communication Layer:                                        │
│  ├─ Fetch API (HTTP requests)                               │
│  ├─ FormData (File uploads)                                 │
│  └─ JSON (Data exchange)                                    │
│                                                               │
│  Server Layer:                                               │
│  ├─ Python HTTP Server                                      │
│  ├─ CORS Middleware                                         │
│  └─ Static File Serving                                     │
│                                                               │
│  Backend Integration:                                        │
│  ├─ FastAPI (REST API)                                      │
│  ├─ Ollama (LLM)                                            │
│  ├─ Qdrant (Vector DB)                                      │
│  ├─ Neo4j (Graph DB)                                        │
│  └─ PostgreSQL (Document DB)                                │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Responsive Design

```
┌─────────────────────────────────────────────────────────────┐
│                    Breakpoint Strategy                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Desktop (> 768px):                                          │
│  ├─ Multi-column layouts                                    │
│  ├─ Side-by-side components                                 │
│  └─ Full-width sections                                     │
│                                                               │
│  Tablet (768px):                                             │
│  ├─ Adjusted grid columns                                   │
│  ├─ Stacked mode selectors                                  │
│  └─ Responsive padding                                      │
│                                                               │
│  Mobile (< 768px):                                           │
│  ├─ Single column layout                                    │
│  ├─ Stacked components                                      │
│  ├─ Touch-friendly targets                                  │
│  └─ Reduced padding                                         │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Performance Optimization

```
┌─────────────────────────────────────────────────────────────┐
│                  Performance Strategies                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Loading:                                                    │
│  ├─ No external dependencies                                │
│  ├─ Minimal CSS/JS files                                    │
│  └─ System fonts (no web fonts)                             │
│                                                               │
│  Runtime:                                                    │
│  ├─ Efficient DOM updates                                   │
│  ├─ Debounced event handlers                                │
│  ├─ Smart polling intervals                                 │
│  └─ Lazy rendering                                          │
│                                                               │
│  Network:                                                    │
│  ├─ Batch API requests                                      │
│  ├─ Cached responses                                        │
│  └─ Optimized polling                                       │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Security Considerations

```
┌─────────────────────────────────────────────────────────────┐
│                    Security Layers                           │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Frontend:                                                   │
│  ├─ Input validation                                        │
│  ├─ File type checking                                      │
│  ├─ XSS prevention (textContent)                            │
│  └─ HTTPS (production)                                      │
│                                                               │
│  Backend:                                                    │
│  ├─ CORS configuration                                      │
│  ├─ Authentication (TODO)                                   │
│  ├─ Rate limiting (TODO)                                    │
│  └─ Input sanitization                                      │
│                                                               │
│  Data:                                                       │
│  ├─ Secure transmission                                     │
│  ├─ No sensitive data in URLs                               │
│  └─ Proper error messages                                   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

**Version**: 1.0.0  
**Last Updated**: March 6, 2026  
**Status**: Production Ready
