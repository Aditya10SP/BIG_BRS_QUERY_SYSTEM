# Graph RAG Query System - Frontend

A modern, user-friendly web interface for the Graph RAG Layer system that allows you to upload banking documents and query them using AI-powered semantic search and knowledge graph traversal.

## Features

### 📤 Document Upload
- **Drag & Drop**: Simply drag files into the upload area
- **Multi-file Support**: Upload multiple documents at once
- **Format Support**: Accepts .docx and .pdf files
- **Real-time Status**: Track upload and processing status for each document
- **Progress Tracking**: Visual progress indicators for uploads

### 💬 Intelligent Querying
- **Multiple Query Modes**:
  - **Auto**: Let AI choose the best retrieval mode
  - **Vector**: Semantic search for factual questions
  - **Graph**: Relationship and dependency queries
  - **Hybrid**: Combined approach for complex questions

- **Rich Responses**:
  - AI-generated answers with inline citations
  - Source document references
  - Faithfulness scores
  - Query execution metrics

### 📊 Document Management
- **Status Dashboard**: Real-time overview of all documents
- **Processing Tracking**: Monitor ingestion pipeline progress
- **Error Handling**: Clear error messages and retry options

## Quick Start

### Prerequisites

1. **Backend API Running**: Make sure the Graph RAG API is running on `http://localhost:8000`
   ```bash
   # From the project root
   python -m uvicorn src.main:app --reload --port 8000
   ```

2. **Required Services**: Ensure all backend services are running:
   - Ollama (LLM)
   - Qdrant (Vector Store)
   - Neo4j (Knowledge Graph)
   - PostgreSQL (Document Store)

### Starting the Frontend

#### Option 1: Using Python Server (Recommended)

```bash
# Navigate to frontend directory
cd frontend

# Make the server executable
chmod +x server.py

# Start the server
python server.py
```

The frontend will be available at: **http://localhost:3000**

#### Option 2: Using Any HTTP Server

```bash
# Using Python's built-in server
cd frontend
python -m http.server 3000

# Or using Node.js http-server
npx http-server -p 3000

# Or using PHP
php -S localhost:3000
```

## Usage Guide

### 1. Upload Documents

1. Click the upload area or drag files into it
2. Select one or more .docx or .pdf files
3. Wait for files to upload and process
4. Monitor status in the file list and status dashboard

**Supported File Types:**
- Microsoft Word (.docx)
- PDF (.pdf)

**Processing Steps:**
- ⏳ Pending: File queued for upload
- 🔄 Processing: Document being parsed, chunked, and indexed
- ✅ Completed: Ready for querying
- ❌ Failed: Error occurred (check logs)

### 2. Ask Questions

1. **Select Query Mode** (or leave on Auto):
   - **Auto**: System automatically chooses the best mode
   - **Vector**: For factual questions like "What is NEFT?"
   - **Graph**: For relationship questions like "What depends on RTGS?"
   - **Hybrid**: For complex questions requiring both approaches

2. **Enter Your Question**:
   - Type your question in the text area
   - Press Ctrl+Enter or click "Ask Question"

3. **Review the Answer**:
   - Read the AI-generated response
   - Check citations to verify sources
   - Review faithfulness score for accuracy
   - View query metrics for performance

### Example Questions

**Factual Questions (Vector Mode):**
- "What is NEFT?"
- "Define transaction limit"
- "Explain RTGS process"

**Relationship Questions (Graph Mode):**
- "What systems does NEFT depend on?"
- "Show all dependencies for RTGS"
- "What integrates with Core Banking System?"

**Comparison Questions (Hybrid Mode):**
- "Compare NEFT and RTGS transaction limits"
- "What are the differences between payment modes?"
- "Compare processing times across systems"

**Conflict Detection:**
- "Are there any conflicts in NEFT transaction limits?"
- "Show contradictions in payment rules"

**Workflow Questions:**
- "Describe the payment authorization workflow"
- "Show the process chain for fund transfers"

## Configuration

### API Endpoint

To change the backend API URL, edit `app.js`:

```javascript
// Configuration
const API_BASE_URL = 'http://localhost:8000';  // Change this
```

### Styling

Customize the look and feel by editing `styles.css`. The design uses CSS variables for easy theming:

```css
:root {
    --primary-color: #4f46e5;
    --primary-hover: #4338ca;
    --success-color: #10b981;
    --error-color: #ef4444;
    /* ... more variables */
}
```

## Architecture

### Frontend Components

```
frontend/
├── index.html      # Main HTML structure
├── styles.css      # Styling and layout
├── app.js          # Application logic
├── server.py       # Development server
└── README.md       # This file
```

### Key Features

1. **Responsive Design**: Works on desktop, tablet, and mobile
2. **Real-time Updates**: Polls backend for document processing status
3. **Error Handling**: Graceful error messages and recovery
4. **CORS Support**: Handles cross-origin requests
5. **Modern UI**: Clean, intuitive interface with smooth animations

## API Integration

The frontend communicates with the backend API using these endpoints:

### POST /ingest
Upload a document for processing
```javascript
FormData: { file: File }
Response: { job_id: string, status: string }
```

### GET /ingest/{job_id}
Check document processing status
```javascript
Response: { 
    status: "pending" | "processing" | "completed" | "failed",
    message: string 
}
```

### POST /query
Submit a query
```javascript
Request: { 
    query_text: string,
    mode?: "VECTOR" | "GRAPH" | "HYBRID",
    top_k?: number,
    max_depth?: number
}
Response: {
    answer: string,
    citations: object,
    faithfulness_score: number,
    retrieval_mode: string,
    metrics: object,
    warnings: string[]
}
```

## Troubleshooting

### "Failed to fetch" Error
- **Cause**: Backend API not running or wrong URL
- **Solution**: Start the backend API and verify it's accessible at `http://localhost:8000`

### Upload Fails
- **Cause**: Invalid file format or backend processing error
- **Solution**: 
  - Check file format (.docx or .pdf only)
  - Check backend logs for errors
  - Ensure all backend services are running

### Query Returns No Results
- **Cause**: No documents uploaded or documents still processing
- **Solution**: 
  - Wait for documents to finish processing (status = Completed)
  - Check that at least one document is successfully ingested

### Slow Query Response
- **Cause**: Complex query or large document corpus
- **Solution**: 
  - Use more specific queries
  - Try VECTOR mode for faster responses
  - Check backend performance metrics

## Browser Support

- Chrome/Edge: ✅ Fully supported
- Firefox: ✅ Fully supported
- Safari: ✅ Fully supported
- Mobile browsers: ✅ Responsive design

## Development

### Adding New Features

1. **New Query Mode**: Add to mode selector in `index.html` and handle in `app.js`
2. **Custom Styling**: Modify CSS variables in `styles.css`
3. **Additional Metrics**: Update `displayResponse()` function in `app.js`

### Testing

1. Start backend API
2. Start frontend server
3. Upload test documents
4. Try various query modes
5. Check browser console for errors

## Security Notes

⚠️ **Important**: This is a development frontend. For production:

1. **CORS**: Configure specific allowed origins instead of `*`
2. **Authentication**: Add user authentication and authorization
3. **File Validation**: Implement server-side file validation
4. **Rate Limiting**: Add rate limiting for API requests
5. **HTTPS**: Use HTTPS in production
6. **Input Sanitization**: Sanitize all user inputs

## Performance Tips

1. **Batch Uploads**: Upload multiple files at once for efficiency
2. **Query Optimization**: Use specific query modes when possible
3. **Browser Cache**: Leverage browser caching for static assets
4. **Minimize Polling**: Adjust polling interval based on expected processing time

## License

Part of the Graph RAG Layer project. See main project LICENSE for details.

## Support

For issues or questions:
1. Check backend API logs
2. Check browser console for errors
3. Review this README
4. Check the main project documentation

---

**Version**: 1.0.0  
**Last Updated**: March 6, 2026  
**Compatibility**: Graph RAG Layer API v0.1.0
