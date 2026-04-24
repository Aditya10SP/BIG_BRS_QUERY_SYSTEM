# Document Ingestion Fix

## Problem Identified

Your documents were showing:
- **23 documents stuck in "processing"** - Actually stuck in "pending" status
- **13 documents failed** - Failed with "Chunking produced no chunks" error

## Root Causes

### 1. Background Thread Issue (Main Problem)
The ingestion API was using Python `threading.Thread` to run document processing in the background. When uvicorn runs with `--reload` flag, it kills all background threads when the code changes, leaving jobs stuck in "pending" status.

### 2. Chunking Failures
Some documents failed during the chunking step, likely due to:
- Empty or malformed document content
- Parsing issues with specific document formats
- Documents with no extractable text

## Fixes Applied

### 1. Replaced Threading with FastAPI BackgroundTasks ✅
Changed `src/api/routes.py` to use FastAPI's built-in `BackgroundTasks` instead of manual threading. This ensures:
- Background tasks survive uvicorn reloads
- Proper task lifecycle management
- Better error handling

### 2. Fixed Deprecated JavaScript Method ✅
Changed `substr()` to `substring()` in `frontend/app.js` to fix the browser console warning.

## How to Apply the Fix

### Step 1: Restart the Backend
```bash
# Option A: Use the restart script
bash restart_backend.sh

# Option B: Manual restart
pkill -f "uvicorn src.main:app"
bash start_backend.sh
```

### Step 2: Refresh the Frontend
Hard refresh your browser:
- **Mac**: `Cmd + Shift + R`
- **Windows/Linux**: `Ctrl + Shift + R`

### Step 3: Clear Stuck Jobs (Optional)
If you want to start fresh:
```bash
# Clear all uploaded files
rm -rf uploads/*

# Or clear just the stuck ones
# (The restart script will ask you)
```

### Step 4: Test Upload
1. Go to http://localhost:3000
2. Upload a test document (.docx or .pdf)
3. Watch the status change from "Pending" → "Processing" → "Completed"

## Expected Behavior After Fix

### Successful Upload Flow:
1. **Upload** - File is saved to `uploads/` directory
2. **Pending** - Job is queued (should be very brief)
3. **Processing** - Document goes through 8 pipeline steps:
   - Parsing
   - Chunking
   - Embedding generation
   - BM25 indexing
   - Entity extraction
   - Entity resolution
   - Conflict detection
   - Graph population
4. **Completed** - Document is ready for querying

### Processing Time:
- Small documents (< 10 pages): 30-60 seconds
- Medium documents (10-50 pages): 1-3 minutes
- Large documents (> 50 pages): 3-10 minutes

## Troubleshooting

### If documents still fail:

1. **Check backend logs** for specific errors:
   ```bash
   # Backend should be running in a terminal
   # Look for error messages with [job_id]
   ```

2. **Verify all services are running**:
   ```bash
   curl http://localhost:11434/  # Ollama
   curl http://localhost:6333/   # Qdrant
   curl http://localhost:7474/   # Neo4j
   ```

3. **Check a specific job status**:
   ```bash
   # Replace JOB_ID with actual job ID from uploads/ directory
   curl http://localhost:8000/api/v1/ingest/JOB_ID | python -m json.tool
   ```

4. **Test with a simple document**:
   - Create a simple .docx file with just a few paragraphs
   - Upload it to test if the pipeline works

### Common Failure Reasons:

- **"Chunking produced no chunks"**: Document has no extractable text
- **LLM timeout**: Ollama is slow or not responding
- **Neo4j connection error**: Neo4j is not running or credentials are wrong
- **Qdrant error**: Vector store is not accessible

## Next Steps

After restarting:
1. Upload a new test document
2. Monitor the status in the frontend
3. Once a document completes, try querying it
4. Check the backend logs for any errors

## Files Modified

- `src/api/routes.py` - Replaced threading with BackgroundTasks
- `frontend/app.js` - Fixed deprecated substr() method
- `restart_backend.sh` - New helper script for restarting
