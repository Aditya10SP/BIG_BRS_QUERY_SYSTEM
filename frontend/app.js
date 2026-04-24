// Configuration
const API_BASE_URL = 'http://localhost:8000';

// State
let uploadedFiles = [];
let jobStatuses = {};

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');
const fileList = document.getElementById('fileList');
const uploadProgress = document.getElementById('uploadProgress');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const queryInput = document.getElementById('queryInput');
const queryBtn = document.getElementById('queryBtn');
const queryBtnText = document.getElementById('queryBtnText');
const queryBtnLoader = document.getElementById('queryBtnLoader');
const responseArea = document.getElementById('responseArea');
const responseContent = document.getElementById('responseContent');
const modeBadge = document.getElementById('modeBadge');
const citations = document.getElementById('citations');
const citationsList = document.getElementById('citationsList');
const metrics = document.getElementById('metrics');
const errorMessage = document.getElementById('errorMessage');

// Status counters
const totalDocs = document.getElementById('totalDocs');
const processingDocs = document.getElementById('processingDocs');
const completedDocs = document.getElementById('completedDocs');
const failedDocs = document.getElementById('failedDocs');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    updateStatusCounters();
});

// Event Listeners
function setupEventListeners() {
    // Upload area click
    uploadArea.addEventListener('click', () => fileInput.click());
    browseBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    // File input change
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);

    // Query button
    queryBtn.addEventListener('click', handleQuery);

    // Enter key in query input
    queryInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && e.ctrlKey) {
            handleQuery();
        }
    });
}

// File Upload Handlers
function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    const files = Array.from(e.dataTransfer.files);
    processFiles(files);
}

function handleFileSelect(e) {
    const files = Array.from(e.target.files);
    processFiles(files);
    fileInput.value = ''; // Reset input
}

function processFiles(files) {
    // Filter valid files
    const validFiles = files.filter(file => {
        const ext = file.name.split('.').pop().toLowerCase();
        return ext === 'docx' || ext === 'pdf';
    });

    if (validFiles.length === 0) {
        showError('Please select valid .docx or .pdf files');
        return;
    }

    validFiles.forEach(file => {
        const fileId = generateId();
        const fileObj = {
            id: fileId,
            file: file,
            name: file.name,
            size: formatFileSize(file.size),
            status: 'pending',
            jobId: null
        };
        uploadedFiles.push(fileObj);
        addFileToList(fileObj);
    });

    updateStatusCounters();
    uploadFiles(validFiles);
}

function addFileToList(fileObj) {
    const fileItem = document.createElement('div');
    fileItem.className = 'file-item';
    fileItem.id = `file-${fileObj.id}`;
    fileItem.innerHTML = `
        <div class="file-info">
            <div class="file-icon">📄</div>
            <div class="file-details">
                <div class="file-name">${fileObj.name}</div>
                <div class="file-size">${fileObj.size}</div>
            </div>
        </div>
        <div class="file-status ${fileObj.status}" id="status-${fileObj.id}">
            <span>⏳</span>
            <span>Pending</span>
        </div>
        <div class="file-actions">
            <button class="btn-icon" onclick="removeFile('${fileObj.id}')" title="Remove">
                ❌
            </button>
        </div>
    `;
    fileList.appendChild(fileItem);
}

async function uploadFiles(files) {
    uploadProgress.style.display = 'block';
    let completed = 0;

    for (const file of files) {
        const fileObj = uploadedFiles.find(f => f.file === file);
        if (!fileObj) continue;

        try {
            updateFileStatus(fileObj.id, 'processing', '⏳', 'Uploading...');

            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch(`${API_BASE_URL}/api/v1/ingest`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Upload failed: ${response.statusText}`);
            }

            const data = await response.json();
            fileObj.jobId = data.job_id;
            fileObj.status = 'processing';

            updateFileStatus(fileObj.id, 'processing', '🔄', 'Processing...');

            // Start polling for status
            pollJobStatus(fileObj);

        } catch (error) {
            console.error('Upload error:', error);
            fileObj.status = 'failed';
            updateFileStatus(fileObj.id, 'failed', '❌', 'Failed');
        }

        completed++;
        const progress = (completed / files.length) * 100;
        progressFill.style.width = `${progress}%`;
        progressText.textContent = `Uploading ${completed} of ${files.length} files...`;
    }

    setTimeout(() => {
        uploadProgress.style.display = 'none';
        progressFill.style.width = '0%';
    }, 2000);

    updateStatusCounters();
}

async function pollJobStatus(fileObj) {
    const maxAttempts = 60; // 5 minutes max
    let attempts = 0;

    const poll = async () => {
        if (attempts >= maxAttempts) {
            fileObj.status = 'failed';
            updateFileStatus(fileObj.id, 'failed', '❌', 'Timeout');
            updateStatusCounters();
            return;
        }

        try {
            const response = await fetch(`${API_BASE_URL}/api/v1/ingest/${fileObj.jobId}`);
            const data = await response.json();

            if (data.status === 'completed') {
                fileObj.status = 'completed';
                updateFileStatus(fileObj.id, 'completed', '✅', 'Completed');
                updateStatusCounters();
            } else if (data.status === 'failed') {
                fileObj.status = 'failed';
                updateFileStatus(fileObj.id, 'failed', '❌', 'Failed');
                updateStatusCounters();
            } else {
                // Still processing, poll again
                attempts++;
                setTimeout(poll, 5000); // Poll every 5 seconds
            }
        } catch (error) {
            console.error('Status poll error:', error);
            attempts++;
            setTimeout(poll, 5000);
        }
    };

    poll();
}

function updateFileStatus(fileId, status, icon, text) {
    const statusEl = document.getElementById(`status-${fileId}`);
    if (statusEl) {
        statusEl.className = `file-status ${status}`;
        statusEl.innerHTML = `<span>${icon}</span><span>${text}</span>`;
    }
}

function removeFile(fileId) {
    uploadedFiles = uploadedFiles.filter(f => f.id !== fileId);
    const fileEl = document.getElementById(`file-${fileId}`);
    if (fileEl) {
        fileEl.remove();
    }
    updateStatusCounters();
}

// Query Handlers
async function handleQuery() {
    const query = queryInput.value.trim();
    if (!query) {
        showError('Please enter a question');
        return;
    }

    // Get selected mode
    const modeRadio = document.querySelector('input[name="mode"]:checked');
    const mode = modeRadio.value;

    // Hide previous results
    responseArea.style.display = 'none';
    errorMessage.style.display = 'none';

    // Show loading state
    queryBtn.disabled = true;
    queryBtnText.style.display = 'none';
    queryBtnLoader.style.display = 'inline-block';

    try {
        const requestBody = {
            query_text: query
        };

        // Only add mode if not AUTO
        if (mode !== 'AUTO') {
            requestBody.mode = mode;
        }

        const response = await fetch(`${API_BASE_URL}/api/v1/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Query failed');
        }

        const data = await response.json();
        displayResponse(data);

    } catch (error) {
        console.error('Query error:', error);
        showError(error.message || 'Failed to process query. Please try again.');
    } finally {
        queryBtn.disabled = false;
        queryBtnText.style.display = 'inline';
        queryBtnLoader.style.display = 'none';
    }
}

function displayResponse(data) {
    // Show response area
    responseArea.style.display = 'block';

    // Display answer
    responseContent.textContent = data.answer;

    // Display mode badge
    modeBadge.textContent = data.retrieval_mode || 'AUTO';

    // Display citations
    if (data.citations && Object.keys(data.citations).length > 0) {
        citations.style.display = 'block';
        citationsList.innerHTML = '';
        
        Object.entries(data.citations).forEach(([id, citation]) => {
            const citationEl = document.createElement('div');
            citationEl.className = 'citation-item';
            citationEl.textContent = `[${id}] ${citation.breadcrumbs || citation.doc_id}`;
            citationsList.appendChild(citationEl);
        });
    } else {
        citations.style.display = 'none';
    }

    // Display metrics
    if (data.metrics) {
        metrics.style.display = 'flex';
        document.getElementById('queryTime').textContent = 
            data.metrics.total_time ? `${data.metrics.total_time.toFixed(2)}s` : '-';
        document.getElementById('faithfulnessScore').textContent = 
            data.faithfulness_score ? `${(data.faithfulness_score * 100).toFixed(1)}%` : '-';
        document.getElementById('chunksRetrieved').textContent = 
            data.metrics.chunks_retrieved || '-';
    } else {
        metrics.style.display = 'none';
    }

    // Display warnings if any
    if (data.warnings && data.warnings.length > 0) {
        const warningEl = document.createElement('div');
        warningEl.className = 'error-message';
        warningEl.style.background = '#fef3c7';
        warningEl.style.borderColor = '#f59e0b';
        warningEl.style.color = '#92400e';
        warningEl.textContent = '⚠️ ' + data.warnings.join(', ');
        responseArea.appendChild(warningEl);
    }
}

function showError(message) {
    errorMessage.textContent = '❌ ' + message;
    errorMessage.style.display = 'block';
    setTimeout(() => {
        errorMessage.style.display = 'none';
    }, 5000);
}

// Status Counters
function updateStatusCounters() {
    const total = uploadedFiles.length;
    const processing = uploadedFiles.filter(f => f.status === 'processing').length;
    const completed = uploadedFiles.filter(f => f.status === 'completed').length;
    const failed = uploadedFiles.filter(f => f.status === 'failed').length;

    totalDocs.textContent = total;
    processingDocs.textContent = processing;
    completedDocs.textContent = completed;
    failedDocs.textContent = failed;
}

// Utility Functions
function generateId() {
    return Date.now().toString(36) + Math.random().toString(36).substring(2);
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

// Example queries for quick testing
const exampleQueries = [
    "What is NEFT?",
    "What systems does RTGS depend on?",
    "Compare NEFT and RTGS transaction limits",
    "What are the conflicts in payment rules?",
    "Show the payment authorization workflow"
];

// Add example query buttons (optional)
function addExampleQueries() {
    const examplesContainer = document.createElement('div');
    examplesContainer.className = 'example-queries';
    examplesContainer.innerHTML = '<p style="margin-bottom: 10px; color: var(--text-secondary); font-size: 0.9rem;">Try these examples:</p>';
    
    exampleQueries.forEach(query => {
        const btn = document.createElement('button');
        btn.className = 'btn-example';
        btn.textContent = query;
        btn.onclick = () => {
            queryInput.value = query;
            queryInput.focus();
        };
        examplesContainer.appendChild(btn);
    });
    
    document.querySelector('.query-input-container').insertBefore(
        examplesContainer,
        queryInput
    );
}

// Uncomment to enable example queries
// addExampleQueries();
