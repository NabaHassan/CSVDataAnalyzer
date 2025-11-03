class CSVAnalyzer {
    constructor() {
        this.selectedFile = null;
        this.isLoading = false;

        this.initializeEventListeners();
    }

    initializeEventListeners() {
        // File upload elements
        this.fileUploadArea = document.getElementById('fileUploadArea');
        this.fileInput = document.getElementById('fileInput');
        this.uploadContent = document.getElementById('uploadContent');
        this.fileInfo = document.getElementById('fileInfo');
        this.fileName = document.getElementById('fileName');
        this.fileDetails = document.getElementById('fileDetails');
        this.removeFileBtn = document.getElementById('removeFileBtn');

        // Query elements
        this.queryForm = document.getElementById('queryForm');
        this.queryInput = document.getElementById('queryInput');
        this.submitButton = document.getElementById('submitButton');

        // Results elements
        this.resultsContent = document.getElementById('resultsContent');

        // Event listeners
        this.setupFileUpload();
        this.setupQueryForm();
    }

    setupFileUpload() {
        // Click to upload
        this.fileUploadArea.addEventListener('click', () => {
            this.fileInput.click();
        });

        // Drag and drop
        this.fileUploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.fileUploadArea.classList.add('drag-over');
        });

        this.fileUploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            this.fileUploadArea.classList.remove('drag-over');
        });

        this.fileUploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            this.fileUploadArea.classList.remove('drag-over');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFileSelect(files[0]);
            }
        });

        // File input change
        this.fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFileSelect(e.target.files[0]);
            }
        });

        // Remove file
        this.removeFileBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            this.removeFile();
        });
    }

    setupQueryForm() {
        this.queryForm.addEventListener('submit', (e) => {
            e.preventDefault();
            if (this.selectedFile && this.queryInput.value.trim()) {
                this.handleQuerySubmit(this.queryInput.value.trim());
            }
        });
    }

    handleFileSelect(file) {
        if (file.type !== 'text/csv' && !file.name.endsWith('.csv')) {
            alert('Please upload a CSV file');
            return;
        }

        this.selectedFile = file;

        // Update UI
        this.uploadContent.style.display = 'none';
        this.fileInfo.style.display = 'block';
        this.fileName.textContent = `📊 ${file.name}`;
        this.fileDetails.textContent = `Size: ${(file.size / 1024).toFixed(2)} KB`;

        // Enable query input
        this.queryInput.disabled = false;
        this.queryInput.placeholder = "Ask a question about your CSV data... (e.g., 'What is the average sales by region?')";

        this.updateSubmitButton();

        console.log('File selected:', file.name);
    }

    removeFile() {
        this.selectedFile = null;
        this.fileInput.value = '';

        // Update UI
        this.uploadContent.style.display = 'block';
        this.fileInfo.style.display = 'none';

        // Disable query input
        this.queryInput.disabled = true;
        this.queryInput.value = '';
        this.queryInput.placeholder = "Please upload a CSV file first to ask questions";

        this.updateSubmitButton();
        this.showPlaceholderResults();
    }

    updateSubmitButton() {
        const hasFile = this.selectedFile !== null;
        const hasQuery = this.queryInput.value.trim() !== '';
        this.submitButton.disabled = !hasFile || !hasQuery || this.isLoading;
    }

    async handleQuerySubmit(query) {
        if (!this.selectedFile || !query) return;

        this.isLoading = true;
        this.updateSubmitButton();
        this.showLoadingResults();

        try {
            const formData = new FormData();
            formData.append('csv_file', this.selectedFile);
            formData.append('user_query', query);

            const response = await fetch('http://localhost:8000/analyze', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            const data = await response.json();
            this.showResults(data);

        } catch (error) {
            console.error('Analysis error:', error);
            this.showError(error.message);
        } finally {
            this.isLoading = false;
            this.updateSubmitButton();
        }
    }

    showLoadingResults() {
        this.resultsContent.innerHTML = `
            <div class="loading-spinner">
                <div class="spinner"></div>
                <div class="step-indicator">
                    <span class="step-badge">Processing</span>
                    <span>Analyzing your data with AI... This may take a minute.</span>
                </div>
            </div>
        `;
        this.resultsContent.className = 'response-card';
    }

   showResults(data) {
    const response = data.formatted_response || data.llm_response || 'No response generated';

    // 🧹 No context UI
    this.resultsContent.innerHTML = `
        <div class="response-text">${response}</div>
    `;
    this.resultsContent.className = 'response-card';
}


    showError(errorMessage) {
        this.resultsContent.innerHTML = `
            <div style="display: flex; align-items: flex-start; gap: 0.5rem;">
                <i class="fas fa-exclamation-circle"></i>
                <div>
                    <strong>Error:</strong>
                    <div class="response-text">${errorMessage}</div>
                </div>
            </div>
        `;
        this.resultsContent.className = 'response-card error-card';
    }

    showPlaceholderResults() {
        this.resultsContent.innerHTML = `
            <div class="placeholder-text">
                <i class="fas fa-check-circle"></i>
                Upload a CSV file and ask a question to see results
            </div>
        `;
        this.resultsContent.className = 'response-card';
    }
}

// Add input event listener for query field
document.addEventListener('DOMContentLoaded', () => {
    const analyzer = new CSVAnalyzer();

    // Enable real-time submit button updates
    document.getElementById('queryInput').addEventListener('input', function() {
        analyzer.updateSubmitButton();
    });
});