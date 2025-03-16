/**
 * Optimized file uploader with batch processing and memory management
 */

class OptimizedUploader {
    constructor(options = {}) {
        // Default options
        this.options = {
            maxFilesPerBatch: 5,
            maxRetries: 3,
            retryDelay: 1000,
            timeout: 30000,
            onProgress: null,
            onComplete: null,
            onError: null,
            ...options
        };
        
        // Internal state
        this.queue = [];
        this.currentBatch = 0;
        this.totalBatches = 0;
        this.uploadedFiles = [];
        this.isUploading = false;
        this.abortController = null;
    }
    
    /**
     * Add files to the upload queue
     * @param {FileList|Array} files - Files to upload
     * @param {string} endpoint - Upload endpoint
     * @param {Object} formData - Additional form data
     */
    addToQueue(files, endpoint, formData = {}) {
        // Convert FileList to Array for easier handling
        const filesArray = Array.from(files);
        
        if (!filesArray.length) {
            return false;
        }
        
        this.queue.push({
            files: filesArray,
            endpoint: endpoint,
            formData: formData,
            retries: 0
        });
        
        // Start processing if not already uploading
        if (!this.isUploading) {
            this.processQueue();
        }
        
        return true;
    }
    
    /**
     * Process the upload queue
     */
    async processQueue() {
        if (!this.queue.length || this.isUploading) {
            return;
        }
        
        this.isUploading = true;
        
        // Get the next item from the queue
        const item = this.queue.shift();
        const { files, endpoint, formData, retries } = item;
        
        // Calculate batches
        const maxFilesPerBatch = this.options.maxFilesPerBatch;
        this.totalBatches = Math.ceil(files.length / maxFilesPerBatch);
        this.currentBatch = 0;
        this.uploadedFiles = [];
        
        try {
            // Process all batches
            for (let i = 0; i < this.totalBatches; i++) {
                this.currentBatch = i;
                
                // Calculate batch boundaries
                const startIdx = i * maxFilesPerBatch;
                const endIdx = Math.min(startIdx + maxFilesPerBatch, files.length);
                const batchFiles = files.slice(startIdx, endIdx);
                
                // Call progress callback
                if (this.options.onProgress) {
                    this.options.onProgress({
                        currentBatch: i + 1,
                        totalBatches: this.totalBatches,
                        startIdx: startIdx + 1,
                        endIdx: endIdx,
                        totalFiles: files.length
                    });
                }
                
                // Upload this batch
                const result = await this.uploadBatch(batchFiles, endpoint, formData);
                
                // Add successful files to our list
                if (result.success && result.files) {
                    this.uploadedFiles = this.uploadedFiles.concat(result.files);
                }
            }
            
            // Call complete callback
            if (this.options.onComplete) {
                this.options.onComplete({
                    success: true,
                    files: this.uploadedFiles
                });
            }
        } catch (error) {
            console.error('Upload error:', error);
            
            // Retry logic
            if (retries < this.options.maxRetries) {
                console.log(`Retrying upload (${retries + 1}/${this.options.maxRetries})`);
                
                // Put the item back in the queue with incremented retries
                this.queue.unshift({
                    ...item,
                    retries: retries + 1
                });
                
                // Wait before retrying
                await new Promise(resolve => setTimeout(resolve, this.options.retryDelay));
            } else {
                // Call error callback
                if (this.options.onError) {
                    this.options.onError(error);
                }
            }
        }
        
        // Reset state
        this.isUploading = false;
        
        // Continue processing queue
        this.processQueue();
    }
    
    /**
     * Upload a batch of files
     * @param {Array} files - Batch of files to upload
     * @param {string} endpoint - Upload endpoint
     * @param {Object} additionalFormData - Additional form data
     * @returns {Promise} - Promise resolving to the upload result
     */
    async uploadBatch(files, endpoint, additionalFormData = {}) {
        // Create form data for this batch
        const formData = new FormData();
        
        // Add files to form data
        files.forEach(file => {
            formData.append('files[]', file);
        });
        
        // Add additional form data
        Object.entries(additionalFormData).forEach(([key, value]) => {
            formData.append(key, value);
        });
        
        // Set up abort controller for timeout
        this.abortController = new AbortController();
        const timeoutId = setTimeout(() => {
            this.abortController.abort();
        }, this.options.timeout);
        
        try {
            // Send the request
            const response = await fetch(endpoint, {
                method: 'POST',
                body: formData,
                signal: this.abortController.signal,
                headers: {
                    'Accept': 'application/json'
                }
            });
            
            // Clear the timeout
            clearTimeout(timeoutId);
            
            // Check response status
            if (!response.ok) {
                throw new Error(`Server returned ${response.status}: ${response.statusText}`);
            }
            
            // Parse response
            const data = await response.json();
            
            // Return result
            return data;
        } catch (error) {
            // Clear the timeout
            clearTimeout(timeoutId);
            
            // Handle abort error
            if (error.name === 'AbortError') {
                throw new Error('Upload timed out. The server took too long to respond.');
            }
            
            // Re-throw other errors
            throw error;
        }
    }
    
    /**
     * Cancel the current upload
     */
    cancel() {
        if (this.abortController) {
            this.abortController.abort();
        }
        
        // Clear the queue
        this.queue = [];
        this.isUploading = false;
    }
} 