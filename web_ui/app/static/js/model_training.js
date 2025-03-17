// Model Training JS

$(document).ready(function() {
    // Training form submission
    $('.model-training-form').on('submit', function(e) {
        e.preventDefault();
        const form = $(this);
        const submitButton = form.find('button[type="submit"]');
        const submitButtonText = submitButton.html();
        const alertContainer = $('#alertContainer');
        
        // Update UI to show loading
        submitButton.prop('disabled', true);
        submitButton.html('<i class="fas fa-spinner fa-spin"></i> Starting training...');
        alertContainer.html('').hide();
        
        // Submit form via AJAX
        $.ajax({
            url: form.attr('action'),
            method: form.attr('method'),
            data: form.serialize(),
            success: function(response) {
                if (response.success) {
                    // Show success message
                    alertContainer.html(
                        `<div class="alert alert-success alert-dismissible fade show" role="alert">
                            <i class="fas fa-check-circle me-2"></i> ${response.message}
                            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                        </div>`
                    ).show();
                    
                    // If there's a redirect URL, redirect after a short delay
                    if (response.redirect) {
                        setTimeout(function() {
                            window.location.href = response.redirect;
                        }, 1500);
                    }
                } else {
                    // Show error message
                    alertContainer.html(
                        `<div class="alert alert-danger alert-dismissible fade show" role="alert">
                            <i class="fas fa-exclamation-triangle me-2"></i> ${response.error || 'An error occurred'}
                            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                        </div>`
                    ).show();
                    
                    // Reset submit button
                    submitButton.prop('disabled', false);
                    submitButton.html(submitButtonText);
                }
            },
            error: function(xhr) {
                let errorMessage = 'An error occurred while starting the training.';
                
                // Try to parse error message from response
                try {
                    const response = JSON.parse(xhr.responseText);
                    if (response && response.error) {
                        errorMessage = response.error;
                    }
                } catch (e) {
                    // Use default error message
                }
                
                // Show error message
                alertContainer.html(
                    `<div class="alert alert-danger alert-dismissible fade show" role="alert">
                        <i class="fas fa-exclamation-triangle me-2"></i> ${errorMessage}
                        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                    </div>`
                ).show();
                
                // Reset submit button
                submitButton.prop('disabled', false);
                submitButton.html(submitButtonText);
            }
        });
    });
    
    // Initialize training progress display if on status page
    if ($('#trainingProgress').length > 0) {
        initTrainingStatusPage();
    }
});

function initTrainingStatusPage() {
    const modelName = $('#trainingProgress').data('model');
    
    // Initial status check
    updateTrainingStatus(modelName);
    
    // Set interval for updates
    const statusInterval = setInterval(function() {
        updateTrainingStatus(modelName, function(status) {
            // If training is complete or errored, stop polling
            if (status.status === 'completed' || status.status === 'error') {
                clearInterval(statusInterval);
            }
        });
    }, 2000);
}

function updateTrainingStatus(modelName, callback) {
    $.ajax({
        url: `/train/models/${modelName}/status`,
        method: 'GET',
        success: function(data) {
            // Update progress bar
            const progress = Math.min(100, Math.max(0, data.progress || 0));
            $('#trainingProgressBar')
                .css('width', progress + '%')
                .attr('aria-valuenow', progress)
                .text(progress + '%');
            
            // Update status message
            $('#trainingStatusMessage').text(data.message || 'Processing...');
            
            // Update status badge
            updateStatusBadge(data.status);
            
            // Update metrics if available
            if (data.metrics) {
                updateMetricsDisplay(data.metrics);
            }
            
            // Call callback if provided
            if (typeof callback === 'function') {
                callback(data);
            }
        },
        error: function(xhr, status, error) {
            console.error('Error fetching training status:', error);
            
            // Call callback if provided
            if (typeof callback === 'function') {
                callback({status: 'error', error: error});
            }
        }
    });
}

function updateStatusBadge(status) {
    const badge = $('#trainingStatusBadge');
    let badgeClass = 'bg-secondary';
    
    // Set badge color based on status
    switch(status) {
        case 'initializing':
        case 'preparing':
            badgeClass = 'bg-info';
            break;
        case 'training':
            badgeClass = 'bg-primary';
            break;
        case 'saving':
            badgeClass = 'bg-warning text-dark';
            break;
        case 'completed':
            badgeClass = 'bg-success';
            break;
        case 'error':
            badgeClass = 'bg-danger';
            break;
    }
    
    // Remove existing color classes and add new one
    badge.removeClass('bg-info bg-primary bg-warning bg-success bg-danger bg-secondary')
        .addClass(badgeClass)
        .text(status);
}

function updateMetricsDisplay(metrics) {
    const metricsContainer = $('#trainingMetrics');
    
    // Only update if we have metrics
    if (!metrics.loss && !metrics.accuracy) {
        return;
    }
    
    // Format metrics for display
    const loss = metrics.loss !== undefined ? metrics.loss.toFixed(4) : '-';
    const accuracy = metrics.accuracy !== undefined ? (metrics.accuracy * 100).toFixed(2) + '%' : '-';
    const valLoss = metrics.val_loss !== undefined ? metrics.val_loss.toFixed(4) : '-';
    const valAccuracy = metrics.val_accuracy !== undefined ? (metrics.val_accuracy * 100).toFixed(2) + '%' : '-';
    
    // Update metrics table
    let html = `
    <div class="table-responsive">
        <table class="table table-bordered table-sm">
            <thead class="table-light">
                <tr>
                    <th>Metric</th>
                    <th>Training</th>
                    <th>Validation</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Loss</td>
                    <td>${loss}</td>
                    <td>${valLoss}</td>
                </tr>
                <tr>
                    <td>Accuracy</td>
                    <td>${accuracy}</td>
                    <td>${valAccuracy}</td>
                </tr>
            </tbody>
        </table>
    </div>`;
    
    // Add epoch info if available
    if (metrics.epoch !== undefined && metrics.total_epochs !== undefined) {
        html += `<div class="mt-2 text-center">
            <small class="text-muted">Epoch ${metrics.epoch} of ${metrics.total_epochs}</small>
        </div>`;
    }
    
    metricsContainer.html(html).show();
} 