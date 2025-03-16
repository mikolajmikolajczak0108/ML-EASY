/**
 * Main JavaScript file for ML-EASY application
 */

// Initialize toasts
document.addEventListener('DOMContentLoaded', function() {
    // Initialize all toasts
    const toastElements = document.querySelectorAll('.toast');
    toastElements.forEach(toastEl => {
        const toast = new bootstrap.Toast(toastEl, {
            autohide: true,
            delay: 3000
        });
    });
    
    // Add fade effect to alerts
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(alert => {
        setTimeout(() => {
            alert.classList.add('show');
        }, 100);
        
        if (alert.classList.contains('alert-dismissible')) {
            const closeBtn = alert.querySelector('.btn-close');
            if (closeBtn) {
                closeBtn.addEventListener('click', () => {
                    alert.classList.remove('show');
                    setTimeout(() => {
                        alert.remove();
                    }, 200);
                });
            }
        }
    });
});

// Global function to show a toast message programmatically
function showToast(message, type = 'success') {
    // Create toast container if it doesn't exist
    let toastContainer = document.querySelector('.toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.className = 'toast-container position-fixed bottom-0 end-0 p-3';
        toastContainer.style.zIndex = '1080';
        document.body.appendChild(toastContainer);
    }
    
    // Create toast element
    const toastEl = document.createElement('div');
    toastEl.className = `toast bg-${type} text-white`;
    toastEl.setAttribute('role', 'alert');
    toastEl.setAttribute('aria-live', 'assertive');
    toastEl.setAttribute('aria-atomic', 'true');
    
    // Set toast content
    toastEl.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                ${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
        </div>
    `;
    
    // Add to container
    toastContainer.appendChild(toastEl);
    
    // Initialize and show toast
    const toast = new bootstrap.Toast(toastEl, {
        autohide: true,
        delay: 3000
    });
    toast.show();
    
    // Remove after hiding
    toastEl.addEventListener('hidden.bs.toast', () => {
        toastEl.remove();
    });
} 