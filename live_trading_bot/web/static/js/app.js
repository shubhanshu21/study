/**
 * Main application JavaScript for RL Trading Bot
 */

// Update service status indicator
function updateServiceStatus() {
  fetch('/api/status')
      .then(response => response.json())
      .then(data => {
          const statusDot = document.getElementById('status-dot');
          const statusText = document.getElementById('status-text');
          
          if (data.service_status === 'running') {
              statusDot.className = 'status-dot online';
              statusText.textContent = 'Service Online';
          } else if (data.service_status === 'starting' || data.service_status === 'stopping') {
              statusDot.className = 'status-dot warning';
              statusText.textContent = `Service ${data.service_status.charAt(0).toUpperCase() + data.service_status.slice(1)}...`;
          } else {
              statusDot.className = 'status-dot offline';
              statusText.textContent = 'Service Offline';
          }
      })
      .catch(error => {
          console.error('Error fetching service status:', error);
          const statusDot = document.getElementById('status-dot');
          const statusText = document.getElementById('status-text');
          statusDot.className = 'status-dot offline';
          statusText.textContent = 'Connection Error';
      });
}

// Format currency
function formatCurrency(amount) {
  return '₹' + parseFloat(amount).toFixed(2).replace(/\d(?=(\d{3})+\.)/g, '$&,');
}

// Format percentage
function formatPercentage(value) {
  return parseFloat(value).toFixed(2) + '%';
}

// Format date
function formatDate(dateString) {
  const date = new Date(dateString);
  return date.toLocaleDateString('en-IN', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
  });
}

// Format date and time
function formatDateTime(dateString) {
  const date = new Date(dateString);
  return date.toLocaleDateString('en-IN', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
  });
}

// Show notification
function showNotification(title, message, type = 'info') {
  // Check if Bootstrap 5 toast is available
  if (typeof bootstrap !== 'undefined' && bootstrap.Toast) {
      // Create toast container if it doesn't exist
      let toastContainer = document.querySelector('.toast-container');
      if (!toastContainer) {
          toastContainer = document.createElement('div');
          toastContainer.className = 'toast-container position-fixed bottom-0 end-0 p-3';
          document.body.appendChild(toastContainer);
      }
      
      // Create toast element
      const toastEl = document.createElement('div');
      toastEl.className = `toast align-items-center text-white bg-${type}`;
      toastEl.setAttribute('role', 'alert');
      toastEl.setAttribute('aria-live', 'assertive');
      toastEl.setAttribute('aria-atomic', 'true');
      
      toastEl.innerHTML = `
          <div class="d-flex">
              <div class="toast-body">
                  <strong>${title}</strong>: ${message}
              </div>
              <button type="button" class="btn-close me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
          </div>
      `;
      
      toastContainer.appendChild(toastEl);
      
      // Initialize and show toast
      const toast = new bootstrap.Toast(toastEl, { autohide: true, delay: 5000 });
      toast.show();
      
      // Remove toast element after it's hidden
      toastEl.addEventListener('hidden.bs.toast', function() {
          this.remove();
      });
  } else {
      // Fallback to alert if Bootstrap toast is not available
      alert(`${title}: ${message}`);
  }
}

// Initialize common functionality
document.addEventListener('DOMContentLoaded', function() {
  // Update service status on page load
  updateServiceStatus();
  
  // Update service status every 30 seconds
  setInterval(updateServiceStatus, 30000);
  
  // Initialize tooltips
  if (typeof bootstrap !== 'undefined' && bootstrap.Tooltip) {
      const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
      tooltipTriggerList.map(function(tooltipTriggerEl) {
          return new bootstrap.Tooltip(tooltipTriggerEl);
      });
  }
  
  // Initialize popovers
  if (typeof bootstrap !== 'undefined' && bootstrap.Popover) {
      const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
      popoverTriggerList.map(function(popoverTriggerEl) {
          return new bootstrap.Popover(popoverTriggerEl);
      });
  }
});