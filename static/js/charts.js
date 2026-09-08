/**
 * Chart.js helper module for Customer Churn Prediction System.
 * Handles palette configurations, responsiveness, and theme transitions.
 */

const AppCharts = {
  colors: {
    primary: '#3b82f6',
    primarySubtle: 'rgba(59, 130, 246, 0.2)',
    success: '#10b981',
    successSubtle: 'rgba(16, 185, 129, 0.2)',
    warning: '#f59e0b',
    warningSubtle: 'rgba(245, 158, 11, 0.2)',
    danger: '#ef4444',
    dangerSubtle: 'rgba(239, 68, 68, 0.2)',
    purple: '#8b5cf6',
    gray: '#94a3b8'
  },

  isDarkMode() {
    return document.documentElement.getAttribute('data-theme') === 'dark';
  },

  getGridColor() {
    return this.isDarkMode() ? 'rgba(255, 255, 255, 0.08)' : 'rgba(0, 0, 0, 0.06)';
  },

  getTextColor() {
    return this.isDarkMode() ? '#9ca3af' : '#64748b';
  }
};
