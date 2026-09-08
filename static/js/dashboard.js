/**
 * Dashboard dynamic charts handler using Chart.js
 */

document.addEventListener('DOMContentLoaded', () => {
  const dataEl = document.getElementById('dashboard-data');
  if (!dataEl) return;

  let dashData;
  try {
    dashData = JSON.parse(dataEl.textContent);
  } catch (e) {
    console.error('Failed to parse dashboard data:', e);
    return;
  }

  const stats = dashData.stats || {};
  const metadata = dashData.metadata || {};

  // 1. Churn Distribution Pie Chart
  const pieCtx = document.getElementById('churnPieChart');
  if (pieCtx) {
    new Chart(pieCtx, {
      type: 'doughnut',
      data: {
        labels: ['Retained Customers', 'Churned Customers'],
        datasets: [{
          data: [stats.retain_count || 5174, stats.churn_count || 1869],
          backgroundColor: [AppCharts.colors.success, AppCharts.colors.danger],
          borderWidth: 2,
          hoverOffset: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'bottom', labels: { color: AppCharts.getTextColor() } },
          tooltip: {
            callbacks: {
              label: (ctx) => ` ${ctx.label}: ${ctx.raw.toLocaleString()} (${((ctx.raw / (stats.total_customers || 7043)) * 100).toFixed(1)}%)`
            }
          }
        },
        cutout: '70%'
      }
    });
  }

  // 2. Churn Rate by Contract Bar Chart
  const contractCtx = document.getElementById('contractChart');
  if (contractCtx && stats.contract_stats) {
    const labels = Object.keys(stats.contract_stats);
    const rates = labels.map(k => stats.contract_stats[k].churn_rate);

    new Chart(contractCtx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Churn Rate (%)',
          data: rates,
          backgroundColor: [AppCharts.colors.danger, AppCharts.colors.warning, AppCharts.colors.success],
          borderRadius: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            beginAtZero: true,
            max: 50,
            ticks: { callback: v => v + '%', color: AppCharts.getTextColor() },
            grid: { color: AppCharts.getGridColor() }
          },
          x: { ticks: { color: AppCharts.getTextColor() }, grid: { display: false } }
        },
        plugins: { legend: { display: false } }
      }
    });
  }

  // 3. Churn Rate by Internet Service Bar Chart
  const netCtx = document.getElementById('internetChart');
  if (netCtx && stats.internet_stats) {
    const labels = Object.keys(stats.internet_stats);
    const rates = labels.map(k => stats.internet_stats[k].churn_rate);

    new Chart(netCtx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Churn Rate (%)',
          data: rates,
          backgroundColor: [AppCharts.colors.primary, AppCharts.colors.danger, AppCharts.colors.success],
          borderRadius: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            beginAtZero: true,
            max: 50,
            ticks: { callback: v => v + '%', color: AppCharts.getTextColor() },
            grid: { color: AppCharts.getGridColor() }
          },
          x: { ticks: { color: AppCharts.getTextColor() }, grid: { display: false } }
        },
        plugins: { legend: { display: false } }
      }
    });
  }

  // 4. Tenure Cohorts Stacked Bar Chart
  const tenureCtx = document.getElementById('tenureCohortChart');
  if (tenureCtx && stats.tenure_cohorts) {
    const cohorts = Object.keys(stats.tenure_cohorts);
    const stayData = cohorts.map(c => stats.tenure_cohorts[c].stay);
    const churnData = cohorts.map(c => stats.tenure_cohorts[c].churn);

    new Chart(tenureCtx, {
      type: 'bar',
      data: {
        labels: cohorts,
        datasets: [
          {
            label: 'Retained',
            data: stayData,
            backgroundColor: AppCharts.colors.success,
            borderRadius: 4
          },
          {
            label: 'Churned',
            data: churnData,
            backgroundColor: AppCharts.colors.danger,
            borderRadius: 4
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { stacked: true, ticks: { color: AppCharts.getTextColor() }, grid: { display: false } },
          y: { stacked: true, ticks: { color: AppCharts.getTextColor() }, grid: { color: AppCharts.getGridColor() } }
        },
        plugins: {
          legend: { position: 'top', labels: { color: AppCharts.getTextColor() } }
        }
      }
    });
  }

  // 5. Feature Importance Horizontal Bar Chart
  const featCtx = document.getElementById('featureImportanceChart');
  if (featCtx && metadata.feature_importance) {
    const topFeats = metadata.feature_importance.slice(0, 8);
    const labels = topFeats.map(f => f.feature);
    const values = topFeats.map(f => f.importance);

    new Chart(featCtx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Relative Importance',
          data: values,
          backgroundColor: AppCharts.colors.primary,
          borderRadius: 4
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { beginAtZero: true, ticks: { color: AppCharts.getTextColor() }, grid: { color: AppCharts.getGridColor() } },
          y: { ticks: { color: AppCharts.getTextColor(), font: { size: 11 } }, grid: { display: false } }
        },
        plugins: { legend: { display: false } }
      }
    });
  }
});
