/**
 * What-If Churn Simulator Client Logic
 */

document.addEventListener('DOMContentLoaded', () => {
  const btnRun = document.getElementById('btnRunSimulation');
  const btnPreset = document.getElementById('btnLoadSimulatorPreset');
  const resultsCard = document.getElementById('resultsCard');

  if (btnPreset) {
    btnPreset.addEventListener('click', () => {
      // Baseline setup
      document.getElementById('base_Contract').value = 'Month-to-month';
      document.getElementById('base_MonthlyCharges').value = 95;
      document.getElementById('base_monthly_val').textContent = '$95';
      document.getElementById('base_tenure').value = 3;
      document.getElementById('base_tenure_val').textContent = '3';
      document.getElementById('base_TechSupport').value = 'No';
      document.getElementById('base_OnlineSecurity').value = 'No';
      document.getElementById('base_InternetService').value = 'Fiber optic';
      document.getElementById('base_PaymentMethod').value = 'Electronic check';

      // Proposed modifications
      document.getElementById('mod_Contract').value = 'One year';
      document.getElementById('mod_MonthlyCharges').value = 70;
      document.getElementById('mod_monthly_val').textContent = '$70';
      document.getElementById('mod_tenure').value = 15;
      document.getElementById('mod_tenure_val').textContent = '15';
      document.getElementById('mod_TechSupport').value = 'Yes';
      document.getElementById('mod_OnlineSecurity').value = 'Yes';
      document.getElementById('mod_PaymentMethod').value = 'Bank transfer (automatic)';

      // Auto trigger simulation
      btnRun.click();
    });
  }

  function getCustomerPayload(prefix) {
    const tenure = parseFloat(document.getElementById(`${prefix}_tenure`).value) || 0;
    const monthly = parseFloat(document.getElementById(`${prefix}_MonthlyCharges`).value) || 0;
    const total = tenure * monthly;

    return {
      customerID: `SIM-${prefix.toUpperCase()}`,
      gender: 'Female',
      SeniorCitizen: '0',
      Partner: 'No',
      Dependents: 'No',
      PhoneService: 'Yes',
      MultipleLines: 'No',
      InternetService: document.getElementById(`${prefix}_InternetService`).value,
      OnlineSecurity: document.getElementById(`${prefix}_OnlineSecurity`).value,
      OnlineBackup: 'No',
      DeviceProtection: 'No',
      TechSupport: document.getElementById(`${prefix}_TechSupport`).value,
      StreamingTV: 'No',
      StreamingMovies: 'No',
      Contract: document.getElementById(`${prefix}_Contract`).value,
      PaperlessBilling: 'Yes',
      PaymentMethod: document.getElementById(`${prefix}_PaymentMethod`).value,
      tenure: tenure,
      MonthlyCharges: monthly,
      TotalCharges: total
    };
  }

  if (btnRun) {
    btnRun.addEventListener('click', async () => {
      btnRun.disabled = true;
      btnRun.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Simulating...';

      const baseline = getCustomerPayload('base');
      const modified = getCustomerPayload('mod');

      try {
        const response = await fetch('/simulate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ baseline, modified })
        });

        const data = await response.json();
        if (data.status === 'success') {
          resultsCard.style.display = 'block';

          // Update Baseline display
          const bProb = data.baseline.probability_percent;
          document.getElementById('baseProbText').textContent = `${bProb}%`;
          document.getElementById('baseCategory').textContent = data.baseline.risk_category;
          document.getElementById('baseCategory').className = `badge bg-${data.baseline.badge_class}-subtle text-${data.baseline.badge_class}`;

          // Update Modified display
          const mProb = data.modified.probability_percent;
          document.getElementById('modProbText').textContent = `${mProb}%`;
          document.getElementById('modCategory').textContent = data.modified.risk_category;
          document.getElementById('modCategory').className = `badge bg-${data.modified.badge_class}-subtle text-${data.modified.badge_class}`;

          // Update Delta
          const delta = data.delta_percent;
          const deltaBadge = document.getElementById('deltaBadge');
          const deltaText = document.getElementById('deltaText');

          if (delta < 0) {
            deltaBadge.className = 'badge bg-success fs-6';
            deltaBadge.textContent = `${delta}% Churn Reduction`;
            deltaText.className = 'fw-bold fs-5 text-success';
            deltaText.textContent = `${delta} percentage pts`;
          } else if (delta > 0) {
            deltaBadge.className = 'badge bg-danger fs-6';
            deltaBadge.textContent = `+${delta}% Churn Increase`;
            deltaText.className = 'fw-bold fs-5 text-danger';
            deltaText.textContent = `+${delta} percentage pts`;
          } else {
            deltaBadge.className = 'badge bg-secondary fs-6';
            deltaBadge.textContent = `0.0% No Change`;
            deltaText.className = 'fw-bold fs-5 text-muted';
            deltaText.textContent = `0.0%`;
          }

          // Narrative
          document.getElementById('narrativeText').textContent = data.narrative;

          // Smooth scroll to results
          resultsCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        } else {
          alert('Simulation failed: ' + (data.message || 'Unknown error'));
        }
      } catch (err) {
        console.error(err);
        alert('Network or server error executing simulation.');
      } finally {
        btnRun.disabled = false;
        btnRun.innerHTML = '<i class="bi bi-play-circle-fill me-2"></i> Execute What-If Simulation';
      }
    });
  }
});
