/**
 * Prediction Form handler with preset loading and auto calculation
 */

function autoCalcTotal() {
  const tenure = parseFloat(document.getElementById('tenure').value) || 0;
  const monthly = parseFloat(document.getElementById('MonthlyCharges').value) || 0;
  const totalField = document.getElementById('TotalCharges');
  
  if (tenure === 0) {
    totalField.value = monthly.toFixed(2);
  } else {
    totalField.value = (tenure * monthly).toFixed(2);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  const btnHigh = document.getElementById('btnLoadHighRisk');
  const btnMed = document.getElementById('btnLoadMedRisk');
  const btnLow = document.getElementById('btnLoadLowRisk');

  function setFormValues(data) {
    for (const [key, value] of Object.entries(data)) {
      const el = document.getElementById(key);
      if (el) {
        el.value = value;
      }
    }
    const tenureEl = document.getElementById('tenure');
    if (tenureEl) {
      document.getElementById('tenureValue').textContent = tenureEl.value;
    }
    autoCalcTotal();
  }

  // 1. High Risk Profile Preset
  if (btnHigh) {
    btnHigh.addEventListener('click', () => {
      setFormValues({
        customerID: 'CUST-HIGH-991',
        gender: 'Female',
        SeniorCitizen: '1',
        Partner: 'No',
        Dependents: 'No',
        PhoneService: 'Yes',
        MultipleLines: 'Yes',
        InternetService: 'Fiber optic',
        OnlineSecurity: 'No',
        OnlineBackup: 'No',
        DeviceProtection: 'No',
        TechSupport: 'No',
        StreamingTV: 'Yes',
        StreamingMovies: 'Yes',
        tenure: 2,
        Contract: 'Month-to-month',
        PaperlessBilling: 'Yes',
        PaymentMethod: 'Electronic check',
        MonthlyCharges: 98.40
      });
    });
  }

  // 2. Moderate Risk Profile Preset
  if (btnMed) {
    btnMed.addEventListener('click', () => {
      setFormValues({
        customerID: 'CUST-MED-442',
        gender: 'Male',
        SeniorCitizen: '0',
        Partner: 'Yes',
        Dependents: 'No',
        PhoneService: 'Yes',
        MultipleLines: 'No',
        InternetService: 'DSL',
        OnlineSecurity: 'Yes',
        OnlineBackup: 'No',
        DeviceProtection: 'Yes',
        TechSupport: 'No',
        StreamingTV: 'No',
        StreamingMovies: 'No',
        tenure: 18,
        Contract: 'One year',
        PaperlessBilling: 'Yes',
        PaymentMethod: 'Mailed check',
        MonthlyCharges: 52.80
      });
    });
  }

  // 3. Low Risk Profile Preset
  if (btnLow) {
    btnLow.addEventListener('click', () => {
      setFormValues({
        customerID: 'CUST-LOW-108',
        gender: 'Male',
        SeniorCitizen: '0',
        Partner: 'Yes',
        Dependents: 'Yes',
        PhoneService: 'Yes',
        MultipleLines: 'Yes',
        InternetService: 'DSL',
        OnlineSecurity: 'Yes',
        OnlineBackup: 'Yes',
        DeviceProtection: 'Yes',
        TechSupport: 'Yes',
        StreamingTV: 'Yes',
        StreamingMovies: 'Yes',
        tenure: 65,
        Contract: 'Two year',
        PaperlessBilling: 'No',
        PaymentMethod: 'Credit card (automatic)',
        MonthlyCharges: 45.20
      });
    });
  }

  // Form submission feedback
  const form = document.getElementById('predictForm');
  const submitBtn = document.getElementById('submitBtn');
  if (form && submitBtn) {
    form.addEventListener('submit', () => {
      submitBtn.disabled = true;
      submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Computing Prediction...';
    });
  }
});
