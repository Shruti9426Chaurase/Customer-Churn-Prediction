"""
Model utilities and inference helper functions.
Handles model loading, risk categorization, feature factor analysis, and retention advice.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

# Default risk thresholds
DEFAULT_THRESHOLDS = {
    'low': 0.30,
    'medium': 0.70
}

_MODEL_CACHE = {
    'model': None,
    'preprocessor': None,
    'metadata': None
}


def get_model_paths():
    """Returns absolute paths to saved model artifacts."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    return {
        'model': os.path.join(models_dir, 'churn_model.pkl'),
        'preprocessor': os.path.join(models_dir, 'preprocessor.pkl'),
        'metadata': os.path.join(models_dir, 'model_metadata.json')
    }


def load_artifacts(force_reload=False):
    """
    Loads and caches the trained model, preprocessor, and metadata.
    """
    global _MODEL_CACHE
    if not force_reload and _MODEL_CACHE['model'] is not None:
        return _MODEL_CACHE['model'], _MODEL_CACHE['preprocessor'], _MODEL_CACHE['metadata']

    paths = get_model_paths()

    if not os.path.exists(paths['model']) or not os.path.exists(paths['preprocessor']):
        return None, None, None

    try:
        model = joblib.load(paths['model'])
        preprocessor = joblib.load(paths['preprocessor'])
        
        metadata = {}
        if os.path.exists(paths['metadata']):
            with open(paths['metadata'], 'r', encoding='utf-8') as f:
                metadata = json.load(f)

        _MODEL_CACHE['model'] = model
        _MODEL_CACHE['preprocessor'] = preprocessor
        _MODEL_CACHE['metadata'] = metadata
        return model, preprocessor, metadata
    except Exception as e:
        print(f"Error loading model artifacts: {e}")
        return None, None, None


def classify_risk(probability: float, thresholds: dict = None) -> tuple:
    """
    Maps churn probability to Risk Category, CSS badge class, and color code.
    Thresholds:
      Low: < 0.30
      Medium: 0.30 <= p < 0.70
      High: >= 0.70
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    low_th = thresholds.get('low', 0.30)
    med_th = thresholds.get('medium', 0.70)

    if probability < low_th:
        return "Low Risk", "success", "#10b981"
    elif probability < med_th:
        return "Medium Risk", "warning", "#f59e0b"
    else:
        return "High Risk", "danger", "#ef4444"


def explain_prediction_factors(customer: dict, top_n: int = 4) -> list:
    """
    Identifies key observable factors correlated with this customer's churn prediction.
    Strictly uses non-causal language.
    """
    factors = []

    # Contract type
    contract = str(customer.get('Contract', '')).strip()
    if 'Month-to-month' in contract:
        factors.append({
            'factor': 'Month-to-Month Contract',
            'detail': 'Customer has no long-term commitment, which is historically associated with higher turnover rates.',
            'impact': 'High Risk Factor'
        })
    elif 'Two year' in contract or 'One year' in contract:
        factors.append({
            'factor': f'{contract} Commitment',
            'detail': 'Long-term contracts correlate strongly with customer loyalty and stability.',
            'impact': 'Protective Factor'
        })

    # Tenure
    try:
        tenure = float(customer.get('tenure', 0))
        if tenure <= 12:
            factors.append({
                'factor': f'Short Customer Tenure ({int(tenure)} months)',
                'detail': 'Customers in their first year exhibit higher churn propensity during the onboarding phase.',
                'impact': 'High Risk Factor'
            })
        elif tenure >= 48:
            factors.append({
                'factor': f'Established Customer Tenure ({int(tenure)} months)',
                'detail': 'Extended tenure reflects brand familiarity and lower sensitivity to competitive offers.',
                'impact': 'Protective Factor'
            })
    except (ValueError, TypeError):
        pass

    # Monthly Charges
    try:
        monthly = float(customer.get('MonthlyCharges', 0))
        if monthly >= 75.0:
            factors.append({
                'factor': f'High Monthly Charges (${monthly:.2f}/mo)',
                'detail': 'Higher recurring monthly bills correlate with increased price shopping and bill sensitivity.',
                'impact': 'High Risk Factor'
            })
        elif monthly <= 35.0:
            factors.append({
                'factor': f'Low Monthly Charges (${monthly:.2f}/mo)',
                'detail': 'Budget-friendly monthly fee provides strong perceived value.',
                'impact': 'Protective Factor'
            })
    except (ValueError, TypeError):
        pass

    # Internet & Support Services
    internet = str(customer.get('InternetService', '')).strip()
    tech_support = str(customer.get('TechSupport', '')).strip()
    online_sec = str(customer.get('OnlineSecurity', '')).strip()

    if internet == 'Fiber optic' and tech_support == 'No':
        factors.append({
            'factor': 'Fiber Optic Service without Tech Support',
            'detail': 'High-speed fiber subscribers without dedicated technical assistance report higher frustration when issues occur.',
            'impact': 'High Risk Factor'
        })
    elif tech_support == 'Yes':
        factors.append({
            'factor': 'Active Tech Support Subscription',
            'detail': 'Subscribers with dedicated technical support experience faster issue resolution and higher satisfaction.',
            'impact': 'Protective Factor'
        })

    if online_sec == 'No' and internet in ['Fiber optic', 'DSL']:
        factors.append({
            'factor': 'No Online Security Add-on',
            'detail': 'Absence of digital security add-ons indicates a transactional, unbundled relationship.',
            'impact': 'Moderate Risk Factor'
        })

    # Payment Method
    payment = str(customer.get('PaymentMethod', '')).strip()
    if 'Electronic check' in payment:
        factors.append({
            'factor': 'Manual Electronic Check Payment',
            'detail': 'Manual recurring payment methods have higher churn correlation compared to automated payment options.',
            'impact': 'Moderate Risk Factor'
        })
    elif 'automatic' in payment.lower():
        factors.append({
            'factor': 'Automated Recurring Payment',
            'detail': 'Automatic billing creates smooth frictionless renewals.',
            'impact': 'Protective Factor'
        })

    # Fallback if factors list is short
    if len(factors) < 2:
        factors.append({
            'factor': 'Standard Account Configuration',
            'detail': 'Customer account profile aligns with standard baseline subscription patterns.',
            'impact': 'Neutral Factor'
        })

    return factors[:top_n]


def generate_retention_recommendations(customer: dict, risk_category: str, probability: float) -> list:
    """
    Produces actionable, practical retention suggestions tailored to customer attributes.
    """
    recs = []

    contract = str(customer.get('Contract', '')).strip()
    monthly = 0.0
    try:
        monthly = float(customer.get('MonthlyCharges', 0))
    except (ValueError, TypeError):
        pass
    tenure = 0
    try:
        tenure = float(customer.get('tenure', 0))
    except (ValueError, TypeError):
        pass
    tech_support = str(customer.get('TechSupport', '')).strip()
    online_sec = str(customer.get('OnlineSecurity', '')).strip()
    payment = str(customer.get('PaymentMethod', '')).strip()

    # Priority 1: Contract migration
    if 'Month-to-month' in contract:
        recs.append({
            'title': 'Annual Contract Incentive',
            'description': 'Propose migrating to a 1-year contract with a 15% discount for the first 3 months or a complimentary device upgrade.',
            'priority': 'Urgent' if probability >= 0.70 else 'Recommended',
            'badge': 'Contract Strategy'
        })

    # Priority 2: Pricing review
    if monthly >= 75.0:
        recs.append({
            'title': 'Loyalty Pricing & Bundle Optimization',
            'description': 'Conduct an account review to customize package services, offering loyalty discount credits without degrading tier speed.',
            'priority': 'Recommended',
            'badge': 'Pricing Strategy'
        })

    # Priority 3: Tech Support & Value-Adds
    if tech_support == 'No':
        recs.append({
            'title': 'Complimentary 60-Day Premium Tech Support',
            'description': 'Provision 60 days of zero-cost 24/7 dedicated technical support to alleviate technical pain points.',
            'priority': 'High Priority',
            'badge': 'Service Engagement'
        })

    if online_sec == 'No':
        recs.append({
            'title': 'Digital Security Suite Trial',
            'description': 'Activate a complimentary 90-day Online Security & Device Protection trial to increase perceived service value.',
            'priority': 'Suggested',
            'badge': 'Value Addition'
        })

    # Priority 4: Onboarding nurture
    if tenure <= 12:
        recs.append({
            'title': 'First-Year VIP Onboarding Call',
            'description': 'Schedule a proactive customer success check-in call to ensure setup satisfaction and resolve early friction.',
            'priority': 'High Priority',
            'badge': 'Customer Success'
        })

    # Priority 5: Autopay incentive
    if 'electronic check' in payment.lower() or 'mailed check' in payment.lower():
        recs.append({
            'title': 'Autopay Bill Credit Incentive',
            'description': 'Offer a one-time $10 account credit upon enrolling in automatic bank transfer or credit card billing.',
            'priority': 'Suggested',
            'badge': 'Billing Friction'
        })

    # Priority 6: General high risk retention desk
    if probability >= 0.70:
        recs.append({
            'title': 'Direct Retention Desk Outreach',
            'description': 'Flag account for immediate proactive concierge outreach before the upcoming billing cycle renewal.',
            'priority': 'Urgent',
            'badge': 'Executive Care'
        })

    if not recs:
        recs.append({
            'title': 'Standard Relationship Nurturing',
            'description': 'Send quarterly loyalty appreciation communications and maintain regular product updates.',
            'priority': 'Routine',
            'badge': 'Account Health'
        })

    return recs[:4]
