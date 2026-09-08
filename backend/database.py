"""
Database configuration and models using Flask-SQLAlchemy.
Stores prediction records and audit history in SQLite.
"""

from datetime import datetime, timezone
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


def utc_now():
    return datetime.now(timezone.utc)


class PredictionHistory(db.Model):
    __tablename__ = 'prediction_history'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    customer_id = db.Column(db.String(64), nullable=False, default="CUST-UNKNOWN")
    
    # Demographics
    gender = db.Column(db.String(16), nullable=True)
    senior_citizen = db.Column(db.String(8), nullable=True)
    partner = db.Column(db.String(8), nullable=True)
    dependents = db.Column(db.String(8), nullable=True)
    
    # Account & Services
    tenure = db.Column(db.Float, nullable=True)
    phone_service = db.Column(db.String(8), nullable=True)
    multiple_lines = db.Column(db.String(32), nullable=True)
    internet_service = db.Column(db.String(32), nullable=True)
    online_security = db.Column(db.String(32), nullable=True)
    online_backup = db.Column(db.String(32), nullable=True)
    device_protection = db.Column(db.String(32), nullable=True)
    tech_support = db.Column(db.String(32), nullable=True)
    streaming_tv = db.Column(db.String(32), nullable=True)
    streaming_movies = db.Column(db.String(32), nullable=True)
    contract = db.Column(db.String(32), nullable=True)
    paperless_billing = db.Column(db.String(8), nullable=True)
    payment_method = db.Column(db.String(64), nullable=True)
    monthly_charges = db.Column(db.Float, nullable=True)
    total_charges = db.Column(db.Float, nullable=True)
    
    # Prediction Outputs
    prediction = db.Column(db.String(20), nullable=False) # "Churn" or "Retain"
    probability = db.Column(db.Float, nullable=False)
    risk_category = db.Column(db.String(20), nullable=False) # "Low Risk", "Medium Risk", "High Risk"
    risk_score = db.Column(db.Integer, nullable=False) # 0 - 100
    model_version = db.Column(db.String(64), nullable=False, default="Random Forest")
    created_at = db.Column(db.DateTime, default=utc_now)

    def to_dict(self):
        return {
            'id': self.id,
            'customer_id': self.customer_id,
            'gender': self.gender,
            'senior_citizen': self.senior_citizen,
            'partner': self.partner,
            'dependents': self.dependents,
            'tenure': self.tenure,
            'phone_service': self.phone_service,
            'multiple_lines': self.multiple_lines,
            'internet_service': self.internet_service,
            'online_security': self.online_security,
            'online_backup': self.online_backup,
            'device_protection': self.device_protection,
            'tech_support': self.tech_support,
            'streaming_tv': self.streaming_tv,
            'streaming_movies': self.streaming_movies,
            'contract': self.contract,
            'paperless_billing': self.paperless_billing,
            'payment_method': self.payment_method,
            'monthly_charges': self.monthly_charges,
            'total_charges': self.total_charges,
            'prediction': self.prediction,
            'probability': round(self.probability, 4),
            'risk_category': self.risk_category,
            'risk_score': self.risk_score,
            'model_version': self.model_version,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else ''
        }
