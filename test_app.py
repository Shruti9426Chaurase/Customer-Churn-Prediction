"""
Comprehensive automated test suite for Customer Churn Prediction System.
Tests Flask routes, ML inference, API endpoints, batch processing, and error handling.
"""

import os
import io
import json
import unittest
import pandas as pd

from app import app
from backend.database import db, PredictionHistory
from backend.prediction import predict_single_customer, predict_batch_dataframe
from ml.model_utils import load_artifacts

class ChurnAppTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        cls.client = app.test_client()
        with app.app_context():
            db.create_all()

    def test_01_model_artifacts_loaded(self):
        """Verify model, preprocessor, and metadata load properly."""
        model, preprocessor, metadata = load_artifacts()
        self.assertIsNotNone(model, "Model should not be None")
        self.assertIsNotNone(preprocessor, "Preprocessor should not be None")
        self.assertIsNotNone(metadata, "Metadata should not be None")
        self.assertIn('model_name', metadata)
        self.assertIn('accuracy', metadata)
        self.assertIn('roc_auc', metadata)
        print("  [PASS] test_01_model_artifacts_loaded")

    def test_02_health_endpoints(self):
        """Verify /health and /api/health."""
        for path in ['/health', '/api/health']:
            res = self.client.get(path)
            self.assertEqual(res.status_code, 200)
            data = json.loads(res.data)
            self.assertEqual(data.get('status'), 'healthy')
        print("  [PASS] test_02_health_endpoints")

    def test_03_page_routes(self):
        """Verify all HTML pages render with 200 OK."""
        pages = [
            '/',
            '/dashboard',
            '/predict',
            '/batch',
            '/analytics',
            '/model',
            '/simulator',
            '/retention',
            '/history',
            '/about'
        ]
        for page in pages:
            res = self.client.get(page)
            self.assertEqual(res.status_code, 200, f"Page {page} failed with status {res.status_code}")
        print("  [PASS] test_03_page_routes")

    def test_04_single_prediction_inference(self):
        """Verify single prediction logic and risk categorization."""
        cust = {
            'customerID': 'TEST-UNIT-HIGH',
            'gender': 'Female',
            'SeniorCitizen': '1',
            'Partner': 'No',
            'Dependents': 'No',
            'tenure': 1,
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'InternetService': 'Fiber optic',
            'OnlineSecurity': 'No',
            'OnlineBackup': 'No',
            'DeviceProtection': 'No',
            'TechSupport': 'No',
            'StreamingTV': 'Yes',
            'StreamingMovies': 'Yes',
            'Contract': 'Month-to-month',
            'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Electronic check',
            'MonthlyCharges': 95.0,
            'TotalCharges': 95.0
        }
        res = predict_single_customer(cust)
        self.assertEqual(res['customer_id'], 'TEST-UNIT-HIGH')
        self.assertIn(res['risk_category'], ['Low Risk', 'Medium Risk', 'High Risk'])
        self.assertGreaterEqual(res['probability'], 0.0)
        self.assertLessEqual(res['probability'], 1.0)
        self.assertGreaterEqual(len(res['factors']), 1)
        self.assertGreaterEqual(len(res['recommendations']), 1)
        print("  [PASS] test_04_single_prediction_inference")

    def test_05_predict_post_route(self):
        """Verify POST /predict form submission and DB insertion."""
        form_data = {
            'customerID': 'TEST-FORM-001',
            'gender': 'Male',
            'SeniorCitizen': '0',
            'Partner': 'Yes',
            'Dependents': 'Yes',
            'tenure': '36',
            'PhoneService': 'Yes',
            'MultipleLines': 'Yes',
            'InternetService': 'DSL',
            'OnlineSecurity': 'Yes',
            'OnlineBackup': 'Yes',
            'DeviceProtection': 'Yes',
            'TechSupport': 'Yes',
            'StreamingTV': 'No',
            'StreamingMovies': 'No',
            'Contract': 'Two year',
            'PaperlessBilling': 'No',
            'PaymentMethod': 'Credit card (automatic)',
            'MonthlyCharges': '45.0',
            'TotalCharges': '1620.0'
        }
        res = self.client.post('/predict', data=form_data, follow_redirects=True)
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'TEST-FORM-001', res.data)
        self.assertIn(b'Attrition Risk Assessment', res.data)

        # Verify record in SQLite database
        with app.app_context():
            rec = PredictionHistory.query.filter_by(customer_id='TEST-FORM-001').first()
            self.assertIsNotNone(rec, "Prediction should be stored in database")
            self.assertEqual(rec.customer_id, 'TEST-FORM-001')
        print("  [PASS] test_05_predict_post_route")

    def test_06_api_predict(self):
        """Verify POST /api/predict JSON API."""
        payload = {
            'customerID': 'API-JSON-002',
            'gender': 'Female',
            'SeniorCitizen': '0',
            'Partner': 'No',
            'Dependents': 'No',
            'tenure': 5,
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'InternetService': 'DSL',
            'OnlineSecurity': 'No',
            'OnlineBackup': 'No',
            'DeviceProtection': 'No',
            'TechSupport': 'No',
            'StreamingTV': 'No',
            'StreamingMovies': 'No',
            'Contract': 'Month-to-month',
            'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Mailed check',
            'MonthlyCharges': 49.0,
            'TotalCharges': 245.0
        }
        res = self.client.post(
            '/api/predict',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(res.status_code, 200)
        resp_data = json.loads(res.data)
        self.assertEqual(resp_data.get('status'), 'success')
        self.assertIn('probability', resp_data.get('data', {}))
        print("  [PASS] test_06_api_predict")

    def test_07_what_if_simulator_api(self):
        """Verify POST /simulate API."""
        base = {
            'Contract': 'Month-to-month',
            'tenure': 2,
            'MonthlyCharges': 90.0,
            'TotalCharges': 180.0,
            'TechSupport': 'No',
            'OnlineSecurity': 'No',
            'InternetService': 'Fiber optic',
            'PaymentMethod': 'Electronic check'
        }
        mod = {
            'Contract': 'Two year',
            'tenure': 20,
            'MonthlyCharges': 60.0,
            'TotalCharges': 1200.0,
            'TechSupport': 'Yes',
            'OnlineSecurity': 'Yes',
            'InternetService': 'Fiber optic',
            'PaymentMethod': 'Bank transfer (automatic)'
        }
        res = self.client.post(
            '/simulate',
            data=json.dumps({'baseline': base, 'modified': mod}),
            content_type='application/json'
        )
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertEqual(data.get('status'), 'success')
        self.assertIn('delta_percent', data)
        self.assertIn('narrative', data)
        self.assertLess(data['modified']['probability_percent'], data['baseline']['probability_percent'])
        print("  [PASS] test_07_what_if_simulator_api")

    def test_08_batch_csv_upload(self):
        """Verify POST /batch CSV upload, processing, and download."""
        sample_path = os.path.join(app.root_path, '..', 'uploads', 'sample_customers.csv')
        self.assertTrue(os.path.exists(sample_path), "sample_customers.csv must exist")

        with open(sample_path, 'rb') as f:
            csv_bytes = f.read()

        data = {
            'file': (io.BytesIO(csv_bytes), 'test_batch_upload.csv')
        }
        res = self.client.post('/batch', data=data, content_type='multipart/form-data')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'Batch Predictions Preview', res.data)
        self.assertIn(b'Download Scored Results', res.data)
        print("  [PASS] test_08_batch_csv_upload")

    def test_09_invalid_inputs_and_errors(self):
        """Verify error handling for invalid files, missing columns, and 404."""
        # 1. 404 test
        res = self.client.get('/non_existent_route_12345')
        self.assertEqual(res.status_code, 404)
        self.assertIn(b'404', res.data)

        # 2. Non-CSV file upload to /batch
        bad_file = {
            'file': (io.BytesIO(b'not a csv content'), 'malicious.exe')
        }
        res_bad = self.client.post('/batch', data=bad_file, content_type='multipart/form-data', follow_redirects=True)
        self.assertIn(b'Invalid file format', res_bad.data)

        # 3. CSV with missing required columns
        broken_csv = io.BytesIO(b'colA,colB\n1,2\n3,4\n')
        broken_file = {'file': (broken_csv, 'broken.csv')}
        res_broken = self.client.post('/batch', data=broken_file, content_type='multipart/form-data', follow_redirects=True)
        self.assertIn(b'Missing required columns', res_broken.data)
        print("  [PASS] test_09_invalid_inputs_and_errors")

    def test_10_api_endpoints(self):
        """Verify /api/model-info, /api/dashboard-stats, /api/history."""
        res_m = self.client.get('/api/model-info')
        self.assertEqual(res_m.status_code, 200)
        self.assertIn('model_name', json.loads(res_m.data))

        res_d = self.client.get('/api/dashboard-stats')
        self.assertEqual(res_d.status_code, 200)
        self.assertIn('total_customers', json.loads(res_d.data))

        res_h = self.client.get('/api/history')
        self.assertEqual(res_h.status_code, 200)
        self.assertIsInstance(json.loads(res_h.data), list)
        print("  [PASS] test_10_api_endpoints")


if __name__ == '__main__':
    print("=" * 60)
    print("RUNNING AUTOMATED TEST SUITE FOR CHURN APPLICATION")
    print("=" * 60)
    unittest.main(verbosity=2)
