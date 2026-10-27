"""
Basic API tests using unittest + Flask test client.

Run with:
    python -m pytest tests/test_api.py -v
or:
    python -m unittest tests.test_api -v
"""

import json
import sys
import os
import unittest

# Ensure root package is on the path when running tests directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app import create_app


# ── Shared valid patient payload ──────────────────────────────────────────────
VALID_PATIENT = {
    "age":                  55,
    "bmi":                  28.5,
    "systolic_bp":          145,
    "diastolic_bp":         90,
    "heart_rate":           80,
    "cholesterol":          220,
    "glucose":              130,
    "hba1c":                7.2,
    "creatinine":           1.2,
    "sodium":               140,
    "potassium":            4.1,
    "wbc":                  7.5,
    "hemoglobin":           13.5,
    "platelets":            250,
    "smoking_status":       0,
    "diabetes":             1,
    "hypertension":         1,
    "heart_disease":        0,
    "num_medications":      5,
    "num_hospitalizations": 2,
    "num_comorbidities":    3,
}


class TestHealthEndpoint(unittest.TestCase):
    def setUp(self):
        self.app   = create_app()
        self.client = self.app.test_client()

    def test_health_returns_200(self):
        r = self.client.get('/api/health')
        self.assertEqual(r.status_code, 200)

    def test_health_has_status_key(self):
        r    = self.client.get('/api/health')
        data = json.loads(r.data)
        self.assertIn('status', data)
        self.assertEqual(data['status'], 'healthy')

    def test_health_has_models_loaded_key(self):
        r    = self.client.get('/api/health')
        data = json.loads(r.data)
        self.assertIn('models_loaded', data)
        self.assertIsInstance(data['models_loaded'], bool)


class TestRootRoute(unittest.TestCase):
    def setUp(self):
        self.app    = create_app()
        self.client = self.app.test_client()

    def test_root_returns_html(self):
        r = self.client.get('/')
        self.assertEqual(r.status_code, 200)
        self.assertIn(b'ClinicalRiskAI', r.data)


class TestPredictEndpoint(unittest.TestCase):
    def setUp(self):
        self.app    = create_app()
        self.client = self.app.test_client()

    def test_predict_missing_body_returns_400(self):
        r = self.client.post(
            '/api/predict',
            data='',
            content_type='application/json',
        )
        # Flask returns 400 when JSON parsing fails on an empty body
        self.assertIn(r.status_code, [400, 422, 503])

    def test_predict_invalid_field_returns_422_or_503(self):
        bad_payload = {'age': 'not_a_number'}
        r = self.client.post(
            '/api/predict',
            data=json.dumps(bad_payload),
            content_type='application/json',
        )
        self.assertIn(r.status_code, [422, 503])

    def test_predict_no_models_returns_503_or_success(self):
        """
        When models are not trained this should return 503.
        If models ARE trained it should return 200 with a valid prediction.
        """
        r = self.client.post(
            '/api/predict',
            data=json.dumps(VALID_PATIENT),
            content_type='application/json',
        )
        if r.status_code == 200:
            data = json.loads(r.data)
            self.assertIn('risk_label', data)
            self.assertIn(data['risk_label'], ['Low', 'Medium', 'High'])
            self.assertIn('probabilities', data)
        else:
            self.assertEqual(r.status_code, 503)


class TestModelsInfoEndpoint(unittest.TestCase):
    def setUp(self):
        self.app    = create_app()
        self.client = self.app.test_client()

    def test_models_info_returns_200(self):
        r = self.client.get('/api/models/info')
        self.assertEqual(r.status_code, 200)

    def test_models_info_has_models_loaded(self):
        r    = self.client.get('/api/models/info')
        data = json.loads(r.data)
        self.assertIn('models_loaded', data)


class TestBatchPredictEndpoint(unittest.TestCase):
    def setUp(self):
        self.app    = create_app()
        self.client = self.app.test_client()

    def test_batch_predict_missing_patients_key(self):
        r = self.client.post(
            '/api/batch-predict',
            data=json.dumps({}),
            content_type='application/json',
        )
        self.assertEqual(r.status_code, 400)

    def test_batch_predict_with_valid_list(self):
        payload = {'patients': [VALID_PATIENT, VALID_PATIENT]}
        r = self.client.post(
            '/api/batch-predict',
            data=json.dumps(payload),
            content_type='application/json',
        )
        # 503 if models not trained, 200 if trained
        self.assertIn(r.status_code, [200, 503])
        if r.status_code == 200:
            data = json.loads(r.data)
            self.assertIn('predictions', data)
            self.assertEqual(data['count'], 2)


class TestPreprocessingUtils(unittest.TestCase):
    def test_validate_patient_data_valid(self):
        from app.utils.preprocessing import validate_patient_data
        ok, errors = validate_patient_data(VALID_PATIENT)
        self.assertTrue(ok, f"Expected valid but got errors: {errors}")

    def test_validate_patient_data_missing_field(self):
        from app.utils.preprocessing import validate_patient_data
        bad = {k: v for k, v in VALID_PATIENT.items() if k != 'age'}
        ok, errors = validate_patient_data(bad)
        self.assertFalse(ok)
        self.assertTrue(any('age' in e for e in errors))

    def test_validate_patient_data_out_of_range(self):
        from app.utils.preprocessing import validate_patient_data
        bad = dict(VALID_PATIENT)
        bad['age'] = 150  # way out of range
        ok, errors = validate_patient_data(bad)
        self.assertFalse(ok)

    def test_extract_features_returns_dataframe(self):
        import pandas as pd
        from app.utils.preprocessing import extract_features, ALL_FEATURES
        df = extract_features(VALID_PATIENT)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(list(df.columns), ALL_FEATURES)
        self.assertEqual(len(df), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
