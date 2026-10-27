"""
Flask API Blueprint – all REST endpoints for the Clinical Risk AI system.
"""

import datetime
import traceback

from flask import Blueprint, request, jsonify, current_app, send_from_directory
import os

from app.models.predictor import RiskPredictor
from app.utils.preprocessing import validate_patient_data

api = Blueprint('api', __name__)

# Shared predictor instance (loaded once per process)
_predictor = None


def get_predictor() -> RiskPredictor:
    global _predictor
    if _predictor is None:
        _predictor = RiskPredictor()
    return _predictor


# ── Root – serve dashboard UI ─────────────────────────────────────────────────
@api.route('/')
def index():
    static_dir = os.path.join(current_app.root_path, '..', 'static')
    return send_from_directory(os.path.abspath(static_dir), 'index.html')


# ── Health check ─────────────────────────────────────────────────────────────
@api.route('/api/health', methods=['GET'])
def health():
    predictor = get_predictor()
    return jsonify({
        'status':        'healthy',
        'models_loaded': predictor.is_ready(),
        'timestamp':     datetime.datetime.now(datetime.timezone.utc).isoformat(),
    })


# ── Single prediction ─────────────────────────────────────────────────────────
@api.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({'error': 'No JSON body provided'}), 400

        patient_data = data.get('patient_data', data)
        model_type   = data.get('model_type', 'gradient_boosting')

        is_valid, errors = validate_patient_data(patient_data)
        if not is_valid:
            return jsonify({'error': 'Validation failed', 'details': errors}), 422

        predictor = get_predictor()
        if not predictor.is_ready():
            return jsonify({
                'error':   'Models not trained yet.',
                'message': 'Please call POST /api/train first.',
            }), 503

        result = predictor.predict(patient_data, model_type=model_type)
        return jsonify(result)

    except Exception as exc:
        traceback.print_exc()
        return jsonify({'error': str(exc)}), 500


# ── Trigger training ─────────────────────────────────────────────────────────
@api.route('/api/train', methods=['POST'])
def train():
    try:
        # Import here to avoid circular imports and keep startup fast
        from ml.train import train_models

        results = train_models()

        # Reload predictor after training
        global _predictor
        _predictor = RiskPredictor()

        return jsonify({
            'status':  'success',
            'message': 'Models trained and saved successfully.',
            'results': results,
        })

    except Exception as exc:
        traceback.print_exc()
        return jsonify({'error': str(exc)}), 500


# ── Model info ────────────────────────────────────────────────────────────────
@api.route('/api/models/info', methods=['GET'])
def models_info():
    predictor = get_predictor()
    return jsonify(predictor.get_model_info())


# ── Feature importance ────────────────────────────────────────────────────────
@api.route('/api/feature-importance', methods=['GET'])
def feature_importance():
    predictor = get_predictor()
    if not predictor.is_ready():
        return jsonify({'error': 'Models not trained yet.'}), 503
    return jsonify(predictor.get_feature_importance())


# ── Batch prediction ──────────────────────────────────────────────────────────
@api.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    try:
        data = request.get_json(silent=True)
        if not data or 'patients' not in data:
            return jsonify({'error': "Body must contain a 'patients' list."}), 400

        patients   = data['patients']
        model_type = data.get('model_type', 'gradient_boosting')

        predictor = get_predictor()
        if not predictor.is_ready():
            return jsonify({'error': 'Models not trained yet.'}), 503

        predictions = []
        for i, patient in enumerate(patients):
            is_valid, errors = validate_patient_data(patient)
            if not is_valid:
                predictions.append({'index': i, 'error': errors})
                continue
            try:
                result = predictor.predict(patient, model_type=model_type)
                result['index'] = i
                predictions.append(result)
            except Exception as exc:
                predictions.append({'index': i, 'error': str(exc)})

        return jsonify({'predictions': predictions, 'count': len(predictions)})

    except Exception as exc:
        traceback.print_exc()
        return jsonify({'error': str(exc)}), 500
