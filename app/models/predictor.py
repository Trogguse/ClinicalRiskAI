"""
RiskPredictor – loads saved models and produces risk predictions with
clinical recommendations.
"""

import os

from config import Config
from ml.pipeline import ClinicalRiskPipeline
from app.utils.preprocessing import extract_features


RISK_LABELS = {0: 'Low', 1: 'Medium', 2: 'High'}


class RiskPredictor:
    """Wraps saved ML models and provides a unified prediction interface."""

    def __init__(self):
        self._pipeline = ClinicalRiskPipeline()
        self._models = {}
        self._label_encoder = None
        self._feature_names = None
        self._load_models()

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _load_models(self):
        """Attempt to load all persisted model artefacts; fail silently."""
        try:
            self._models['gradient_boosting'] = self._pipeline.load_model('gradient_boosting_model')
            self._models['random_forest']     = self._pipeline.load_model('random_forest_model')
            self._label_encoder               = self._pipeline.load_model('label_encoder')
            self._feature_names               = self._pipeline.load_model('feature_names')
        except FileNotFoundError:
            self._models = {}

    def is_ready(self) -> bool:
        """Return True if at least the gradient boosting model is loaded."""
        return 'gradient_boosting' in self._models and 'random_forest' in self._models

    # ── Public API ────────────────────────────────────────────────────────────
    def predict(self, patient_data: dict, model_type: str = 'gradient_boosting') -> dict:
        """
        Predict clinical risk for a single patient.

        Parameters
        ----------
        patient_data : dict  – validated patient feature values
        model_type   : str   – 'gradient_boosting' | 'random_forest'

        Returns
        -------
        dict with risk_level, risk_label, confidence, probabilities, recommendations
        """
        if not self.is_ready():
            # Reload in case training just completed
            self._load_models()
            if not self.is_ready():
                raise RuntimeError("Models are not loaded. Please train models first.")

        if model_type not in self._models:
            model_type = 'gradient_boosting'

        model = self._models[model_type]
        X_df  = extract_features(patient_data)

        risk_level_enc = int(model.predict(X_df)[0])
        probabilities  = model.predict_proba(X_df)[0].tolist()

        # Map encoded label back if LabelEncoder is available
        if self._label_encoder is not None:
            risk_level = int(self._label_encoder.inverse_transform([risk_level_enc])[0])
        else:
            risk_level = risk_level_enc

        confidence    = round(max(probabilities) * 100, 2)
        risk_label    = RISK_LABELS.get(risk_level, 'Unknown')

        # Map probability indices to human labels
        n_classes = len(probabilities)
        if self._label_encoder is not None:
            class_labels = [
                RISK_LABELS.get(int(c), str(c))
                for c in self._label_encoder.classes_[:n_classes]
            ]
        else:
            class_labels = [RISK_LABELS.get(i, str(i)) for i in range(n_classes)]

        prob_dict = {label: round(p * 100, 2) for label, p in zip(class_labels, probabilities)}

        recommendations = self.get_recommendations(risk_level, patient_data)

        return {
            'risk_level':       risk_level,
            'risk_label':       risk_label,
            'confidence':       confidence,
            'probabilities':    prob_dict,
            'recommendations':  recommendations,
            'model_used':       model_type,
        }

    def get_recommendations(self, risk_level: int, patient_data: dict) -> list:
        """Return a list of clinical recommendation strings based on risk and patient features."""
        recs = []

        if risk_level == 0:  # Low
            recs.append("✅ Maintain current healthy lifestyle habits.")
            recs.append("📅 Schedule routine annual check-up.")
            recs.append("🏃 Continue regular physical activity (150 min/week).")
            recs.append("🥗 Sustain a balanced, nutrient-rich diet.")

        elif risk_level == 1:  # Medium
            recs.append("⚠️ Increased monitoring recommended – follow up in 3–6 months.")
            recs.append("📊 Consider lifestyle modifications to reduce risk factors.")
            if float(patient_data.get('bmi', 0)) > 30:
                recs.append("⚖️ Weight management programme advised (BMI > 30).")
            if float(patient_data.get('systolic_bp', 0)) > 140:
                recs.append("💊 Blood pressure review and possible medication adjustment.")
            if float(patient_data.get('glucose', 0)) > 126:
                recs.append("🩸 Fasting glucose elevated – diabetes screening recommended.")
            if int(float(patient_data.get('smoking_status', 0))) == 1:
                recs.append("🚭 Smoking cessation programme strongly advised.")
            recs.append("🏋️ Increase physical activity; target 30 min/day moderate exercise.")
            recs.append("🫀 Cardiology referral if symptoms develop.")

        else:  # High
            recs.append("🚨 HIGH RISK – urgent clinical evaluation required.")
            recs.append("🏥 Consider hospital admission or urgent specialist referral.")
            recs.append("💊 Comprehensive medication review by attending physician.")
            if float(patient_data.get('creatinine', 0)) > 1.5:
                recs.append("🫘 Elevated creatinine – nephrology consultation recommended.")
            if float(patient_data.get('hba1c', 0)) > 7.5:
                recs.append("🩸 Poor glycaemic control – endocrinology review needed.")
            if float(patient_data.get('systolic_bp', 0)) > 160:
                recs.append("❤️ Severe hypertension – immediate antihypertensive therapy.")
            if int(float(patient_data.get('heart_disease', 0))) == 1:
                recs.append("🫀 Known heart disease – cardiology follow-up within 1 week.")
            if int(float(patient_data.get('smoking_status', 0))) == 1:
                recs.append("🚭 Smoking cessation is critical at this risk level.")
            recs.append("📋 Develop a comprehensive care management plan.")
            recs.append("🔬 Complete laboratory panel and imaging as indicated.")

        return recs

    def get_model_info(self) -> dict:
        """Return metadata about loaded models."""
        info = {'models_loaded': self.is_ready()}
        if self.is_ready():
            info['available_models'] = list(self._models.keys())
            info['feature_count']    = len(self._feature_names) if self._feature_names else 0
        return info

    def get_feature_importance(self) -> dict:
        """Return feature importances for all loaded models."""
        result = {}
        for name, model in self._models.items():
            try:
                classifier    = model.named_steps['classifier']
                preprocessor  = model.named_steps['preprocessor']
                importances   = classifier.feature_importances_
                try:
                    feature_names_out = list(preprocessor.get_feature_names_out())
                except Exception:
                    feature_names_out = [f"f{i}" for i in range(len(importances))]

                pairs = sorted(
                    zip(feature_names_out, importances),
                    key=lambda x: x[1],
                    reverse=True,
                )[:15]
                result[name] = {k: round(float(v), 6) for k, v in pairs}
            except Exception as exc:
                result[name] = {'error': str(exc)}
        return result
