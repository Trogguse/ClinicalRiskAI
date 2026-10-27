"""
Input validation and feature extraction utilities.

The feature order defined here MUST match the column order used in
data/generate_data.py and ml/pipeline.py.
"""

import pandas as pd

# ── Feature order (must match generate_data.py) ───────────────────────────────
NUMERIC_FEATURES = [
    'age', 'bmi', 'systolic_bp', 'diastolic_bp', 'heart_rate',
    'cholesterol', 'glucose', 'hba1c', 'creatinine',
    'sodium', 'potassium', 'wbc', 'hemoglobin', 'platelets',
    'num_medications', 'num_hospitalizations', 'num_comorbidities',
]

CATEGORICAL_FEATURES = [
    'smoking_status', 'diabetes', 'hypertension', 'heart_disease',
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ── Valid ranges for each feature ─────────────────────────────────────────────
FEATURE_RANGES = {
    'age':                  (20,   90),
    'bmi':                  (10,   60),
    'systolic_bp':          (60,  250),
    'diastolic_bp':         (40,  150),
    'heart_rate':           (30,  200),
    'cholesterol':          (100, 400),
    'glucose':              (40,  600),
    'hba1c':                (3.0, 15.0),
    'creatinine':           (0.2, 15.0),
    'sodium':               (120, 160),
    'potassium':            (2.0,  8.0),
    'wbc':                  (1.0, 30.0),
    'hemoglobin':           (5.0, 22.0),
    'platelets':            (50,  800),
    'num_medications':      (0,   20),
    'num_hospitalizations': (0,   20),
    'num_comorbidities':    (0,   15),
    'smoking_status':       (0,    1),
    'diabetes':             (0,    1),
    'hypertension':         (0,    1),
    'heart_disease':        (0,    1),
}


def validate_patient_data(data: dict):
    """
    Validate patient data dict.

    Returns
    -------
    (is_valid: bool, errors: list[str])
    """
    errors = []

    # Check required fields
    for feature in ALL_FEATURES:
        if feature not in data:
            errors.append(f"Missing required field: '{feature}'")

    if errors:
        return False, errors

    # Check types and ranges
    for feature in ALL_FEATURES:
        try:
            value = float(data[feature])
        except (TypeError, ValueError):
            errors.append(f"Field '{feature}' must be a number, got: {data[feature]!r}")
            continue

        lo, hi = FEATURE_RANGES[feature]
        if not (lo <= value <= hi):
            errors.append(
                f"Field '{feature}' value {value} is outside expected range [{lo}, {hi}]"
            )

    return len(errors) == 0, errors


def extract_features(patient_data: dict) -> pd.DataFrame:
    """
    Convert a patient data dict to a single-row pandas DataFrame
    with columns in the exact order expected by the trained pipeline.
    """
    row = {feature: float(patient_data[feature]) for feature in ALL_FEATURES}
    return pd.DataFrame([row], columns=ALL_FEATURES)
