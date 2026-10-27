import numpy as np
import pandas as pd


def generate_data(n_samples=5000, random_state=42):
    """
    Generate realistic synthetic clinical data for risk stratification.

    Returns a pandas DataFrame with clinical features and a risk_level target
    (0=Low, 1=Medium, 2=High) with approximate class distribution of 60/25/15%.
    """
    rng = np.random.RandomState(random_state)

    # ── Demographics ──────────────────────────────────────────────────────────
    age = rng.uniform(20, 90, n_samples)
    bmi = rng.uniform(15, 45, n_samples)

    # ── Vitals ────────────────────────────────────────────────────────────────
    systolic_bp = rng.uniform(90, 200, n_samples)
    diastolic_bp = rng.uniform(60, 120, n_samples)
    heart_rate = rng.uniform(50, 120, n_samples)

    # ── Lab Values ────────────────────────────────────────────────────────────
    cholesterol = rng.uniform(150, 300, n_samples)
    glucose = rng.uniform(70, 300, n_samples)
    hba1c = rng.uniform(4.0, 12.0, n_samples)
    creatinine = rng.uniform(0.5, 5.0, n_samples)
    sodium = rng.uniform(135, 145, n_samples)
    potassium = rng.uniform(3.5, 5.5, n_samples)
    wbc = rng.uniform(4.0, 11.0, n_samples)
    hemoglobin = rng.uniform(8.0, 18.0, n_samples)
    platelets = rng.uniform(150, 400, n_samples)

    # ── Medical History (binary) ───────────────────────────────────────────────
    smoking_status = rng.binomial(1, 0.30, n_samples)
    diabetes = rng.binomial(1, 0.20, n_samples)
    hypertension = rng.binomial(1, 0.35, n_samples)
    heart_disease = rng.binomial(1, 0.15, n_samples)

    # ── Clinical Complexity ───────────────────────────────────────────────────
    num_medications = rng.randint(0, 16, n_samples)
    num_hospitalizations = rng.randint(0, 11, n_samples)
    num_comorbidities = rng.randint(0, 9, n_samples)

    # ── Risk Score (used to derive target) ────────────────────────────────────
    # Combines clinical signals linearly then adds noise so the label is
    # meaningful but not deterministic.
    risk_score = (
        0.03 * age
        + 0.05 * bmi
        + 0.02 * systolic_bp
        + 0.01 * diastolic_bp
        + 0.01 * heart_rate
        + 0.02 * cholesterol
        + 0.03 * glucose
        + 0.5  * hba1c
        + 0.3  * creatinine
        + 0.2  * smoking_status
        + 0.5  * diabetes
        + 0.3  * hypertension
        + 0.8  * heart_disease
        + 0.2  * num_medications
        + 0.4  * num_hospitalizations
        + 0.3  * num_comorbidities
        + rng.normal(0, 2, n_samples)  # stochastic noise
    )

    # Normalise to [0, 1]
    score_min, score_max = risk_score.min(), risk_score.max()
    risk_score_norm = (risk_score - score_min) / (score_max - score_min)

    # Threshold to achieve ≈ 60 / 25 / 15 class split
    risk_level = np.zeros(n_samples, dtype=int)
    risk_level[risk_score_norm >= 0.60] = 1   # Medium
    risk_level[risk_score_norm >= 0.85] = 2   # High

    df = pd.DataFrame({
        'age': age,
        'bmi': bmi,
        'systolic_bp': systolic_bp,
        'diastolic_bp': diastolic_bp,
        'heart_rate': heart_rate,
        'cholesterol': cholesterol,
        'glucose': glucose,
        'hba1c': hba1c,
        'creatinine': creatinine,
        'sodium': sodium,
        'potassium': potassium,
        'wbc': wbc,
        'hemoglobin': hemoglobin,
        'platelets': platelets,
        'smoking_status': smoking_status,
        'diabetes': diabetes,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'num_medications': num_medications,
        'num_hospitalizations': num_hospitalizations,
        'num_comorbidities': num_comorbidities,
        'risk_level': risk_level,
    })

    return df


if __name__ == '__main__':
    df = generate_data()
    print(df.head())
    print("\nClass distribution:")
    print(df['risk_level'].value_counts(normalize=True).sort_index())
