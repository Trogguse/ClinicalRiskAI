import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Allow running this file directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import Config
from data.generate_data import generate_data
from ml.pipeline import ClinicalRiskPipeline, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from ml.evaluate import evaluate_model


ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def train_models():
    """
    Full training workflow:
    1. Generate / load synthetic clinical data
    2. Split into train / test
    3. Apply SMOTE on training set
    4. Train Gradient Boosting and Random Forest with GridSearchCV
    5. Persist models and metadata
    6. Return a dict with training results
    """
    print("=" * 60)
    print("Clinical Risk AI – Model Training")
    print("=" * 60)

    os.makedirs(Config.MODEL_DIR, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\n[1/5] Generating synthetic clinical data …")
    df = generate_data(n_samples=Config.N_SAMPLES, random_state=Config.RANDOM_STATE)

    feature_names = ALL_FEATURES
    X = df[feature_names].values
    y = df['risk_level'].values

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    print(f"  Dataset shape : {df.shape}")
    print(f"  Class counts  : {dict(zip(le.classes_, np.bincount(y_enc)))}")

    # ── Train / test split ────────────────────────────────────────────────────
    print("\n[2/5] Splitting data …")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc,
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_STATE,
        stratify=y_enc,
    )

    # ── SMOTE ─────────────────────────────────────────────────────────────────
    print("\n[3/5] Applying SMOTE …")
    pipeline_obj = ClinicalRiskPipeline()
    X_train_res, y_train_res = pipeline_obj.apply_smote(X_train, y_train)
    print(f"  After SMOTE : {X_train_res.shape[0]} samples")

    # Convert back to DataFrame so the ColumnTransformer can use column names
    X_train_df = pd.DataFrame(X_train_res, columns=feature_names)
    X_test_df  = pd.DataFrame(X_test,      columns=feature_names)

    results = {}

    # ── Gradient Boosting ─────────────────────────────────────────────────────
    print("\n[4/5] Training Gradient Boosting (GridSearchCV) …")
    gb_pipeline = pipeline_obj.build_gradient_boosting()
    gb_best = pipeline_obj.optimize_hyperparameters(
        gb_pipeline, X_train_df, y_train_res, model_type='gradient_boosting'
    )
    pipeline_obj.save_model(gb_best, 'gradient_boosting_model')

    gb_metrics = evaluate_model(gb_best, X_test_df, y_test)
    results['gradient_boosting'] = gb_metrics
    print(f"  GB Accuracy : {gb_metrics['accuracy']:.4f}")
    print(f"  GB ROC-AUC  : {gb_metrics['roc_auc']:.4f}")

    # ── Random Forest ─────────────────────────────────────────────────────────
    print("\n[4/5] Training Random Forest (GridSearchCV) …")
    rf_pipeline = pipeline_obj.build_random_forest()
    rf_best = pipeline_obj.optimize_hyperparameters(
        rf_pipeline, X_train_df, y_train_res, model_type='random_forest'
    )
    pipeline_obj.save_model(rf_best, 'random_forest_model')

    rf_metrics = evaluate_model(rf_best, X_test_df, y_test)
    results['random_forest'] = rf_metrics
    print(f"  RF Accuracy : {rf_metrics['accuracy']:.4f}")
    print(f"  RF ROC-AUC  : {rf_metrics['roc_auc']:.4f}")

    # ── Persist metadata ──────────────────────────────────────────────────────
    print("\n[5/5] Saving metadata …")
    pipeline_obj.save_model(le, 'label_encoder')
    pipeline_obj.save_model(feature_names, 'feature_names')

    print("\n[OK] Training complete!")
    return results


if __name__ == '__main__':
    train_models()
