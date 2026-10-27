import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)

from ml.pipeline import NUMERIC_FEATURES, CATEGORICAL_FEATURES

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained sklearn Pipeline on held-out data.

    Parameters
    ----------
    model   : fitted sklearn Pipeline
    X_test  : pandas DataFrame with the correct feature columns
    y_test  : array-like of integer labels

    Returns
    -------
    dict with accuracy, precision, recall, f1, confusion_matrix,
         roc_auc, feature_importances, classification_report
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc  = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
    rec  = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
    f1   = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
    cm   = confusion_matrix(y_test, y_pred).tolist()

    # ROC-AUC (macro OvR)
    try:
        roc_auc = float(roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro'))
    except Exception:
        roc_auc = 0.0

    # Feature importances (top 10) ─────────────────────────────────────────────
    feature_importances = {}
    try:
        classifier = model.named_steps['classifier']
        importances = classifier.feature_importances_

        # The ColumnTransformer expands categorical features via OHE,
        # so we use the preprocessor's get_feature_names_out() to align.
        preprocessor = model.named_steps['preprocessor']
        try:
            feature_out_names = list(preprocessor.get_feature_names_out())
        except Exception:
            feature_out_names = [f"f{i}" for i in range(len(importances))]

        pairs = sorted(
            zip(feature_out_names, importances),
            key=lambda x: x[1],
            reverse=True,
        )[:10]
        feature_importances = {k: round(float(v), 6) for k, v in pairs}
    except Exception:
        pass

    # Classification report as dict ───────────────────────────────────────────
    clf_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    # Convert numpy floats to plain Python floats for JSON serialisation
    clf_report_clean = {}
    for k, v in clf_report.items():
        if isinstance(v, dict):
            clf_report_clean[str(k)] = {kk: float(vv) for kk, vv in v.items()}
        else:
            clf_report_clean[str(k)] = float(v)

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'confusion_matrix': cm,
        'roc_auc': roc_auc,
        'feature_importances': feature_importances,
        'classification_report': clf_report_clean,
    }
