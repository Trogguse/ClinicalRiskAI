import os
import joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

from config import Config


# ── Feature column definitions ────────────────────────────────────────────────
NUMERIC_FEATURES = [
    'age', 'bmi', 'systolic_bp', 'diastolic_bp', 'heart_rate',
    'cholesterol', 'glucose', 'hba1c', 'creatinine',
    'sodium', 'potassium', 'wbc', 'hemoglobin', 'platelets',
    'num_medications', 'num_hospitalizations', 'num_comorbidities',
]

CATEGORICAL_FEATURES = [
    'smoking_status', 'diabetes', 'hypertension', 'heart_disease',
]


class ClinicalRiskPipeline:
    """End-to-end ML pipeline for clinical risk stratification."""

    def __init__(self):
        os.makedirs(Config.MODEL_DIR, exist_ok=True)

    # ── Preprocessor ──────────────────────────────────────────────────────────
    def build_preprocessor(self):
        """Return a ColumnTransformer that scales numeric and encodes categorical features."""
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler()),
        ])
        categorical_transformer = Pipeline(steps=[
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ])
        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES),
        ])
        return preprocessor

    # ── Model builders ────────────────────────────────────────────────────────
    def build_gradient_boosting(self):
        """Return a scikit-learn Pipeline with preprocessor + GradientBoostingClassifier."""
        return Pipeline(steps=[
            ('preprocessor', self.build_preprocessor()),
            ('classifier', GradientBoostingClassifier(random_state=Config.RANDOM_STATE)),
        ])

    def build_random_forest(self):
        """Return a scikit-learn Pipeline with preprocessor + RandomForestClassifier."""
        return Pipeline(steps=[
            ('preprocessor', self.build_preprocessor()),
            ('classifier', RandomForestClassifier(
                random_state=Config.RANDOM_STATE,
            )),
        ])

    # ── SMOTE ─────────────────────────────────────────────────────────────────
    def apply_smote(self, X, y):
        """
        Apply SMOTE to the *raw* (pre-preprocessing) feature matrix to
        address class imbalance.  Returns resampled X, y as numpy arrays.
        """
        smote = SMOTE(random_state=Config.RANDOM_STATE)
        X_res, y_res = smote.fit_resample(X, y)
        return X_res, y_res

    # ── Hyperparameter optimisation ───────────────────────────────────────────
    def optimize_hyperparameters(self, pipeline, X, y, model_type='gradient_boosting'):
        """
        Run GridSearchCV over the given pipeline.

        Parameters
        ----------
        model_type : str  'gradient_boosting' | 'random_forest'

        Returns
        -------
        best estimator (fitted Pipeline)
        """
        if model_type == 'gradient_boosting':
            param_grid = {
                'classifier__n_estimators': [80, 120],
                'classifier__learning_rate': [0.1],
                'classifier__max_depth': [3, 4],
            }
        else:  # random_forest
            param_grid = {
                'classifier__n_estimators': [80, 120],
                'classifier__max_depth': [10, 15],
                'classifier__min_samples_split': [2, 5],
            }

        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=Config.CV_FOLDS,
            scoring='f1_weighted',
            n_jobs=4,
            verbose=1,
            refit=True,
        )
        grid_search.fit(X, y)
        print(f"[{model_type}] Best params: {grid_search.best_params_}")
        return grid_search.best_estimator_

    # ── Persistence ───────────────────────────────────────────────────────────
    def save_model(self, model, name: str):
        """Persist a model (or any object) to the models directory using joblib."""
        path = os.path.join(Config.MODEL_DIR, f"{name}.pkl")
        joblib.dump(model, path)
        print(f"Saved model to {path}")
        return path

    def load_model(self, name: str):
        """Load a previously saved model from the models directory."""
        path = os.path.join(Config.MODEL_DIR, f"{name}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        return joblib.load(path)
