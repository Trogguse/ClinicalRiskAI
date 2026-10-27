# ClinicalRiskAI 🧬

**Clinical Predictive Analytics & Risk Stratification System**

A production-ready Flask application that uses ensemble machine learning (Gradient Boosting + Random Forest) to stratify patient risk into Low / Medium / High categories based on 21 clinical features.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Trogguse/ClinicalRiskAI)
[![GitHub license](https://img.shields.io/github/license/Trogguse/ClinicalRiskAI)](https://github.com/Trogguse/ClinicalRiskAI)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

---

## 🌐 Live Deployment on Render

Deploy this application live to Render with **1 Click**:

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Trogguse/ClinicalRiskAI)

### Manual Render Setup:
1. Go to [dashboard.render.com](https://dashboard.render.com) and click **New +** > **Web Service**.
2. Connect your GitHub repository: `https://github.com/Trogguse/ClinicalRiskAI`.
3. Configure the settings:
   - **Name**: `clinical-risk-ai`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn run:app`
4. Click **Deploy Web Service**.
5. Once deployed, access the dashboard at: `https://clinical-risk-ai.onrender.com` (or your custom Render service name).

---

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Machine Learning](#machine-learning)
- [Docker](#docker)
- [Tests](#tests)

---

## ✨ Features

- **Two ML models**: Gradient Boosting & Random Forest with hyperparameter tuning via GridSearchCV
- **SMOTE** oversampling to handle class imbalance
- **Beautiful dashboard UI** with real-time risk gauge, probability bars, and clinical recommendations
- **REST API** with 6 endpoints including batch prediction
- **Docker-ready** with `Dockerfile` and `docker-compose.yml`

---

## 📁 Project Structure

```
ClinicalRiskAI/
├── app/
│   ├── __init__.py          # Flask app factory
│   ├── routes.py            # API Blueprint (all endpoints)
│   ├── models/
│   │   └── predictor.py     # RiskPredictor class
│   └── utils/
│       └── preprocessing.py # Validation & feature extraction
├── ml/
│   ├── pipeline.py          # ClinicalRiskPipeline class
│   ├── train.py             # Training workflow
│   └── evaluate.py          # Model evaluation metrics
├── data/
│   └── generate_data.py     # Synthetic data generator
├── models/                  # Saved .pkl artefacts (created at runtime)
├── static/
│   └── index.html           # Dashboard UI
├── tests/
│   └── test_api.py          # Unit tests
├── run.py                   # Application entry point
├── config.py                # Configuration
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## 🚀 Quick Start

### 1. Create and activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the server

```bash
python run.py
```

The server starts at **http://localhost:5000**.

### 4. Train the models

Open your browser at **http://localhost:5000** and click **"Train Models"**, or call the API directly:

```bash
curl -X POST http://localhost:5000/api/train
```

Training takes approximately 2–5 minutes (GridSearchCV with 5-fold CV).  
After training, `gradient_boosting_model.pkl`, `random_forest_model.pkl`, `label_encoder.pkl`, and `feature_names.pkl` will be saved in the `models/` directory.

### 5. Open the dashboard

Navigate to **http://localhost:5000** – the dashboard auto-checks model status and prompts you to train if needed.

---

## 🔌 API Reference

### `GET /api/health`
Returns server and model status.

**Response**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "timestamp": "2024-01-01T00:00:00Z"
}
```

---

### `POST /api/predict`
Predict risk for a single patient.

**Request body**
```json
{
  "age": 55,
  "bmi": 28.5,
  "systolic_bp": 145,
  "diastolic_bp": 90,
  "heart_rate": 80,
  "cholesterol": 220,
  "glucose": 130,
  "hba1c": 7.2,
  "creatinine": 1.2,
  "sodium": 140,
  "potassium": 4.1,
  "wbc": 7.5,
  "hemoglobin": 13.5,
  "platelets": 250,
  "smoking_status": 0,
  "diabetes": 1,
  "hypertension": 1,
  "heart_disease": 0,
  "num_medications": 5,
  "num_hospitalizations": 2,
  "num_comorbidities": 3,
  "model_type": "gradient_boosting"
}
```

**Response**
```json
{
  "risk_level": 1,
  "risk_label": "Medium",
  "confidence": 74.3,
  "probabilities": { "Low": 20.1, "Medium": 74.3, "High": 5.6 },
  "recommendations": ["⚠️ Increased monitoring recommended …"],
  "model_used": "gradient_boosting"
}
```

---

### `POST /api/train`
Trigger model training. Returns training metrics.

---

### `GET /api/models/info`
Returns metadata about loaded models (available models, feature count).

---

### `GET /api/feature-importance`
Returns top-15 feature importances for each loaded model.

---

### `POST /api/batch-predict`
Predict risk for multiple patients.

**Request body**
```json
{
  "patients": [ { … }, { … } ],
  "model_type": "random_forest"
}
```

**Response**
```json
{
  "predictions": [ { "index": 0, "risk_label": "Low", … }, … ],
  "count": 2
}
```

---

## 🤖 Machine Learning

### Features (21 total)

| Category          | Features |
|-------------------|---------|
| Demographics      | age, bmi |
| Vitals            | systolic_bp, diastolic_bp, heart_rate |
| Lab values        | cholesterol, glucose, hba1c, creatinine, sodium, potassium, wbc, hemoglobin, platelets |
| Medical history   | smoking_status, diabetes, hypertension, heart_disease |
| Clinical complexity | num_medications, num_hospitalizations, num_comorbidities |

### Risk Labels

| Level | Label  | Approx. prevalence |
|-------|--------|--------------------|
| 0     | Low    | ~60%               |
| 1     | Medium | ~25%               |
| 2     | High   | ~15%               |

### Pipeline

1. **Data generation** – 5 000 synthetic patients with realistic distributions
2. **Train/test split** – 80 / 20 stratified
3. **SMOTE** – oversample minority classes in the training set
4. **ColumnTransformer** – StandardScaler (numeric) + OneHotEncoder (binary flags)
5. **GridSearchCV** – 5-fold CV, `f1_weighted` scoring
6. **Evaluation** – accuracy, precision, recall, F1, ROC-AUC, confusion matrix

---

## 🐳 Docker

```bash
# Build and start
docker-compose up --build

# Stop
docker-compose down
```

The service runs on port **5000** and mounts `./models` so trained models persist between restarts.

---

## 🧪 Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Or with unittest directly
python -m unittest tests.test_api -v
```

Tests cover health endpoint, root route, prediction validation, batch prediction, and preprocessing utilities.

---

## 📝 License

MIT – see `LICENSE` for details.
