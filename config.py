import os


class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'clinical-risk-ai-secret-2024')
    MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
    DEBUG = os.environ.get('DEBUG', 'True') == 'True'
    HOST = os.environ.get('HOST', '0.0.0.0')
    PORT = int(os.environ.get('PORT', 5000))

    # ML Config
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    CV_FOLDS = 3
    N_SAMPLES = 2500
