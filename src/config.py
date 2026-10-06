URL = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"

COUNTRY = "France"

START_DATE = "2020-03-01"
END_DATE = "2023-12-31"

HORIZON = 7

TRAIN_RATIO = 0.8
N_ESTIMATORS = 100
RANDOM_STATE = 42

RAW_DATA_PATH = "data/raw/owid_covid.csv"
PROCESSED_DATA_PATH = "data/processed/covid_features.csv"

MODEL_PATH = "models/covid_rf_model.joblib"
METRICS_PATH = "metrics/train_metrics.json"

MLFLOW_EXPERIMENT_NAME = "covid_prediction_experiment"

# Registre de modÃ¨les MLflow
MLFLOW_MODEL_NAME = "covid_rf_model"
ALIAS_CANDIDATE = "candidate"
ALIAS_PRODUCTION = "production"

PROMOTION_PATH = "metrics/promotion.json"

# Monitoring en production
PREDICTION_LOG_PATH = "data/monitoring/predictions.csv"
MONITORING_SUMMARY_PATH = "data/monitoring/monitoring_summary.json"
MONITORING_REPORTS_DIR = "data/monitoring/reports"
MLFLOW_MONITORING_EXPERIMENT_NAME = "covid_monitoring"

MONITORING_WINDOW = 500          # dernières prédictions analysées
MONITORING_MIN_ROWS = 30         # en dessous, pas de verdict de dérive
DRIFT_SHARE_THRESHOLD = 0.5      # part de variables en dérive qui déclenche l'alerte
PERFORMANCE_DEGRADATION_FACTOR = 1.5   # MAE relative en production / à l'entraînement
