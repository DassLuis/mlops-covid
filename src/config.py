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
