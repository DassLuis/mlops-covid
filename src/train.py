import os
import sys
import joblib
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import load_data, preprocess_country_data
from src.features import create_features
from src.config import (
    URL,
    COUNTRY,
    START_DATE,
    END_DATE,
    HORIZON,
    RANDOM_STATE,
    MODEL_PATH,
    MLFLOW_EXPERIMENT_NAME,
)


def train_model():
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    print("Chargement des données...")
    df = load_data(URL)

    print("Prétraitement des données...")
    df_country = preprocess_country_data(
        df=df,
        country=COUNTRY,
        start_date=START_DATE,
        end_date=END_DATE,
    )

    print("Création des features...")
    df_model, X, y, features = create_features(
        df_country=df_country,
        horizon=HORIZON,
    )

    train_size = int(len(df_model) * 0.8)

    X_train = X.iloc[:train_size]
    X_test = X.iloc[train_size:]

    y_train = y.iloc[:train_size]
    y_test = y.iloc[train_size:]

    n_estimators = 100

    print("Entraînement du modèle...")
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    with mlflow.start_run():
        mlflow.log_param("country", COUNTRY)
        mlflow.log_param("start_date", START_DATE)
        mlflow.log_param("end_date", END_DATE)
        mlflow.log_param("horizon", HORIZON)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("train_size", train_size)
        mlflow.log_param("features", features)

        model.fit(X_train, y_train)

        print("Évaluation du modèle...")
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f"MAE : {mae:,.2f}")
        print(f"RMSE : {rmse:,.2f}")

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)

        os.makedirs("models", exist_ok=True)

        model_package = {
            "model": model,
            "features": features,
            "country": COUNTRY,
            "horizon": HORIZON,
            "mae": mae,
            "rmse": rmse,
        }

        joblib.dump(model_package, MODEL_PATH)

        # mlflow.log_artifact(MODEL_PATH)
        #mlflow.sklearn.log_model(model, name="random_forest_model")

        print(f"Modèle sauvegardé dans {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
