import joblib
import pandas as pd

from src.config import MODEL_PATH


def load_model(model_path: str = MODEL_PATH):
    model_package = joblib.load(model_path)
    return model_package


def make_prediction(input_data: dict, model_package: dict):
    model = model_package["model"]
    features = model_package["features"]

    input_df = pd.DataFrame([input_data])
    input_df = input_df[features]

    prediction = model.predict(input_df)[0]

    return float(prediction)