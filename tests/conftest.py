import os
import tempfile

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.config import COUNTRY, HORIZON, MODEL_PATH, RANDOM_STATE
from src.features import FEATURES

TEST_API_KEY = "cle-reservee-aux-tests"

# Les tests ne doivent ni dépendre du serveur MLflow, ni écrire dans le journal
# de production, ni connaître la vraie clé : ces variables sont fixées avant
# l'import de l'API.
os.environ["MODEL_SOURCE"] = "local"
os.environ["API_KEY"] = TEST_API_KEY
os.environ["PREDICTION_LOG_PATH"] = os.path.join(
    tempfile.mkdtemp(prefix="covid_tests_"), "predictions.csv"
)


def build_stand_in_model():
    """Le modèle est suivi par DVC, pas par git : sur une machine qui n'a que le
    code (intégration continue), un petit modèle de remplacement suffit à tester
    l'API, qui ne vérifie pas la valeur des prédictions."""
    rng = np.random.default_rng(RANDOM_STATE)
    X = pd.DataFrame(rng.uniform(0, 1000, size=(60, len(FEATURES))), columns=FEATURES)
    y = X["lag_7"] * 0.9

    model = RandomForestRegressor(n_estimators=5, random_state=RANDOM_STATE).fit(X, y)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "features": list(FEATURES),
            "country": COUNTRY,
            "horizon": HORIZON,
            "mae": 0.0,
            "rmse": 0.0,
        },
        MODEL_PATH,
    )


if not os.path.exists(MODEL_PATH):
    build_stand_in_model()
