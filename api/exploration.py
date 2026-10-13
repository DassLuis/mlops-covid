"""Données d'exploration servies à l'application Streamlit.

L'application ne lit aucun fichier : la série, le jeu de test, les statistiques
et les essais d'entraînement passent tous par l'API. Les chiffres sont calculés
avec le pipeline du projet et le modèle réellement servi.
"""
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import (
    COUNTRY,
    END_DATE,
    HORIZON,
    METRICS_PATH,
    MONITORING_SUMMARY_PATH,
    PROMOTION_PATH,
    RANDOM_STATE,
    RAW_DATA_PATH,
    START_DATE,
    TRAIN_RATIO,
)
from src.features import TARGET, create_features
from src.preprocess import preprocess_country_data

_cache = {}


class DataUnavailable(RuntimeError):
    """Les données brutes n'ont pas encore été téléchargées par le pipeline."""


def scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(
            1 - ((y_true - y_pred) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum()
        ),
    }


def load_features(raw_path: str = None):
    raw_path = raw_path or RAW_DATA_PATH

    if not os.path.exists(raw_path):
        raise DataUnavailable(
            f"{raw_path} est absent : lancer le pipeline (dvc repro) d'abord."
        )

    raw = pd.read_csv(raw_path)
    df_country = preprocess_country_data(raw, COUNTRY, START_DATE, END_DATE)
    df_model, X, y, features = create_features(df_country, HORIZON)

    return len(raw), len(df_country), df_model.reset_index(drop=True), features


def records(df: pd.DataFrame) -> list:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    return json.loads(out.to_json(orient="records"))


def build_exploration(model_package: dict, raw_path: str = None) -> dict:
    n_raw, n_country, df_model, features = load_features(raw_path)

    model = model_package["model"]
    dates = pd.to_datetime(df_model["date"])
    X, y = df_model[features], df_model[TARGET]

    n = len(df_model)
    train_size = int(n * TRAIN_RATIO)

    X_test = X.iloc[train_size:]
    y_test = y.iloc[train_size:].to_numpy()
    y_pred = model.predict(X_test)

    error = y_pred - y_test
    abs_error = np.abs(error)
    published = y_test > 0
    naive = X_test["lag_1"].to_numpy()

    series = pd.DataFrame({"date": dates, "new_cases": df_model["new_cases"]})

    test = X_test.reset_index(drop=True).copy()
    test.insert(0, "date", dates.iloc[train_size:].reset_index(drop=True))
    test["y_true"] = y_test
    test["y_pred"] = y_pred
    test["erreur"] = error

    stats = {
        "n_raw": int(n_raw),
        "n_country": int(n_country),
        "n_total": int(n),
        "n_train": int(train_size),
        "n_test": int(n - train_size),
        "date_min": str(dates.min().date()),
        "date_max": str(dates.max().date()),
        "split_date": str(dates.iloc[train_size].date()),
        "country": COUNTRY,
        "horizon": HORIZON,
        "random_state": RANDOM_STATE,
        "n_estimators": int(getattr(model, "n_estimators", 0)),
        "model_version": model_package["version"],
        **scores(y_test, y_pred),
        "zero_share_all": float((series["new_cases"] == 0).mean() * 100),
        "n_zeros_all": int((series["new_cases"] == 0).sum()),
        "n_zero_days_test": int((~published).sum()),
        "n_publication_days_test": int(published.sum()),
        "mae_zero_days": float(abs_error[~published].mean()),
        "mae_nz": float(abs_error[published].mean()),
        "rmse_nz": float(np.sqrt((error[published] ** 2).mean())),
        "mape_nz": float(np.mean(np.abs(error[published] / y_test[published])) * 100),
        "sse_share_nz": float((error[published] ** 2).sum() / (error ** 2).sum() * 100),
        "mae_naive": float(np.mean(np.abs(naive - y_test))),
        "rmse_naive": float(np.sqrt(np.mean((naive - y_test) ** 2))),
        "mae_zero": float(np.mean(np.abs(y_test))),
        "peak_all": float(series["new_cases"].max()),
        "peak_date": str(dates.iloc[int(series["new_cases"].idxmax())].date()),
        "last_publication": str(test.loc[published, "date"].max().date()),
        "trailing_zeros": int(len(y_test) - 1 - np.where(published)[0][-1]),
        "features": list(features),
        "importances": {
            f: float(v) for f, v in zip(features, model.feature_importances_)
        },
    }

    return {
        "stats": stats,
        "series": records(series),
        "test": records(test),
    }


def get_exploration(model_package: dict, raw_path: str = None) -> dict:
    """Calcul fait une fois, puis refait si le modèle servi ou les données changent."""
    raw_path = raw_path or RAW_DATA_PATH
    key = (
        model_package["version"],
        raw_path,
        os.path.getmtime(raw_path) if os.path.exists(raw_path) else None,
    )

    if _cache.get("key") != key:
        _cache["value"] = build_exploration(model_package, raw_path)
        _cache["key"] = key

    return _cache["value"]


def trial_training(
    n_estimators: int,
    train_share: float,
    max_depth,
    raw_path: str = None,
) -> dict:
    """Essai d'entraînement avec d'autres réglages. Le modèle obtenu n'est ni
    enregistré ni servi : il ne sert qu'à comparer ses métriques à la référence."""
    _, _, df_model, features = load_features(raw_path)

    X, y = df_model[features], df_model[TARGET]
    train_size = int(len(df_model) * train_share)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    start = time.time()
    model.fit(X.iloc[:train_size], y.iloc[:train_size])
    y_pred = model.predict(X.iloc[train_size:])
    dates = pd.to_datetime(df_model["date"])

    return {
        **scores(y.iloc[train_size:].to_numpy(), y_pred),
        "n_train": int(train_size),
        "n_test": int(len(df_model) - train_size),
        "last_train_date": str(dates.iloc[train_size - 1].date()),
        "duration_seconds": round(time.time() - start, 2),
    }


def read_json(path: str):
    if not os.path.exists(path):
        return None

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def mlops_status() -> dict:
    """Derniers résultats écrits par le pipeline : entraînement, promotion, monitoring."""
    return {
        "entrainement": read_json(METRICS_PATH),
        "promotion": read_json(PROMOTION_PATH),
        "monitoring": read_json(MONITORING_SUMMARY_PATH),
    }
