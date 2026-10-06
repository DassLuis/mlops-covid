"""Prépare les données consommées par l'application Streamlit.

Rejoue exactement le pipeline du projet (src/preprocess, src/features, mêmes
hyperparamètres que src/train.py) et fige le résultat dans presentation/data/,
pour que la présentation démarre instantanément et fonctionne hors ligne.

Usage :
    python presentation/build_data.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.preprocess import load_data, preprocess_country_data  # noqa: E402
from src.features import create_features  # noqa: E402
from src.config import (  # noqa: E402
    URL, COUNTRY, START_DATE, END_DATE, HORIZON, RANDOM_STATE,
)

DATA_DIR = os.path.join(ROOT, "presentation", "data")
RAW_CACHE = os.path.join(DATA_DIR, "owid_raw.parquet")


def get_raw() -> pd.DataFrame:
    if os.path.exists(RAW_CACHE):
        print("Cache OWID trouvé.")
        return pd.read_parquet(RAW_CACHE)
    print("Téléchargement des données OWID (une seule fois)...")
    df = load_data(URL)
    df.to_parquet(RAW_CACHE, index=False)
    print(f"  {len(df):,} lignes téléchargées.".replace(",", " "))
    return df


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    raw = get_raw()
    n_raw = len(raw)

    df_country = preprocess_country_data(raw, COUNTRY, START_DATE, END_DATE)
    df_model, X, y, features = create_features(df_country, HORIZON)

    dates = pd.to_datetime(df_model["date"]).reset_index(drop=True)
    n = len(df_model)
    train_size = int(n * 0.8)

    X_tr, X_te = X.iloc[:train_size], X.iloc[train_size:]
    y_tr, y_te = y.iloc[:train_size], y.iloc[train_size:]

    model = RandomForestRegressor(
        n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
    )
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    yte = y_te.to_numpy()

    mae = float(mean_absolute_error(yte, pred))
    rmse = float(np.sqrt(mean_squared_error(yte, pred)))
    err = pred - yte
    abse = np.abs(err)
    r2 = float(1 - ((yte - pred) ** 2).sum() / ((yte - yte.mean()) ** 2).sum())

    nz = yte > 0
    naive = X_te["lag_1"].to_numpy()

    # ---- série complète (graphiques 1 et 2) ----
    serie = pd.DataFrame({"date": dates, "new_cases": df_model["new_cases"].to_numpy()})
    serie.to_parquet(os.path.join(DATA_DIR, "serie.parquet"), index=False)

    # ---- jeu de test complet : entrées, cible, prédiction ----
    test = X_te.reset_index(drop=True).copy()
    test.insert(0, "date", dates[train_size:].reset_index(drop=True))
    test["y_true"] = yte
    test["y_pred"] = pred
    test["erreur"] = err
    test.to_parquet(os.path.join(DATA_DIR, "test.parquet"), index=False)

    # ---- jeu d'entraînement, pour le réentraînement en direct ----
    train = X.reset_index(drop=True).copy()
    train.insert(0, "date", dates)
    train["target"] = y.to_numpy()
    train.to_parquet(os.path.join(DATA_DIR, "modele_complet.parquet"), index=False)

    stats = {
        "n_raw": int(n_raw),
        "n_country": int(len(df_country)),
        "n_total": int(n),
        "n_train": int(train_size),
        "n_test": int(n - train_size),
        "date_min": str(dates.min().date()),
        "date_max": str(dates.max().date()),
        "split_date": str(dates[train_size].date()),
        "country": COUNTRY,
        "horizon": HORIZON,
        "random_state": RANDOM_STATE,
        "n_estimators": 100,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "zero_share_all": float((serie["new_cases"] == 0).mean() * 100),
        "n_zeros_all": int((serie["new_cases"] == 0).sum()),
        "n_zero_days_test": int((~nz).sum()),
        "n_publication_days_test": int(nz.sum()),
        "mae_zero_days": float(abse[~nz].mean()),
        "mae_nz": float(abse[nz].mean()),
        "rmse_nz": float(np.sqrt((err[nz] ** 2).mean())),
        "mape_nz": float(np.mean(np.abs(err[nz] / yte[nz])) * 100),
        "sse_share_nz": float((err[nz] ** 2).sum() / (err ** 2).sum() * 100),
        "mae_naive": float(np.mean(np.abs(naive - yte))),
        "rmse_naive": float(np.sqrt(np.mean((naive - yte) ** 2))),
        "mae_zero": float(np.mean(np.abs(yte))),
        "peak_all": float(serie["new_cases"].max()),
        "peak_date": str(dates[serie["new_cases"].idxmax()].date()),
        "last_publication": str(test.loc[nz, "date"].max().date()),
        "trailing_zeros": int(len(yte) - 1 - np.where(nz)[0][-1]),
        "features": list(features),
        "importances": {f: float(v) for f, v in zip(features, model.feature_importances_)},
    }
    with open(os.path.join(DATA_DIR, "stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)

    print(f"\nÉcrit dans {DATA_DIR} :")
    for name in ("serie.parquet", "test.parquet", "modele_complet.parquet", "stats.json"):
        p = os.path.join(DATA_DIR, name)
        print(f"  {name:26s} {os.path.getsize(p) / 1024:7.0f} Ko")
    print(f"\nMAE {mae:,.2f} · RMSE {rmse:,.2f} · R² {r2:.4f}".replace(",", " "))


if __name__ == "__main__":
    main()
