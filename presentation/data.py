"""Chargement des données figées par build_data.py, et modèle du projet."""

import json
import os
import sys

import joblib
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(HERE, "data")
MODEL_PATH = os.path.join(ROOT, "models", "covid_rf_model.joblib")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

FR_MOIS = {
    1: "janvier", 2: "février", 3: "mars", 4: "avril", 5: "mai", 6: "juin",
    7: "juillet", 8: "août", 9: "septembre", 10: "octobre", 11: "novembre",
    12: "décembre",
}
FR_MOIS_COURT = {
    1: "janv", 2: "févr", 3: "mars", 4: "avr", 5: "mai", 6: "juin",
    7: "juil", 8: "août", 9: "sept", 10: "oct", 11: "nov", 12: "déc",
}


def date_fr(d, court: bool = False) -> str:
    m = FR_MOIS_COURT[d.month] if court else FR_MOIS[d.month]
    return f"{d.day} {m} {d.year}"


class DonneesManquantes(RuntimeError):
    pass


def _need(path: str):
    if not os.path.exists(path):
        raise DonneesManquantes(
            "Données absentes. Lancez d'abord :\n\n"
            "    python presentation/build_data.py"
        )
    return path


@st.cache_data(show_spinner=False)
def load_stats() -> dict:
    with open(_need(os.path.join(DATA_DIR, "stats.json")), encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data(show_spinner=False)
def load_serie() -> pd.DataFrame:
    df = pd.read_parquet(_need(os.path.join(DATA_DIR, "serie.parquet")))
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def load_test() -> pd.DataFrame:
    df = pd.read_parquet(_need(os.path.join(DATA_DIR, "test.parquet")))
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def load_complet() -> pd.DataFrame:
    df = pd.read_parquet(_need(os.path.join(DATA_DIR, "modele_complet.parquet")))
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def load_etat_mlops() -> dict:
    """Derniers résultats écrits par le projet : entraînement, promotion, monitoring.

    Un fichier absent donne None : la page affiche alors la règle sans le chiffre.
    """
    fichiers = {
        "entrainement": ("metrics", "train_metrics.json"),
        "promotion": ("metrics", "promotion.json"),
        "monitoring": ("data", "monitoring", "monitoring_summary.json"),
    }
    etat = {}
    for cle, morceaux in fichiers.items():
        chemin = os.path.join(ROOT, *morceaux)
        if os.path.exists(chemin):
            with open(chemin, encoding="utf-8") as fh:
                etat[cle] = json.load(fh)
        else:
            etat[cle] = None
    return etat


@st.cache_resource(show_spinner=False)
def load_model():
    """Charge le modèle réellement livré par le projet (models/*.joblib)."""
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def predire(entrees: dict):
    """Prédiction locale via le modèle du projet. Renvoie None s'il est absent."""
    paquet = load_model()
    if paquet is None:
        return None
    features = paquet["features"]
    X = pd.DataFrame([entrees])[features]
    return float(paquet["model"].predict(X)[0])
