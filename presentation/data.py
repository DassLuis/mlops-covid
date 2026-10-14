"""Accès aux données du projet : l'application ne lit aucun fichier, tout passe
par l'API déployée."""

import os

import pandas as pd
import requests
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

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


def reglages_api() -> tuple:
    """Adresse et clé de l'API, lues dans l'environnement ou dans le fichier .env
    du projet : la clé n'est écrite nulle part dans le code."""
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(ROOT, ".env"))
    except ImportError:
        pass
    return os.getenv("API_URL", "http://localhost:8000"), os.getenv("API_KEY", "")


def _api(methode: str, chemin: str, delai: int = 60, **kwargs):
    url, cle = reglages_api()
    try:
        reponse = requests.request(
            methode, f"{url.rstrip('/')}{chemin}",
            headers={"x-api-key": cle}, timeout=delai, **kwargs,
        )
    except requests.RequestException as exc:
        raise DonneesManquantes(
            f"API injoignable sur {url} ({type(exc).__name__}). "
            "Démarrez la pile avec `docker compose up -d`."
        ) from exc
    if reponse.status_code == 401:
        raise DonneesManquantes(
            "L'API refuse la clé : vérifiez API_KEY dans le fichier .env."
        )
    if reponse.status_code != 200:
        raise DonneesManquantes(
            f"L'API répond {reponse.status_code} sur {chemin} : {reponse.text[:200]}"
        )
    return reponse.json()


def _tableau(chemin: str) -> pd.DataFrame:
    df = pd.DataFrame(_api("GET", chemin))
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False, ttl=300)
def load_stats() -> dict:
    return _api("GET", "/data/stats")


@st.cache_data(show_spinner=False, ttl=300)
def load_serie() -> pd.DataFrame:
    return _tableau("/data/series")


@st.cache_data(show_spinner=False, ttl=300)
def load_test() -> pd.DataFrame:
    return _tableau("/data/test")


@st.cache_data(show_spinner=False, ttl=60)
def load_etat_mlops() -> dict:
    """Derniers résultats du pipeline : entraînement, promotion, monitoring."""
    return _api("GET", "/mlops/status")


@st.cache_data(show_spinner=False, ttl=15)
def etat_services() -> dict:
    """État de chaque service de la pile et alertes actives, vus par l'API."""
    return _api("GET", "/mlops/services", delai=20)


def liens_outils() -> dict:
    """Adresses des outils, telles que le navigateur de l'utilisateur les voit."""
    reglages_api()  # charge le fichier .env quand l'application tourne hors Docker
    defauts = {
        "api": "http://localhost:8000/docs",
        "mlflow": "http://localhost:5000",
        "airflow": "http://localhost:8080",
        "minio": "http://localhost:9001",
        "prometheus": "http://localhost:9090",
        "alertmanager": "http://localhost:9093",
        "grafana": "http://localhost:3000",
    }
    return {cle: os.getenv(f"LIEN_{cle.upper()}", url) for cle, url in defauts.items()}


@st.cache_data(show_spinner=False, ttl=30)
def etat_api():
    """Carte d'identité du modèle servi, ou None si l'API ne répond pas."""
    try:
        return _api("GET", "/health", delai=6)
    except DonneesManquantes:
        return None


def predire(entrees: dict):
    """Prévision demandée à l'API, sans journalisation. None si elle ne répond pas."""
    try:
        return float(_api("POST", "/simulate", delai=10,
                          json=entrees)["prediction_new_cases_h_plus_7"])
    except DonneesManquantes:
        return None


def essai_entrainement(n_estimators: int, part: float, profondeur=None) -> dict:
    """Essai d'entraînement calculé par l'API, avec d'autres réglages."""
    return _api("POST", "/lab/train", delai=180, json={
        "n_estimators": n_estimators,
        "train_share": part,
        "max_depth": profondeur,
    })
