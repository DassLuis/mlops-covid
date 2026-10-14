"""État des services de la pile, pour la page Supervision de l'application.

L'API interroge chaque service sur le réseau Docker ; l'application n'en contacte
aucun directement. Une adresse se remplace par la variable SUPERVISION_<CLÉ>.
"""
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

DELAI = 4  # secondes accordées à chaque service pour répondre

# clé, nom affiché, rôle, adresse de santé sur le réseau Docker
SERVICES = [
    ("mlflow", "MLflow", "Suivi des expériences et registre", "http://mlflow:5000/health"),
    ("training", "Service d'entraînement", "Étapes du pipeline",
     "http://training:8000/health"),
    ("airflow", "Airflow", "Orchestration des DAG", "http://airflow-webserver:8080/health"),
    ("minio", "MinIO", "Stockage des versions DVC", "http://minio:9000/minio/health/live"),
    ("prometheus", "Prometheus", "Métriques et règles d'alerte",
     "http://prometheus:9090/-/healthy"),
    ("alertmanager", "Alertmanager", "Acheminement des alertes",
     "http://alertmanager:9093/-/healthy"),
    ("grafana", "Grafana", "Tableau de bord", "http://grafana:3000/api/health"),
]

ALERTES_URL = "http://prometheus:9090/api/v1/alerts"


def adresse(cle: str, defaut: str) -> str:
    return os.getenv(f"SUPERVISION_{cle.upper()}", defaut)


def sonder(url: str) -> dict:
    """Interroge une adresse de santé : en ligne, et en combien de temps."""
    debut = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=DELAI) as reponse:
            en_ligne = reponse.status < 400
    except Exception:
        en_ligne = False
    return {"ok": en_ligne, "latence_ms": round((time.perf_counter() - debut) * 1000)}


def alertes_actives() -> list:
    """Alertes déclenchées dans Prometheus ; liste vide s'il ne répond pas."""
    try:
        with urllib.request.urlopen(adresse("alertes", ALERTES_URL), timeout=DELAI) as r:
            donnees = json.load(r)
    except Exception:
        return []

    return [
        {
            "nom": a["labels"].get("alertname", ""),
            "gravite": a["labels"].get("severity", ""),
            "etat": a.get("state", ""),
            "resume": a.get("annotations", {}).get("summary", ""),
        }
        for a in donnees.get("data", {}).get("alerts", [])
    ]


def etat_services() -> dict:
    with ThreadPoolExecutor(max_workers=len(SERVICES)) as pool:
        resultats = list(pool.map(
            lambda s: sonder(adresse(s[0], s[3])), SERVICES))

    return {
        "services": [
            {"cle": cle, "nom": nom, "role": role, **resultat}
            for (cle, nom, role, _), resultat in zip(SERVICES, resultats)
        ],
        "alertes": alertes_actives(),
    }
