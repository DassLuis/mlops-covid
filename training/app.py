"""Service d'entraînement : exécute les étapes du pipeline à la demande d'Airflow.

Airflow orchestre, ce service calcule. Chaque étape est une commande connue
d'avance ; le service n'exécute rien d'autre.
"""
import os
import subprocess
import sys
import threading

from fastapi import Depends, FastAPI, HTTPException

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.auth import verify_api_key

PROJECT_DIR = os.getenv("PROJECT_DIR", os.getcwd())
OUTPUT_TAIL = 6000  # caractères de sortie renvoyés à Airflow

STEPS = {
    # Les trois étapes de dvc.yaml : DVC ne rejoue que ce qui a changé
    "ingest": ["dvc", "repro", "--single-item", "--force", "ingest"],
    "prepare": ["dvc", "repro", "--single-item", "prepare"],
    "train": ["dvc", "repro", "--single-item", "train"],
    "promote": [sys.executable, "src/promote.py"],
    "push": ["dvc", "push"],
    "monitor": [sys.executable, "src/monitor.py"],
}

# DVC verrouille le dépôt : une seule étape à la fois
_lock = threading.Lock()

app = FastAPI(
    title="Service d'entraînement COVID - MLOps",
    description="Étapes du pipeline : ingestion, préparation, entraînement, "
                "promotion, versioning, monitoring.",
    version="1.0.0",
)


def build_command(name: str, force: bool) -> list:
    command = list(STEPS[name])

    # Réentraîner même si ni les données ni le code n'ont changé
    if force and name == "train":
        command.insert(command.index("train"), "--force")

    return command


@app.get("/health")
def health():
    return {"status": "ok", "steps": list(STEPS)}


@app.post("/steps/{name}", dependencies=[Depends(verify_api_key)])
def run_step(name: str, force: bool = False):
    if name not in STEPS:
        raise HTTPException(status_code=404, detail=f"Étape inconnue : {name}")

    with _lock:
        result = subprocess.run(
            build_command(name, force),
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
        )

    report = {
        "step": name,
        "returncode": result.returncode,
        "output": (result.stdout + result.stderr)[-OUTPUT_TAIL:],
    }

    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=report)

    return report
