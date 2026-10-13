import os
import subprocess

from fastapi.testclient import TestClient

from training import app as training_app
from training.app import app, build_command


client = TestClient(app)

# Clé fixée par tests/conftest.py avant l'import du service
HEADERS = {"x-api-key": os.environ["API_KEY"]}


def fake_run(returncode: int, output: str):
    def run(command, **kwargs):
        return subprocess.CompletedProcess(command, returncode, stdout=output, stderr="")

    return run


def test_health_lists_the_steps():
    response = client.get("/health")

    assert response.status_code == 200
    assert "train" in response.json()["steps"]


def test_step_requires_api_key():
    assert client.post("/steps/train").status_code == 401


def test_unknown_step_is_rejected():
    response = client.post("/steps/rm-rf", headers=HEADERS)

    assert response.status_code == 404


def test_force_only_applies_to_training():
    assert "--force" in build_command("train", force=True)
    assert "--force" not in build_command("train", force=False)
    assert build_command("prepare", force=True) == build_command("prepare", force=False)


def test_successful_step_returns_its_output(monkeypatch):
    monkeypatch.setattr(training_app.subprocess, "run", fake_run(0, "étape terminée"))

    response = client.post("/steps/prepare", headers=HEADERS)

    assert response.status_code == 200
    assert response.json() == {
        "step": "prepare",
        "returncode": 0,
        "output": "étape terminée",
    }


def test_failed_step_is_reported_to_the_orchestrator(monkeypatch):
    monkeypatch.setattr(training_app.subprocess, "run", fake_run(1, "échec du script"))

    response = client.post("/steps/train", headers=HEADERS)

    # Airflow doit voir l'échec : curl --fail renvoie alors un code d'erreur
    assert response.status_code == 500
    assert response.json()["detail"]["returncode"] == 1
