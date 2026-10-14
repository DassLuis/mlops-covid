import os

from fastapi.testclient import TestClient

from api import supervision
from api.app import app


client = TestClient(app)

# Clé fixée par tests/conftest.py avant l'import de l'API
HEADERS = {"x-api-key": os.environ["API_KEY"]}


def test_services_require_api_key():
    assert client.get("/mlops/services").status_code == 401


def test_services_report_each_component(monkeypatch):
    # MLflow répond, tous les autres services sont hors ligne
    monkeypatch.setattr(
        supervision, "sonder",
        lambda url: {"ok": "mlflow" in url, "latence_ms": 3},
    )
    monkeypatch.setattr(supervision, "alertes_actives", lambda: [
        {"nom": "DeriveDesDonnees", "gravite": "warning", "etat": "firing", "resume": ""},
    ])

    response = client.get("/mlops/services", headers=HEADERS)

    assert response.status_code == 200

    etat = response.json()
    services = {s["cle"]: s for s in etat["services"]}

    assert len(services) == len(supervision.SERVICES)
    assert services["mlflow"]["ok"] is True
    assert services["grafana"]["ok"] is False
    assert etat["alertes"][0]["nom"] == "DeriveDesDonnees"


def test_unreachable_service_is_reported_offline():
    # Une adresse qui ne répond pas ne doit pas faire échouer la page
    assert supervision.sonder("http://127.0.0.1:9/health")["ok"] is False
