import os

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import exploration
from api.app import app
from src.config import COUNTRY, START_DATE
from src.features import FEATURES


client = TestClient(app)

# Clé fixée par tests/conftest.py avant l'import de l'API
HEADERS = {"x-api-key": os.environ["API_KEY"]}


@pytest.fixture
def raw_data(tmp_path, monkeypatch):
    """Données brutes de substitution, au format OWID : une publication par
    semaine et des zéros le reste du temps, comme la série réelle."""
    days = 140
    dates = pd.date_range(START_DATE, periods=days)

    raw = pd.DataFrame(
        {
            "location": COUNTRY,
            "date": dates.strftime("%Y-%m-%d"),
            "new_cases": [1000.0 + i if i % 7 == 0 else 0.0 for i in range(days)],
            "new_deaths": 5.0,
            "people_vaccinated": 100.0,
            "stringency_index": 50.0,
        }
    )

    path = tmp_path / "owid_covid.csv"
    raw.to_csv(path, index=False)

    monkeypatch.setattr(exploration, "RAW_DATA_PATH", str(path))
    exploration._cache.clear()

    return raw


def test_data_endpoints_require_api_key():
    for path in ("/data/stats", "/data/series", "/data/test", "/mlops/status"):
        assert client.get(path).status_code == 401


def test_data_stats_describes_the_dataset(raw_data):
    response = client.get("/data/stats", headers=HEADERS)

    assert response.status_code == 200

    stats = response.json()

    # 7 premiers jours sans lag_7, 7 derniers sans cible
    assert stats["n_total"] == len(raw_data) - 14
    assert stats["n_train"] + stats["n_test"] == stats["n_total"]
    assert stats["features"] == FEATURES
    assert set(stats["importances"]) == set(FEATURES)
    assert stats["n_publication_days_test"] > 0


def test_data_series_and_test_set_are_consistent(raw_data):
    series = client.get("/data/series", headers=HEADERS).json()
    test = client.get("/data/test", headers=HEADERS).json()
    stats = client.get("/data/stats", headers=HEADERS).json()

    assert len(series) == stats["n_total"]
    assert len(test) == stats["n_test"]
    assert {"date", "y_true", "y_pred", "erreur", *FEATURES} <= set(test[0])


def test_data_endpoints_report_missing_data(tmp_path, monkeypatch):
    monkeypatch.setattr(exploration, "RAW_DATA_PATH", str(tmp_path / "absent.csv"))
    exploration._cache.clear()

    assert client.get("/data/stats", headers=HEADERS).status_code == 503


def test_simulate_predicts_without_logging():
    payload = {feature: 100.0 for feature in FEATURES}
    log_path = os.environ["PREDICTION_LOG_PATH"]
    before = os.path.getsize(log_path) if os.path.exists(log_path) else 0

    response = client.post("/simulate", json=payload, headers=HEADERS)

    assert response.status_code == 200
    assert "prediction_new_cases_h_plus_7" in response.json()

    # Une simulation ne doit pas alimenter le suivi de dérive
    after = os.path.getsize(log_path) if os.path.exists(log_path) else 0
    assert after == before


def test_lab_train_returns_trial_metrics(raw_data):
    response = client.post(
        "/lab/train",
        json={"n_estimators": 10, "train_share": 0.8, "max_depth": 3},
        headers=HEADERS,
    )

    assert response.status_code == 200

    trial = response.json()

    assert trial["n_train"] + trial["n_test"] == len(raw_data) - 14
    assert trial["mae"] >= 0


def test_lab_train_rejects_out_of_range_settings():
    response = client.post(
        "/lab/train",
        json={"n_estimators": 5000},
        headers=HEADERS,
    )

    assert response.status_code == 422


def test_mlops_status_lists_the_three_reports():
    response = client.get("/mlops/status", headers=HEADERS)

    assert response.status_code == 200
    assert set(response.json()) == {"entrainement", "promotion", "monitoring"}
