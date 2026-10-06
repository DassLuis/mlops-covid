from fastapi.testclient import TestClient

from api.app import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_predict_without_api_key():
    payload = {
        "lag_1": 1200,
        "lag_2": 1100,
        "lag_3": 1300,
        "lag_4": 1250,
        "lag_5": 1400,
        "lag_6": 1350,
        "lag_7": 1500,
        "rolling_mean_7": 1300,
        "new_deaths": 25,
        "people_vaccinated": 40000000,
        "stringency_index": 45.5,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 401


def test_predict_with_api_key():
    payload = {
        "lag_1": 1200,
        "lag_2": 1100,
        "lag_3": 1300,
        "lag_4": 1250,
        "lag_5": 1400,
        "lag_6": 1350,
        "lag_7": 1500,
        "rolling_mean_7": 1300,
        "new_deaths": 25,
        "people_vaccinated": 40000000,
        "stringency_index": 45.5,
    }

    headers = {
        "x-api-key": "mlops-covid-secret-key",
    }

    response = client.post(
        "/predict",
        json=payload,
        headers=headers,
    )

    assert response.status_code == 200
    assert "prediction_new_cases_h_plus_7" in response.json()