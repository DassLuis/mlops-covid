import pandas as pd
import pytest

from src.features import FEATURES, TARGET, create_features


def make_country_data(days: int = 30) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2021-01-01", periods=days),
            "new_cases": range(days),
            "new_deaths": 1.0,
            "people_vaccinated": 100.0,
            "stringency_index": 50.0,
        }
    )


def test_create_features_builds_lags_and_target():
    df_model, X, y, features = create_features(make_country_data(), horizon=7)

    assert features == FEATURES
    assert list(X.columns) == FEATURES

    row = df_model.iloc[0]

    # Sur une série 0, 1, 2... chaque valeur se déduit du jour courant
    assert row["lag_1"] == row["new_cases"] - 1
    assert row["lag_7"] == row["new_cases"] - 7
    assert row[TARGET] == row["new_cases"] + 7


def test_create_features_drops_incomplete_rows():
    df_model, X, y, _ = create_features(make_country_data(days=30), horizon=7)

    # 7 premiers jours sans lag_7, 7 derniers sans cible
    assert len(df_model) == 30 - 7 - 7
    assert not X.isna().any().any()
    assert not y.isna().any()


def test_is_drifted_depends_on_the_method():
    # Evidently n'est installé que dans l'image Airflow, où tourne le monitoring
    pytest.importorskip("evidently")
    from src.monitor import is_drifted

    assert is_drifted("Wasserstein distance (normed)", score=0.3, threshold=0.1)
    assert not is_drifted("Wasserstein distance (normed)", score=0.05, threshold=0.1)
    assert is_drifted("K-S p_value", score=0.01, threshold=0.05)
    assert not is_drifted("K-S p_value", score=0.5, threshold=0.05)


def test_relative_mae():
    pytest.importorskip("evidently")
    from src.monitor import relative_mae

    assert relative_mae(50.0, 100.0) == 0.5
    assert relative_mae(50.0, 0.0) is None
