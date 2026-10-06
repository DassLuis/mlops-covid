import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import preprocess_country_data
from src.features import create_features, TARGET
from src.config import (
    COUNTRY,
    START_DATE,
    END_DATE,
    HORIZON,
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
)


def prepare_data():
    print("Lecture des données brutes...")
    df = pd.read_csv(RAW_DATA_PATH)

    print("Prétraitement des données...")
    df_country = preprocess_country_data(
        df=df,
        country=COUNTRY,
        start_date=START_DATE,
        end_date=END_DATE,
    )

    print("Création des features...")
    df_model, X, y, features = create_features(
        df_country=df_country,
        horizon=HORIZON,
    )

    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df_model[["date"] + features + [TARGET]].to_csv(
        PROCESSED_DATA_PATH, index=False, lineterminator="\n"
    )

    print(f"{len(df_model):,} lignes sauvegardées dans {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    prepare_data()
