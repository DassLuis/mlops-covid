import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import load_data
from src.config import URL, RAW_DATA_PATH


def ingest_data():
    print("Téléchargement des données OWID...")
    df = load_data(URL)

    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    df.to_csv(RAW_DATA_PATH, index=False, lineterminator="\n")

    print(f"{len(df):,} lignes sauvegardées dans {RAW_DATA_PATH}")


if __name__ == "__main__":
    ingest_data()
