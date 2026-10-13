"""Envoie à l'API des requêtes tirées des données réelles, pour alimenter le monitoring.

    python scripts/simulate_traffic.py --scenario normal   # période d'entraînement
    python scripts/simulate_traffic.py --scenario recent   # année 2023 : dérive réelle
"""
import argparse
import os
import sys

import pandas as pd
import requests
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import FEATURES
from src.config import PROCESSED_DATA_PATH, RANDOM_STATE, TRAIN_RATIO

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario", choices=["normal", "recent"], default="normal")
    parser.add_argument("--n", type=int, default=200, help="nombre de requêtes")
    parser.add_argument("--url", default=os.getenv("API_URL", "http://localhost:8000"))
    args = parser.parse_args()

    df = pd.read_csv(PROCESSED_DATA_PATH)
    train_size = int(len(df) * TRAIN_RATIO)

    # "normal" : jours semblables à ceux vus à l'entraînement ;
    # "recent" : jours postérieurs, que le modèle n'a jamais vus.
    period = df.iloc[:train_size] if args.scenario == "normal" else df.iloc[train_size:]
    sample = period.sample(n=min(args.n, len(period)), random_state=RANDOM_STATE)

    api_key = os.getenv("API_KEY")

    if not api_key:
        sys.exit("API_KEY n'est pas définie : la renseigner dans le fichier .env.")

    headers = {"x-api-key": api_key}
    errors = 0

    for _, row in sample.iterrows():
        payload = {feature: float(row[feature]) for feature in FEATURES}
        payload["date"] = str(row["date"])[:10]

        response = requests.post(f"{args.url}/predict", json=payload, headers=headers, timeout=10)

        if response.status_code != 200:
            errors += 1

    print(f"Scénario {args.scenario} : {len(sample)} requêtes envoyées, {errors} en erreur.")


if __name__ == "__main__":
    main()
