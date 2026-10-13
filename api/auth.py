import os
import secrets

from dotenv import load_dotenv
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

load_dotenv()

# La clé vient de l'environnement (fichier .env, non versionné). Sans elle,
# l'API refuse de démarrer plutôt que d'accepter une clé connue de tous.
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise RuntimeError(
        "API_KEY n'est pas définie : copier .env.example en .env et y renseigner une clé."
    )

API_KEY_NAME = "x-api-key"

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def verify_api_key(api_key: str = Security(api_key_header)):
    # Comparaison en temps constant : la durée de la réponse ne renseigne pas sur la clé
    if not api_key or not secrets.compare_digest(api_key.encode(), API_KEY.encode()):
        raise HTTPException(
            status_code=401,
            detail="Clé API invalide ou manquante",
        )

    return api_key
