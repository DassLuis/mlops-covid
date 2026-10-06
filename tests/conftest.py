import os
import tempfile

# Les tests ne doivent ni dépendre du serveur MLflow ni écrire dans le journal
# de production : ces variables sont fixées avant l'import de l'API.
os.environ["MODEL_SOURCE"] = "local"
os.environ["PREDICTION_LOG_PATH"] = os.path.join(
    tempfile.mkdtemp(prefix="covid_tests_"), "predictions.csv"
)
