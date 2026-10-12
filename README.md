# COVID-19 MLOps Project

## Description

Ce projet a été réalisé dans le cadre d'un projet MLOps.

L'objectif est de prédire le nombre de nouveaux cas de COVID-19 à J+7 à partir des données publiques de **Our World In Data (OWID)**.

Le projet couvre l'ensemble d'une chaîne MLOps :

- Prétraitement des données et Feature Engineering
- Entraînement d'un modèle de Machine Learning
- **Versioning des données et des modèles** avec DVC et MLflow
- **Registre de modèles** et cycle de vie (candidat, production, rejet) avec MLflow et Airflow
- API REST avec FastAPI, qui sert le modèle en production du registre
- **Monitoring en production** : métriques Prometheus, tableaux de bord Grafana, dérive et performance avec Evidently
- Orchestration avec Apache Airflow
- Conteneurisation avec Docker
- Tests unitaires avec Pytest

---

# Architecture du projet

```
mlops-covid/
│
├── airflow/dags/
│   ├── retrain_covid_model.py    # DAG : entraînement, validation, promotion
│   └── monitor_covid_model.py    # DAG : dérive et performance, réentraînement
├── api/
│   ├── app.py                    # API FastAPI
│   ├── auth.py                   # clé API
│   ├── metrics.py                # métriques Prometheus
│   └── prediction_log.py         # journal des prédictions
├── data/
│   ├── raw/                      # données OWID (suivies par DVC)
│   ├── processed/                # features (suivies par DVC)
│   └── monitoring/               # journal des prédictions, rapports Evidently
├── docker/                       # Dockerfiles
├── metrics/                      # métriques d'entraînement et de promotion
├── models/                       # modèle entraîné (suivi par DVC)
├── monitoring/
│   ├── prometheus/               # collecte et règles d'alerte
│   ├── alertmanager/             # regroupement et acheminement des alertes
│   └── grafana/                  # source de données et tableau de bord
├── scripts/
│   └── simulate_traffic.py       # trafic de démonstration
├── src/
│   ├── config.py
│   ├── ingest.py                 # étape DVC 1 : téléchargement
│   ├── preprocess.py
│   ├── features.py
│   ├── prepare.py                # étape DVC 2 : prétraitement et features
│   ├── train.py                  # étape DVC 3 : entraînement et enregistrement
│   ├── promote.py                # comparaison candidat / production
│   ├── monitor.py                # rapports Evidently
│   └── predict.py
├── tests/
│
├── dvc.yaml                      # pipeline DVC
├── dvc.lock                      # empreintes des données, du code et du modèle
├── docker-compose.yml
├── requirements.txt              # API
├── requirements-dev.txt          # poste de développement (API + DVC)
├── requirements-airflow.txt      # image Airflow (DVC, Evidently)
└── .env
```

---

# Technologies utilisées

- Python 3.11
- Pandas, NumPy, Scikit-Learn
- FastAPI, Uvicorn
- DVC, MinIO (stockage objet compatible S3)
- MLflow (suivi des expériences et registre de modèles)
- Apache Airflow
- Prometheus, Alertmanager, Grafana
- Evidently
- Docker, Docker Compose
- Pytest

---

# Jeu de données

Source officielle :

https://github.com/owid/covid-19-data

Les données sont téléchargées depuis le dépôt OWID par l'étape `ingest` du pipeline DVC, puis versionnées.

---

# Modèle de Machine Learning

Modèle utilisé :

**RandomForestRegressor**

Objectif :

Prédire le nombre de nouveaux cas de COVID-19 à J+7.

Métriques utilisées :

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)

---

# Installation

Cloner le projet puis se placer dans son dossier.

Copier `.env.example` en `.env` et y renseigner la clé API.

Lancer ensuite :

```bash
docker compose up -d --build
```

Le premier lancement peut prendre plusieurs minutes.

Si un port est déjà utilisé sur la machine, il se change sans toucher aux fichiers :

```bash
API_PORT=8010 AIRFLOW_PORT=8081 docker compose up -d
```

(variables disponibles : `API_PORT`, `MLFLOW_PORT`, `AIRFLOW_PORT`, `PROMETHEUS_PORT`, `ALERTMANAGER_PORT`, `GRAFANA_PORT`, `MINIO_PORT`, `MINIO_CONSOLE_PORT`)

Pour travailler hors Docker (pipeline DVC, tests) :

```bash
pip install -r requirements-dev.txt
```

---

# Accès aux services

| Service | Adresse | Rôle |
|---|---|---|
| API FastAPI | http://localhost:8000/docs | prédictions, `/health`, `/metrics` |
| MLflow | http://localhost:5000 | expériences, registre de modèles, historique du monitoring |
| Airflow | http://localhost:8080 (admin / admin) | DAG d'entraînement et de monitoring |
| Prometheus | http://localhost:9090 | métriques et alertes |
| Alertmanager | http://localhost:9093 | alertes regroupées et acheminées |
| Grafana | http://localhost:3000 (admin / admin) | tableau de bord « MLOps COVID - API et modèle » |
| MinIO | http://localhost:9001 (identifiants du `.env`) | stockage distant de DVC, bucket `dvc` |

---

# Authentification de l'API

Les endpoints `/predict` et `/model/reload` sont protégés par une clé API.

Ajouter l'en-tête suivant :

```
x-api-key: mlops-covid-secret-key
```

---

# Exemple d'appel de l'API

Depuis Swagger ou avec curl :

```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-H "x-api-key: mlops-covid-secret-key" \
-d '{
  "lag_1":1200,
  "lag_2":1100,
  "lag_3":1300,
  "lag_4":1250,
  "lag_5":1400,
  "lag_6":1350,
  "lag_7":1500,
  "rolling_mean_7":1300,
  "new_deaths":25,
  "people_vaccinated":40000000,
  "stringency_index":45.5,
  "date":"2023-03-01"
}'
```

La réponse indique la prédiction et la version du modèle qui l'a produite.

Le champ `date` est facultatif : il indique le jour auquel correspondent les données, ce qui permet au monitoring de comparer plus tard la prédiction à la valeur réellement observée.

| Endpoint | Rôle |
|---|---|
| `GET /health` | état de l'API, version et origine du modèle chargé |
| `POST /predict` | prédiction à J+7 |
| `POST /model/reload` | recharge le modèle en production du registre |
| `GET /metrics` | métriques au format Prometheus |

---

# Versioning des données et des modèles

Le versioning repose sur trois outils qui se complètent :

- **Git** versionne le code, la configuration et le fichier `dvc.lock`
- **DVC** versionne les fichiers volumineux : données brutes, données préparées, modèle
- **MLflow** versionne chaque entraînement et chaque modèle, et garde le lien avec les données

## Pipeline DVC

Le pipeline est décrit dans `dvc.yaml` :

```
ingest  ──►  prepare  ──►  train
```

| Étape | Script | Sortie |
|---|---|---|
| `ingest` | `src/ingest.py` | `data/raw/owid_covid.csv` |
| `prepare` | `src/prepare.py` | `data/processed/covid_features.csv` |
| `train` | `src/train.py` | `models/covid_rf_model.joblib`, `metrics/train_metrics.json` |

```bash
dvc repro          # rejoue uniquement les étapes dont le code, les données ou les paramètres ont changé
dvc dag            # affiche le graphe du pipeline
dvc metrics show   # affiche les métriques du dernier entraînement
dvc push           # envoie données et modèle vers le stockage DVC
dvc pull           # les récupère sur une autre machine
```

`dvc.lock` contient l'empreinte (md5) de chaque fichier de données, de chaque script et des paramètres de `src/config.py`. Un commit git désigne donc une version exacte du code, des données et du modèle.

Pour revenir à une version antérieure :

```bash
git checkout <commit>
dvc checkout
```

## Stockage distant

Le stockage DVC est un bucket S3, `s3://dvc`, servi par le conteneur **MinIO**. DVC lui parle comme à Amazon S3 : passer à un S3 hébergé revient à changer l'adresse et les identifiants, sans toucher au pipeline.

L'adresse et les identifiants ne sont pas écrits dans le dépôt, ils passent par l'environnement. Dans Airflow, `docker-compose.yml` les fournit. Sur le poste de développement, avant un `dvc push` ou un `dvc pull` :

```bash
export AWS_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID=minioadmin          # MINIO_ROOT_USER du .env
export AWS_SECRET_ACCESS_KEY=minioadmin      # MINIO_ROOT_PASSWORD du .env
```

Le bucket est créé au premier démarrage par le service `minio-init`.

## Lien entre données et modèle dans MLflow

Chaque run d'entraînement enregistre dans MLflow :

- les paramètres et les métriques (MAE, RMSE)
- le modèle lui-même
- le jeu de données utilisé (onglet « Datasets » du run)
- l'empreinte DVC des données (`dvc_data_md5`) et le commit git (`git_commit`)

À partir de n'importe quel modèle, on retrouve ainsi les données et le code exacts qui l'ont produit.

---

# Registre de modèles et cycle de vie

Chaque entraînement crée une nouvelle version du modèle `covid_rf_model` dans le **registre MLflow** (onglet « Models »). Deux alias suivent le cycle de vie :

| Alias | Signification |
|---|---|
| `candidate` | dernière version entraînée, en attente de validation |
| `production` | version servie par l'API |

Le script `src/promote.py` évalue le candidat et le modèle en production **sur le même jeu de test**. Le candidat n'obtient l'alias `production` que si sa MAE est inférieure ou égale. Chaque version porte une étiquette `validation_status` : `promu`, `rejeté` ou `remplacé`. La décision est écrite dans `metrics/promotion.json`.

Un retour arrière se fait en déplaçant l'alias `production` vers une version antérieure, depuis l'interface MLflow, puis en appelant `POST /model/reload`.

## DAG `retrain_covid_model`

Planifié chaque semaine, ou lancé à la main depuis Airflow (**Trigger DAG**) :

```
ingest_data ► prepare_data ► train_model ► promote_model ► version_artifacts ► reload_api
```

| Tâche | Action |
|---|---|
| `ingest_data` | télécharge les données (`dvc repro ingest`) |
| `prepare_data` | prétraitement et features, rejoués seulement si les données ont changé |
| `train_model` | entraîne et enregistre une version candidate, seulement si les données, le code ou les paramètres ont changé |
| `promote_model` | compare le candidat au modèle en production |
| `version_artifacts` | envoie données et modèle vers le stockage MinIO (`dvc push`) |
| `reload_api` | demande à l'API de recharger le modèle en production |

Pour forcer un entraînement sans changement de données : **Trigger DAG w/ config**, paramètre `force_train` à `true`.

L'API charge le modèle qui porte l'alias `production`. Si le registre est injoignable, elle se replie sur le fichier `models/covid_rf_model.joblib` ; `/health` indique toujours la version et l'origine du modèle chargé.

---

# Monitoring en production

## Métriques de l'API (Prometheus)

L'API expose ses métriques sur `/metrics`, collectées toutes les 15 secondes par Prometheus :

| Métrique | Contenu |
|---|---|
| `covid_api_requests_total` | requêtes par endpoint et code HTTP |
| `covid_api_request_duration_seconds` | temps de réponse |
| `covid_predictions_total` | prédictions servies, par version du modèle |
| `covid_prediction_value` | distribution des prédictions |
| `covid_model_info`, `covid_model_loaded` | modèle chargé |
| `covid_data_drift_share`, `covid_data_drift_detected` | résultat du dernier contrôle de dérive |
| `covid_production_mae`, `covid_performance_degraded` | performance mesurée en production |

Les règles d'alerte (`monitoring/prometheus/alert_rules.yml`) couvrent : API indisponible, aucun modèle chargé, taux d'erreur élevé, latence élevée, dérive des données, performance dégradée, monitoring en retard.

Prometheus transmet les alertes déclenchées à **Alertmanager**, qui les regroupe par nom, évite les doublons (une API indisponible masque l'alerte « aucun modèle chargé ») et les achemine vers le destinataire `equipe_mlops`. Sans canal renseigné, elles se consultent dans son interface. Pour les recevoir par mail ou sur Slack, décommenter le bloc correspondant dans `monitoring/alertmanager/alertmanager.yml` et y mettre ses identifiants.

Le tableau de bord Grafana est créé automatiquement au démarrage.

## Dérive et performance (Evidently)

Chaque prédiction est journalisée dans `data/monitoring/predictions.csv`. Le script `src/monitor.py` produit deux rapports Evidently :

- **Dérive des données** : les entrées reçues en production sont comparées aux données d'entraînement. Une dérive est déclarée quand au moins la moitié des variables ont changé de distribution.
- **Performance** : les prédictions datées sont rapprochées de la valeur réellement observée à J+7. La MAE est rapportée au niveau moyen des cas, pour rester comparable d'une période à l'autre ; la performance est dite dégradée au-delà de 1,5 fois la valeur de référence.

Les rapports HTML sont écrits dans `data/monitoring/reports/`. Chaque contrôle est aussi enregistré dans MLflow (expérience `covid_monitoring`), avec ses métriques et ses rapports : l'historique de la dérive et de la performance s'y consulte dans le temps.

## DAG `monitor_covid_model`

Planifié chaque jour :

```
run_monitoring ► check_results ─┬─► trigger_retraining   (dérive ou performance dégradée)
                                └─► no_action
```

En cas de dérive ou de baisse de performance, il déclenche le DAG `retrain_covid_model`.

## Démonstration

```bash
python scripts/simulate_traffic.py --scenario normal    # jours semblables à l'entraînement
python scripts/simulate_traffic.py --scenario recent    # année 2023, jamais vue par le modèle
```

Puis lancer le DAG `monitor_covid_model` dans Airflow. Le premier scénario ne déclenche rien ; le second fait apparaître une dérive réelle (les données de 2023 ne ressemblent plus à celles de l'entraînement), visible dans Grafana, dans l'alerte Prometheus `DeriveDesDonnees` et dans le rapport Evidently.

Si l'API n'est pas sur le port 8000, ajouter `--url http://localhost:<port>`.

---

# Tests

Lancer les tests :

```bash
python -m pytest -v
```

Les tests vérifient notamment :

- endpoint `/health`
- authentification de l'API
- endpoint `/predict`, avec et sans date
- endpoint `/metrics`
- rechargement du modèle
- construction des features (décalages, cible, lignes incomplètes)
- règles de dérive et de performance (dans l'image Airflow, où Evidently est installé)

---

# Docker

Le projet est découpé en plusieurs services :

- API FastAPI
- MLflow
- Airflow Webserver
- Airflow Scheduler
- MinIO
- Prometheus
- Alertmanager
- Grafana

Tous les services sont orchestrés via :

```
docker-compose.yml
```

---

# Structure MLOps

```
                 Données OWID
                      │
                      ▼
   ┌──────── Pipeline DVC (Airflow) ────────┐
   │  ingest  ►  prepare  ►  train          │──► versions des données et du modèle (MinIO)
   └────────────────────┬───────────────────┘
                        ▼
              Registre MLflow
        candidat ──(comparaison)──► production
                        │
                        ▼
                     FastAPI ──► Prédictions
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼
   Prometheus / Grafana        Journal des prédictions
   (trafic, latence)                  │
          │                           │
          ▼                           │
     Alertmanager                     │
     (alertes)                        │
                                      ▼
                           Evidently (Airflow, chaque jour)
                           dérive et performance
                                      │
                                      └──► réentraînement si nécessaire
```

---

# Présentation du projet

Le dossier `presentation/` contient le support de soutenance, sous trois formes
issues des mêmes données.

## Application Streamlit

Présentation interactive en 8 sections, thème sombre, avec un simulateur en
direct.

Installation puis lancement :

```bash
pip install -r requirements-streamlit.txt
python presentation/build_data.py       # une seule fois : fige les données
python presentation/build_assets.py     # une seule fois : bannière et fond de page
streamlit run presentation/app.py
```

L'application s'ouvre sur http://localhost:8501

Navigation par la barre latérale : Introduction, Données & Prétraitement,
Modèle & Entraînement, Architecture MLOps, Résultats & Analyse, Interprétabilité,
Simulateur Interactif, Conclusion. Un stepper suit l'avancement, et
l'interrupteur **Effets visuels** coupe les animations si la machine de
projection peine.

Le simulateur propose quatre modes :

- **Curseurs** — les onze variables réglées à la main, prévision recalculée en
  direct par `models/covid_rf_model.joblib`
- **Rejouer une date** — un jour de publication du jeu de test, ses entrées
  réelles, la prévision et la valeur observée à J+7
- **API réelle** — vraie requête `POST /predict` avec l'en-tête `x-api-key`,
  code HTTP et réponse JSON affichés (nécessite les conteneurs démarrés)
- **Réentraîner** — nombre d'arbres, part d'apprentissage et profondeur
  réglables, métriques comparées à celles du projet

Les trois premiers modes fonctionnent hors ligne, sans Docker.

### Organisation du code

| Fichier | Rôle |
|---|---|
| `presentation/app.py` | pages, navigation, simulateur |
| `presentation/studio_css.py` | thème sombre, animations, composants |
| `presentation/studio_charts.py` | graphiques Plotly |
| `presentation/studio_archi.py` | schéma d'architecture (SVG) |
| `presentation/data.py` | chargement des données figées et du modèle |
| `presentation/build_data.py` | rejoue le pipeline et fige les résultats |
| `presentation/build_assets.py` | dessine la bannière et le fond de page |

Une **version claire** de la présentation, calée sur la charte du PDF et
découpée en 19 diapositives, reste disponible :

```bash
streamlit run presentation/app_clair.py --theme.base light
```

### Dépendances externes

Font Awesome et la police Outfit sont chargées par CDN : prévoyez une connexion
le jour de la soutenance, ou la mise en cache du navigateur. Tout le reste
fonctionne hors ligne.

Pour remplacer l'illustration animée de la barre latérale par une animation
Lottie, déposez un fichier `presentation/assets/lottie_sidebar.json` : il sera
utilisé automatiquement.

### Fond de page

Les pages sont posées sur une illustration de poumons parcourus d'un réseau de
neurones, dessinée par `presentation/build_assets.py` dans
`presentation/assets/banner_lungs.jpg`. Elle est fixée à la fenêtre, atténuée et
éteinte vers les bords pour que le texte reste lisible en projection.

Pour utiliser votre propre visuel, déposez-le sous
`presentation/assets/banner_lungs.png` : il est prioritaire sur l'image
dessinée, et `build_assets.py` ne l'écrase jamais. L'intensité se règle sur une
seule ligne, dans `presentation/app.py` :

```python
fond_ecran(opacite=0.45)   # 0 = fond éteint, 1 = image en pleine force
```

## Support imprimable

- `presentation/presentation-mlops-covid.html` — à ouvrir dans un navigateur
- `presentation/presentation-mlops-covid.pdf` — 19 pages au format 16:9

## Guides de lecture

Deux guides accompagnent l'application Streamlit, au choix selon le temps
disponible.

| Guide | Format | Contenu |
|---|---|---|
| `presentation/guide-simple-streamlit.html` · `.pdf` | 10 pages A4 | **Une page par section** : à quoi elle sert, ce qu'on voit à l'écran, la phrase à dire, le chiffre à retenir. Une capture par section. |
| `presentation/guide-lecture-streamlit.html` · `.pdf` | 30 pages A4 | Version détaillée : lecture graphique par graphique, pièges à éviter, questions du jury. |

Le guide simple se termine par un mémo : les six chiffres qui reviennent et les
trois questions les plus probables, avec leur réponse en une phrase.

## Origine des chiffres

Aucune valeur du support n'est saisie à la main. `presentation/build_data.py`
rejoue le pipeline du projet (`src/preprocess.py`, `src/features.py`, mêmes
hyperparamètres que `src/train.py`) et fige le résultat dans
`presentation/data/`. Les métriques obtenues sont identiques à celles des runs
MLflow : MAE 942,72 et RMSE 4 262,00.

---

# Auteur

Projet réalisé par

**Saadia Szapiro**

dans le cadre d'un projet pédagogique MLOps.