# COVID-19 MLOps Project

## Description

Ce projet a été réalisé dans le cadre d'un projet MLOps.

L'objectif est de prédire le nombre de nouveaux cas de COVID-19 à J+7 à partir des données publiques de **Our World In Data (OWID)**.

Le projet couvre l'ensemble d'une chaîne MLOps :

- Prétraitement des données
- Feature Engineering
- Entraînement d'un modèle de Machine Learning
- API REST avec FastAPI
- Suivi des expériences avec MLflow
- Orchestration avec Apache Airflow
- Conteneurisation avec Docker
- Tests unitaires avec Pytest

---

# Architecture du projet

```
mlops-covid/
│
├── airflow/                  # DAG Airflow
├── api/                      # API FastAPI
├── data/                     # Données
├── docker/                   # Dockerfiles
├── models/                   # Modèle entraîné
├── src/                      # Scripts Python
│   ├── config.py
│   ├── preprocess.py
│   ├── features.py
│   ├── train.py
│   └── predict.py
│
├── tests/                    # Tests unitaires
│
├── docker-compose.yml
├── requirements.txt
├── requirements-airflow.txt
├── README.md
└── .env
```

---

# Technologies utilisées

- Python 3.11
- Pandas
- NumPy
- Scikit-Learn
- FastAPI
- Uvicorn
- MLflow
- Apache Airflow
- Docker
- Docker Compose
- Pytest

---

# Jeu de données

Source officielle :

https://github.com/owid/covid-19-data

Les données sont téléchargées automatiquement depuis le dépôt OWID.

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

Lancer ensuite :

```bash
docker compose up --build
```

ou

```bash
docker compose up -d
```

Le premier lancement peut prendre plusieurs minutes.

---

# Accès aux services

## API FastAPI

Documentation Swagger :

http://localhost:8000/docs

---

## MLflow

Interface de suivi des expériences :

http://localhost:5000

---

## Airflow

http://localhost:8080

Connexion :

```
username : admin
password : admin
```

---

# Authentification de l'API

L'endpoint `/predict` est protégé par une clé API.

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
  "stringency_index":45.5
}'
```

---

# Réentraîner automatiquement le modèle

Le projet utilise Apache Airflow.

Dans Airflow :

- ouvrir le DAG **retrain_covid_model**
- cliquer sur **Trigger DAG**

Le pipeline effectue automatiquement :

- téléchargement des données
- prétraitement
- feature engineering
- entraînement du modèle
- calcul des métriques
- enregistrement d'un nouveau run dans MLflow

---

# Suivi des expériences

Chaque entraînement est automatiquement enregistré dans MLflow.

Les informations sauvegardées sont notamment :

- paramètres du modèle
- métriques (MAE, RMSE)
- date d'entraînement
- comparaison des différents runs

---

# Tests

Lancer les tests :

```bash
python -m pytest -v
```

Les tests vérifient notamment :

- endpoint `/health`
- authentification de l'API
- endpoint `/predict`

---

# Docker

Le projet est découpé en plusieurs services :

- API FastAPI
- MLflow
- Airflow Webserver
- Airflow Scheduler

Tous les services sont orchestrés via :

```
docker-compose.yml
```

---

# Structure MLOps

Le pipeline MLOps est le suivant :

```
Données OWID
        │
        ▼
Prétraitement
        │
        ▼
Feature Engineering
        │
        ▼
Random Forest
        │
        ▼
Sauvegarde du modèle
        │
        ▼
FastAPI
        │
        ▼
Prédictions
```

L'orchestration automatique est assurée par Airflow.

Les expériences sont suivies dans MLflow.

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
découpée en 17 diapositives, reste disponible :

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
- `presentation/presentation-mlops-covid.pdf` — 17 pages au format 16:9

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