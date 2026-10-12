"""Les dix-neuf diapositives de la présentation.

Le contenu reprend mot pour mot le support PDF. Seule la diapositive 16
(Démonstration) gagne un laboratoire interactif : elle est le moment où, en
soutenance, on montre la chaîne en fonctionnement.
"""

import json
import re

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

import charts
from data import date_fr, load_etat_mlops, load_model, predire
from studio_archi import ARCHI_SVG_CLAIR
from theme import (
    ACCENT, ACCENT_DIM, INK2, INK3, WARM, GOOD,
    bullets, card, cols, foot, icon, num, rail, stat, stats_row, steps, table,
    block, NBSP,
)


def w(html: str) -> None:
    """Écrit un fragment HTML brut (une seule ligne, pour le parseur Streamlit)."""
    st.markdown(re.sub(r"\s+", " ", html).strip(), unsafe_allow_html=True)


def _titre(txt: str) -> str:
    return f'<div class="h2">{txt}</div>'


# ══════════════════════════ 01 · couverture ══════════════════════════
def s01(S, D):
    w(rail("virus", "Projet MLOps · Soutenance", "Examen DTS"))
    w(block(
        '<div class="h1">Prédire les cas de COVID-19<br>à sept jours</div>',
        '<p class="lead">Une chaîne MLOps complète, des données publiques mondiales '
        'jusqu’à une API d’inférence conteneurisée : modèle versionné, promu par '
        'comparaison, surveillé en production et réentraîné automatiquement.</p>',
        '<div style="height:1.6rem"></div>',
        stats_row(
            stat(num(S["n_total"]), "Jours de données", "j"),
            stat("11", "Variables construites"),
            stat("8", "Services conteneurisés"),
            stat(num(S["mae"], 0), "MAE sur le test", "cas", kind="hi"),
            cols=4,
        ),
        '<div style="height:1.4rem"></div>',
        '<div class="cols" style="grid-template-columns:repeat(4,1fr);gap:1.6rem;'
        'border-top:1px solid var(--rule);padding-top:1.1rem">'
        '<div><p class="stat-l" style="margin:0 0 .35rem">Réalisé par</p>'
        '<p class="txt">Luvengani Dass<br>Saadia Szapiro</p></div>'
        '<div><p class="stat-l" style="margin:0 0 .35rem">Données</p>'
        '<p class="txt">Our World in Data<br>France · 2020 → 2023</p></div>'
        '<div><p class="stat-l" style="margin:0 0 .35rem">Modèle</p>'
        '<p class="txt">Random Forest<br>11 variables · horizon J+7</p></div>'
        '<div><p class="stat-l" style="margin:0 0 .35rem">Livrable</p>'
        '<p class="txt">API · DVC · MLflow · Airflow<br>Prometheus · Evidently</p></div>'
        '</div>',
    ))


# ══════════════════════════ 02 · le problème ══════════════════════════
def s02(S, D):
    w(rail("cible", "Le problème", "Contexte"))
    w(block(
        _titre("Anticiper une semaine à l’avance, avec ce que l’on sait aujourd’hui"),
        cols(
            bullets([
                "<strong>La question posée.</strong> À partir de l’historique connu le "
                "jour J, estimer le nombre de cas qui sera publié le jour J+7.",
                "<strong>Pourquoi sept jours.</strong> C’est le délai utile : une semaine "
                "suffit pour ajuster une capacité de dépistage, un planning hospitalier ou "
                "une campagne de rappel, et reste assez court pour que le passé récent "
                "porte de l’information.",
                "<strong>La contrainte de rigueur.</strong> Aucune variable postérieure au "
                "jour J ne doit entrer dans le modèle, sous peine de fuite temporelle — une "
                "erreur qui produit d’excellents scores et un modèle inutilisable.",
            ]),
            card("Formulation",
                 '<p class="txt">Régression supervisée sur série temporelle. Une ligne = un '
                 'jour. La cible est la valeur observée sept jours plus tard.</p>',
                 icon_name="cible")
            + card("Ce que ce n’est pas",
                   '<p class="small">Pas un modèle épidémiologique compartimental (SIR, '
                   'SEIR). Pas un modèle causal : il apprend des régularités statistiques, '
                   'il n’explique pas la transmission.</p>', kind="plain"),
            layout="c21",
        ),
    ))


# ══════════════════════════ 03 · les données ══════════════════════════
def s03(S, D):
    w(rail("donnees", "Les données", "Our World in Data"))
    g, d = st.columns([1, 1.35], gap="large")
    with g:
        w(block(
            _titre(f"De {num(S['n_raw'])} lignes mondiales<br>à "
                   f"{num(S['n_total'])} jours français"),
            '<p class="txt">Six colonnes retenues sur l’export public d’OWID, filtrées '
            'sur la France, triées chronologiquement, puis bornées à la fenêtre '
            'd’étude.</p>',
            table([
                ("new_cases", "cas publiés — la cible"),
                ("new_deaths", "décès publiés"),
                ("people_vaccinated", "personnes vaccinées, cumul"),
                ("stringency_index", "sévérité des restrictions, 0–100"),
            ]),
            '<p class="small">Valeurs manquantes comblées par 0, cas négatifs ramenés '
            'à 0 (corrections rétroactives des bulletins).</p>',
        ))
    with d:
        st.plotly_chart(
            charts.serie_complete(D["serie"], S["split_date"], S),
            use_container_width=True, config={"displayModeBar": False},
            key="c_serie",
        )
        w('<p class="small">Fenêtre d’étude : '
          f'{date_fr(pd.Timestamp(S["date_min"]))} → '
          f'{date_fr(pd.Timestamp(S["date_max"]))}. Le découpage est strictement '
          'chronologique — le modèle n’apprend que sur le passé.</p>')


# ══════════════════════════ 04 · diagnostic ══════════════════════════
def s04(S, D):
    w(rail("peigne", "Les données", "Diagnostic"))
    w(block(
        _titre("88 % des journées valent zéro"),
        '<p class="txt" style="max-width:96ch">La France ne publie ses chiffres qu’une '
        'fois par semaine. Les six autres jours sont absents du fichier source — et le '
        'prétraitement les remplace par 0. La série n’est donc pas quotidienne : c’est un '
        '<strong>total hebdomadaire posé sur un seul jour</strong>, entouré de six '
        'zéros.</p>',
    ))
    st.plotly_chart(
        charts.peigne(D["serie"], "2022-09-01", "2022-11-01"),
        use_container_width=True, config={"displayModeBar": False}, key="c_peigne",
    )
    w(block(stats_row(
        stat(num(S["n_zeros_all"]), "Jours à zéro, jeu complet",
             f"/{NBSP}{num(S['n_total'])}"),
        stat(num(S["n_zero_days_test"]), "Jours à zéro, jeu de test",
             f"/{NBSP}{num(S['n_test'])}", kind="warn"),
        stat(str(S["n_publication_days_test"]), "Vraies publications à prédire",
             kind="hi"),
    )))


# ══════════════════════════ 05 · feature engineering ══════════════════════════
def s05(S, D):
    w(rail("fenetre", "Préparation", "Feature engineering"))
    fen = (
        '<div style="display:flex;align-items:center;gap:.35rem;flex-wrap:wrap">'
        + "".join(
            f'<div style="background:var(--accent-bg);color:{ACCENT_DIM};'
            'font-family:\'IBM Plex Mono\',monospace;font-size:.72rem;padding:.55rem .5rem;'
            f'min-width:3.1rem;text-align:center">lag_{k}</div>'
            for k in range(7, 1, -1)
        )
        + f'<div style="background:{ACCENT};color:#fff;font-family:\'IBM Plex Mono\','
        'monospace;font-size:.72rem;padding:.55rem .5rem;min-width:3.1rem;'
        'text-align:center">lag_1</div>'
        + f'<div style="color:{ACCENT};font-size:1.1rem;padding:0 .3rem">→</div>'
        + '<div style="background:var(--warm-bg);color:var(--warm);'
        'font-family:\'IBM Plex Mono\',monospace;font-size:.72rem;padding:.55rem .7rem;'
        'text-align:center">CIBLE J+7</div>'
        "</div>"
    )
    w(block(
        _titre("Onze variables construites par une fenêtre glissante"),
        '<p class="stat-l" style="margin:.4rem 0 .5rem">Entrées — 7 jours connus</p>',
        fen,
        '<p class="small">+ la moyenne mobile de ces sept jours, puis trois variables '
        'de contexte du jour J.</p>',
        '<div style="height:.6rem"></div>',
        cols(
            card("Le décalage de la cible",
                 '<div class="code" style="background:transparent;border:none;padding:0">'
                 'df[<span class="s">"target"</span>] = '
                 'df[<span class="s">"new_cases"</span>].shift(-7)</div>'
                 '<p class="small" style="margin-top:.6rem">Chaque ligne apprend à relier '
                 'son passé immédiat à ce qui arrivera une semaine plus tard.</p>')
            + card("Contexte du jour J",
                   '<p class="txt"><code>new_deaths</code> · '
                   '<code>people_vaccinated</code> · <code>stringency_index</code></p>',
                   kind="plain"),
            card("Après nettoyage",
                 f'<p class="txt"><strong>{num(S["n_total"])}</strong> lignes '
                 f'exploitables sur {num(S["n_country"])} jours de calendrier. '
                 f'{num(S["n_country"] - S["n_total"])} journées, en tête et en queue de '
                 'série, sont écartées : les décalages ou la cible y sont '
                 'incalculables.</p>', kind="plain")
            + '<p class="small">Aucune de ces onze entrées n’est postérieure au jour J : '
              'la contrainte de la diapositive 02 est tenue par construction.</p>',
        ),
    ))


# ══════════════════════════ 06 · le modèle ══════════════════════════
def s06(S, D):
    w(rail("arbre", "Le modèle", "Apprentissage"))
    w(block(
        _titre("Une forêt aléatoire, entraînée dans l’ordre du temps"),
        cols(
            table([
                ("RandomForestRegressor", "scikit-learn"),
                ("n_estimators", f"{S['n_estimators']} arbres"),
                ("random_state", f"{S['random_state']} — résultats reproductibles"),
                ("profondeur", "non bornée (défaut)"),
            ], head=("Choix", "Valeur"))
            + '<p class="small">Un ensemble d’arbres capte des seuils et des interactions '
              'non linéaires sans exiger de normalisation ni d’hypothèse de distribution — '
              'un point de départ solide sur onze variables hétérogènes.</p>',
            card("Le point critique : pas de mélange",
                 '<p class="txt">Le découpage est <strong>chronologique</strong>, jamais '
                 'aléatoire. Mélanger les lignes ferait apprendre au modèle des jours '
                 'postérieurs à ceux qu’il doit prédire : les scores s’envoleraient et le '
                 'modèle ne vaudrait rien en production.</p>', kind="warm",
                 icon_name="limite")
            + stats_row(
                stat(num(S["n_train"]), "Apprentissage", "jours",
                     "mars 2020 → mars 2023"),
                stat(num(S["n_test"]), "Test", "jours", "mars → déc. 2023", kind="hi"),
                cols=2),
        ),
    ))


# ══════════════════════════ 07 · métriques ══════════════════════════
def s07(S, D):
    w(rail("courbe", "Résultats", "Métriques sur le jeu de test"))
    w(block(
        _titre("Le modèle bat nettement les références naïves"),
        stats_row(
            stat(num(S["mae"], 0), "MAE — erreur absolue moyenne", "cas", kind="hi"),
            stat(num(S["rmse"], 0), "RMSE — erreur quadratique", "cas", kind="warn"),
            stat(num(S["r2"], 2), "R² — variance expliquée"),
        ),
    ))
    g, d = st.columns([1, 1.15], gap="large")
    with g:
        st.plotly_chart(charts.baselines(S), use_container_width=True,
                        config={"displayModeBar": False}, key="c_base")
    with d:
        gain_naif = (1 - S["mae"] / S["mae_naive"]) * 100
        gain_zero = (1 - S["mae"] / S["mae_zero"]) * 100
        w(block(bullets([
            f"<strong>−{gain_naif:.0f} % d’erreur</strong> face à la référence « répéter "
            f"la dernière valeur connue » ({num(S['mae_naive'], 0)} cas de MAE).",
            f"<strong>−{gain_zero:.0f} % d’erreur</strong> face à la référence « toujours "
            f"prédire zéro » ({num(S['mae_zero'], 0)} cas) — la barre la plus exigeante, "
            "vu la structure des données.",
            f"Le <strong>rapport RMSE / MAE de "
            f"{S['rmse']/S['mae']:.1f}".replace(".", ",") + "</strong> est le vrai signal "
            "à lire : il dit que l’erreur n’est pas répartie uniformément. La diapositive "
            "09 montre où elle se concentre.",
        ])))


# ══════════════════════════ 08 · prédit vs réel ══════════════════════════
def s08(S, D):
    w(rail("courbe", "Résultats", "Prédit vs réel"))
    w(block(_titre("Le modèle a compris le rythme, pas encore l’amplitude")))
    st.plotly_chart(charts.predit_vs_reel(D["test"]), use_container_width=True,
                    config={"displayModeBar": False}, key="c_pvr")
    a, b, c = st.columns(3, gap="large")
    for col, (t, txt) in zip((a, b, c), [
        ("Le rythme est juste.", "Les pics prédits tombent exactement les jours de "
         "publication : la périodicité hebdomadaire est parfaitement restituée."),
        ("L’amplitude est surestimée.", "Sur les treize publications, la prédiction "
         "dépasse le réel à chaque fois — le modèle reste ancré sur les niveaux, plus "
         "élevés, vus à l’entraînement."),
        ("Puis plus rien.", f"La France cesse toute publication après le "
         f"{date_fr(pd.Timestamp(S['last_publication']))} : les "
         f"{S['trailing_zeros']} derniers jours du test sont des zéros que le modèle "
         "prédit sans effort."),
    ]):
        with col:
            w(f'<p class="small"><strong>{t}</strong> {txt}</p>')


# ══════════════════════════ 09 · analyse de l'erreur ══════════════════════════
def s09(S, D):
    w(rail("ecart", "Résultats", "Analyse de l’erreur"))
    w(block(_titre("Toute l’erreur se concentre sur treize journées")))
    st.plotly_chart(charts.jours_publication(D["test"]), use_container_width=True,
                    config={"displayModeBar": False}, key="c_13j")
    w(block(
        stats_row(
            stat(num(S["mae_zero_days"], 0), f"MAE sur les {S['n_zero_days_test']} "
                 "jours sans publication", "cas"),
            stat(num(S["mae_nz"], 0), f"MAE sur les {S['n_publication_days_test']} "
                 "jours de publication", "cas", kind="warn"),
            stat(num(S["sse_share_nz"], 0), "Part de l’erreur quadratique due à ces "
                 "13 jours", "%", kind="warn"),
        ),
        '<div style="height:.8rem"></div>',
        '<p class="txt" style="max-width:104ch"><strong>Voilà pourquoi la MAE est '
        'flatteuse :</strong> moyennée sur '
        f'{S["n_test"]} jours dont {S["n_zero_days_test"]} sont triviaux, elle dilue '
        f'l’erreur réelle d’un facteur {num(S["mae_nz"] / S["mae"], 1)}.</p>',
    ))


# ══════════════════════════ 10 · importances ══════════════════════════
def s10(S, D):
    w(rail("poids", "Interprétation", "Poids des variables"))
    g, d = st.columns([1, 1.15], gap="large")
    imp = S["importances"]
    with g:
        st.plotly_chart(charts.importances(S), use_container_width=True,
                        config={"displayModeBar": False}, key="c_imp")
    with d:
        w(block(
            _titre("Le modèle a redécouvert seul la cadence hebdomadaire"),
            bullets([
                f"<strong>lag_7 pèse {imp['lag_7']*100:.1f} %</strong>".replace(".", ",")
                + " à lui seul : la dernière publication connue, exactement sept jours "
                "plus tôt, est de loin le meilleur prédicteur.",
                "<strong>lag_1 à lag_6 pèsent 0,0 %</strong> — pas « presque zéro », zéro "
                "exact. Ces six variables valent 0 six fois sur sept : elles ne portent "
                "strictement aucune information.",
                f"La <strong>vaccination cumulée "
                f"({imp['people_vaccinated']*100:.1f} %)</strong>".replace(".", ",")
                + " joue en réalité le rôle d’un indicateur de calendrier : elle croît de "
                "façon monotone et situe la date dans l’épidémie.",
            ]),
            '<p class="small">Ce graphique est le meilleur diagnostic du projet : il '
            'confirme, depuis l’intérieur du modèle, le défaut de structure identifié à '
            'la diapositive 04.</p>',
        ))


# ══════════════════════════ 11 · architecture ══════════════════════════
def s11(S, D):
    w(rail("conteneur", "Architecture", "docker compose"))
    w(block(
        _titre("Huit services, trois boucles : entraîner, servir, surveiller"),
        f'<div style="margin-top:.6rem">{ARCHI_SVG_CLAIR}</div>',
        '<p class="small">Trois exigences, trois réponses : le versioning des données et '
        'des modèles (DVC, MinIO, MLflow), le registre et son cycle de vie (MLflow, '
        'Airflow), le monitoring en production (Prometheus, Alertmanager, Grafana, '
        'Evidently). Les diapositives 13 à 15 les détaillent.</p>',
    ))


# ══════════════════════════ 12 · l'API ══════════════════════════
def s12(S, D):
    w(rail("api", "Mise en service", "API FastAPI"))
    w(block(
        _titre("Cinq points d’entrée, une clé, une réponse"),
        cols(
            table([
                ("GET /health", "modèle chargé, version, origine"),
                ("POST /predict", "prévision — protégé par clé"),
                ("POST /model/reload", "recharge la version en production — protégé"),
                ("GET /metrics", "métriques pour Prometheus"),
                ("GET /docs", "documentation Swagger générée"),
            ], head=("Endpoint", "Rôle"))
            + bullets([
                "<strong>Validation Pydantic</strong> des onze champs : un type erroné "
                "est rejeté avant d’atteindre le modèle.",
                "<strong>Clé API</strong> en en-tête <code>x-api-key</code> ; absente ou "
                "incorrecte, la requête reçoit un <code>401</code>.",
                "<strong>Modèle lu dans le registre</strong> : la version qui porte "
                "l’alias <code>production</code> ; si MLflow est injoignable, repli sur "
                "le fichier local.",
            ]),
            '<p class="stat-l" style="margin:0 0 .4rem">Requête</p>'
            '<div class="code">POST /predict<br>'
            'x-api-key: mlops-covid-secret-key<br><br>{<br>'
            '&nbsp;&nbsp;<span class="s">"lag_1"</span>: 1200, '
            '<span class="c">… "lag_7"</span>: 1500,<br>'
            '&nbsp;&nbsp;<span class="s">"rolling_mean_7"</span>: 1300,<br>'
            '&nbsp;&nbsp;<span class="s">"new_deaths"</span>: 25,<br>'
            '&nbsp;&nbsp;<span class="s">"people_vaccinated"</span>: 40000000,<br>'
            '&nbsp;&nbsp;<span class="s">"stringency_index"</span>: 45.5,<br>'
            '&nbsp;&nbsp;<span class="s">"date"</span>: '
            '<span class="s">"2023-03-01"</span><br>}</div>'
            '<p class="stat-l" style="margin:.9rem 0 .4rem">Réponse</p>'
            '<div class="code" style="border-left-color:var(--warm)">200 OK<br><br>'
            '{ <span class="s">"prediction_new_cases_h_plus_7"</span>: 5300.66,<br>'
            '&nbsp;&nbsp;<span class="s">"model_version"</span>: '
            '<span class="s">"4"</span> }</div>',
        ),
    ))


# ══════════════════════════ 13 · versioning ══════════════════════════
def s13(S, D):
    ent = load_etat_mlops()["entrainement"]
    trace = ""
    if ent:
        trace = table([
            ("Version du modèle", f"covid_rf_model v{ent['model_version']}"),
            ("Empreinte des données", f"{ent['data_md5'][:12]}…"),
            ("Run MLflow", f"{ent['mlflow_run_id'][:12]}…"),
        ], head=("Dernier entraînement", ""))
    w(rail("cycle", "Versioning", "DVC · MinIO · MLflow"))
    w(block(
        _titre("À partir d’un modèle, retrouver ses données et son code"),
        cols(
            card("DVC — les données et le modèle", bullets([
                "Un pipeline en trois étapes : <code>ingest</code>, <code>prepare</code>, "
                "<code>train</code>.",
                "<code>dvc.lock</code> garde l’<strong>empreinte</strong> de chaque "
                "fichier, de chaque script et des paramètres : un commit git désigne "
                "une version exacte du code, des données et du modèle.",
                "<code>dvc repro</code> ne rejoue que les étapes dont une entrée a "
                "changé ; <code>dvc push</code> envoie les fichiers dans "
                "<strong>MinIO</strong>, un stockage compatible S3.",
            ]))
            + '<p class="small">Retour arrière : <code>git checkout</code> puis '
              '<code>dvc checkout</code>.</p>',
            card("MLflow — le lien entre modèle et données", bullets([
                "Chaque entraînement enregistre ses <strong>9 paramètres</strong> et ses "
                "<strong>2 métriques</strong>, MAE et RMSE.",
                "Il enregistre aussi le jeu de données utilisé, son empreinte DVC et "
                "le commit git.",
            ], warm=True), kind="warm")
            + trace,
        ),
    ))


# ══════════════════════════ 14 · registre ══════════════════════════
def s14(S, D):
    promo = load_etat_mlops()["promotion"]
    decision = ""
    if promo:
        decision = table([
            ("Version en production", f"v{promo['production_version']}"),
            ("MAE du dernier candidat", f"{num(promo['candidate_mae'], 2)} cas"),
            ("Décision", promo["reason"]),
        ], head=("Dernière promotion", ""))
    w(rail("cycle", "Registre de modèles", "MLflow · Airflow"))
    w(block(
        _titre("Un modèle n’entre en production que s’il fait au moins aussi bien"),
        cols(
            card("MLflow — un cycle de vie porté par deux alias", bullets([
                "<code>candidate</code> : la dernière version entraînée, à valider.",
                "<code>production</code> : la version servie par l’API.",
                "Le candidat et la production sont évalués <strong>sur le même jeu de "
                "test</strong> ; le candidat ne passe que si sa MAE est inférieure ou "
                "égale. Chaque version porte son verdict : promu, rejeté, remplacé.",
            ]))
            + '<p class="small">Les alias remplacent les étapes Staging / Production, '
              'que MLflow a dépréciées. Retour arrière : déplacer l’alias '
              '<code>production</code>, puis appeler <code>POST /model/reload</code>.</p>',
            card("Airflow — le DAG retrain_covid_model", bullets([
                "Six tâches : <code>ingest_data</code>, <code>prepare_data</code>, "
                "<code>train_model</code>, <code>promote_model</code>, "
                "<code>version_artifacts</code>, <code>reload_api</code>.",
                "Planifié <code>@weekly</code>, à la cadence de publication des "
                "données ; déclenchable à la main.",
                "<code>reload_api</code> fait charger la nouvelle production à l’API, "
                "sans redémarrage.",
            ], warm=True), kind="warm")
            + decision,
        ),
    ))


# ══════════════════════════ 15 · monitoring ══════════════════════════
def s15(S, D):
    mon = load_etat_mlops()["monitoring"]
    controle = ""
    if mon and mon.get("status") == "ok":
        controle = table([
            ("Prédictions analysées", num(mon["analysed_rows"])),
            ("Variables en dérive",
             f"{mon['drifted_features_count']} sur {len(S['features'])}"),
            ("Réentraînement recommandé", "oui" if mon["retrain_recommended"] else "non"),
        ], head=(f"Dernier contrôle · {mon['timestamp'][:10]}", ""))
    w(rail("qualite", "Monitoring", "Prometheus · Evidently · Airflow"))
    w(block(
        _titre("Surveiller le service, surveiller le modèle, réagir seul"),
        cols(
            card("Prometheus — le service", bullets([
                "L’API expose <code>/metrics</code>, collecté toutes les "
                "<strong>15 secondes</strong> : requêtes, temps de réponse, prédictions "
                "par version du modèle.",
                "<strong>7 règles d’alerte</strong> : API indisponible, aucun modèle "
                "chargé, taux d’erreur, latence, dérive, performance dégradée, "
                "monitoring en retard.",
                "<strong>Alertmanager</strong> regroupe les alertes et les achemine ; "
                "<strong>Grafana</strong> les affiche.",
            ]))
            + '<p class="small">Le canal de notification (mail, Slack) reste à '
              'brancher.</p>',
            card("Evidently — le modèle", bullets([
                "Chaque prédiction est journalisée avec ses entrées.",
                "<strong>Dérive</strong> : alerte si au moins la moitié des variables "
                "ont changé de distribution par rapport à l’entraînement.",
                "<strong>Performance</strong> : les prévisions datées sont rapprochées "
                "de la valeur observée à J+7 ; alerte au-delà de 1,5 fois l’erreur "
                "relative de référence.",
                "Le DAG <code>monitor_covid_model</code> tourne <code>@daily</code> et "
                "<strong>relance l’entraînement</strong> si nécessaire.",
            ], warm=True), kind="warm")
            + controle,
        ),
    ))


# ══════════════════════════ 16 · démonstration (interactive) ══════════════════════════
FEATURES = [f"lag_{k}" for k in range(1, 8)] + [
    "rolling_mean_7", "new_deaths", "people_vaccinated", "stringency_index",
]


def _tab_deroule():
    w(block(cols(
        steps([
            "<strong><code>docker compose up -d</code></strong><br>Les huit conteneurs "
            "démarrent ensemble : API, MLflow, MinIO, Airflow, Prometheus, "
            "Alertmanager et Grafana.",
            "<strong>Swagger sur <code>:8000/docs</code></strong><br>L’API se documente "
            "seule : les onze champs attendus, leurs types et un exemple pré-rempli.",
            "<strong><code>GET /health</code></strong><br>Le modèle répond avec sa propre "
            "carte d’identité : version, origine (le registre), pays, horizon, MAE et "
            "RMSE.",
            "<strong><code>POST /predict</code> sans clé → 401</strong><br>La protection "
            "se vérifie en une requête : sans en-tête <code>x-api-key</code>, rien ne "
            "passe.",
        ]),
        steps([
            "<strong><code>POST /predict</code> avec la clé → 200</strong><br>Sur la "
            "charge utile d’exemple, le modèle renvoie <strong>5 300,66 cas</strong> "
            "à J+7.",
            "<strong>MLflow sur <code>:5000</code></strong><br>Les runs côte à côte, et "
            "dans l’onglet Models les versions avec leurs alias <code>candidate</code> "
            "et <code>production</code>.",
            "<strong>Airflow sur <code>:8080</code> → Trigger DAG</strong><br>"
            "<code>retrain_covid_model</code> se déclenche à la main ; ses six tâches "
            "passent au vert, de l’entraînement à la promotion.",
            "<strong>Grafana sur <code>:3000</code></strong><br>Après un trafic de "
            "démonstration et le DAG <code>monitor_covid_model</code>, la dérive "
            "apparaît sur le tableau de bord et l’alerte dans Alertmanager.",
        ], start=4),
    )))
    w(card("Plan de repli",
           '<p class="small">Si un service tarde à démarrer, les trois onglets suivants '
           'rejouent le même récit sans Docker : ils appellent directement le modèle '
           '<code>models/covid_rf_model.joblib</code> livré par le projet.</p>',
           kind="plain", icon_name="qualite"))


def _tab_curseurs(S, D):
    test = D["test"]
    pub = test[test["y_true"] > 0]

    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        w('<p class="stat-l" style="margin:0 0 .3rem">Point de départ</p>')
        choix = st.selectbox(
            "Charger les entrées réelles d’un jour de publication",
            options=list(range(len(pub))),
            format_func=lambda i: date_fr(pub.iloc[i]["date"]),
            key="lab_preset", label_visibility="collapsed",
        )
    base = pub.iloc[choix]

    if st.session_state.get("lab_base_date") != str(base["date"]):
        st.session_state["lab_base_date"] = str(base["date"])
        for f in FEATURES:
            st.session_state[f"sl_{f}"] = float(base[f])

    st.markdown("")
    g, d = st.columns([1.25, 1], gap="large")

    with g:
        w('<p class="stat-l" style="margin:0 0 .5rem">Les sept publications précédentes'
          '</p>')
        lag_cols = st.columns(4)
        for k in range(1, 8):
            with lag_cols[(k - 1) % 4]:
                st.slider(
                    f"lag_{k}", 0.0, 120000.0,
                    value=float(st.session_state.get(f"sl_lag_{k}", base[f"lag_{k}"])),
                    step=500.0, key=f"sl_lag_{k}", format="%.0f",
                )
        w('<p class="stat-l" style="margin:.7rem 0 .5rem">Moyenne mobile et contexte</p>')
        m1, m2 = st.columns(2)
        with m1:
            st.slider("rolling_mean_7", 0.0, 120000.0,
                      value=float(st.session_state.get("sl_rolling_mean_7",
                                                       base["rolling_mean_7"])),
                      step=500.0, key="sl_rolling_mean_7", format="%.0f")
            st.slider("new_deaths", 0.0, 2000.0,
                      value=float(st.session_state.get("sl_new_deaths",
                                                       base["new_deaths"])),
                      step=5.0, key="sl_new_deaths", format="%.0f")
        with m2:
            st.slider("people_vaccinated", 0.0, 60_000_000.0,
                      value=float(st.session_state.get("sl_people_vaccinated",
                                                       base["people_vaccinated"])),
                      step=250_000.0, key="sl_people_vaccinated", format="%.0f")
            st.slider("stringency_index", 0.0, 100.0,
                      value=float(st.session_state.get("sl_stringency_index",
                                                       base["stringency_index"])),
                      step=0.5, key="sl_stringency_index", format="%.1f")

    entrees = {f: float(st.session_state[f"sl_{f}"]) for f in FEATURES}
    valeur = predire(entrees)

    with d:
        if valeur is None:
            st.warning(
                "Modèle introuvable : `models/covid_rf_model.joblib`. "
                "Lancez `python src/train.py` pour le régénérer."
            )
            return
        w(block(stats_row(
            stat(num(valeur, 0), "Prévision à J+7", "cas", kind="hi"), cols=1)))
        ecart = valeur - float(base["y_true"])
        w(block(
            '<div style="height:.5rem"></div>',
            card("Comparaison avec la journée chargée",
                 f'<p class="txt">Réel observé le '
                 f'{date_fr(base["date"] + pd.Timedelta(days=7))} : '
                 f'<strong>{num(base["y_true"], 0)}</strong> cas.<br>'
                 f'Écart de la prévision actuelle : '
                 f'<strong style="color:{WARM if ecart > 0 else GOOD}">'
                 f'{"+" if ecart >= 0 else "−"}{num(abs(ecart), 0)}</strong> cas.</p>',
                 kind="plain"),
        ))
        st.plotly_chart(
            charts.jauge_prediction(valeur, test["y_true"].to_numpy()),
            use_container_width=True, config={"displayModeBar": False},
            key="c_jauge",
        )
        w('<p class="small">La boîte situe les treize publications réellement observées '
          'sur le jeu de test ; le losange, votre prévision.</p>')


def _tab_rejouer(S, D):
    test = D["test"]
    pub = test[test["y_true"] > 0].reset_index(drop=True)
    i = st.select_slider(
        "Jour de publication à rejouer",
        options=list(range(len(pub))),
        format_func=lambda k: date_fr(pub.iloc[k]["date"], court=True),
        key="lab_rejouer",
    )
    ligne = pub.iloc[i]
    reel, prev = float(ligne["y_true"]), float(ligne["y_pred"])
    ecart = prev - reel
    rel = ecart / reel * 100

    w(block(stats_row(
        stat(num(reel, 0), "Réel observé à J+7", "cas", kind="hi"),
        stat(num(prev, 0), "Prédit par le modèle", "cas", kind="warn"),
        stat(("+" if ecart >= 0 else "−") + num(abs(ecart), 0), "Écart", "cas",
             f"soit {'+' if rel >= 0 else '−'}{num(abs(rel), 0)} % du réel"),
    )))
    st.plotly_chart(charts.contexte_date(pub, i), use_container_width=True,
                    config={"displayModeBar": False}, key="c_ctx")
    w('<p class="small">Les barres claires sont les sept entrées connues le jour '
      f'{date_fr(ligne["date"])} ; seule celle de J−7 est non nulle, ce qui illustre '
      'directement la diapositive 10.</p>')


def _tab_api(S, D):
    g, d = st.columns([1, 1.2], gap="large")
    with g:
        w('<p class="stat-l" style="margin:0 0 .4rem">Cible de la requête</p>')
        url = st.text_input("URL de l’API", "http://localhost:8000", key="api_url",
                            label_visibility="collapsed")
        cle = st.text_input("Clé API (en-tête x-api-key)", "mlops-covid-secret-key",
                            key="api_key")
        envoyer_cle = st.checkbox("Envoyer l’en-tête x-api-key", value=True,
                                  key="api_send_key")
        w('<p class="small">Décochez la case pour démontrer le refus : l’API doit '
          'répondre <code>401</code>.</p>')
        b1, b2 = st.columns(2)
        go_health = b1.button("GET /health", key="api_health")
        go_pred = b2.button("POST /predict", key="api_pred", type="primary")

    charge = {
        "lag_1": 1200, "lag_2": 1100, "lag_3": 1300, "lag_4": 1250, "lag_5": 1400,
        "lag_6": 1350, "lag_7": 1500, "rolling_mean_7": 1300, "new_deaths": 25,
        "people_vaccinated": 40000000, "stringency_index": 45.5,
    }

    with d:
        w('<p class="stat-l" style="margin:0 0 .4rem">Réponse du service</p>')
        if not (go_health or go_pred):
            st.info(
                "Aucune requête envoyée. Les conteneurs doivent tourner "
                "(`docker compose up -d`) pour que l’appel aboutisse.",
            )
            corps = json.dumps(charge, indent=4, ensure_ascii=False)
            st.code(
                f"curl -X POST {url.rstrip('/')}/predict \\\n"
                '  -H "Content-Type: application/json" \\\n'
                f'  -H "x-api-key: {cle}" \\\n'
                f"  -d '{corps}'",
                language="bash",
            )
            return

        import requests

        try:
            if go_health:
                r = requests.get(f"{url.rstrip('/')}/health", timeout=6)
            else:
                entetes = {"x-api-key": cle} if envoyer_cle else {}
                r = requests.post(f"{url.rstrip('/')}/predict", json=charge,
                                  headers=entetes, timeout=6)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Impossible de joindre {url} — {type(exc).__name__}.")
            w('<p class="small">Démarrez la pile avec <code>docker compose up -d</code>, '
              'ou utilisez les onglets « Curseurs » et « Rejouer une date » qui '
              'fonctionnent hors ligne.</p>')
            return

        ok = r.status_code == 200
        w(block(stats_row(
            stat(str(r.status_code), "Code HTTP",
                 kind="good" if ok else "warn"),
            stat("oui" if envoyer_cle or go_health else "non", "Clé envoyée"),
            stat(f"{r.elapsed.total_seconds()*1000:.0f}", "Temps de réponse", "ms"),
        )))
        st.markdown("")
        try:
            st.json(r.json())
        except ValueError:
            st.code(r.text)


def _tab_reentrainer(S, D):
    complet = D["complet"]
    dates = complet["date"]

    g, d = st.columns([1, 1.2], gap="large")
    with g:
        w('<p class="stat-l" style="margin:0 0 .4rem">Paramètres de l’essai</p>')
        n_est = st.slider("Nombre d’arbres", 10, 300, S["n_estimators"], 10,
                          key="rt_n")
        part = st.slider("Part d’apprentissage (%)", 60, 95, 80, 5, key="rt_split")
        borne = st.checkbox("Borner la profondeur des arbres", value=False, key="rt_d")
        prof = st.slider("Profondeur maximale", 2, 30, 12, 1, key="rt_prof",
                         disabled=not borne)
        lancer = st.button("Lancer l’entraînement", key="rt_go", type="primary")
        w('<p class="small">L’entraînement rejoue exactement le pipeline du projet sur '
          'les données figées. Le découpage reste chronologique.</p>')

    with d:
        w('<p class="stat-l" style="margin:0 0 .4rem">Résultat</p>')
        if not lancer:
            st.info(
                f"Référence du projet : MAE {num(S['mae'], 2)} · "
                f"RMSE {num(S['rmse'], 2)} · R² {num(S['r2'], 4)}, "
                f"obtenus avec {S['n_estimators']} arbres et 80 % d’apprentissage.",
            )
            return

        n = len(complet)
        ts = int(n * part / 100)
        X = complet[S["features"]]
        y = complet["target"]
        with st.spinner("Entraînement en cours…"):
            mdl = RandomForestRegressor(
                n_estimators=n_est,
                max_depth=prof if borne else None,
                random_state=S["random_state"], n_jobs=-1,
            )
            mdl.fit(X.iloc[:ts], y.iloc[:ts])
            p = mdl.predict(X.iloc[ts:])
            yv = y.iloc[ts:].to_numpy()

        essai = {
            "mae": float(mean_absolute_error(yv, p)),
            "rmse": float(np.sqrt(mean_squared_error(yv, p))),
            "r2": float(1 - ((yv - p) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()),
        }
        d_mae = essai["mae"] - S["mae"]

        identique = abs(d_mae) < 0.5
        w(block(stats_row(
            stat(num(essai["mae"], 0), "MAE de l’essai", "cas",
                 kind="good" if d_mae < 0 else ("" if identique else "warn")),
            stat(num(essai["rmse"], 0), "RMSE de l’essai", "cas"),
            stat(num(essai["r2"], 2), "R² de l’essai"),
        )))
        st.plotly_chart(charts.comparaison_metriques(S, essai),
                        use_container_width=True,
                        config={"displayModeBar": False}, key="c_cmp")
        if identique:
            verdict = ("Ces paramètres sont ceux du projet : l’essai <strong>reproduit "
                       "la référence au centième près</strong> — la graine fixée fait "
                       "son travail.")
        else:
            sens = "descend sous" if d_mae < 0 else "remonte au-dessus de"
            verdict = (f"La MAE {sens} la référence, de "
                       f"<strong>{num(abs(d_mae), 0)}</strong> cas.")
        w(f'<p class="small">Apprentissage sur {num(ts)} jours '
          f'(jusqu’au {date_fr(dates.iloc[ts - 1], court=True)}), test sur '
          f'{num(n - ts)} jours. {verdict}</p>')


def s16(S, D):
    w(rail("direct", "Démonstration", "En direct · laboratoire interactif"))
    w(block(
        _titre("La chaîne en fonctionnement, du démarrage au réentraînement"),
        '<p class="txt" style="max-width:96ch">Les quatre onglets de droite ne rejouent '
        'aucun résultat enregistré : ils appellent le modèle et l’API réellement livrés '
        'par le projet.</p>',
    ))
    t1, t2, t3, t4, t5 = st.tabs([
        "Déroulé de la démo", "Curseurs", "Rejouer une date", "API réelle",
        "Réentraîner",
    ])
    with t1:
        _tab_deroule()
    with t2:
        _tab_curseurs(S, D)
    with t3:
        _tab_rejouer(S, D)
    with t4:
        _tab_api(S, D)
    with t5:
        _tab_reentrainer(S, D)


# ══════════════════════════ 17 · qualité ══════════════════════════
def s17(S, D):
    w(rail("qualite", "Qualité", "Tests & conteneurisation"))
    w(block(
        _titre("Ce qui est vérifié, et ce qui ne l’est pas encore"),
        cols(
            stats_row(
                stat("8 / 8", "Tests pytest réussis", kind="good"),
                stat("0", "Échec"),
                cols=2)
            + table([
                ("/health", "200 · modèle chargé"),
                ("/predict sans clé", "401"),
                ("/predict avec clé, avec ou sans date", "200 · prévision"),
                ("/metrics · /model/reload", "200"),
                ("Construction des variables", "décalages, cible, lignes incomplètes"),
            ], head=("Test", "Attendu"))
            + '<p class="small">Les tests de l’API s’exécutent contre l’application '
              'réelle via le <code>TestClient</code> de FastAPI. Deux tests de plus, sur '
              'les règles de dérive et de performance, s’exécutent dans l’image Airflow, '
              'où Evidently est installé.</p>',
            card("Conteneurisation",
                 '<p class="txt">Trois images construites (API, MLflow, Airflow), cinq '
                 'images publiques et huit services orchestrés par un seul '
                 '<code>docker compose up</code>. Les volumes assurent la persistance : '
                 'modèle, base MLflow, métadonnées Airflow, stockage MinIO.</p>',
                 kind="plain", icon_name="conteneur")
            + card("Angles morts assumés", bullets([
                "Aucun test sur le téléchargement ni sur l’entraînement lui-même.",
                "Pas de test d’intégration bout en bout sur la chaîne de "
                "réentraînement, ni d’intégration continue.",
            ], warm=True), kind="warm"),
        ),
    ))


# ══════════════════════════ 18 · limites ══════════════════════════
def s18(S, D):
    w(rail("limite", "Regard critique", "Limites"))
    w(block(
        _titre("Six limites que nous assumons"),
        cols(
            steps([
                "<strong>La cadence hebdomadaire n’est pas traitée.</strong> Travailler "
                "sur une série « quotidienne » composée à 88 % de zéros rend les "
                "métriques globales flatteuses et peu informatives.",
                "<strong>Six variables sur onze sont inertes.</strong> lag_1 à lag_6 ne "
                "portent aucune information : le modèle en supporte le coût sans aucun "
                "bénéfice.",
                "<strong>Surestimation systématique en décrue.</strong> Sur les treize "
                f"publications testées, l’erreur relative moyenne atteint "
                f"{num(S['mape_nz'], 0)} %.",
            ]),
            steps([
                "<strong>Tout tourne sur une seule machine.</strong> MinIO, MLflow et "
                "Airflow partagent le même poste : ni sauvegarde distante, ni reprise "
                "si la machine tombe.",
                "<strong>Le secret est en clair.</strong> La clé API figure dans "
                "<code>docker-compose.yml</code> et dans le dépôt.",
                "<strong>Les alertes n’ont pas de destinataire.</strong> Elles sont "
                "déclenchées, regroupées et routées, mais aucun canal (mail, Slack) "
                "n’est branché.",
            ], start=3),
        ),
        '<div style="height:.6rem"></div>',
        '<p class="small" style="max-width:104ch">Ces limites ne sont pas des oublis : '
        'elles ont été identifiées en analysant les résultats, et c’est cette analyse '
        'qui a produit les diapositives 04, 09 et 10.</p>',
    ))


# ══════════════════════════ 19 · perspectives ══════════════════════════
def s19(S, D):
    w(rail("cap", "Conclusion", "Prochaines étapes"))
    hebdo = round(S["n_total"] / 7) + 3
    w(block(
        _titre("La chaîne tient ; c’est le signal qu’il faut maintenant redresser"),
        cols(
            '<p class="card-lab" style="color:var(--accent-dim)">Sur la modélisation</p>'
            + bullets([
                "<strong>Rééchantillonner en pas hebdomadaire.</strong> Une série de "
                f"{hebdo} points denses remplacerait {num(S['n_total'])} points dont "
                "88 % de zéros — les décalages redeviendraient tous porteurs "
                "d’information.",
                "<strong>Évaluer sur les seuls jours de publication</strong>, et publier "
                "systématiquement l’écart à une référence naïve.",
                "<strong>Valider par TimeSeriesSplit</strong> plutôt que par un unique "
                "découpage, et comparer la forêt à une régression régularisée et à un "
                "gradient boosting.",
            ]),
            '<p class="card-lab" style="color:var(--warm)">Sur l’industrialisation</p>'
            + bullets([
                "<strong>Brancher un canal de notification</strong> sur Alertmanager, "
                "mail ou Slack.",
                "<strong>Héberger le stockage et le registre</strong> : un S3 et une base "
                "MLflow hors du poste, sauvegardés ; pour DVC, seule l’adresse change.",
                "<strong>Sortir les secrets du dépôt</strong>, dans un coffre ou des "
                "variables injectées au déploiement.",
                "<strong>Brancher les tests sur une intégration continue</strong> et "
                "couvrir la chaîne de réentraînement de bout en bout.",
            ], warm=True),
        ),
        '<div style="height:.4rem"></div>',
        '<p class="lead" style="max-width:100ch">L’infrastructure MLOps est en place et '
        'fonctionne de bout en bout. Le travail restant porte désormais sur la donnée et '
        'sur l’évaluation — c’est-à-dire sur ce qui fait la valeur d’une prévision.</p>',
    ))


# ══════════════════════════ table des matières ══════════════════════════
SLIDES = [
    ("Titre", "virus", s01),
    ("Le problème", "cible", s02),
    ("Les données", "donnees", s03),
    ("Diagnostic des données", "peigne", s04),
    ("Préparation", "fenetre", s05),
    ("Le modèle", "arbre", s06),
    ("Résultats · métriques", "courbe", s07),
    ("Résultats · prédit vs réel", "courbe", s08),
    ("Analyse de l’erreur", "ecart", s09),
    ("Interprétation", "poids", s10),
    ("Architecture MLOps", "conteneur", s11),
    ("Mise en service · API", "api", s12),
    ("Versioning", "cycle", s13),
    ("Registre de modèles", "cycle", s14),
    ("Monitoring", "qualite", s15),
    ("Démonstration en direct", "direct", s16),
    ("Qualité", "qualite", s17),
    ("Regard critique", "limite", s18),
    ("Conclusion", "cap", s19),
]
TOTAL = len(SLIDES)
