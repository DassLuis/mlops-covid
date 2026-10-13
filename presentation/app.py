"""Prédiction COVID-19 à J+7 — présentation MLOps.

Lancement :
    streamlit run presentation/app.py

L'application ne lit aucun fichier de données : les chiffres, les graphiques et
le simulateur passent tous par l'API du projet (voir presentation/data.py).
"""

import json
import os
import re
import sys
import time

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# =========================
# CONFIG GLOBALE
# =========================
st.set_page_config(
    page_title="COVID-19 · Prédiction à J+7",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",
)

ASSETS_DIR = os.path.join(HERE, "assets")

from studio_css import (  # noqa: E402
    CUSTOM_CSS, STYLE_WOW, CYAN, VIOLET, AMBRE, VERT, ROUGE, ARDOISE, ARDOISE_2,
)
import studio_charts as ch  # noqa: E402
from data import (  # noqa: E402
    DonneesManquantes, date_fr, essai_entrainement, etat_api, load_etat_mlops,
    load_serie, load_stats, load_test, predire, reglages_api,
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =========================
# PETITES FONCTIONS UTILES
# =========================
def get_img_base64(path: str):
    """Encode une image du dossier assets/ pour l'injecter en HTML."""
    if not os.path.exists(path):
        return None
    import base64
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii")


@st.cache_data(show_spinner=False)
def fond_encode():
    """Cherche l'image de fond dans assets/ et l'encode une fois pour toutes.

    `banner_lungs.png` l'emporte sur le `.jpg` dessiné par build_assets.py :
    déposer sa propre illustration suffit donc à remplacer celle-ci.
    """
    for ext, mime in (("png", "png"), ("jpg", "jpeg"), ("jpeg", "jpeg"),
                      ("webp", "webp")):
        b64 = get_img_base64(os.path.join(ASSETS_DIR, f"banner_lungs.{ext}"))
        if b64:
            return mime, b64
    return None


def fond_ecran(opacite: float = 0.45) -> None:
    """Pose les poumons en arrière-plan de la page, derrière tout le contenu.

    L'image est fixée à la fenêtre et s'éteint vers les bords : le texte des
    cartes reste lisible, y compris projeté.
    """
    trouve = fond_encode()
    if not trouve:
        return
    mime, b64 = trouve
    voile = ("radial-gradient(ellipse 78% 70% at 50% 30%,"
             " rgba(0,0,0,1) 0%, rgba(0,0,0,0.55) 52%, rgba(0,0,0,0) 82%)")
    st.markdown(
        f"""
        <style>
        .stApp::before {{
            content: "";
            position: fixed;
            inset: 0;
            z-index: 0;
            pointer-events: none;
            background-image: url("data:image/{mime};base64,{b64}");
            background-size: contain;
            background-position: center top;
            background-repeat: no-repeat;
            opacity: {opacite};
            -webkit-mask-image: {voile};
            mask-image: {voile};
        }}
        /* le contenu et la barre latérale passent devant le fond */
        [data-testid="stHeader"], [data-testid="stSidebar"],
        [data-testid="stMain"], [data-testid="stBottom"],
        [data-testid="stAppViewContainer"] > section {{
            position: relative;
            z-index: 1;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


fond_ecran()


def show_image(name: str, caption: str = "", width: int | None = None) -> None:
    """Affiche une image d'assets/ avec l'effet verre et le survol animé."""
    img_b64 = get_img_base64(os.path.join(ASSETS_DIR, name))
    if not img_b64:
        st.info(f"Image `{name}` introuvable dans `assets/`.")
        return
    w = f"width:{width}px;" if width else "width:100%;"
    st.markdown(
        f'<div class="glass-img-container stagger-1" style="{w}">'
        f'<div class="img-reveal-shine"><img src="data:image/png;base64,{img_b64}">'
        + (f'<div class="img-caption-overlay">{caption}</div>' if caption else "")
        + "</div></div>",
        unsafe_allow_html=True,
    )


def card(wow: bool = True):
    """Ouvre une carte de verre qui enveloppe réellement le bloc Streamlit."""
    box = st.container()
    with box:
        st.markdown(
            f'<span class="card-flag {"reveal" if wow else ""}"></span>',
            unsafe_allow_html=True,
        )
    return box


def animate_metric(label: str, value: float, dec: int = 0, unite: str = "",
                   delta: float | None = None, steps: int = 22,
                   speed: float = 0.012):
    """Montée progressive du chiffre, pour l'effet dashboard.

    Le formatage suit la convention française (espace insécable pour les
    milliers, virgule décimale), comme partout ailleurs dans l'application.
    """
    def fr(v: float) -> str:
        return f"{v:,.{dec}f}".replace(",", " ").replace(".", ",")

    ph = st.empty()
    steps = max(8, min(steps, 40))
    suffixe = f" {unite}" if unite else ""
    for i in range(1, steps + 1):
        v = (value * i) / steps
        if delta is None:
            ph.metric(label, fr(v) + suffixe)
        else:
            d = (delta * i) / steps
            ph.metric(label, fr(v) + suffixe, f"{'+' if d >= 0 else ''}{fr(d)}")
        time.sleep(speed)
    return ph


def metric_fr(col, label: str, valeur, unite: str = "", anime: bool = False,
              dec: int = 0) -> None:
    """st.metric avec séparateur de milliers français, animé si demandé."""
    with col:
        if anime and isinstance(valeur, (int, float)):
            ph = st.empty()
            for i in range(1, 19):
                v = valeur * i / 18
                ph.metric(label, f"{v:,.{dec}f}".replace(",", " ").replace(".", ",")
                          + (f" {unite}" if unite else ""))
                time.sleep(0.012)
        else:
            txt = (f"{valeur:,.{dec}f}".replace(",", " ").replace(".", ",")
                   if isinstance(valeur, (int, float)) else str(valeur))
            st.metric(label, txt + (f" {unite}" if unite else ""))


def num(v, dec: int = 0) -> str:
    return f"{v:,.{dec}f}".replace(",", " ").replace(".", ",")


def html(fragment: str) -> None:
    """Écrit du HTML brut sur une seule ligne.

    Streamlit passe le contenu dans son parseur markdown : un bloc multi-lignes
    (un SVG, par exemple) y perd ses balises. Aplatir règle le problème.
    """
    st.markdown(re.sub(r"\s+", " ", fragment).strip(), unsafe_allow_html=True)


def titre(icone: str, texte: str, niveau: str = "h2") -> None:
    st.markdown(
        f'<{niveau}><i class="fa-solid {icone}" '
        f'style="color:{CYAN}; margin-right:15px;"></i>{texte}</{niveau}>',
        unsafe_allow_html=True,
    )


def kv(paires) -> None:
    lignes = "".join(
        f'<div class="kv-row"><span class="kv-k">{k}</span>'
        f'<span class="kv-v">{v}</span></div>' for k, v in paires
    )
    st.markdown(f'<div class="kv">{lignes}</div>', unsafe_allow_html=True)


def note(texte: str, ton: str = "") -> None:
    st.markdown(f'<div class="note {ton}">{texte}</div>', unsafe_allow_html=True)


def pills(items) -> None:
    st.markdown(
        "".join(f'<span class="pill {c}">{t}</span>' for t, c in items),
        unsafe_allow_html=True,
    )


def finding(col, icone: str, couleur: str, titre_txt: str, corps: str) -> None:
    with col:
        st.markdown(
            f'<div class="finding"><div class="finding-h">'
            f'<i class="fa-solid {icone}" style="color:{couleur}"></i>{titre_txt}</div>'
            f'<div class="finding-b">{corps}</div></div>',
            unsafe_allow_html=True,
        )


def apply_page_fx(page_name: str, wow: bool) -> None:
    """Réglages d'animation légèrement différents selon la page."""
    if not wow:
        return
    if page_name == "Introduction":
        css = ":root{ --cardIn:0.75s; --hoverLift:-5px; }"
    elif page_name == "Résultats & Analyse":
        css = ":root{ --cardIn:0.60s; --hoverLift:-4px; }"
    elif page_name == "Simulateur Interactif":
        css = ":root{ --cardIn:0.50s; --hoverLift:-3px; }"
    else:
        css = ":root{ --cardIn:0.60s; --hoverLift:-4px; }"
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# =========================
# DONNÉES DU PROJET
# =========================
try:
    S = load_stats()
    SERIE = load_serie()
    TEST = load_test()
except DonneesManquantes as exc:
    st.error(str(exc))
    st.stop()

PUB = TEST[TEST["y_true"] > 0].reset_index(drop=True)
FEATURES = [f"lag_{k}" for k in range(1, 8)] + [
    "rolling_mean_7", "new_deaths", "people_vaccinated", "stringency_index",
]

# =========================
# SIDEBAR : NAVIGATION + INFOS
# =========================
with st.sidebar:
    st.markdown(
        '<h2><i class="fa-solid fa-compass" '
        f'style="color:{CYAN}; margin-right:12px;"></i>Navigation</h2>',
        unsafe_allow_html=True,
    )

    # ---------- ILLUSTRATION ANIMÉE ----------
    # Un fichier Lottie déposé dans assets/ prend le dessus ; sinon on affiche
    # l'onde de diffusion en CSS, qui ne dépend d'aucun composant externe.
    lottie_path = os.path.join(ASSETS_DIR, "lottie_sidebar.json")
    rendu_lottie = False
    if os.path.exists(lottie_path):
        try:
            with open(lottie_path, "r", encoding="utf-8") as fh:
                anim = fh.read()
            components.html(
                "<style>html,body{background:transparent;margin:0;overflow:hidden}"
                "#lot{width:100%;height:100%}</style><div id=\"lot\"></div>"
                '<script src="https://cdnjs.cloudflare.com/ajax/libs/'
                'bodymovin/5.12.2/lottie.min.js"></script>'
                "<script>lottie.loadAnimation({container:"
                "document.getElementById('lot'),renderer:'svg',loop:true,"
                f"autoplay:true,animationData:{anim}}});</script>",
                height=150,
            )
            rendu_lottie = True
        except Exception:
            rendu_lottie = False
    if not rendu_lottie:
        st.markdown(
            '<div class="lottie-box"><div class="onde">'
            "<i></i><i></i><i></i><b></b></div></div>",
            unsafe_allow_html=True,
        )

    # ---------- PANNEAU AUTEURS + TOGGLE ----------
    st.markdown('<div class="sidebar-panel">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-auth-title">Équipe projet</div>',
                unsafe_allow_html=True)
    st.markdown(
        """
        <div class="author-card">
            <div class="author-avatar">DL</div>
            <div class="author-info">
                <span class="author-name">Luvengani Dass</span>
                <span class="author-role">Data Scientist</span>
            </div>
        </div>
        <div class="author-card">
            <div class="author-avatar" style="background:linear-gradient(135deg,#fbbf24 0%,#f59e0b 100%);">SS</div>
            <div class="author-info">
                <span class="author-name">Saadia Szapiro</span>
                <span class="author-role">MLOps Engineer</span>
            </div>
        </div>
        <div style="height:1px; background:rgba(255,255,255,0.05); margin:1rem 0;"></div>
        <div class="sidebar-auth-title">Préférences studio</div>
        """,
        unsafe_allow_html=True,
    )
    wow_mode = st.toggle("Effets visuels", value=True,
                         help="Animations avancées et effets de lumière",
                         key="wow_mode")
    st.markdown("</div>", unsafe_allow_html=True)

    PAGES = (
        "Introduction",
        "Données & Prétraitement",
        "Modèle & Entraînement",
        "Architecture MLOps",
        "Résultats & Analyse",
        "Interprétabilité du Modèle",
        "Simulateur Interactif",
        "Conclusion",
    )
    def _depuis_url() -> int:
        """?page=4 ouvre directement la quatrième section — pratique pour un lien."""
        try:
            return max(1, min(len(PAGES), int(st.query_params.get("page")))) - 1
        except (TypeError, ValueError):
            return 0

    page = st.radio("Sections", PAGES, index=_depuis_url(),
                    label_visibility="collapsed")

    # =========================
    # PIPELINE DE SOUTENANCE
    # =========================
    PIPELINE_STEPS = [
        ("Introduction", "Vue d'ensemble"),
        ("Données & Prétraitement", "Données & diagnostic"),
        ("Modèle & Entraînement", "Apprentissage"),
        ("Architecture MLOps", "Industrialisation"),
        ("Résultats & Analyse", "Performance"),
        ("Interprétabilité du Modèle", "Explicabilité"),
        ("Simulateur Interactif", "Simulation live"),
        ("Conclusion", "Synthèse finale"),
    ]
    step_index = next((i for i, (p, _) in enumerate(PIPELINE_STEPS) if p == page), 0)

    if "pipeline_max_step" not in st.session_state:
        st.session_state.pipeline_max_step = 0
    if step_index > st.session_state.pipeline_max_step:
        st.session_state.pipeline_max_step = step_index

    st.markdown('<div class="sidebar-separator"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-auth-title">Pipeline étude</div>',
                unsafe_allow_html=True)

    stepper = '<div class="premium-pipeline-container"><div class="stepper-track">'
    for i, (_, label) in enumerate(PIPELINE_STEPS):
        cls = ""
        if i == step_index:
            cls = "active"
        elif i < step_index:
            cls = "completed"
        elif i <= st.session_state.pipeline_max_step:
            cls = "visited"
        stepper += (f'<div class="stepper-step"><div class="stepper-node {cls}"></div>'
                    f'<div class="stepper-text {cls}">{label}</div></div>')
    stepper += "</div></div>"
    st.markdown(stepper, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-separator"></div>', unsafe_allow_html=True)

    pdf = os.path.join(HERE, "presentation-mlops-covid.pdf")
    if os.path.exists(pdf):
        with open(pdf, "rb") as fh:
            st.download_button("Support PDF · 19 pages", fh.read(),
                               file_name="presentation-mlops-covid.pdf",
                               mime="application/pdf", use_container_width=True)

# =========================
# ACTIVATION WOW + FX PAR PAGE
# =========================
if wow_mode:
    st.markdown(STYLE_WOW, unsafe_allow_html=True)
apply_page_fx(page, wow_mode)

# =========================
# ANIMATION « CHANGEMENT DE PAGE »
# =========================
if "prev_page" not in st.session_state:
    st.session_state.prev_page = page
if page != st.session_state.prev_page:
    with st.spinner("Chargement de la section..."):
        time.sleep(0.25)
    st.session_state.prev_page = page

# =========================
# BANNIÈRE AVEC OVERLAY TEXTE
# =========================
banner_b64 = get_img_base64(os.path.join(ASSETS_DIR, "banner_covid.png"))
if banner_b64:
    st.markdown(
        f"""
        <div style="position:relative; width:100%; height:190px; border-radius:18px;
                    overflow:hidden; margin-bottom:1.4rem;
                    box-shadow:0 18px 40px rgba(2,6,23,0.85);
                    border:1px solid rgba(255,255,255,0.08);">
            <div class="banner-shimmer"></div>
            <div class="banner-breath-overlay"></div>
            <div class="border-beam"></div>
            <img src="data:image/png;base64,{banner_b64}"
                 style="width:100%; height:100%; object-fit:cover; filter:brightness(0.55);">
            <div style="position:absolute; inset:0; display:flex; flex-direction:column;
                        justify-content:center; padding-left:2.5rem; z-index:6;">
                <div style="font-size:0.9rem; letter-spacing:0.2em; text-transform:uppercase;
                            color:{CYAN}; font-weight:600;">
                    Projet MLOps · Séries temporelles
                </div>
                <div class="banner-glow-text"
                     style="font-size:2.2rem; font-weight:800; margin-top:0.4rem; color:#ffffff;">
                    Prédiction des cas COVID-19 à sept jours
                </div>
                <div style="font-size:1.1rem; color:{ARDOISE}; margin-top:0.6rem; font-weight:400;">
                    Our World in Data · France · Random Forest · API conteneurisée
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------- INTRODUCTION ----------
if page == "Introduction":
    c1, c2 = st.columns([2, 1], gap="large")

    with c1:
        with card(wow_mode):
            titre("fa-thumbtack", "Contexte et objectifs du projet", "h3")
            st.markdown(
                """
                Pendant la pandémie, **anticiper la charge à une semaine** conditionne
                tout le reste : capacité de dépistage, planning hospitalier, campagnes
                de rappel. Ce projet construit une **chaîne MLOps complète** qui répond
                à une question précise :

                <div style="margin:14px 0 18px; padding:14px 18px;
                            background:rgba(56,189,248,0.06);
                            border-left:3px solid #38bdf8; border-radius:0 12px 12px 0;">
                    <b>À partir de l'historique connu le jour J, combien de cas seront
                    publiés le jour J+7 ?</b>
                </div>
                """,
                unsafe_allow_html=True,
            )
            titre("fa-bullseye", "Objectifs", "h3")
            st.markdown(
                """
                - Construire un pipeline de bout en bout :
                  **Téléchargement ➜ Prétraitement ➜ Variables ➜ Random Forest ➜ API**
                - Comparer le modèle à des **références naïves**, et pas seulement à
                  lui-même
                - Évaluer par des métriques précises : *MAE, RMSE, R²*
                - **Industrialiser** : conteneurisation, versioning des données et
                  des modèles, registre de modèles, monitoring en production,
                  réentraînement automatique
                - Porter un **regard critique** sur les résultats obtenus
                """
            )

    with c2:
        with card(wow_mode):
            titre("fa-circle-info", "En un coup d'œil", "h3")
            kv([
                ("Source", "Our World in Data"),
                ("Périmètre", f"France · {S['date_min'][:4]} → {S['date_max'][:4]}"),
                ("Horizon", f"J+{S['horizon']}"),
                ("Modèle", f"RandomForest · {S['n_estimators']} arbres"),
                ("Variables", f"{len(S['features'])}"),
                ("Services", "10 conteneurs Docker"),
            ])
            st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
            pills([("Python 3.11", ""), ("scikit-learn", ""), ("FastAPI", "violet"),
                   ("MLflow", "violet"), ("DVC", "violet"), ("Airflow", "amber"),
                   ("Prometheus", "amber"), ("Evidently", "amber"),
                   ("Docker", "green")])

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    metric_fr(m1, "Jours de données", S["n_total"], "j")
    metric_fr(m2, "Variables construites", len(S["features"]))
    metric_fr(m3, "MAE sur le test", S["mae"], "cas")
    metric_fr(m4, "Services conteneurisés", 10)

    st.markdown("---")
    st.markdown(
        f'<div style="color:{ARDOISE}; font-size:0.95rem;">'
        f'<i class="fa-solid fa-terminal" style="color:{CYAN}; margin-right:8px;"></i>'
        "Utilisez le menu latéral pour parcourir le projet : données, modèle, "
        "architecture, résultats, et le simulateur qui appelle le modèle en direct."
        "</div>",
        unsafe_allow_html=True,
    )

# ---------- DONNÉES & PRÉTRAITEMENT ----------
elif page == "Données & Prétraitement":
    titre("fa-magnifying-glass-chart", "Intelligence des données & prétraitement")

    with card(wow_mode):
        t1, t2, t3 = st.tabs(["Jeu de données", "Diagnostic hebdomadaire",
                              "Feature engineering"])

        with t1:
            g, d = st.columns([1, 1.5], gap="large")
            with g:
                titre("fa-database", "De 429 435 lignes à 1 387 jours", "h3")
                st.markdown(
                    "Six colonnes retenues sur l'export public d'OWID, filtrées sur la "
                    "France, triées chronologiquement, puis bornées à la fenêtre d'étude."
                )
                kv([
                    ("<code>new_cases</code>", "cas publiés — la cible"),
                    ("<code>new_deaths</code>", "décès publiés"),
                    ("<code>people_vaccinated</code>", "vaccinés, cumul"),
                    ("<code>stringency_index</code>", "sévérité des restrictions"),
                ])
                st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
                note("Valeurs manquantes comblées par 0, cas négatifs ramenés à 0 — "
                     "les bulletins publient des corrections rétroactives.")
            with d:
                st.plotly_chart(ch.serie_complete(SERIE, S["split_date"], S),
                                use_container_width=True,
                                config={"displayModeBar": False})
                note(f"Fenêtre d'étude : {date_fr(pd.Timestamp(S['date_min']))} → "
                     f"{date_fr(pd.Timestamp(S['date_max']))}. Le découpage est "
                     "strictement chronologique — le modèle n'apprend que sur le passé.")

            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            a, b, c, e = st.columns(4)
            metric_fr(a, "Lignes OWID brutes", S["n_raw"])
            metric_fr(b, "Jours France retenus", S["n_country"])
            metric_fr(c, "Lignes exploitables", S["n_total"])
            metric_fr(e, "Écartées (bords de série)", S["n_country"] - S["n_total"])

        with t2:
            titre("fa-wave-square", "88 % des journées valent zéro", "h3")
            st.markdown(
                "La France ne publie ses chiffres **qu'une fois par semaine**. Les six "
                "autres jours sont absents du fichier source, et le prétraitement les "
                "remplace par 0. La série n'est donc pas quotidienne : c'est un "
                "**total hebdomadaire posé sur un seul jour**, entouré de six zéros."
            )
            st.plotly_chart(ch.peigne(SERIE, "2022-09-01", "2022-11-01"),
                            use_container_width=True,
                            config={"displayModeBar": False})
            a, b, c = st.columns(3)
            metric_fr(a, "Jours à zéro, jeu complet", S["n_zeros_all"])
            metric_fr(b, "Jours à zéro, jeu de test", S["n_zero_days_test"])
            metric_fr(c, "Vraies publications à prédire", S["n_publication_days_test"])
            st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
            note("Ce constat conditionne toute la lecture des résultats : il explique "
                 "l'écart entre MAE et RMSE, et le poids des variables. Il revient aux "
                 "pages Résultats et Interprétabilité.", "amber")

        with t3:
            titre("fa-layer-group", "Onze variables par fenêtre glissante", "h3")
            lags = "".join(
                f'<span class="pill" style="margin-right:6px">lag_{k}</span>'
                for k in range(7, 0, -1)
            )
            st.markdown(
                f'<div style="display:flex;align-items:center;flex-wrap:wrap;'
                f'gap:2px;margin:.6rem 0 1rem">{lags}'
                f'<span style="color:{CYAN};font-size:1.3rem;margin:0 .5rem">→</span>'
                f'<span class="pill amber">CIBLE J+7</span></div>',
                unsafe_allow_html=True,
            )
            g, d = st.columns([1, 1], gap="large")
            with g:
                st.markdown("**Le décalage de la cible**")
                st.code('df["target"] = df["new_cases"].shift(-7)', language="python")
                note("Chaque ligne apprend à relier son passé immédiat à ce qui "
                     "arrivera une semaine plus tard.")
            with d:
                st.markdown("**Les onze entrées**")
                kv([
                    ("7 décalages", "lag_1 … lag_7"),
                    ("1 moyenne mobile", "rolling_mean_7"),
                    ("3 variables de contexte",
                     "new_deaths · people_vaccinated · stringency_index"),
                ])
                note("Aucune de ces entrées n'est postérieure au jour J : la contrainte "
                     "anti-fuite est tenue par construction.", "")

# ---------- MODÈLE & ENTRAÎNEMENT ----------
elif page == "Modèle & Entraînement":
    titre("fa-sitemap", "Modèle et stratégie d'apprentissage")

    g, d = st.columns([1, 1], gap="large")

    with g:
        with card(wow_mode):
            titre("fa-tree", "Une forêt aléatoire", "h3")
            kv([
                ("Algorithme", "RandomForestRegressor"),
                ("Bibliothèque", "scikit-learn"),
                ("n_estimators", f"{S['n_estimators']} arbres"),
                ("random_state", f"{S['random_state']} — reproductible"),
                ("Profondeur", "non bornée (défaut)"),
                ("Métriques", "MAE · RMSE"),
            ])
            st.markdown("<div style='height:.7rem'></div>", unsafe_allow_html=True)
            note("Un ensemble d'arbres capte des seuils et des interactions non "
                 "linéaires sans exiger de normalisation ni d'hypothèse de "
                 "distribution — un point de départ solide sur onze variables "
                 "hétérogènes.")

    with d:
        with card(wow_mode):
            titre("fa-triangle-exclamation", "Le point critique : pas de mélange", "h3")
            st.markdown(
                "Le découpage est **chronologique**, jamais aléatoire. Mélanger les "
                "lignes ferait apprendre au modèle des jours postérieurs à ceux qu'il "
                "doit prédire : les scores s'envoleraient et le modèle ne vaudrait "
                "rien en production."
            )
            a, b = st.columns(2)
            metric_fr(a, "Apprentissage", S["n_train"], "j")
            metric_fr(b, "Test", S["n_test"], "j")
            st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
            note(f"Apprentissage jusqu'au "
                 f"{date_fr(pd.Timestamp(S['split_date']) - pd.Timedelta(days=1))}, "
                 f"test du {date_fr(pd.Timestamp(S['split_date']))} au "
                 f"{date_fr(pd.Timestamp(S['date_max']))}.", "amber")

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    with card(wow_mode):
        titre("fa-ranking-star", "Le modèle face à deux références simples", "h3")
        g2, d2 = st.columns([1.1, 1], gap="large")
        with g2:
            st.plotly_chart(ch.baselines(S), use_container_width=True,
                            config={"displayModeBar": False})
        with d2:
            gain_naif = (1 - S["mae"] / S["mae_naive"]) * 100
            gain_zero = (1 - S["mae"] / S["mae_zero"]) * 100
            st.markdown(
                f"""
                - **−{gain_naif:.0f} % d'erreur** face à « répéter la dernière valeur
                  connue » ({num(S['mae_naive'])} cas de MAE)
                - **−{gain_zero:.0f} % d'erreur** face à « toujours prédire zéro »
                  ({num(S['mae_zero'])} cas) — la barre la plus exigeante, vu la
                  structure des données
                - Comparer à une référence naïve est indispensable : sur une série
                  composée à 88 % de zéros, une MAE basse ne prouve rien à elle seule
                """
            )

# ---------- ARCHITECTURE MLOPS ----------
elif page == "Architecture MLOps":
    titre("fa-diagram-project", "Architecture MLOps")

    with card(wow_mode):
        E = load_etat_mlops()
        t1, t3, t4, t5, t2 = st.tabs(["Vue d'ensemble", "Versioning", "Registre",
                                      "Monitoring", "API FastAPI"])

        with t1:
            titre("fa-cubes", "Dix services, trois boucles : entraîner, servir, "
                  "surveiller", "h3")
            from studio_archi import ARCHI_SVG
            html(ARCHI_SVG)
            note("Trois exigences, trois réponses : le <b>versioning</b> des données et "
                 "des modèles (DVC, MinIO, MLflow), le <b>registre</b> et son cycle de "
                 "vie (MLflow, Airflow), le <b>monitoring</b> en production (Prometheus, "
                 "Alertmanager, Grafana, Evidently). Chacune a son onglet.")

        with t2:
            g, d = st.columns([1, 1], gap="large")
            with g:
                titre("fa-plug", "Cinq points d'entrée, une clé", "h3")
                kv([
                    ("<code>GET /health</code>", "modèle chargé, version, origine"),
                    ("<code>POST /predict</code>", "prévision — protégé par clé"),
                    ("<code>POST /model/reload</code>",
                     "recharge la version en production — protégé"),
                    ("<code>GET /metrics</code>", "métriques pour Prometheus"),
                    ("<code>GET /docs</code>", "documentation Swagger générée"),
                ])
                st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
                st.markdown(
                    """
                    - **Validation Pydantic** des onze champs : un type erroné est
                      rejeté avant d'atteindre le modèle
                    - **Clé API** en en-tête `x-api-key` ; absente ou incorrecte,
                      la requête reçoit un `401`
                    - **Modèle lu dans le registre** : la version qui porte l'alias
                      `production` ; si MLflow est injoignable, repli sur le fichier
                      local
                    - **Date facultative** : elle permet de comparer plus tard la
                      prévision à la valeur observée
                    """
                )
            with d:
                st.markdown("**Requête**")
                st.code(
                    'POST /predict\n'
                    'x-api-key: <votre clé>\n\n'
                    '{\n'
                    '  "lag_1": 1200, ... "lag_7": 1500,\n'
                    '  "rolling_mean_7": 1300,\n'
                    '  "new_deaths": 25,\n'
                    '  "people_vaccinated": 40000000,\n'
                    '  "stringency_index": 45.5,\n'
                    '  "date": "2023-03-01"\n'
                    '}',
                    language="json",
                )
                st.markdown("**Réponse**")
                st.code('200 OK\n\n'
                        '{ "prediction_new_cases_h_plus_7": 5300.66,\n'
                        '  "model_version": "5" }',
                        language="json")

        # ----- versioning : DVC + MLflow -----
        with t3:
            ent = E["entrainement"]
            g, d = st.columns([1, 1], gap="large")
            with g:
                titre("fa-code-branch", "DVC — les données et le modèle", "h3")
                kv([
                    ("<code>ingest</code>", "data/raw/owid_covid.csv"),
                    ("<code>prepare</code>", "data/processed/covid_features.csv"),
                    ("<code>train</code>", "models/covid_rf_model.joblib"),
                ])
                st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
                st.markdown(
                    """
                    - `dvc.lock` garde l'**empreinte** de chaque fichier de données, de
                      chaque script et des paramètres : un commit git désigne une
                      version exacte du code, des données et du modèle
                    - `dvc repro` ne rejoue que les étapes dont une entrée a changé
                    - `dvc push` envoie les fichiers dans **MinIO**, un stockage
                      compatible S3 (`s3://dvc`)
                    - Retour arrière : `git checkout <commit>` puis `dvc checkout`
                    """
                )
            with d:
                titre("fa-link", "MLflow — le lien entre modèle et données", "h3")
                st.markdown(
                    """
                    Chaque entraînement enregistre dans MLflow ses **9 paramètres**,
                    ses **2 métriques**, le jeu de données utilisé, et deux
                    étiquettes : l'empreinte DVC des données et le commit git.
                    """
                )
                if ent:
                    kv([
                        ("Version du modèle", f"covid_rf_model v{ent['model_version']}"),
                        ("Empreinte des données", f"<code>{ent['data_md5'][:12]}…</code>"),
                        ("Run MLflow", f"<code>{ent['mlflow_run_id'][:12]}…</code>"),
                        ("Lignes d'apprentissage · de test",
                         f"{num(ent['train_rows'])} · {num(ent['test_rows'])}"),
                    ])
                note("Git versionne le code, DVC les fichiers lourds, MLflow chaque "
                     "entraînement. À partir d'un modèle, on retrouve les données et le "
                     "code qui l'ont produit.")

        # ----- registre : MLflow + Airflow -----
        with t4:
            promo = E["promotion"]
            g, d = st.columns([1, 1], gap="large")
            with g:
                titre("fa-box-archive", "Un cycle de vie porté par deux alias", "h3")
                kv([
                    ("<code>candidate</code>", "dernière version entraînée, à valider"),
                    ("<code>production</code>", "version servie par l'API"),
                    ("<code>validation_status</code>", "promu · rejeté · remplacé"),
                ])
                st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
                st.markdown(
                    """
                    - **Règle de promotion** : le candidat et la production sont
                      évalués sur le même jeu de test ; le candidat ne passe que si sa
                      MAE est inférieure ou égale
                    - **Retour arrière** : déplacer l'alias `production` vers une
                      version antérieure, puis appeler `POST /model/reload`
                    """
                )
                note("Les alias remplacent les étapes Staging / Production / Archived, "
                     "que MLflow a dépréciées : <code>candidate</code> joue le rôle de "
                     "Staging, l'étiquette <code>remplacé</code> celui d'Archived.",
                     "amber")
            with d:
                titre("fa-rotate", "Airflow — le DAG retrain_covid_model", "h3")
                st.code(
                    "ingest_data ▸ prepare_data ▸ train_model\n"
                    "            ▸ promote_model ▸ version_artifacts ▸ reload_api",
                    language="text",
                )
                st.markdown(
                    """
                    - Planifié `@weekly`, à la cadence de publication des données ;
                      déclenchable à la main
                    - Airflow n'exécute aucun calcul : chaque tâche appelle le
                      **service d'entraînement**, dans son propre conteneur
                    - `train_model` crée une version `candidate` ; `promote_model`
                      applique la règle ; `reload_api` fait charger la nouvelle
                      production à l'API, sans redémarrage
                    """
                )
                if promo:
                    kv([
                        ("Version en production", f"v{promo['production_version']}"),
                        ("MAE du dernier candidat",
                         f"{num(promo['candidate_mae'], 2)} cas"),
                        ("Dernière décision", promo["reason"]),
                    ])

        # ----- monitoring : Prometheus + Evidently + Airflow -----
        with t5:
            mon = E["monitoring"]
            g, d = st.columns([1, 1], gap="large")
            with g:
                titre("fa-gauge-high", "Prometheus — le service sous surveillance", "h3")
                st.markdown(
                    """
                    - L'API expose `/metrics`, collecté toutes les **15 secondes** :
                      requêtes, temps de réponse, prédictions par version du modèle
                    - **7 règles d'alerte** : API indisponible, aucun modèle chargé,
                      taux d'erreur, latence, dérive des données, performance
                      dégradée, monitoring en retard
                    - **Alertmanager** regroupe les alertes et les achemine ;
                      **Grafana** les affiche sur un tableau de bord créé au
                      démarrage
                    """
                )
                note("Le canal de notification (mail, Slack) reste à brancher : les "
                     "alertes sont routées et consultables, pas encore envoyées.",
                     "amber")
            with d:
                titre("fa-wave-square", "Evidently — le modèle sous surveillance", "h3")
                st.markdown(
                    """
                    - Chaque prédiction est **journalisée** avec ses entrées
                    - **Dérive** : les entrées reçues sont comparées aux données
                      d'entraînement ; alerte si au moins la moitié des variables ont
                      changé de distribution
                    - **Performance** : les prévisions datées sont rapprochées de la
                      valeur observée à J+7 ; alerte au-delà de 1,5 fois l'erreur
                      relative de référence
                    - Le DAG `monitor_covid_model` tourne `@daily` et **relance
                      l'entraînement** en cas de dérive ou de dégradation
                    """
                )
                if mon and mon.get("status") == "ok":
                    kv([
                        ("Dernier contrôle", f"{mon['timestamp'][:10]} · "
                                             f"{num(mon['analysed_rows'])} prédictions"),
                        ("Variables en dérive",
                         f"{mon['drifted_features_count']} sur {len(S['features'])}"),
                        ("MAE en production",
                         f"{num(mon['production_mae'], 2)} cas"
                         if mon.get("production_mae") is not None else "non mesurée"),
                        ("Réentraînement recommandé",
                         "oui" if mon["retrain_recommended"] else "non"),
                    ])

# ---------- RÉSULTATS & ANALYSE ----------
elif page == "Résultats & Analyse":
    titre("fa-chart-line", "Performance et analyse de l'erreur")

    with card(wow_mode):
        t1, t2, t3 = st.tabs(["Métriques", "Prédit vs réel", "Les 13 jours qui comptent"])

        with t1:
            a, b, c = st.columns(3)
            if wow_mode:
                with a:
                    animate_metric("MAE", S["mae"], unite="cas")
                with b:
                    animate_metric("RMSE", S["rmse"], unite="cas")
                with c:
                    animate_metric("R²", S["r2"], dec=2)
            else:
                metric_fr(a, "MAE (cas)", S["mae"])
                metric_fr(b, "RMSE (cas)", S["rmse"])
                metric_fr(c, "R²", S["r2"], dec=2)

            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            g, d = st.columns([1, 1.1], gap="large")
            with g:
                st.plotly_chart(ch.regimes_erreur(S), use_container_width=True,
                                config={"displayModeBar": False})
            with d:
                ratio = num(S["rmse"] / S["mae"], 1)
                st.markdown(
                    f"""
                    **Le rapport RMSE / MAE de {ratio} est le vrai signal à lire.**

                    La RMSE pénalise les grosses erreurs ; un rapport aussi élevé dit
                    que l'erreur n'est pas répartie uniformément. En séparant les deux
                    régimes, on voit pourquoi : sur les {S['n_zero_days_test']} jours
                    sans publication, l'erreur moyenne est de {num(S['mae_zero_days'])}
                    cas ; sur les {S['n_publication_days_test']} jours de publication,
                    elle atteint **{num(S['mae_nz'])} cas**.
                    """
                )
                note(f"Ces {S['n_publication_days_test']} journées concentrent "
                     f"{S['sse_share_nz']:.0f} % de l'erreur quadratique totale.",
                     "red")

        with t2:
            st.plotly_chart(ch.predit_vs_reel(TEST), use_container_width=True,
                            config={"displayModeBar": False})
            f1, f2, f3 = st.columns(3, gap="medium")
            finding(f1, "fa-clock-rotate-left", VERT, "Le rythme est juste",
                    "Les pics prédits tombent exactement les jours de publication : "
                    "la périodicité hebdomadaire est parfaitement restituée.")
            finding(f2, "fa-arrow-trend-up", AMBRE, "L'amplitude est surestimée",
                    "Sur les treize publications, la prédiction dépasse le réel à "
                    "chaque fois — le modèle reste ancré sur les niveaux, plus élevés, "
                    "vus à l'entraînement.")
            finding(f3, "fa-ban", ROUGE, "Puis plus rien",
                    f"La France cesse toute publication après le "
                    f"{date_fr(pd.Timestamp(S['last_publication']))} : les "
                    f"{S['trailing_zeros']} derniers jours du test sont des zéros que "
                    "le modèle prédit sans effort.")

        with t3:
            st.plotly_chart(ch.jours_publication(TEST), use_container_width=True,
                            config={"displayModeBar": False})
            a, b, c = st.columns(3)
            metric_fr(a, f"MAE · {S['n_zero_days_test']} jours sans publication",
                      S["mae_zero_days"], "cas")
            metric_fr(b, f"MAE · {S['n_publication_days_test']} jours de publication",
                      S["mae_nz"], "cas")
            metric_fr(c, "Part de l'erreur quadratique", S["sse_share_nz"], "%")
            st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
            note(f"<b>Voilà pourquoi la MAE globale est flatteuse :</b> moyennée sur "
                 f"{S['n_test']} jours dont {S['n_zero_days_test']} sont triviaux, elle "
                 f"dilue l'erreur réelle d'un facteur "
                 f"{num(S['mae_nz'] / S['mae'], 1)}.", "red")

# ---------- INTERPRÉTABILITÉ ----------
elif page == "Interprétabilité du Modèle":
    titre("fa-brain", "Ce que le modèle a réellement appris")

    with card(wow_mode):
        g, d = st.columns([1.1, 1], gap="large")
        with g:
            st.plotly_chart(ch.importances(S), use_container_width=True,
                            config={"displayModeBar": False})
        with d:
            imp = S["importances"]
            titre("fa-lightbulb", "Le modèle a redécouvert la cadence hebdomadaire",
                  "h3")
            pc = {k: num(v * 100, 1) for k, v in imp.items()}
            st.markdown(
                f"""
                - **`lag_7` pèse {pc['lag_7']} %** à lui seul : la dernière publication
                  connue, exactement sept jours plus tôt, est de loin le meilleur
                  prédicteur
                - **`lag_1` à `lag_6` pèsent 0,0 %** — pas « presque zéro », zéro exact.
                  Ces six variables valent 0 six fois sur sept : elles ne portent
                  strictement aucune information
                - **`rolling_mean_7` ({pc['rolling_mean_7']} %)** lisse la tendance
                  récente
                - **`people_vaccinated` ({pc['people_vaccinated']} %)** joue en réalité
                  le rôle d'un indicateur de calendrier : elle croît de façon monotone
                  et situe la date dans l'épidémie
                """
            )
            note("Ce graphique est le meilleur diagnostic du projet : il confirme, "
                 "depuis l'intérieur du modèle, le défaut de structure vu à la page "
                 "Données.", "amber")

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3, gap="medium")
    finding(f1, "fa-magnifying-glass", CYAN, "Une preuve interne",
            "Aucune hypothèse extérieure n'est nécessaire : l'importance des "
            "variables suffit à révéler la périodicité des publications.")
    finding(f2, "fa-scissors", AMBRE, "Six variables inutiles",
            "lag_1 à lag_6 pourraient être retirées sans perte : le modèle serait "
            "plus léger et plus lisible.")
    finding(f3, "fa-calendar-days", VIOLET, "Un proxy de calendrier",
            "La vaccination cumulée n'agit pas comme un facteur causal, mais comme "
            "une horloge — un biais à connaître avant de généraliser.")

# ---------- SIMULATEUR INTERACTIF ----------
elif page == "Simulateur Interactif":
    titre("fa-flask-vial", "Simulateur de prévision en direct")

    sante = etat_api()
    if sante is None or not sante.get("model_loaded"):
        st.warning("L'API ne sert aucun modèle pour l'instant. Démarrez la pile avec "
                   "`docker compose up -d`.")
    else:
        st.success(f"**API connectée** — modèle version {sante['model_version']}, "
                   f"{S['n_estimators']} arbres, horizon J+{S['horizon']}, "
                   f"origine : {sante['model_source']}.")

    with card(wow_mode):
        t1, t2, t3, t4 = st.tabs(["Curseurs", "Rejouer une date", "API réelle",
                                  "Réentraîner"])

        # ---- Curseurs ----
        with t1:
            titre("fa-sliders", "Régler les onze entrées à la main", "h3")
            choix = st.selectbox(
                "Charger les entrées réelles d'un jour de publication",
                options=list(range(len(PUB))),
                format_func=lambda i: date_fr(PUB.iloc[i]["date"]),
                key="lab_preset",
            )
            base = PUB.iloc[choix]
            if st.session_state.get("lab_base_date") != str(base["date"]):
                st.session_state["lab_base_date"] = str(base["date"])
                for f in FEATURES:
                    st.session_state[f"sl_{f}"] = float(base[f])

            g, d = st.columns([1.25, 1], gap="large")
            with g:
                st.markdown("**Les sept publications précédentes**")
                lag_cols = st.columns(4)
                for k in range(1, 8):
                    with lag_cols[(k - 1) % 4]:
                        st.slider(f"lag_{k}", 0.0, 120000.0,
                                  value=float(st.session_state.get(
                                      f"sl_lag_{k}", base[f"lag_{k}"])),
                                  step=500.0, key=f"sl_lag_{k}", format="%.0f")
                st.markdown("**Moyenne mobile et contexte**")
                m1, m2 = st.columns(2)
                with m1:
                    st.slider("rolling_mean_7", 0.0, 120000.0,
                              value=float(st.session_state.get(
                                  "sl_rolling_mean_7", base["rolling_mean_7"])),
                              step=500.0, key="sl_rolling_mean_7", format="%.0f")
                    st.slider("new_deaths", 0.0, 2000.0,
                              value=float(st.session_state.get(
                                  "sl_new_deaths", base["new_deaths"])),
                              step=5.0, key="sl_new_deaths", format="%.0f")
                with m2:
                    st.slider("people_vaccinated", 0.0, 60_000_000.0,
                              value=float(st.session_state.get(
                                  "sl_people_vaccinated", base["people_vaccinated"])),
                              step=250_000.0, key="sl_people_vaccinated", format="%.0f")
                    st.slider("stringency_index", 0.0, 100.0,
                              value=float(st.session_state.get(
                                  "sl_stringency_index", base["stringency_index"])),
                              step=0.5, key="sl_stringency_index", format="%.1f")

            entrees = {f: float(st.session_state[f"sl_{f}"]) for f in FEATURES}
            valeur = predire(entrees)

            with d:
                if valeur is None:
                    st.warning("Modèle indisponible.")
                else:
                    reel = float(base["y_true"])
                    ecart = valeur - reel
                    coul = ROUGE if ecart > 0 else VERT
                    st.markdown(
                        f'<div style="background:rgba(56,189,248,0.08);'
                        f'border-left:5px solid {CYAN}; padding:1.5rem;'
                        'border-radius:0 14px 14px 0; margin-bottom:18px;">'
                        f'<div style="font-size:0.78rem; text-transform:uppercase;'
                        f'color:{CYAN}; letter-spacing:0.1em; font-weight:700;">'
                        "Prévision à J+7</div>"
                        f'<div style="font-size:2.4rem; font-weight:800; color:#fff;'
                        f'line-height:1.15;">{num(valeur)} <span style="font-size:1rem;'
                        f'color:{ARDOISE_2}">cas</span></div>'
                        f'<div style="font-size:0.95rem; color:{ARDOISE}; margin-top:6px">'
                        f'Réel observé le '
                        f'{date_fr(base["date"] + pd.Timedelta(days=7))} : '
                        f'<b>{num(reel)}</b> cas · écart '
                        f'<span style="color:{coul}; font-weight:700">'
                        f'{"+" if ecart >= 0 else "−"}{num(abs(ecart))}</span></div>'
                        "</div>",
                        unsafe_allow_html=True,
                    )
                    st.plotly_chart(
                        ch.jauge_prediction(valeur, TEST["y_true"].to_numpy()),
                        use_container_width=True, config={"displayModeBar": False})
                    note("La boîte situe les treize publications réellement observées "
                         "sur le jeu de test ; le losange, votre prévision.")

        # ---- Rejouer une date ----
        with t2:
            titre("fa-clock-rotate-left", "Rejouer une journée du jeu de test", "h3")
            i = st.select_slider(
                "Jour de publication",
                options=list(range(len(PUB))),
                format_func=lambda k: date_fr(PUB.iloc[k]["date"], court=True),
                key="lab_rejouer",
            )
            ligne = PUB.iloc[i]
            reel, prev = float(ligne["y_true"]), float(ligne["y_pred"])
            ecart = prev - reel
            a, b, c = st.columns(3)
            metric_fr(a, "Réel observé à J+7", reel, "cas")
            metric_fr(b, "Prédit par le modèle", prev, "cas")
            with c:
                # surestimer est un défaut : la flèche montante doit être rouge.
                st.metric("Écart",
                          f"{'+' if ecart >= 0 else '−'}{num(abs(ecart))} cas",
                          f"{ecart / reel * 100:+.0f} % du réel",
                          delta_color="inverse")
            st.plotly_chart(ch.contexte_date(PUB, i), use_container_width=True,
                            config={"displayModeBar": False})
            note(f"Les barres claires sont les sept entrées connues le "
                 f"{date_fr(ligne['date'])} ; <b>seule celle de J−7 est non nulle</b>, "
                 "ce qui illustre directement la page Interprétabilité.", "amber")

        # ---- API réelle ----
        with t3:
            titre("fa-server", "Appeler l'API du projet", "h3")
            g, d = st.columns([1, 1.2], gap="large")
            charge = {
                "lag_1": 1200, "lag_2": 1100, "lag_3": 1300, "lag_4": 1250,
                "lag_5": 1400, "lag_6": 1350, "lag_7": 1500, "rolling_mean_7": 1300,
                "new_deaths": 25, "people_vaccinated": 40000000,
                "stringency_index": 45.5,
            }
            with g:
                url_defaut, cle_defaut = reglages_api()
                url = st.text_input("URL du service", url_defaut, key="api_url")
                cle = st.text_input("Clé API (en-tête x-api-key)", cle_defaut,
                                    type="password", key="api_key")
                envoyer_cle = st.checkbox("Envoyer l'en-tête x-api-key", value=True,
                                          key="api_send_key")
                note("Décochez la case pour démontrer le refus : l'API doit "
                     "répondre <code>401</code>.")
                st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
                b1, b2 = st.columns(2)
                go_health = b1.button("GET /health", key="api_health")
                go_pred = b2.button("POST /predict", key="api_pred")

            with d:
                if not (go_health or go_pred):
                    st.info("Aucune requête envoyée. Les conteneurs doivent tourner "
                            "(`docker compose up -d`) pour que l'appel aboutisse.")
                    st.code(
                        f"curl -X POST {url.rstrip('/')}/predict \\\n"
                        '  -H "Content-Type: application/json" \\\n'
                        f'  -H "x-api-key: {cle}" \\\n'
                        f"  -d '{json.dumps(charge, indent=4)}'",
                        language="bash",
                    )
                else:
                    try:
                        if go_health:
                            r = requests.get(f"{url.rstrip('/')}/health", timeout=6)
                        else:
                            h = {"x-api-key": cle} if envoyer_cle else {}
                            r = requests.post(f"{url.rstrip('/')}/predict",
                                              json=charge, headers=h, timeout=6)
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Impossible de joindre {url} — "
                                 f"{type(exc).__name__}.")
                        note("Démarrez la pile avec <code>docker compose up -d</code> : "
                             "toute l'application passe par l'API.", "red")
                    else:
                        ok = r.status_code == 200
                        coul = VERT if ok else AMBRE
                        st.markdown(
                            f'<div style="background:rgba(255,255,255,0.03);'
                            f'border-left:5px solid {coul}; padding:1.2rem;'
                            'border-radius:0 14px 14px 0; margin-bottom:14px;">'
                            f'<span style="font-size:2rem;font-weight:800;'
                            f'color:{coul}">{r.status_code}</span>'
                            f'<span style="color:{ARDOISE}; margin-left:14px">'
                            f'{r.elapsed.total_seconds()*1000:.0f} ms · clé '
                            f'{"envoyée" if (envoyer_cle or go_health) else "omise"}'
                            "</span></div>",
                            unsafe_allow_html=True,
                        )
                        try:
                            st.json(r.json())
                        except ValueError:
                            st.code(r.text)

        # ---- Réentraîner ----
        with t4:
            titre("fa-arrows-rotate", "Relancer l'entraînement", "h3")
            g, d = st.columns([1, 1.2], gap="large")
            with g:
                n_est = st.slider("Nombre d'arbres", 10, 300, S["n_estimators"], 10,
                                  key="rt_n")
                part = st.slider("Part d'apprentissage (%)", 60, 95, 80, 5,
                                 key="rt_split")
                borne = st.checkbox("Borner la profondeur des arbres", value=False,
                                    key="rt_d")
                prof = st.slider("Profondeur maximale", 2, 30, 12, 1, key="rt_prof",
                                 disabled=not borne)
                lancer = st.button("Lancer l'entraînement", key="rt_go")
                note("L'essai est calculé par l'API, avec le pipeline du projet. Le "
                     "découpage reste chronologique, et le modèle servi ne change pas.")

            with d:
                if not lancer:
                    st.info(f"Référence du projet : MAE {num(S['mae'], 2)} · "
                            f"RMSE {num(S['rmse'], 2)} · R² {num(S['r2'], 4)}, "
                            f"avec {S['n_estimators']} arbres et 80 % "
                            "d'apprentissage.")
                else:
                    essai = None
                    with st.status("Entraînement en cours...", expanded=True) as etat:
                        st.write("Demande envoyée à l'API (POST /lab/train)...")
                        try:
                            essai = essai_entrainement(
                                n_est, part / 100, prof if borne else None)
                        except DonneesManquantes as exc:
                            etat.update(label="Échec de l'essai", state="error")
                            st.error(str(exc))
                        else:
                            st.write(f"Apprentissage sur {num(essai['n_train'])} "
                                     f"jours, test sur {num(essai['n_test'])}.")
                            etat.update(
                                label=f"Terminé en "
                                      f"{num(essai['duration_seconds'], 1)} s",
                                state="complete", expanded=False)
                    if essai is None:
                        st.stop()
                    d_mae = essai["mae"] - S["mae"]
                    a, b, c = st.columns(3)
                    with a:
                        # pas de delta affiché quand l'essai retombe sur la référence
                        ecart_txt = (None if abs(d_mae) < 0.5
                                     else f"{'+' if d_mae > 0 else '−'}"
                                          f"{num(abs(d_mae))} cas")
                        st.metric("MAE de l'essai", f"{num(essai['mae'])} cas",
                                  ecart_txt, delta_color="inverse")
                    metric_fr(b, "RMSE de l'essai", essai["rmse"], "cas")
                    metric_fr(c, "R² de l'essai", essai["r2"], dec=2)
                    st.plotly_chart(ch.comparaison_metriques(S, essai),
                                    use_container_width=True,
                                    config={"displayModeBar": False})
                    if abs(d_mae) < 0.5:
                        note("Ces paramètres sont ceux du projet : l'essai "
                             "<b>reproduit la référence au centième près</b> — la "
                             "graine fixée fait son travail.")
                    else:
                        sens = "descend sous" if d_mae < 0 else "remonte au-dessus de"
                        note(f"La MAE {sens} la référence, de "
                             f"<b>{num(abs(d_mae))}</b> cas.", "amber")

# ---------- CONCLUSION ----------
elif page == "Conclusion":
    titre("fa-flag-checkered", "Synthèse et perspectives")

    with card(wow_mode):
        t1, t2, t3 = st.tabs(["Qualité & tests", "Limites assumées", "Perspectives"])

        with t1:
            g, d = st.columns([1, 1], gap="large")
            with g:
                titre("fa-vial-circle-check", "Ce qui est vérifié", "h3")
                a, b = st.columns(2)
                metric_fr(a, "Tests pytest réussis", "25 / 25")
                metric_fr(b, "Échecs", 0)
                st.markdown("<div style='height:.7rem'></div>", unsafe_allow_html=True)
                kv([
                    ("<code>/health</code>", "200 · modèle chargé"),
                    ("<code>/predict</code> sans clé, ou avec une mauvaise clé", "401"),
                    ("<code>/predict</code> avec clé, avec ou sans date",
                     "200 · prévision"),
                    ("<code>/metrics</code> · <code>/model/reload</code>", "200"),
                    ("Données d'exploration, simulation, essais", "8 tests"),
                    ("Service d'entraînement", "6 tests"),
                    ("Construction des variables", "décalages, cible, lignes incomplètes"),
                ])
                note("Les tests de l'API s'exécutent contre l'application réelle via le "
                     "<code>TestClient</code> de FastAPI. Deux des 25 tests, sur les règles de "
                     "dérive et de performance, s'exécutent dans le service "
                     "d'entraînement, où Evidently est installé. GitHub Actions "
                     "relance l'ensemble à chaque push.")
                st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
                titre("fa-docker", "Conteneurisation", "h3")
                st.markdown(
                    "Quatre images construites (API, MLflow, entraînement, Streamlit), six "
                    "images publiques (Airflow, MinIO et son client, Prometheus, "
                    "Alertmanager, Grafana) et dix "
                    "services orchestrés par un seul `docker compose up`. Les "
                    "volumes assurent la persistance : modèle, base MLflow, "
                    "métadonnées Airflow, stockage MinIO."
                )
            with d:
                st.plotly_chart(ch.radar_projet(S), use_container_width=True,
                                config={"displayModeBar": False})
                note("Lecture honnête du projet : l'infrastructure est solide, la "
                     "couverture de tests et la qualité du signal le sont moins. "
                     "<b>Ces six notes sur 5 sont une appréciation argumentée, pas "
                     "une mesure calculée</b> — contrairement à tout le reste de "
                     "l'application.", "amber")

        with t2:
            titre("fa-triangle-exclamation", "Six limites que nous assumons", "h3")
            l1, l2, l3 = st.columns(3, gap="medium")
            finding(l1, "fa-wave-square", ROUGE, "01 · Cadence non traitée",
                    "Travailler sur une série « quotidienne » composée à 88 % de zéros "
                    "rend les métriques globales flatteuses et peu informatives.")
            finding(l2, "fa-ban", ROUGE, "02 · Six variables inertes",
                    "lag_1 à lag_6 ne portent aucune information : le modèle en "
                    "supporte le coût sans aucun bénéfice.")
            finding(l3, "fa-arrow-trend-down", AMBRE, "03 · Surestimation en décrue",
                    f"Sur les {S['n_publication_days_test']} publications testées, "
                    f"l'erreur relative moyenne atteint {S['mape_nz']:.0f} %.")
            st.markdown("<div style='height:.9rem'></div>", unsafe_allow_html=True)
            l4, l5, l6 = st.columns(3, gap="medium")
            finding(l4, "fa-server", AMBRE, "04 · Tout sur une seule machine",
                    "MinIO, MLflow et Airflow tournent sur le même poste : ni "
                    "sauvegarde distante, ni reprise si la machine tombe.")
            finding(l5, "fa-key", AMBRE, "05 · Authentification minimale",
                    "Une seule clé, partagée par tous les clients, sans expiration : "
                    "impossible de savoir qui appelle, ni de retirer un accès.")
            finding(l6, "fa-bell-slash", AMBRE, "06 · Alertes sans destinataire",
                    "Les alertes sont déclenchées, regroupées et routées, mais aucun "
                    "canal (mail, Slack) n'est branché : il faut aller les lire.")
            st.markdown("<div style='height:.9rem'></div>", unsafe_allow_html=True)
            note("Ces limites ne sont pas des oublis : elles ont été identifiées en "
                 "analysant les résultats, et c'est cette analyse qui a produit les "
                 "pages Données, Résultats et Interprétabilité.")

        with t3:
            hebdo = round(S["n_total"] / 7) + 3
            g, d = st.columns([1, 1], gap="large")
            with g:
                titre("fa-chart-simple", "Sur la modélisation", "h3")
                st.markdown(
                    f"""
                    - **Rééchantillonner en pas hebdomadaire.** Une série de {hebdo}
                      points denses remplacerait {num(S['n_total'])} points dont 88 %
                      de zéros — les décalages redeviendraient tous porteurs
                      d'information
                    - **Évaluer sur les seuls jours de publication**, et publier
                      systématiquement l'écart à une référence naïve
                    - **Valider par TimeSeriesSplit** plutôt que par un unique
                      découpage, et comparer la forêt à une régression régularisée et à
                      un gradient boosting
                    """
                )
            with d:
                titre("fa-gears", "Sur l'industrialisation", "h3")
                st.markdown(
                    """
                    - **Brancher un canal de notification** sur Alertmanager (mail ou
                      Slack) — c'est la réponse directe à la limite 06
                    - **Héberger le stockage et le registre** : un S3 et une base
                      MLflow hors du poste, sauvegardés ; pour DVC, seule l'adresse
                      change
                    - **Renforcer l'authentification** : un jeton par client, à durée
                      limitée (OAuth2 et JWT)
                    - **Prolonger l'intégration continue** jusqu'au déploiement, et
                      couvrir la chaîne de réentraînement par un test de bout en bout
                    """
                )
            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            st.markdown(
                f'<div style="background:rgba(56,189,248,0.06);'
                f'border:1px solid rgba(56,189,248,0.18); border-radius:18px;'
                'padding:1.4rem 1.6rem;">'
                f'<div style="font-size:1.15rem; color:#f1f5f9; font-weight:600;'
                'line-height:1.6;">'
                "L'infrastructure MLOps est en place et fonctionne de bout en bout. "
                "Le travail restant porte désormais sur la donnée et sur l'évaluation "
                "— c'est-à-dire sur ce qui fait la valeur d'une prévision.</div>"
                f'<div style="color:{ARDOISE_2}; font-size:0.85rem; margin-top:12px;">'
                "Luvengani Dass · Saadia Szapiro — merci de votre attention</div>"
                "</div>",
                unsafe_allow_html=True,
            )
