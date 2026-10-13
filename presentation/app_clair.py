"""Présentation Streamlit du projet MLOps COVID-19.

Lancement :
    streamlit run presentation/app.py

Version claire de la présentation. Elle ne lit aucun fichier de données : les
chiffres et le laboratoire de la diapositive 16 passent par l'API du projet.
"""

import os
import sys

import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from data import (  # noqa: E402
    DonneesManquantes, load_serie, load_stats, load_test,
)
from theme import CSS, foot, icon, progress  # noqa: E402
import slides as SL  # noqa: E402

st.set_page_config(
    page_title="Prédire le COVID à J+7",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CSS, unsafe_allow_html=True)

def _depuis_url() -> int:
    """?d=7 ouvre directement la diapositive 7 — pratique pour un lien direct."""
    brut = st.query_params.get("d")
    try:
        return max(1, min(SL.TOTAL, int(brut)))
    except (TypeError, ValueError):
        return 1


if "slide" not in st.session_state:
    st.session_state.slide = _depuis_url()


def aller(n: int) -> None:
    st.session_state.slide = max(1, min(SL.TOTAL, n))
    st.query_params["d"] = str(st.session_state.slide)


# ─────────────────────────── données ───────────────────────────
try:
    S = load_stats()
    D = {"serie": load_serie(), "test": load_test()}
except DonneesManquantes as exc:
    st.error(str(exc))
    st.stop()

idx = st.session_state.slide
label, ico, rendu = SL.SLIDES[idx - 1]

# ─────────────────────────── barre latérale ───────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="display:flex;align-items:center;gap:.5rem;margin:.2rem 0 .1rem">'
        + icon("virus", 20)
        + '<span style="font-family:Archivo,sans-serif;font-weight:700;font-size:1.02rem;'
        'letter-spacing:-.01em;color:#101720">Prédire le COVID à J+7</span></div>'
        '<p class="small" style="margin:0 0 1rem">Soutenance MLOps · '
        'Luvengani Dass &amp; Saadia Szapiro</p>',
        unsafe_allow_html=True,
    )

    st.markdown('<p class="sb-t">Sommaire</p>', unsafe_allow_html=True)
    choix = st.radio(
        "Sommaire",
        options=list(range(1, SL.TOTAL + 1)),
        index=idx - 1,
        format_func=lambda n: f"{n:02d} · {SL.SLIDES[n - 1][0]}",
        label_visibility="collapsed",
        key="nav_radio",
    )
    if choix != idx:
        aller(choix)
        st.rerun()

    st.markdown("---")
    st.markdown('<p class="sb-t">Support imprimable</p>', unsafe_allow_html=True)
    pdf = os.path.join(HERE, "presentation-mlops-covid.pdf")
    if os.path.exists(pdf):
        with open(pdf, "rb") as fh:
            st.download_button(
                "Télécharger le PDF (19 pages)", fh.read(),
                file_name="presentation-mlops-covid.pdf", mime="application/pdf",
                use_container_width=True,
            )
    st.markdown(
        '<p class="small">Les chiffres affichés proviennent du pipeline du projet, '
        'rejoué à l’identique — aucune valeur n’est saisie à la main.</p>',
        unsafe_allow_html=True,
    )

# ─────────────────────────── diapositive ───────────────────────────
st.markdown(progress(idx, SL.TOTAL), unsafe_allow_html=True)

rendu(S, D)

st.markdown(foot(label, idx, SL.TOTAL), unsafe_allow_html=True)

# ─────────────────────────── navigation ───────────────────────────
p, milieu, s = st.columns([1, 3, 1])
with p:
    st.button("◀  Précédent", key="prev", disabled=idx == 1,
              on_click=aller, args=(idx - 1,), use_container_width=True)
with s:
    st.button("Suivant  ▶", key="next", disabled=idx == SL.TOTAL,
              on_click=aller, args=(idx + 1,), use_container_width=True,
              type="primary")
