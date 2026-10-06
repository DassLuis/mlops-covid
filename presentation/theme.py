"""Charte visuelle de la présentation : palette, typographie, icônes, fragments HTML.

Les valeurs reprennent à l'identique celles du support PDF, pour que les deux
livrables forment un seul système visuel.
"""

# ─────────────────────────── palette ───────────────────────────
PAPER = "#ffffff"
GROUND = "#dfe4ea"
INK = "#101720"
INK2 = "#495563"
INK3 = "#7d8896"
RULE = "#dde2e9"
RULE_FIRM = "#b9c2cd"
ACCENT = "#2a78d6"
ACCENT_DIM = "#1c5cab"
ACCENT_SOFT = "#9ec5f4"
ACCENT_BG = "#eaf2fd"
WARM = "#d2501f"
WARM_CHART = "#eb6834"
WARM_BG = "#fceee8"
GOOD = "#14805a"
GOOD_BG = "#e6f5ef"
PANEL = "#f4f6f9"
GRID = "#e3e2dd"

F_DISPLAY = '"Archivo", "Helvetica Neue", Arial, sans-serif'
F_BODY = '"IBM Plex Sans", "Segoe UI", Helvetica, Arial, sans-serif'
F_MONO = '"IBM Plex Mono", Consolas, monospace'


# ─────────────────────────── icônes ───────────────────────────
# Jeu dessiné au trait fin, 24×24, stroke currentColor — cohérent d'un bout
# à l'autre et rendu identique sur tous les systèmes.
def _svg(body: str) -> str:
    return (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" '
        'aria-hidden="true">' + body + "</svg>"
    )


ICONS = {
    # virus : cercle + spicules — le sujet
    "virus": _svg(
        '<circle cx="12" cy="12" r="5"/><path d="M12 7V3.5M12 17v3.5M7 12H3.5M17 12h3.5'
        'M8.5 8.5 6 6M15.5 8.5 18 6M8.5 15.5 6 18M15.5 15.5 18 18"/>'
    ),
    # question : cible et flèche — le problème posé
    "cible": _svg(
        '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3.4"/>'
        '<path d="M12 4V2M12 22v-2M4 12H2M22 12h-2"/>'
    ),
    # base de données — les données brutes
    "donnees": _svg(
        '<ellipse cx="12" cy="6" rx="7.5" ry="3"/>'
        '<path d="M4.5 6v12c0 1.7 3.4 3 7.5 3s7.5-1.3 7.5-3V6"/>'
        '<path d="M4.5 12c0 1.7 3.4 3 7.5 3s7.5-1.3 7.5-3"/>'
    ),
    # peigne hebdomadaire — le diagnostic
    "peigne": _svg(
        '<path d="M3 20h18"/><path d="M5 20v-5M9 20v-9M13 20v-6M17 20v-11M21 20v-4"/>'
    ),
    # fenêtre glissante — la préparation
    "fenetre": _svg(
        '<path d="M3 5v14M7 5v14"/><rect x="10" y="7" width="11" height="10" rx="1"/>'
        '<path d="M14 12h4M16.5 9.8 18.6 12l-2.1 2.2"/>'
    ),
    # arbre de décision — le modèle
    "arbre": _svg(
        '<circle cx="12" cy="4.5" r="2.2"/><circle cx="6" cy="13" r="2.2"/>'
        '<circle cx="18" cy="13" r="2.2"/><circle cx="18" cy="20" r="1.8"/>'
        '<path d="M10.6 6.3 7.2 11.2M13.4 6.3l3.4 4.9M18 15.2v3"/>'
    ),
    # courbe de résultats
    "courbe": _svg(
        '<path d="M3 20h18"/><path d="M3 20V4"/>'
        '<path d="M6 16l3.5-5 3 3L20 6"/><circle cx="20" cy="6" r="1.6"/>'
    ),
    # écart mesuré — analyse de l'erreur
    "ecart": _svg(
        '<path d="M3 20h18"/><path d="M6 20V9M11 20V5M16 20v-8M21 20v-4"/>'
        '<path d="M6 9 21 4" stroke-dasharray="3 2.5"/>'
    ),
    # poids des variables
    "poids": _svg(
        '<path d="M4 4v16"/><path d="M4 7h14M4 12h8M4 17h4"/>'
    ),
    # conteneurs — architecture
    "conteneur": _svg(
        '<rect x="3" y="10" width="5.5" height="5.5" rx=".6"/>'
        '<rect x="9.2" y="10" width="5.5" height="5.5" rx=".6"/>'
        '<rect x="9.2" y="4" width="5.5" height="5.5" rx=".6"/>'
        '<path d="M2 18.5c1.6 1.7 4 2.5 7.5 2.5 6 0 9.5-2.8 10.5-7 -1.6.8-3.2.8-4.5.2"/>'
    ),
    # prise réseau — l'API
    "api": _svg(
        '<path d="M9 3v5M15 3v5"/><rect x="6" y="8" width="12" height="6" rx="1.5"/>'
        '<path d="M12 14v3.5a3.5 3.5 0 0 0 3.5 3.5H17"/>'
    ),
    # cycle de réentraînement
    "cycle": _svg(
        '<path d="M20.5 12a8.5 8.5 0 1 1-2.6-6.1"/><path d="M20.8 3.6v5h-5"/>'
        '<path d="M12 8v4.4l2.8 1.8"/>'
    ),
    # démonstration en direct
    "direct": _svg(
        '<circle cx="12" cy="12" r="2.6"/>'
        '<path d="M7.4 7.4a6.5 6.5 0 0 0 0 9.2M16.6 16.6a6.5 6.5 0 0 0 0-9.2"/>'
        '<path d="M4.6 4.6a10.4 10.4 0 0 0 0 14.8M19.4 19.4a10.4 10.4 0 0 0 0-14.8"/>'
    ),
    # contrôle qualité
    "qualite": _svg(
        '<path d="M12 3l7 3v5.5c0 4.3-2.9 8.1-7 9.5-4.1-1.4-7-5.2-7-9.5V6z"/>'
        '<path d="M9 12l2.2 2.2L15.5 10"/>'
    ),
    # limites assumées
    "limite": _svg(
        '<path d="M12 4.5 21 19H3z"/><path d="M12 10v4"/><circle cx="12" cy="16.6" r=".9"/>'
    ),
    # cap à tenir — conclusion
    "cap": _svg(
        '<circle cx="12" cy="12" r="8.5"/><path d="M15.5 8.5 10.8 10.8 8.5 15.5l4.7-2.3z"/>'
    ),
    # laboratoire — simulation
    "labo": _svg(
        '<path d="M10 3h4M11 3v6.2L5.6 18A2 2 0 0 0 7.3 21h9.4a2 2 0 0 0 1.7-3L13 9.2V3"/>'
        '<path d="M8.4 14h7.2"/>'
    ),
    "cle": _svg(
        '<circle cx="7.5" cy="12" r="3.5"/><path d="M11 12h9.5M17 12v3M20 12v2.4"/>'
    ),
    "curseur": _svg(
        '<path d="M4 7h16M4 12h16M4 17h16"/><circle cx="9" cy="7" r="2.1"/>'
        '<circle cx="15" cy="12" r="2.1"/><circle cx="7" cy="17" r="2.1"/>'
    ),
    "calendrier": _svg(
        '<rect x="3.5" y="5" width="17" height="16" rx="1.5"/>'
        '<path d="M3.5 10h17M8 3v4M16 3v4"/><circle cx="12" cy="15" r="1.4"/>'
    ),
}


def icon(name: str, size: int = 18, color: str = ACCENT) -> str:
    """Renvoie l'icône en HTML, dimensionnée et colorée."""
    svg = ICONS.get(name, ICONS["virus"])
    return (
        f'<span class="ico" style="width:{size}px;height:{size}px;color:{color}">'
        f"{svg}</span>"
    )


# ─────────────────────────── feuille de style ───────────────────────────
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {{
  --paper:{PAPER}; --ground:{GROUND}; --ink:{INK}; --ink2:{INK2}; --ink3:{INK3};
  --rule:{RULE}; --rule-firm:{RULE_FIRM}; --accent:{ACCENT}; --accent-dim:{ACCENT_DIM};
  --accent-bg:{ACCENT_BG}; --warm:{WARM}; --warm-bg:{WARM_BG};
  --good:{GOOD}; --good-bg:{GOOD_BG}; --panel:{PANEL};
}}

/* ── fond « bureau » et diapositive « papier » ── */
[data-testid="stAppViewContainer"] {{ background:{GROUND}; }}
[data-testid="stHeader"] {{ background:transparent; }}
#MainMenu, [data-testid="stToolbar"] {{ visibility:hidden; }}
footer {{ visibility:hidden; }}

[data-testid="stMainBlockContainer"], .main .block-container {{
  background:{PAPER};
  max-width:1240px;
  padding:2.6rem 3.2rem 2.2rem;
  margin-top:1.1rem; margin-bottom:2.6rem;
  box-shadow:0 1px 2px rgba(16,23,32,.14), 0 14px 38px rgba(16,23,32,.10);
}}

html, body, [data-testid="stAppViewContainer"] * {{ font-family:{F_BODY}; }}

/* ── animations d'entrée ── */
@keyframes rise {{ from {{ opacity:0; transform:translateY(15px); }}
                   to   {{ opacity:1; transform:none; }} }}
@keyframes fadein {{ from {{ opacity:0; }} to {{ opacity:1; }} }}
@keyframes wipe {{ from {{ transform:scaleX(0); }} to {{ transform:scaleX(1); }} }}
@keyframes pop {{ 0% {{ opacity:0; transform:scale(.94); }}
                  70% {{ transform:scale(1.015); }}
                  100% {{ opacity:1; transform:scale(1); }} }}

.anim > * {{ animation:rise .52s cubic-bezier(.22,.72,.24,1) both; }}
.anim > *:nth-child(1) {{ animation-delay:.02s; }}
.anim > *:nth-child(2) {{ animation-delay:.09s; }}
.anim > *:nth-child(3) {{ animation-delay:.16s; }}
.anim > *:nth-child(4) {{ animation-delay:.23s; }}
.anim > *:nth-child(5) {{ animation-delay:.30s; }}
.anim > *:nth-child(6) {{ animation-delay:.37s; }}
.anim > *:nth-child(7) {{ animation-delay:.44s; }}
.anim > *:nth-child(8) {{ animation-delay:.51s; }}

[data-testid="stPlotlyChart"] {{ animation:fadein .75s .18s both; }}
[data-testid="stMetric"] {{ animation:pop .5s .1s both; }}

@media (prefers-reduced-motion:reduce) {{
  .anim > *, [data-testid="stPlotlyChart"], [data-testid="stMetric"],
  .stat, .rail-rule i {{ animation:none !important; }}
}}

/* ── rail supérieur ── */
.rail {{ display:flex; align-items:center; justify-content:space-between;
        gap:1.4rem; padding-bottom:.55rem; }}
.rail-l {{ display:flex; align-items:center; gap:.55rem; }}
.ico {{ display:inline-flex; align-items:center; justify-content:center;
        vertical-align:-.22em; }}
.ico svg {{ width:100%; height:100%; }}
.eyebrow {{ font-family:{F_MONO}; font-size:.72rem; font-weight:500;
            letter-spacing:.16em; text-transform:uppercase; color:{ACCENT}; }}
.eyebrow.dim {{ color:{INK3}; }}
.rail-rule {{ position:relative; height:1px; background:{RULE};
              margin:0 0 1.5rem; overflow:hidden; }}
.rail-rule i {{ position:absolute; inset:0; background:{ACCENT};
                transform-origin:left; animation:wipe .7s cubic-bezier(.3,.8,.3,1) both; }}

/* ── typographie ── */
.h1 {{ font-family:{F_DISPLAY}; font-weight:700; font-size:3.35rem; line-height:1.04;
       letter-spacing:-.028em; color:{INK}; margin:0; text-wrap:balance; }}
.h2 {{ font-family:{F_DISPLAY}; font-weight:700; font-size:1.92rem; line-height:1.14;
       letter-spacing:-.018em; color:{INK}; margin:0 0 .55rem; text-wrap:balance; }}
.lead {{ font-size:1.08rem; line-height:1.56; color:{INK2}; margin:.7rem 0 0;
         max-width:64ch; }}
.txt {{ font-size:.94rem; line-height:1.62; color:{INK2}; margin:0; }}
.small {{ font-size:.83rem; line-height:1.55; color:{INK3}; margin:0; }}
.txt strong, .small strong, .lead strong, li strong {{ color:{INK}; font-weight:600; }}
code, .mono {{ font-family:{F_MONO}; font-size:.88em; color:{INK};
               background:transparent; }}

/* ── listes ── */
ul.bul {{ margin:0; padding:0; list-style:none; display:flex; flex-direction:column;
          gap:.7rem; }}
ul.bul > li {{ position:relative; padding-left:1.25rem; font-size:.94rem;
               line-height:1.58; color:{INK2}; }}
ul.bul > li::before {{ content:""; position:absolute; left:0; top:.62em; width:.4rem;
                       height:.4rem; border-radius:50%; background:{ACCENT}; opacity:.8; }}
ul.bul.warm > li::before {{ background:{WARM}; }}

ol.steps {{ margin:0; padding:0; list-style:none; counter-reset:st;
            display:flex; flex-direction:column; gap:.85rem; }}
ol.steps > li {{ counter-increment:st; position:relative; padding-left:2.1rem;
                 font-size:.92rem; line-height:1.55; color:{INK2}; }}
ol.steps > li::before {{ content:counter(st,decimal-leading-zero); position:absolute;
  left:0; top:.1em; font-family:{F_MONO}; font-size:.76rem; font-weight:600;
  color:{ACCENT}; letter-spacing:.02em; }}

/* ── cartouches ── */
.card {{ background:{ACCENT_BG}; padding:1.05rem 1.2rem; }}
.card.warm {{ background:{WARM_BG}; }}
.card.plain {{ background:{PANEL}; }}
.card.good {{ background:{GOOD_BG}; }}
.card-lab {{ font-family:{F_MONO}; font-size:.7rem; font-weight:600; letter-spacing:.13em;
             text-transform:uppercase; color:{ACCENT_DIM}; margin:0 0 .5rem;
             display:flex; align-items:center; gap:.45rem; }}
.card.warm .card-lab {{ color:{WARM}; }}
.card.plain .card-lab {{ color:{INK3}; }}
.card.good .card-lab {{ color:{GOOD}; }}

/* ── tuiles de chiffres ── */
.stats {{ display:grid; gap:2px; }}
.stat {{ background:{PANEL}; padding:1rem 1.15rem; animation:pop .5s both; }}
.stat.hi {{ background:{ACCENT_BG}; }}
.stat.warn {{ background:{WARM_BG}; }}
.stat.good {{ background:{GOOD_BG}; }}
.stat-v {{ font-family:{F_DISPLAY}; font-weight:700; font-size:2.1rem; line-height:1;
           letter-spacing:-.02em; color:{INK}; font-variant-numeric:tabular-nums; }}
.stat.hi .stat-v {{ color:{ACCENT_DIM}; }}
.stat.warn .stat-v {{ color:{WARM}; }}
.stat.good .stat-v {{ color:{GOOD}; }}
.stat-v .u {{ font-size:.42em; font-weight:600; letter-spacing:0; color:{INK3};
              margin-left:.25em; }}
.stat-l {{ font-family:{F_MONO}; font-size:.65rem; font-weight:500; letter-spacing:.1em;
           text-transform:uppercase; color:{INK3}; margin-top:.5rem; }}
.stat-s {{ font-size:.8rem; color:{INK2}; margin-top:.3rem; }}

/* ── tableaux ── */
table.t {{ width:100%; border-collapse:collapse; font-size:.88rem; }}
table.t th {{ text-align:left; font-family:{F_MONO}; font-weight:600; font-size:.66rem;
  letter-spacing:.11em; text-transform:uppercase; color:{INK3};
  padding:0 .9rem .5rem 0; border-bottom:1px solid {RULE_FIRM}; }}
table.t td {{ padding:.5rem .9rem .5rem 0; border-bottom:1px solid {RULE};
              color:{INK2}; vertical-align:top; }}
table.t td.k {{ font-family:{F_MONO}; color:{INK}; white-space:nowrap; }}
table.t tr:last-child td {{ border-bottom:none; }}

/* ── code ── */
div.code {{ margin:0; background:{PANEL}; padding:.9rem 1.05rem; overflow-x:auto;
  font-family:{F_MONO}; font-size:.82rem; line-height:1.7; color:{INK};
  border-left:2px solid {ACCENT}; white-space:pre-wrap; }}
div.code .s {{ color:{ACCENT_DIM}; }}
div.code .c {{ color:{INK3}; }}

/* Streamlit colore les <code> en ligne : on reprend la main. */
[data-testid="stMarkdownContainer"] code, [data-testid="stMarkdown"] code {{
  font-family:{F_MONO} !important; font-size:.86em !important;
  color:{INK} !important; background:{PANEL} !important;
  padding:.05em .3em; border-radius:2px; }}

/* ── grille interne ── */
.cols {{ display:grid; gap:2.1rem; align-items:start; }}
.c11 {{ grid-template-columns:1fr 1fr; }}
.c21 {{ grid-template-columns:1.35fr 1fr; }}
.stack {{ display:flex; flex-direction:column; gap:1rem; }}

/* ── pied de diapositive ── */
.foot {{ display:flex; align-items:baseline; justify-content:space-between;
  border-top:1px solid {RULE}; margin-top:1.6rem; padding-top:.6rem;
  font-family:{F_MONO}; font-size:.68rem; letter-spacing:.05em; color:{INK3}; }}
.foot .n {{ font-weight:600; color:{INK2}; font-variant-numeric:tabular-nums; }}

/* ── barre latérale ── */
[data-testid="stSidebar"] {{ background:{PAPER}; border-right:1px solid {RULE}; }}
[data-testid="stSidebar"] .sb-t {{ font-family:{F_MONO}; font-size:.66rem;
  font-weight:600; letter-spacing:.15em; text-transform:uppercase; color:{INK3};
  margin:.2rem 0 .5rem; }}
[data-testid="stSidebar"] [data-testid="stRadio"] label p {{ font-size:.83rem !important; }}
[data-testid="stSidebar"] [data-testid="stRadio"] label {{ padding:.05rem 0; }}

/* progression */
.prog {{ height:3px; background:{RULE}; margin:.15rem 0 1rem; }}
.prog i {{ display:block; height:100%; background:{ACCENT};
           transition:width .5s cubic-bezier(.3,.8,.3,1); }}

/* ── contrôles Streamlit ── */
.stButton > button {{
  font-family:{F_MONO}; font-size:.76rem; font-weight:600; letter-spacing:.06em;
  border-radius:2px; border:1px solid {RULE_FIRM}; background:{PAPER}; color:{INK2};
  padding:.4rem .9rem; transition:all .18s ease;
}}
.stButton > button:hover {{ border-color:{ACCENT}; color:{ACCENT_DIM};
                            background:{ACCENT_BG}; transform:translateY(-1px); }}
.stButton > button[kind="primary"] {{ background:{ACCENT}; border-color:{ACCENT};
                                      color:#fff; }}
.stButton > button[kind="primary"]:hover {{ background:{ACCENT_DIM};
                                            border-color:{ACCENT_DIM}; color:#fff; }}
[data-testid="stTabs"] button p {{ font-family:{F_MONO}; font-size:.76rem;
  letter-spacing:.05em; }}
[data-testid="stMetricValue"] {{ font-family:{F_DISPLAY}; font-weight:700;
                                 letter-spacing:-.02em; }}
[data-testid="stMetricLabel"] p {{ font-family:{F_MONO}; font-size:.66rem !important;
  letter-spacing:.1em; text-transform:uppercase; color:{INK3}; }}
</style>
"""


# ─────────────────────────── fragments HTML ───────────────────────────
NBSP = " "  # espace insécable, séparateur de milliers français


def num(v, dec: int = 0) -> str:
    """Formate un nombre à la française : 16 576 · 942,72 · 0,68."""
    s = f"{v:,.{dec}f}".replace(",", NBSP).replace(".", ",")
    return s


def rail(icon_name: str, left: str, right: str = "") -> str:
    return (
        '<div class="rail"><div class="rail-l">'
        + icon(icon_name, 17)
        + f'<span class="eyebrow">{left}</span></div>'
        + f'<span class="eyebrow dim">{right}</span></div>'
        + '<div class="rail-rule"><i></i></div>'
    )


def foot(label: str, index: int, total: int) -> str:
    return (
        f'<div class="foot"><span>{label}</span>'
        f'<span class="n">{index:02d}{NBSP}/{NBSP}{total:02d}</span></div>'
    )


def stat(value: str, label: str, unit: str = "", sub: str = "", kind: str = "") -> str:
    u = f'<span class="u">{unit}</span>' if unit else ""
    s = f'<div class="stat-s">{sub}</div>' if sub else ""
    k = f" {kind}" if kind else ""
    return (
        f'<div class="stat{k}"><div class="stat-v">{value}{u}</div>'
        f'<div class="stat-l">{label}</div>{s}</div>'
    )


def stats_row(*tiles: str, cols: int = 3) -> str:
    return (
        f'<div class="stats" style="grid-template-columns:repeat({cols},1fr)">'
        + "".join(tiles)
        + "</div>"
    )


def card(label: str, body: str, kind: str = "", icon_name: str = "") -> str:
    k = f" {kind}" if kind else ""
    ic = icon(icon_name, 14, "currentColor") if icon_name else ""
    return (
        f'<div class="card{k}"><p class="card-lab">{ic}{label}</p>{body}</div>'
    )


def bullets(items, warm: bool = False) -> str:
    cls = "bul warm" if warm else "bul"
    return f'<ul class="{cls}">' + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


def steps(items, start: int = 0) -> str:
    reset = f' style="counter-reset:st {start}"' if start else ""
    return f'<ol class="steps"{reset}>' + "".join(f"<li>{i}</li>" for i in items) + "</ol>"


def table(rows, head=None) -> str:
    h = ""
    if head:
        h = "<tr>" + "".join(f"<th>{c}</th>" for c in head) + "</tr>"
    body = "".join(
        "<tr>" + "".join(
            f'<td class="k">{c}</td>' if j == 0 else f"<td>{c}</td>"
            for j, c in enumerate(r)
        ) + "</tr>"
        for r in rows
    )
    return f'<table class="t">{h}{body}</table>'


def block(*parts: str, anim: bool = True) -> str:
    """Regroupe des fragments dans un conteneur animé en cascade."""
    cls = "anim" if anim else ""
    return f'<div class="{cls}">' + "".join(parts) + "</div>"


def cols(*columns: str, layout: str = "c11") -> str:
    return f'<div class="cols {layout}">' + "".join(
        f'<div class="stack">{c}</div>' for c in columns
    ) + "</div>"


def progress(index: int, total: int) -> str:
    pct = index / total * 100
    return f'<div class="prog"><i style="width:{pct:.1f}%"></i></div>'
