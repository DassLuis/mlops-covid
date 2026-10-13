"""Schéma d'architecture MLOps, décliné pour le thème sombre et le thème clair.

Chaque flèche porte son étiquette : ce qui circule, et à quelle cadence.
"""

SOMBRE = {
    "police": "Outfit,sans-serif",
    "rayon": 12,
    "enceinte_fond": "rgba(255,255,255,0.02)", "enceinte_bord": "rgba(255,255,255,0.12)",
    "discret": "#64748b", "etiquette": "#94a3b8",
    # (fond, bord, titre, sous-titre)
    "bleu": ("rgba(56,189,248,0.08)", "rgba(56,189,248,0.22)", "#38bdf8", "#7dd3fc"),
    "violet": ("rgba(139,92,246,0.10)", "rgba(139,92,246,0.28)", "#a78bfa", "#8b5cf6"),
    "chaud": ("rgba(251,191,36,0.10)", "rgba(251,191,36,0.28)", "#fbbf24", "#fcd34d"),
    "neutre": ("rgba(255,255,255,0.04)", "rgba(255,255,255,0.10)", "#f1f5f9", "#64748b"),
    "plein": ("#38bdf8", "#38bdf8", "#020617", "#0c4a6e"),
    "trait": "#38bdf8", "trait_violet": "#8b5cf6", "trait_chaud": "#fbbf24",
    "halo": True,
}

CLAIR = {
    "police": "IBM Plex Sans,sans-serif",
    "rayon": 0,
    "enceinte_fond": "none", "enceinte_bord": "#b9c2cd",
    "discret": "#7d8896", "etiquette": "#495563",
    "bleu": ("#eaf2fd", "none", "#1c5cab", "#1c5cab"),
    "violet": ("#efeafc", "none", "#5b3fb0", "#5b3fb0"),
    "chaud": ("#fceee8", "none", "#d2501f", "#d2501f"),
    "neutre": ("#f4f6f9", "none", "#101720", "#7d8896"),
    "plein": ("#2a78d6", "none", "#ffffff", "#cde2fb"),
    "trait": "#2a78d6", "trait_violet": "#5b3fb0", "trait_chaud": "#d2501f",
    "halo": False,
}


def schema(p: dict) -> str:
    police = f'font-family="{p["police"]}"'

    def boite(x, y, titre, sous, ton, w=170, h=50):
        fond, bord, c_titre, c_sous = p[ton]
        cx = x + w / 2
        halo = ' filter="url(#glow)"' if ton == "plein" and p["halo"] else ""
        return (
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{p["rayon"]}" '
            f'fill="{fond}" stroke="{bord}"{halo}/>'
            f'<text x="{cx}" y="{y + 22}" {police} font-size="13" font-weight="700" '
            f'fill="{c_titre}" text-anchor="middle">{titre}</text>'
            f'<text x="{cx}" y="{y + 38}" {police} font-size="10.5" fill="{c_sous}" '
            f'text-anchor="middle">{sous}</text>'
        )

    def fleche(points, trait="trait", tirets=False):
        tete = "fa2" if trait == "trait_chaud" else "fc"
        pointille = ' stroke-dasharray="4 3"' if tirets else ""
        return (f'<polyline points="{points}" fill="none" stroke="{p[trait]}" '
                f'stroke-width="1.6"{pointille} marker-end="url(#{tete})"/>')

    def etiquette(x, y, texte, ancre="start"):
        return (f'<text x="{x}" y="{y}" {police} font-size="11" '
                f'fill="{p["etiquette"]}" text-anchor="{ancre}">{texte}</text>')

    return f"""
<svg viewBox="0 0 1000 480" role="img" style="width:100%;height:auto;margin:.6rem 0"
 aria-label="Architecture : Airflow appelle le service d'entraînement, qui télécharge les données
 OWID, envoie données et modèle dans MinIO et enregistre une nouvelle version dans le
 registre MLflow. L'API FastAPI sert la version en production. Chaque prédiction est
 journalisée ; Evidently y cherche chaque jour une dérive et peut relancer
 l'entraînement. Prometheus collecte les métriques de l'API et transmet ses alertes à
 Alertmanager ; Grafana les affiche. L'application Streamlit passe par l'API.">
<defs>
  <marker id="fc" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7"
   orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{p["trait"]}"/></marker>
  <marker id="fa2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7"
   orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z"
   fill="{p["trait_chaud"]}"/></marker>
  <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
    <feGaussianBlur stdDeviation="6" result="b"/>
    <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>
</defs>

<!-- enceinte docker compose -->
<rect x="196" y="8" width="796" height="390" rx="{p["rayon"] and 18}"
 fill="{p["enceinte_fond"]}" stroke="{p["enceinte_bord"]}" stroke-width="1"
 stroke-dasharray="6 5"/>
<text x="212" y="30" {police} font-size="11"
 fill="{p["discret"]}" letter-spacing="1.6">DOCKER COMPOSE · 10 SERVICES</text>

<!-- source externe et client -->
{boite(0, 150, "OWID · GitHub", "owid-covid-data.csv", "neutre", w=160)}
{boite(830, 424, "Utilisateur", "navigateur", "neutre", w=160, h=48)}
{boite(830, 326, "Streamlit", ":8501 · interface", "violet", w=160, h=48)}

<!-- colonne orchestration et versioning -->
{boite(226, 48, "Airflow", "2 DAG · @weekly · @daily", "bleu")}
{boite(226, 150, "Service d'entraînement", "DVC : ingest ▸ prepare ▸ train", "plein")}
{boite(226, 262, "MinIO · S3", "données et modèle versionnés", "chaud")}

<!-- colonne registre et contrôle -->
{boite(466, 48, "Evidently", "dérive · performance", "bleu")}
{boite(466, 150, "MLflow · registre", "candidate ▸ production", "violet")}
{boite(466, 262, "Alertmanager · Grafana", ":9093 · :3000", "bleu")}

<!-- colonne service -->
{boite(706, 48, "Journal des prédictions", "predictions.csv", "chaud")}
{boite(706, 150, "API FastAPI", ":8000 · /predict · /metrics", "bleu")}
{boite(706, 262, "Prometheus", ":9090 · 7 règles d'alerte", "bleu")}

<!-- ═══ entraînement et versioning ═══ -->
{fleche("160,175 222,175")}
{etiquette(80, 222, "téléchargement du CSV", "middle")}
{fleche("311,100 311,146")}
{etiquette(322, 128, "POST /steps")}
{fleche("311,202 311,258", "trait_chaud")}
{etiquette(322, 235, "dvc push")}
{fleche("398,175 462,175", "trait_violet")}
{etiquette(430, 167, "version", "middle")}
{fleche("638,175 702,175", "trait_chaud")}
{etiquette(670, 167, "production", "middle")}

<!-- ═══ boucle de monitoring ═══ -->
{fleche("791,146 791,102")}
{etiquette(802, 128, "chaque prédiction")}
{fleche("702,73 640,73")}
{etiquette(671, 65, "@daily", "middle")}
{fleche("462,73 400,73", tirets=True)}
{etiquette(431, 65, "réentraîner", "middle")}
{fleche("551,100 551,146", "trait_violet")}
{etiquette(562, 128, "historique")}

<!-- ═══ métriques et alertes ═══ -->
{fleche("791,202 791,258")}
{etiquette(802, 235, "collecte · 15 s")}
{fleche("702,287 640,287")}
{etiquette(671, 279, "alertes", "middle")}

<!-- ═══ application ═══ -->
{fleche("910,420 910,378")}
{fleche("910,322 910,175 880,175")}
{etiquette(918, 250, "API")}
</svg>
"""


ARCHI_SVG = schema(SOMBRE)
ARCHI_SVG_CLAIR = schema(CLAIR)
