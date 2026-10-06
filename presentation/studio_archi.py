"""Schéma d'architecture MLOps, décliné pour le thème sombre.

Chaque flèche porte son étiquette : ce qui circule, et à quelle cadence.
"""

ARCHI_SVG = """
<svg viewBox="0 0 1000 400" role="img" style="width:100%;height:auto;margin:.6rem 0"
 aria-label="Architecture : l'interface Airflow déclenche le planificateur, qui relance
 chaque semaine l'entraînement ; celui-ci télécharge les données OWID, journalise ses
 paramètres et métriques dans MLflow et écrit le modèle sur un volume partagé que l'API
 FastAPI charge au démarrage pour répondre aux clients.">
<defs>
  <marker id="fc" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7"
   orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#38bdf8"/></marker>
  <marker id="fa2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7"
   orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#fbbf24"/></marker>
  <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
    <feGaussianBlur stdDeviation="6" result="b"/>
    <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>
</defs>

<!-- enceinte docker compose -->
<rect x="196" y="8" width="768" height="296" rx="18" fill="rgba(255,255,255,0.02)"
 stroke="rgba(255,255,255,0.12)" stroke-width="1" stroke-dasharray="6 5"/>
<text x="212" y="30" font-family="Outfit,sans-serif" font-size="11"
 fill="#64748b" letter-spacing="1.6">DOCKER COMPOSE · 4 SERVICES</text>

<!-- source externe -->
<rect x="0" y="164" width="160" height="54" rx="12" fill="rgba(255,255,255,0.04)"
 stroke="rgba(255,255,255,0.10)"/>
<text x="80" y="189" font-family="Outfit,sans-serif" font-size="13" font-weight="700"
 fill="#f1f5f9" text-anchor="middle">OWID · GitHub</text>
<text x="80" y="206" font-family="Outfit,sans-serif" font-size="10.5"
 fill="#64748b" text-anchor="middle">owid-covid-data.csv</text>

<!-- colonne airflow + entrainement -->
<rect x="240" y="34" width="150" height="48" rx="12" fill="rgba(56,189,248,0.08)"
 stroke="rgba(56,189,248,0.22)"/>
<text x="315" y="55" font-family="Outfit,sans-serif" font-size="12.5" font-weight="700"
 fill="#38bdf8" text-anchor="middle">airflow-webserver</text>
<text x="315" y="71" font-family="Outfit,sans-serif" font-size="10.5"
 fill="#7dd3fc" text-anchor="middle">:8080</text>

<rect x="240" y="106" width="150" height="48" rx="12" fill="rgba(56,189,248,0.08)"
 stroke="rgba(56,189,248,0.22)"/>
<text x="315" y="127" font-family="Outfit,sans-serif" font-size="12.5" font-weight="700"
 fill="#38bdf8" text-anchor="middle">airflow-scheduler</text>
<text x="315" y="143" font-family="Outfit,sans-serif" font-size="10.5"
 fill="#7dd3fc" text-anchor="middle">retrain_covid_model</text>

<rect x="240" y="178" width="150" height="54" rx="12" fill="#38bdf8" filter="url(#glow)"/>
<text x="315" y="202" font-family="Outfit,sans-serif" font-size="13" font-weight="800"
 fill="#020617" text-anchor="middle">Entraînement</text>
<text x="315" y="219" font-family="Outfit,sans-serif" font-size="10.5"
 fill="#0c4a6e" text-anchor="middle">src/train.py</text>

<!-- mlflow -->
<rect x="470" y="66" width="184" height="54" rx="12" fill="rgba(139,92,246,0.10)"
 stroke="rgba(139,92,246,0.28)"/>
<text x="562" y="90" font-family="Outfit,sans-serif" font-size="13" font-weight="700"
 fill="#a78bfa" text-anchor="middle">MLflow</text>
<text x="562" y="107" font-family="Outfit,sans-serif" font-size="10.5"
 fill="#8b5cf6" text-anchor="middle">:5000 · base SQLite</text>

<!-- volume modele -->
<rect x="470" y="178" width="184" height="54" rx="12" fill="rgba(251,191,36,0.10)"
 stroke="rgba(251,191,36,0.28)"/>
<text x="562" y="202" font-family="Outfit,sans-serif" font-size="13" font-weight="700"
 fill="#fbbf24" text-anchor="middle">Volume partagé</text>
<text x="562" y="219" font-family="Outfit,sans-serif" font-size="10.5"
 fill="#fcd34d" text-anchor="middle">models/*.joblib</text>

<!-- api -->
<rect x="744" y="178" width="184" height="54" rx="12" fill="rgba(56,189,248,0.08)"
 stroke="rgba(56,189,248,0.22)"/>
<text x="836" y="202" font-family="Outfit,sans-serif" font-size="13" font-weight="700"
 fill="#38bdf8" text-anchor="middle">API FastAPI</text>
<text x="836" y="219" font-family="Outfit,sans-serif" font-size="10.5"
 fill="#7dd3fc" text-anchor="middle">:8000</text>

<!-- client -->
<rect x="744" y="336" width="184" height="48" rx="12" fill="rgba(255,255,255,0.04)"
 stroke="rgba(255,255,255,0.10)"/>
<text x="836" y="365" font-family="Outfit,sans-serif" font-size="12.5" font-weight="700"
 fill="#f1f5f9" text-anchor="middle">Client · Swagger · curl</text>

<!-- ═══ flux ═══ -->
<line x1="160" y1="205" x2="236" y2="205" stroke="#38bdf8" stroke-width="1.6"
 marker-end="url(#fc)"/>
<text x="80" y="242" font-family="Outfit,sans-serif" font-size="11"
 fill="#94a3b8" text-anchor="middle">téléchargement du CSV</text>

<line x1="315" y1="84" x2="315" y2="102" stroke="#38bdf8" stroke-width="1.6"
 marker-end="url(#fc)"/>
<text x="328" y="99" font-family="Outfit,sans-serif" font-size="11"
 fill="#94a3b8">Trigger DAG manuel</text>

<line x1="315" y1="156" x2="315" y2="174" stroke="#38bdf8" stroke-width="1.6"
 marker-end="url(#fc)"/>
<text x="328" y="171" font-family="Outfit,sans-serif" font-size="11"
 fill="#94a3b8">@weekly</text>

<polyline points="392,190 448,190 448,93 466,93" fill="none" stroke="#8b5cf6"
 stroke-width="1.6" marker-end="url(#fc)"/>
<text x="562" y="54" font-family="Outfit,sans-serif" font-size="11"
 fill="#94a3b8" text-anchor="middle">9 paramètres · MAE · RMSE</text>

<line x1="392" y1="218" x2="466" y2="218" stroke="#fbbf24" stroke-width="1.6"
 marker-end="url(#fa2)"/>
<text x="429" y="250" font-family="Outfit,sans-serif" font-size="11"
 fill="#94a3b8" text-anchor="middle">joblib.dump</text>

<line x1="656" y1="205" x2="740" y2="205" stroke="#fbbf24" stroke-width="1.6"
 marker-end="url(#fa2)"/>
<text x="698" y="250" font-family="Outfit,sans-serif" font-size="11"
 fill="#94a3b8" text-anchor="middle">chargé au démarrage</text>

<line x1="860" y1="332" x2="860" y2="236" stroke="#38bdf8" stroke-width="1.6"
 marker-end="url(#fc)"/>
<text x="872" y="274" font-family="Outfit,sans-serif" font-size="11"
 fill="#94a3b8">POST /predict</text>
<text x="872" y="290" font-family="Outfit,sans-serif" font-size="10.5"
 fill="#64748b">x-api-key</text>

<line x1="812" y1="236" x2="812" y2="332" stroke="#38bdf8" stroke-width="1.6"
 stroke-dasharray="4 3" marker-end="url(#fc)"/>
<text x="800" y="286" font-family="Outfit,sans-serif" font-size="11"
 fill="#94a3b8" text-anchor="end">prévision J+7</text>
</svg>
"""
