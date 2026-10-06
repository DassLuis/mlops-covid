"""Graphiques Plotly de la présentation.

Reprennent exactement les figures du support PDF : mêmes données, mêmes couleurs,
mêmes lectures. L'interactivité (survol, zoom) est le seul ajout.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from theme import (
    ACCENT, ACCENT_DIM, ACCENT_SOFT, WARM_CHART, INK, INK2, INK3,
    GRID, PAPER, F_BODY, F_MONO, num,
)
from data import FR_MOIS_COURT

HOVER = dict(
    bgcolor=PAPER,
    bordercolor=GRID,
    font=dict(family=F_BODY, size=12, color=INK),
)


def _base(height: int = 340, legend: bool = False, margin=None) -> dict:
    return dict(
        height=height,
        paper_bgcolor=PAPER,
        plot_bgcolor=PAPER,
        font=dict(family=F_BODY, size=12, color=INK2),
        margin=margin or dict(l=8, r=8, t=8, b=8),
        hoverlabel=HOVER,
        showlegend=legend,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
            font=dict(size=12, color=INK2),
        ),
        dragmode=False,
    )


def _axes(fig, ytitle: str = "", ytick="~s"):
    fig.update_xaxes(
        showgrid=False, showline=True, linecolor=GRID, linewidth=1,
        ticks="outside", tickcolor=GRID, ticklen=4,
        tickfont=dict(size=11, color=INK3),
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False,
        showline=False, tickformat=ytick,
        tickfont=dict(size=11, color=INK3),
        title=dict(text=ytitle, font=dict(size=11, color=INK3)),
    )
    return fig


def _mois_fr(dates: pd.Series):
    """Ticks mensuels en français (Plotly ne localise pas)."""
    ticks = pd.date_range(
        dates.min().normalize().replace(day=1), dates.max(), freq="MS"
    )
    return list(ticks), [FR_MOIS_COURT[d.month] for d in ticks]


# ─────────────────────── 1 · série complète et découpage ───────────────────────
def serie_complete(serie: pd.DataFrame, split_date, stats: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=serie["date"], y=serie["new_cases"], mode="lines",
        line=dict(color=ACCENT, width=1.2), name="Cas publiés",
        hovertemplate="%{x|%d/%m/%Y}<br><b>%{y:,.0f}</b> cas<extra></extra>",
    ))
    fig.add_vrect(
        x0=split_date, x1=serie["date"].max(),
        fillcolor=WARM_CHART, opacity=0.09, line_width=0, layer="below",
    )
    fig.add_vline(x=split_date, line=dict(color=WARM_CHART, width=2, dash="dash"))

    ymax = float(serie["new_cases"].max())
    fig.add_annotation(
        x=serie["date"].iloc[len(serie) // 5], y=ymax * 1.09,
        text=f"<b>Apprentissage · {stats['n_train']} jours</b>",
        showarrow=False, font=dict(size=12, color=ACCENT_DIM),
    )
    fig.add_annotation(
        x=pd.Timestamp(split_date) + pd.Timedelta(days=140), y=ymax * 1.09,
        text=f"<b>Test · {stats['n_test']} j</b>",
        showarrow=False, font=dict(size=12, color=WARM_CHART),
    )
    fig.add_annotation(
        x=pd.Timestamp(stats["peak_date"]), y=ymax,
        text=f"Pic Omicron · {ymax/1e6:.2f} M<br>(total de la semaine)".replace(".", ","),
        showarrow=True, arrowhead=0, arrowcolor=INK3, arrowwidth=1,
        ax=-80, ay=30, align="right", xanchor="right",
        font=dict(size=11, color=INK2),
    )
    fig.update_layout(**_base(340))
    fig.update_yaxes(range=[0, ymax * 1.18])
    annees = pd.date_range(serie["date"].min().normalize().replace(month=1, day=1),
                           serie["date"].max() + pd.Timedelta(days=370), freq="YS")
    fig.update_xaxes(tickmode="array", tickvals=list(annees),
                     ticktext=[str(d.year) for d in annees])
    return _axes(fig, "cas publiés")


# ─────────────────────── 2 · le peigne hebdomadaire ───────────────────────
def peigne(serie: pd.DataFrame, debut: str, fin: str) -> go.Figure:
    m = serie[(serie["date"] >= debut) & (serie["date"] <= fin)]
    fig = go.Figure(go.Bar(
        x=m["date"], y=m["new_cases"],
        marker=dict(color=ACCENT, line_width=0), width=0.65 * 86400000,
        hovertemplate="%{x|%a %d/%m}<br><b>%{y:,.0f}</b> cas<extra></extra>",
    ))
    ymax = float(m["new_cases"].max()) if len(m) else 1.0
    fig.add_annotation(
        xref="paper", yref="paper", x=0.01, y=0.98, xanchor="left", yanchor="top",
        text="<b>6 jours à 0 entre chaque publication</b>",
        showarrow=False, font=dict(size=13, color=WARM_CHART),
    )
    fig.update_layout(**_base(320))
    fig.update_yaxes(range=[0, ymax * 1.3])
    dim = [d for d in m["date"] if d.weekday() == 6]
    fig.update_xaxes(
        tickmode="array", tickvals=dim,
        ticktext=[f"{d.day:02d} {FR_MOIS_COURT[d.month]}" for d in dim],
    )
    return _axes(fig, "cas publiés")


# ─────────────────────── 3 · prédit vs réel ───────────────────────
def predit_vs_reel(test: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=test["date"], y=test["y_true"], mode="lines", name="Réel à J+7",
        line=dict(color=ACCENT, width=2.6),
        hovertemplate="%{x|%d/%m/%Y}<br>Réel <b>%{y:,.0f}</b><extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=test["date"], y=test["y_pred"], mode="lines", name="Prédit par le modèle",
        line=dict(color=WARM_CHART, width=2, dash="dash"),
        hovertemplate="%{x|%d/%m/%Y}<br>Prédit <b>%{y:,.0f}</b><extra></extra>",
    ))
    fig.add_annotation(
        xref="paper", yref="paper", x=0.55, y=0.42, xanchor="left", yanchor="top",
        text="Plus aucune publication à partir de juillet 2023 :<br>"
             "les 189 derniers jours du test valent 0",
        showarrow=False, align="left", font=dict(size=11.5, color=INK2),
    )
    fig.update_layout(**_base(340, legend=True), hovermode="x unified")
    tv, tt = _mois_fr(test["date"])
    fig.update_xaxes(tickmode="array", tickvals=tv, ticktext=tt)
    return _axes(fig, "cas à J+7")


# ─────────────────────── 4 · les 13 jours de publication ───────────────────────
def jours_publication(test: pd.DataFrame) -> go.Figure:
    nz = test[test["y_true"] > 0].reset_index(drop=True)
    lab = [f"{d.day:02d} {FR_MOIS_COURT[d.month]}" for d in nz["date"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=lab, y=nz["y_true"], name="Réel", marker=dict(color=ACCENT, line_width=0),
        hovertemplate="%{x}<br>Réel <b>%{y:,.0f}</b><extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=lab, y=nz["y_pred"], name="Prédit",
        marker=dict(color=WARM_CHART, line_width=0),
        hovertemplate="%{x}<br>Prédit <b>%{y:,.0f}</b><extra></extra>",
    ))
    fig.update_layout(**_base(330, legend=True), barmode="group", bargap=0.28,
                      bargroupgap=0.08)
    return _axes(fig, "cas à J+7")


# ─────────────────────── 5 · le modèle face aux références ───────────────────────
def baselines(stats: dict) -> go.Figure:
    noms = ["Répéter la<br>dernière valeur", "Toujours<br>prédire 0",
            "Random Forest<br>(notre modèle)"]
    vals = [stats["mae_naive"], stats["mae_zero"], stats["mae"]]
    fig = go.Figure(go.Bar(
        x=noms, y=vals,
        marker=dict(color=[ACCENT_SOFT, ACCENT_SOFT, ACCENT], line_width=0),
        text=[num(v) for v in vals], textposition="outside",
        textfont=dict(family=F_BODY, size=13, color=INK2),
        hovertemplate="%{x}<br>MAE <b>%{y:,.0f}</b> cas<extra></extra>",
    ))
    fig.update_layout(**_base(330), bargap=0.42)
    fig.update_yaxes(range=[0, max(vals) * 1.2])
    return _axes(fig, "erreur absolue moyenne (cas)")


# ─────────────────────── 6 · poids des variables ───────────────────────
def importances(stats: dict) -> go.Figure:
    imp = pd.Series(stats["importances"]).sort_values()
    pct = imp.values * 100
    cols = [ACCENT if v >= pct.max() * 0.999 else ACCENT_SOFT for v in pct]
    fig = go.Figure(go.Bar(
        x=pct, y=list(imp.index), orientation="h",
        marker=dict(color=cols, line_width=0),
        text=[f"{v:.1f} %".replace(".", ",") for v in pct], textposition="outside",
        textfont=dict(family=F_BODY, size=11, color=INK2),
        hovertemplate="%{y}<br><b>%{x:.1f} %</b> du poids<extra></extra>",
    ))
    fig.update_layout(**_base(360, margin=dict(l=8, r=52, t=8, b=8)))
    fig.update_xaxes(
        showgrid=True, gridcolor=GRID, zeroline=False, showline=False,
        ticksuffix=" %", range=[0, pct.max() * 1.22],
        tickfont=dict(size=11, color=INK3),
    )
    fig.update_yaxes(
        showgrid=False, showline=False, ticks="",
        tickfont=dict(family=F_MONO, size=11, color=INK2),
    )
    return fig


# ─────────────────── laboratoire · contexte d'une date rejouée ───────────────────
def contexte_date(test: pd.DataFrame, i: int) -> go.Figure:
    """Les sept entrées du jour choisi, puis le réel et le prédit à J+7."""
    ligne = test.iloc[i]
    lags = [ligne[f"lag_{k}"] for k in range(7, 0, -1)]
    x = [f"J−{k}" for k in range(7, 0, -1)] + ["J+7"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x[:7], y=lags, name="Entrées connues",
        marker=dict(color=ACCENT_SOFT, line_width=0),
        hovertemplate="%{x}<br><b>%{y:,.0f}</b> cas<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=["J+7"], y=[ligne["y_true"]], name="Réel à J+7",
        marker=dict(color=ACCENT, line_width=0),
        hovertemplate="Réel <b>%{y:,.0f}</b><extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=["J+7"], y=[ligne["y_pred"]], name="Prédit",
        marker=dict(color=WARM_CHART, line_width=0),
        hovertemplate="Prédit <b>%{y:,.0f}</b><extra></extra>",
    ))
    fig.update_layout(**_base(300, legend=True), barmode="group", bargap=0.3)
    return _axes(fig, "cas")


# ─────────────────── laboratoire · comparaison de réentraînements ───────────────
def comparaison_metriques(reference: dict, essai: dict) -> go.Figure:
    noms = ["MAE", "RMSE"]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=noms, y=[reference["mae"], reference["rmse"]], name="Modèle du projet",
        marker=dict(color=ACCENT_SOFT, line_width=0),
        text=[num(reference["mae"], 0), num(reference["rmse"], 0)],
        textposition="outside", textfont=dict(size=12, color=INK2),
        hovertemplate="%{x} référence <b>%{y:,.0f}</b><extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=noms, y=[essai["mae"], essai["rmse"]], name="Votre essai",
        marker=dict(color=ACCENT, line_width=0),
        text=[num(essai["mae"], 0), num(essai["rmse"], 0)],
        textposition="outside", textfont=dict(size=12, color=INK2),
        hovertemplate="%{x} essai <b>%{y:,.0f}</b><extra></extra>",
    ))
    top = max(reference["rmse"], essai["rmse"]) * 1.22
    fig.update_layout(**_base(300, legend=True), barmode="group", bargap=0.42)
    fig.update_yaxes(range=[0, top])
    return _axes(fig, "cas")


# ─────────────────── laboratoire · position d'une prédiction ───────────────────
def jauge_prediction(valeur: float, reference: np.ndarray) -> go.Figure:
    """Situe une prévision parmi les valeurs réellement observées sur le test."""
    obs = reference[reference > 0]
    fig = go.Figure()
    fig.add_trace(go.Box(
        x=obs, name="", boxpoints=False, marker=dict(color=ACCENT_SOFT),
        line=dict(color=ACCENT_SOFT, width=1.4), fillcolor=ACCENT_SOFT,
        hovertemplate="Publications observées<extra></extra>", orientation="h",
    ))
    fig.add_trace(go.Scatter(
        x=[valeur], y=[""], mode="markers", name="Votre prévision",
        marker=dict(color=WARM_CHART, size=15, symbol="diamond",
                    line=dict(color=PAPER, width=2)),
        hovertemplate="Prévision <b>%{x:,.0f}</b><extra></extra>",
    ))
    fig.update_layout(**_base(150, margin=dict(l=8, r=8, t=8, b=28)))
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_xaxes(
        showgrid=True, gridcolor=GRID, zeroline=False, showline=False,
        tickformat="~s", tickfont=dict(size=11, color=INK3),
    )
    return fig
