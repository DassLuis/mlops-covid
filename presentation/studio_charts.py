"""Graphiques Plotly du thème studio : template sombre, fonds transparents.

Mêmes données et mêmes lectures que le support PDF ; seul l'habillage change.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from studio_css import CYAN, VIOLET, AMBRE, VERT, ROUGE, ARDOISE, ARDOISE_2
from data import FR_MOIS_COURT

GRILLE = "rgba(255,255,255,0.06)"
LIGNE = "rgba(255,255,255,0.12)"
POLICE = "Outfit, sans-serif"


def _base(height: int = 360, legend: bool = False, margin=None) -> dict:
    return dict(
        height=height,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=POLICE, size=13, color=ARDOISE),
        margin=margin or dict(l=10, r=10, t=10, b=10),
        hoverlabel=dict(bgcolor="rgba(2,6,23,0.92)", bordercolor=LIGNE,
                        font=dict(family=POLICE, size=13, color="#f1f5f9")),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right",
                    x=1, bgcolor="rgba(0,0,0,0)", borderwidth=0,
                    font=dict(size=12, color=ARDOISE)),
        dragmode=False,
    )


def _axes(fig, ytitle: str = "", ytick="~s"):
    fig.update_xaxes(showgrid=False, showline=True, linecolor=LIGNE, linewidth=1,
                     ticks="outside", tickcolor=LIGNE, ticklen=4, zeroline=False,
                     tickfont=dict(size=11.5, color=ARDOISE_2))
    fig.update_yaxes(showgrid=True, gridcolor=GRILLE, gridwidth=1, zeroline=False,
                     showline=False, tickformat=ytick,
                     tickfont=dict(size=11.5, color=ARDOISE_2),
                     title=dict(text=ytitle, font=dict(size=11.5, color=ARDOISE_2)))
    return fig


def _mois_fr(dates: pd.Series):
    t = pd.date_range(dates.min().normalize().replace(day=1), dates.max(), freq="MS")
    return list(t), [FR_MOIS_COURT[d.month] for d in t]


# ─────────────────────── série complète et découpage ───────────────────────
def serie_complete(serie: pd.DataFrame, split_date, stats: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=serie["date"], y=serie["new_cases"], mode="lines", name="Cas publiés",
        line=dict(color=CYAN, width=1.3),
        fill="tozeroy", fillcolor="rgba(56,189,248,0.10)",
        hovertemplate="%{x|%d/%m/%Y}<br><b>%{y:,.0f}</b> cas<extra></extra>",
    ))
    fig.add_vrect(x0=split_date, x1=serie["date"].max(), fillcolor=AMBRE,
                  opacity=0.07, line_width=0, layer="below")
    fig.add_vline(x=split_date, line=dict(color=AMBRE, width=2, dash="dash"))

    ymax = float(serie["new_cases"].max())
    fig.add_annotation(x=serie["date"].iloc[len(serie) // 5], y=ymax * 1.09,
                       text=f"<b>Apprentissage · {stats['n_train']} jours</b>",
                       showarrow=False, font=dict(size=12.5, color=CYAN))
    fig.add_annotation(x=pd.Timestamp(split_date) + pd.Timedelta(days=140),
                       y=ymax * 1.09, text=f"<b>Test · {stats['n_test']} j</b>",
                       showarrow=False, font=dict(size=12.5, color=AMBRE))
    fig.add_annotation(
        x=pd.Timestamp(stats["peak_date"]), y=ymax,
        text=f"Pic Omicron · {ymax/1e6:.2f} M".replace(".", ","),
        showarrow=True, arrowhead=0, arrowcolor=ARDOISE_2, arrowwidth=1,
        ax=-70, ay=28, align="right", xanchor="right",
        font=dict(size=11.5, color=ARDOISE),
    )
    fig.update_layout(**_base(360))
    fig.update_yaxes(range=[0, ymax * 1.2])
    annees = pd.date_range(serie["date"].min().normalize().replace(month=1, day=1),
                           serie["date"].max() + pd.Timedelta(days=370), freq="YS")
    fig.update_xaxes(tickmode="array", tickvals=list(annees),
                     ticktext=[str(d.year) for d in annees])
    return _axes(fig, "cas publiés")


# ─────────────────────── le peigne hebdomadaire ───────────────────────
def peigne(serie: pd.DataFrame, debut: str, fin: str) -> go.Figure:
    m = serie[(serie["date"] >= debut) & (serie["date"] <= fin)]
    fig = go.Figure(go.Bar(
        x=m["date"], y=m["new_cases"], width=0.6 * 86400000,
        marker=dict(color=CYAN, line_width=0),
        hovertemplate="%{x|%d/%m}<br><b>%{y:,.0f}</b> cas<extra></extra>",
    ))
    ymax = float(m["new_cases"].max()) if len(m) else 1.0
    fig.add_annotation(xref="paper", yref="paper", x=0.01, y=0.98,
                       xanchor="left", yanchor="top",
                       text="<b>6 jours à 0 entre chaque publication</b>",
                       showarrow=False, font=dict(size=13, color=AMBRE))
    fig.update_layout(**_base(340))
    fig.update_yaxes(range=[0, ymax * 1.32])
    dim = [d for d in m["date"] if d.weekday() == 6]
    fig.update_xaxes(tickmode="array", tickvals=dim,
                     ticktext=[f"{d.day:02d} {FR_MOIS_COURT[d.month]}" for d in dim])
    return _axes(fig, "cas publiés")


# ─────────────────────── prédit vs réel ───────────────────────
def predit_vs_reel(test: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=test["date"], y=test["y_true"], mode="lines", name="Réel à J+7",
        line=dict(color=CYAN, width=2.6),
        hovertemplate="%{x|%d/%m/%Y}<br>Réel <b>%{y:,.0f}</b><extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=test["date"], y=test["y_pred"], mode="lines", name="Prédit",
        line=dict(color=AMBRE, width=2, dash="dash"),
        hovertemplate="%{x|%d/%m/%Y}<br>Prédit <b>%{y:,.0f}</b><extra></extra>",
    ))
    fig.add_annotation(xref="paper", yref="paper", x=0.55, y=0.44,
                       xanchor="left", yanchor="top", align="left",
                       text="Plus aucune publication à partir de juillet 2023 :<br>"
                            "les 189 derniers jours du test valent 0",
                       showarrow=False, font=dict(size=12, color=ARDOISE_2))
    fig.update_layout(**_base(360, legend=True), hovermode="x unified")
    tv, tt = _mois_fr(test["date"])
    fig.update_xaxes(tickmode="array", tickvals=tv, ticktext=tt)
    return _axes(fig, "cas à J+7")


# ─────────────────────── les 13 jours de publication ───────────────────────
def jours_publication(test: pd.DataFrame) -> go.Figure:
    nz = test[test["y_true"] > 0].reset_index(drop=True)
    lab = [f"{d.day:02d} {FR_MOIS_COURT[d.month]}" for d in nz["date"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=lab, y=nz["y_true"], name="Réel",
                         marker=dict(color=CYAN, line_width=0),
                         hovertemplate="%{x}<br>Réel <b>%{y:,.0f}</b><extra></extra>"))
    fig.add_trace(go.Bar(x=lab, y=nz["y_pred"], name="Prédit",
                         marker=dict(color=AMBRE, line_width=0),
                         hovertemplate="%{x}<br>Prédit <b>%{y:,.0f}</b><extra></extra>"))
    fig.update_layout(**_base(350, legend=True), barmode="group",
                      bargap=0.3, bargroupgap=0.08)
    return _axes(fig, "cas à J+7")


# ─────────────────────── références naïves ───────────────────────
def baselines(stats: dict) -> go.Figure:
    noms = ["Répéter la<br>dernière valeur", "Toujours<br>prédire 0",
            "Random Forest<br>(notre modèle)"]
    vals = [stats["mae_naive"], stats["mae_zero"], stats["mae"]]
    fig = go.Figure(go.Bar(
        x=noms, y=vals,
        marker=dict(color=["rgba(148,163,184,0.35)", "rgba(148,163,184,0.35)", CYAN],
                    line_width=0),
        text=[f"{v:,.0f}".replace(",", " ") for v in vals],
        textposition="outside",
        textfont=dict(family=POLICE, size=14, color="#f1f5f9"),
        hovertemplate="%{x}<br>MAE <b>%{y:,.0f}</b> cas<extra></extra>",
    ))
    fig.update_layout(**_base(340), bargap=0.45)
    fig.update_yaxes(range=[0, max(vals) * 1.22])
    return _axes(fig, "erreur absolue moyenne (cas)")


# ─────────────────────── poids des variables ───────────────────────
def importances(stats: dict) -> go.Figure:
    imp = pd.Series(stats["importances"]).sort_values()
    pct = imp.values * 100
    couleurs = [CYAN if v >= pct.max() * 0.999
                else ("rgba(56,189,248,0.45)" if v > 0 else "rgba(148,163,184,0.18)")
                for v in pct]
    fig = go.Figure(go.Bar(
        x=pct, y=list(imp.index), orientation="h",
        marker=dict(color=couleurs, line_width=0),
        text=[f"{v:.1f} %".replace(".", ",") for v in pct], textposition="outside",
        textfont=dict(family=POLICE, size=12, color=ARDOISE),
        hovertemplate="%{y}<br><b>%{x:.1f} %</b> du poids<extra></extra>",
    ))
    fig.update_layout(**_base(420, margin=dict(l=10, r=60, t=10, b=10)))
    fig.update_xaxes(showgrid=True, gridcolor=GRILLE, zeroline=False, showline=False,
                     ticksuffix=" %", range=[0, pct.max() * 1.24],
                     tickfont=dict(size=11.5, color=ARDOISE_2))
    fig.update_yaxes(showgrid=False, showline=False, ticks="",
                     tickfont=dict(size=12.5, color=ARDOISE))
    return fig


# ─────────────────────── répartition de l'erreur ───────────────────────
def regimes_erreur(stats: dict) -> go.Figure:
    """Deux régimes : les jours sans publication, et les jours de publication."""
    noms = [f"{stats['n_zero_days_test']} jours<br>sans publication",
            f"{stats['n_publication_days_test']} jours<br>de publication"]
    vals = [stats["mae_zero_days"], stats["mae_nz"]]
    fig = go.Figure(go.Bar(
        x=noms, y=vals, marker=dict(color=[VERT, ROUGE], line_width=0),
        text=[f"{v:,.0f}".replace(",", " ") for v in vals],
        textposition="outside",
        textfont=dict(family=POLICE, size=14, color="#f1f5f9"),
        hovertemplate="%{x}<br>MAE <b>%{y:,.0f}</b> cas<extra></extra>",
    ))
    fig.update_layout(**_base(340), bargap=0.55)
    fig.update_yaxes(range=[0, max(vals) * 1.22])
    return _axes(fig, "erreur absolue moyenne (cas)")


# ─────────────────────── laboratoire ───────────────────────
def contexte_date(pub: pd.DataFrame, i: int) -> go.Figure:
    ligne = pub.iloc[i]
    lags = [ligne[f"lag_{k}"] for k in range(7, 0, -1)]
    x = [f"J−{k}" for k in range(7, 0, -1)]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=lags, name="Entrées connues",
                         marker=dict(color="rgba(56,189,248,0.4)", line_width=0),
                         hovertemplate="%{x}<br><b>%{y:,.0f}</b> cas<extra></extra>"))
    fig.add_trace(go.Bar(x=["J+7"], y=[ligne["y_true"]], name="Réel à J+7",
                         marker=dict(color=CYAN, line_width=0),
                         hovertemplate="Réel <b>%{y:,.0f}</b><extra></extra>"))
    fig.add_trace(go.Bar(x=["J+7"], y=[ligne["y_pred"]], name="Prédit",
                         marker=dict(color=AMBRE, line_width=0),
                         hovertemplate="Prédit <b>%{y:,.0f}</b><extra></extra>"))
    fig.update_layout(**_base(320, legend=True), barmode="group", bargap=0.3)
    return _axes(fig, "cas")


def jauge_prediction(valeur: float, reference: np.ndarray) -> go.Figure:
    obs = reference[reference > 0]
    fig = go.Figure()
    fig.add_trace(go.Box(x=obs, name="", boxpoints=False, orientation="h",
                         marker=dict(color="rgba(56,189,248,0.5)"),
                         line=dict(color="rgba(56,189,248,0.7)", width=1.5),
                         fillcolor="rgba(56,189,248,0.15)",
                         hovertemplate="Publications observées<extra></extra>"))
    fig.add_trace(go.Scatter(x=[valeur], y=[""], mode="markers",
                             name="Votre prévision",
                             marker=dict(color=AMBRE, size=17, symbol="diamond",
                                         line=dict(color="#020617", width=2)),
                             hovertemplate="Prévision <b>%{x:,.0f}</b><extra></extra>"))
    fig.update_layout(**_base(160, margin=dict(l=10, r=10, t=10, b=30)))
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_xaxes(showgrid=True, gridcolor=GRILLE, zeroline=False, showline=False,
                     tickformat="~s", tickfont=dict(size=11.5, color=ARDOISE_2))
    return fig


def comparaison_metriques(reference: dict, essai: dict) -> go.Figure:
    noms = ["MAE", "RMSE"]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=noms, y=[reference["mae"], reference["rmse"]], name="Modèle du projet",
        marker=dict(color="rgba(148,163,184,0.35)", line_width=0),
        text=[f"{reference['mae']:,.0f}".replace(",", " "),
              f"{reference['rmse']:,.0f}".replace(",", " ")],
        textposition="outside", textfont=dict(size=13, color=ARDOISE),
        hovertemplate="%{x} référence <b>%{y:,.0f}</b><extra></extra>"))
    fig.add_trace(go.Bar(
        x=noms, y=[essai["mae"], essai["rmse"]], name="Votre essai",
        marker=dict(color=CYAN, line_width=0),
        text=[f"{essai['mae']:,.0f}".replace(",", " "),
              f"{essai['rmse']:,.0f}".replace(",", " ")],
        textposition="outside", textfont=dict(size=13, color="#f1f5f9"),
        hovertemplate="%{x} essai <b>%{y:,.0f}</b><extra></extra>"))
    fig.update_layout(**_base(320, legend=True), barmode="group", bargap=0.45)
    fig.update_yaxes(range=[0, max(reference["rmse"], essai["rmse"]) * 1.25])
    return _axes(fig, "cas")


def radar_projet(stats: dict) -> go.Figure:
    """Signature du projet : où la chaîne est solide, où elle l'est moins."""
    axes = ["Reproductibilité", "Conteneurisation", "Traçabilité",
            "Automatisation", "Tests", "Qualité du signal"]
    notes = [5, 5, 4, 4, 2, 2]
    fig = go.Figure(go.Scatterpolar(
        r=notes + [notes[0]], theta=axes + [axes[0]], fill="toself",
        fillcolor="rgba(56,189,248,0.16)",
        line=dict(color=CYAN, width=2),
        marker=dict(color=CYAN, size=7),
        hovertemplate="%{theta}<br><b>%{r}</b> / 5<extra></extra>",
    ))
    fig.update_layout(**_base(380))
    fig.update_layout(polar=dict(
        bgcolor="rgba(0,0,0,0)",
        radialaxis=dict(visible=True, range=[0, 5], gridcolor=GRILLE,
                        tickfont=dict(size=10, color=ARDOISE_2), linecolor=GRILLE),
        angularaxis=dict(gridcolor=GRILLE, linecolor=LIGNE,
                         tickfont=dict(size=12, color=ARDOISE)),
    ))
    return fig
