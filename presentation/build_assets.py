"""Génère les images de l'application : bannière et fond de page.

La bannière est dessinée à partir de la vraie courbe épidémique du projet, pas
d'une image d'illustration. Le fond de page — deux poumons parcourus d'un réseau
de neurones — est dessiné lui aussi, dans la charte du projet.

Usage :
    python presentation/build_assets.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon
from matplotlib.path import Path as MplPath
from PIL import Image
from scipy.interpolate import splev, splprep
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
DATA = os.path.join(HERE, "data")

CYAN = "#38bdf8"
VIOLET = "#8b5cf6"
NUIT_1 = "#020617"
NUIT_2 = "#0f172a"
BLEU = "#0ea5e9"
AMBRE = "#f59e0b"
ORANGE = "#fb923c"
CREME = "#fde68a"
LILAS = "#e9d5ff"

# Fond de page : dimensions, cadrage, et forme d'un lobe pulmonaire.
FOND_L, FOND_H, FOND_DPI = 1800, 900, 100
FOND_Y = 1.25
FOND_X = FOND_Y * FOND_L / FOND_H
LOBE = [(-0.16, 0.88), (-0.44, 0.98), (-0.72, 0.78), (-0.88, 0.40),
        (-0.94, -0.05), (-0.86, -0.50), (-0.68, -0.88), (-0.44, -1.02),
        (-0.28, -0.80), (-0.22, -0.30), (-0.19, 0.22), (-0.24, 0.62)]


def banniere() -> None:
    """Courbe épidémique en néon cyan sur le dégradé nuit de l'application."""
    serie = pd.read_parquet(os.path.join(DATA, "serie.parquet"))
    serie["date"] = pd.to_datetime(serie["date"])
    y = serie["new_cases"].to_numpy(dtype=float)
    x = np.arange(len(y))

    fig = plt.figure(figsize=(17, 2.5), dpi=120)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    # dégradé de fond, en diagonale
    grad = np.linspace(0, 1, 256).reshape(1, -1)
    cmap = LinearSegmentedColormap.from_list("nuit", [NUIT_1, NUIT_2])
    ax.imshow(grad, extent=[0, len(y), 0, y.max() * 1.05], aspect="auto",
              cmap=cmap, zorder=0)

    # halo : la même courbe, épaissie et transparente, tracée plusieurs fois
    for lw, alpha in ((14, 0.05), (9, 0.08), (5, 0.14)):
        ax.plot(x, y, color=CYAN, lw=lw, alpha=alpha, zorder=2,
                solid_capstyle="round")
    ax.fill_between(x, y, color=CYAN, alpha=0.10, zorder=1, lw=0)
    ax.plot(x, y, color=CYAN, lw=1.5, alpha=0.95, zorder=3)

    # voile violet à droite, pour rappeler le second accent
    voile = np.linspace(0, 1, 256).reshape(1, -1)
    cmap_v = LinearSegmentedColormap.from_list(
        "voile", [(0, 0, 0, 0), tuple(int(VIOLET[i:i + 2], 16) / 255
                                      for i in (1, 3, 5)) + (0.16,)])
    ax.imshow(voile, extent=[0, len(y), 0, y.max() * 1.05], aspect="auto",
              cmap=cmap_v, zorder=4)

    ax.set_xlim(0, len(y))
    ax.set_ylim(0, y.max() * 1.05)

    os.makedirs(ASSETS, exist_ok=True)
    p = os.path.join(ASSETS, "banner_covid.png")
    fig.savefig(p, facecolor=NUIT_1, dpi=110)
    plt.close(fig)
    print(f"  banner_covid.png       {os.path.getsize(p) / 1024:6.0f} Ko")


def _contour(cote: int, k: float = 1.0, n: int = 700) -> np.ndarray:
    """Contour lissé et fermé d'un lobe (cote -1 à gauche, +1 à droite)."""
    p = np.array(LOBE)
    x = p[:, 0] * 1.28 * (1 if cote < 0 else -1)
    y = p[:, 1] * 1.10
    tck, _ = splprep([np.r_[x, x[0]], np.r_[y, y[0]]], s=0, per=True)
    return np.array(splev(np.linspace(0, 1, n), tck)).T * k


def _toile():
    """Figure au cadrage du fond, sur noir : sert de calque."""
    fig = plt.figure(figsize=(FOND_L / FOND_DPI, FOND_H / FOND_DPI), dpi=FOND_DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-FOND_X, FOND_X)
    ax.set_ylim(-FOND_Y, FOND_Y)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.patch.set_facecolor("black")
    return fig, ax


def _pixels(fig) -> np.ndarray:
    """Rastérise un calque en tableau RVB entre 0 et 1."""
    fig.canvas.draw()
    a = np.asarray(fig.canvas.buffer_rgba(), dtype=float)[..., :3] / 255.0
    plt.close(fig)
    return a


def _semis(poly: np.ndarray, n: int, rng) -> np.ndarray:
    """Tire n points au hasard à l'intérieur du lobe."""
    chemin = MplPath(poly)
    lo, hi = poly.min(0), poly.max(0)
    points = []
    while len(points) < n:
        tirage = rng.uniform(lo, hi, size=(4 * n, 2))
        points.extend(tirage[chemin.contains_points(tirage)].tolist())
    return np.array(points[:n])


def _calque_lumiere(rng) -> np.ndarray:
    """Ce qui brille : réseau ambré, rebord cyan, trachée et bronches."""
    fig, ax = _toile()

    for cote in (-1, 1):
        poly = _contour(cote)
        masque = Polygon(poly, closed=True, transform=ax.transData)
        artistes = []

        n = 210
        pts = _semis(poly, n, rng)

        # arêtes : chaque nœud relié à ses voisins proches
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        i, j = np.where((d > 0) & (d < 0.20))
        aretes = [[pts[a], pts[b]] for a, b in zip(i, j) if a < b]
        lc = LineCollection(aretes, colors=ORANGE, linewidths=0.5, alpha=0.30,
                            zorder=4)
        ax.add_collection(lc)
        artistes.append(lc)
        artistes.append(ax.scatter(pts[:, 0], pts[:, 1], s=3.5, color=CREME,
                                   alpha=0.6, lw=0, zorder=5))

        # foyers : nappes chaudes localisées
        foyers = pts[rng.choice(n, 7, replace=False)]
        for fx, fy in foyers:
            for taille, couleur, alpha in ((30000, "#9a3412", 0.14),
                                           (9000, ORANGE, 0.12),
                                           (1400, AMBRE, 0.16)):
                artistes.append(ax.scatter([fx], [fy], s=taille, color=couleur,
                                           alpha=alpha, lw=0, zorder=2))

        # étoiles : nœuds vifs et leurs rayons
        etoiles = np.vstack([foyers, pts[rng.choice(n, 9, replace=False)]])
        artistes.append(ax.scatter(etoiles[:, 0], etoiles[:, 1], s=30,
                                   color="#fff7ed", alpha=0.9, lw=0, zorder=7))
        rayons = []
        for ex, ey in etoiles:
            for t in rng.uniform(0, 2 * np.pi, 11):
                r = rng.uniform(0.04, 0.14)
                rayons.append([(ex, ey), (ex + r * np.cos(t), ey + r * np.sin(t))])
        lr = LineCollection(rayons, colors=CREME, linewidths=0.45, alpha=0.32,
                            zorder=5)
        ax.add_collection(lr)
        artistes.append(lr)

        for artiste in artistes:        # rien ne déborde du lobe
            artiste.set_clip_path(masque)

        ax.plot(poly[:, 0], poly[:, 1], color=CYAN, lw=2.0, alpha=0.8, zorder=6)

    # trachée, bronches souches et anneaux cartilagineux
    ax.plot([0, 0], [FOND_Y, 0.60], color=LILAS, lw=15, alpha=0.55,
            solid_capstyle="round", zorder=8)
    for cote in (-1, 1):
        tck, _ = splprep([np.array([0, 0.17 * cote, 0.34 * cote]),
                          np.array([0.62, 0.44, 0.24])], s=0, k=2)
        courbe = np.array(splev(np.linspace(0, 1, 60), tck))
        ax.plot(courbe[0], courbe[1], color=LILAS, lw=11, alpha=0.5,
                solid_capstyle="round", zorder=8)
    for y in np.arange(0.64, FOND_Y, 0.085):
        ax.plot([-0.05, 0.05], [y, y], color="#faf5ff", lw=3.0, alpha=0.45,
                zorder=9)

    return _pixels(fig)


def _calque_fond() -> np.ndarray:
    """La nuit, la cage thoracique et le verre bleu des poumons."""
    fig, ax = _toile()

    grad = np.linspace(0, 1, 256).reshape(-1, 1)
    ax.imshow(grad, extent=[-FOND_X, FOND_X, -FOND_Y, FOND_Y], aspect="auto",
              origin="lower", zorder=0,
              cmap=LinearSegmentedColormap.from_list("nuit", ["#050b18", NUIT_1]))

    for k in range(11):             # côtes, à peine lisibles
        y = FOND_Y - k * 0.23
        t = np.linspace(0, np.pi, 200)
        ax.plot(2.3 * np.cos(t), y - 0.28 * np.sin(t), color="#1d4ed8", lw=12,
                alpha=0.07, zorder=1)

    for cote in (-1, 1):
        for k, alpha in ((1.13, 0.15), (1.05, 0.13), (1.0, 0.12)):
            poly = _contour(cote, k=k)
            ax.fill(poly[:, 0], poly[:, 1], color=BLEU, alpha=alpha, lw=0,
                    zorder=2)

    return _pixels(fig)


def fond_poumons(forcer: bool = False) -> None:
    """Fond de page : deux poumons parcourus d'un réseau de neurones.

    L'image n'est dessinée que si aucune n'est déjà en place : déposer une
    photographie sous `assets/banner_lungs.png` suffit à la remplacer.
    """
    if not forcer:
        for ext in ("png", "jpg", "jpeg", "webp"):
            if os.path.exists(os.path.join(ASSETS, f"banner_lungs.{ext}")):
                print(f"  banner_lungs.{ext:<4}      conservé (image déjà en place)")
                return

    rng = np.random.default_rng(11)
    lumiere = _calque_lumiere(rng)
    fond = gaussian_filter(_calque_fond(), (7, 7, 0))

    # halo : la même lumière, étalée trois fois, du serré au très large
    halo = (gaussian_filter(lumiere, (4, 4, 0)) * 0.85
            + gaussian_filter(lumiere, (16, 16, 0)) * 0.95
            + gaussian_filter(lumiere, (52, 52, 0)) * 0.85)

    img = fond + halo + lumiere * 0.65
    img = 1.0 - np.exp(-1.15 * img)     # compression douce des hautes lumières

    # vignettage : les bords s'éteignent, le centre reste
    yy, xx = np.mgrid[0:FOND_H, 0:FOND_L]
    r = np.hypot((xx / FOND_L - 0.5) * 1.5, (yy / FOND_H - 0.5) * 1.35)
    img *= np.clip(1.0 - 0.75 * r ** 2, 0, 1)[..., None]

    # JPEG plutôt que PNG : l'image est réinjectée dans la page à chaque
    # rechargement, autant qu'elle reste légère.
    os.makedirs(ASSETS, exist_ok=True)
    p = os.path.join(ASSETS, "banner_lungs.jpg")
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(
        p, quality=88, optimize=True, progressive=True)
    print(f"  banner_lungs.jpg       {os.path.getsize(p) / 1024:6.0f} Ko")


if __name__ == "__main__":
    print(f"Écriture dans {ASSETS} :")
    banniere()
    fond_poumons()
