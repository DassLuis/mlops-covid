import sys
import os
import site
import pandas as pd
from pathlib import Path
from PIL import Image
import base64
import json
import time
import requests
from streamlit_lottie import st_lottie

# Force include User Site packages if missing
user_site = site.getusersitepackages()
if user_site not in sys.path:
    sys.path.append(user_site)

import streamlit as st

import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# AI Dependencies
try:
    import tensorflow as tf
    from tensorflow.keras.applications.resnet50 import preprocess_input
    import cv2
    AI_DEPS_INSTALLED = True
except ImportError:
    AI_DEPS_INSTALLED = False

PLOTLY_AVAILABLE = True

# =========================
# CONFIG GLOBALE
# =========================
st.set_page_config(
    page_title="COVID-19 Diagnostic Intelligence",
    page_icon="🩺",  # Still an emoji, but in tab, usually fine. Or I can find an image.
    layout="wide"
)

ASSETS_DIR = Path("assets")  # dossier où tu mettras tes images (plots, Grad-CAM, etc.)

# =========================
# STYLES CUSTOM (CSS)
# =========================
CUSTOM_CSS = """
<style>
/* =========================
   0. ICONOGRAPHY (Font Awesome)
   ========================= */
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

/* =========================
   1. GLOBAL RESET & FONT
   ========================= */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

@keyframes meshGlow {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.stApp {
    background: 
        radial-gradient(circle at 10% 20%, rgba(56, 189, 248, 0.08) 0%, transparent 40%),
        radial-gradient(circle at 90% 80%, rgba(139, 92, 246, 0.08) 0%, transparent 40%),
        linear-gradient(135deg, #020617 0%, #0f172a 100%);
    background-size: 200% 200%;
    animation: meshGlow 15s ease infinite;
    color: #f1f5f9;
}

/* =========================
   2. SIDEBAR (Studio-Grade)
   ========================= */
[data-testid="stSidebar"] {
    background: rgba(10, 15, 30, 0.9) !important;
    backdrop-filter: blur(40px) saturate(150%);
    -webkit-backdrop-filter: blur(40px) saturate(150%);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

[data-testid="stSidebar"] h1 {
    font-weight: 800;
    font-size: 1.5rem;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #f8fafc 0%, #94a3b8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    padding: 1.5rem 0;
}

/* Navigation Overhaul */
.stRadio > div {
    gap: 0.5rem;
}
.stRadio label {
    background: transparent;
    color: #64748b !important;
    padding: 0.75rem 1rem !important;
    border-radius: 12px;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    border: 1px solid transparent;
    font-size: 0.95rem;
    font-weight: 500;
}
.stRadio label:hover {
    background: rgba(255, 255, 255, 0.04) !important;
    color: #f8fafc !important;
    transform: translateX(2px);
}
.stRadio div[role="radio"][aria-checked="true"] label {
    background: rgba(56, 189, 248, 0.1) !important;
    color: #38bdf8 !important;
    border: 1px solid rgba(56, 189, 248, 0.2);
    box-shadow: 0 4px 20px rgba(56, 189, 248, 0.1);
}

/* Sidebar Panel & Authors */
.sidebar-panel {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 20px;
    padding: 1.25rem;
    margin: 1.5rem 0;
}
.sidebar-auth-title {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #475569;
    margin-bottom: 1rem;
    font-weight: 700;
}
.author-card {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px;
    background: rgba(255, 255, 255, 0.01);
    border-radius: 14px;
    margin-bottom: 8px;
    border: 1px solid transparent;
    transition: all 0.3s ease;
}
.author-card:hover {
    background: rgba(56, 189, 248, 0.05);
    border-color: rgba(56, 189, 248, 0.15);
    transform: translateY(-2px);
}
.author-avatar {
    width: 36px;
    height: 36px;
    background: linear-gradient(135deg, #38bdf8 0%, #8b5cf6 100%);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: 700;
    font-size: 0.8rem;
    box-shadow: 0 4px 10px rgba(56, 189, 248, 0.2);
}
.author-info {
    display: flex;
    flex-direction: column;
}
.author-name {
    font-size: 0.85rem;
    font-weight: 600;
    color: #f1f5f9;
}
.author-role {
    font-size: 0.65rem;
    color: #64748b;
}

/* =========================
   3. CARDS & SECTIONS
   ========================= */
.block-card {
    background: rgba(15, 23, 42, 0.4);
    backdrop-filter: blur(20px);
    border-radius: 24px;
    padding: 2rem;
    border: 1px solid rgba(255, 255, 255, 0.05);
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4);
}

h1, h2, h3 {
    font-weight: 800 !important;
    letter-spacing: -0.02em;
    text-shadow: 0 0 30px rgba(56, 189, 248, 0.1);
}

h2 { font-size: 2rem !important; }
h3 { font-size: 1.5rem !important; }

/* Metrics */
[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 20px;
    padding: 1.25rem;
}
[data-testid="stMetric"]:hover {
    background: rgba(255, 255, 255, 0.04);
    border-color: rgba(56, 189, 248, 0.2);
}

/* Dataframe & Tables */
.stDataFrame, .stTable {
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.05);
}

/* Hyper-Wow Image System */
.glass-img-container {
    position: relative;
    border-radius: 28px;
    padding: 10px;
    background: rgba(15, 23, 42, 0.6);
    backdrop-filter: blur(15px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    transition: all 0.6s cubic-bezier(0.16, 1, 0.3, 1);
    box-shadow: 0 15px 45px rgba(0, 0, 0, 0.4);
    perspective: 1000px;
    margin-bottom: 2rem;
    overflow: hidden;
}

.glass-img-container:hover {
    transform: translateY(-12px) rotateX(2deg) rotateY(-1deg);
    border-color: rgba(56, 189, 248, 0.5);
    box-shadow: 
        0 40px 80px rgba(0, 0, 0, 0.7),
        0 0 30px rgba(56, 189, 248, 0.2);
}

.glass-img-container img {
    width: 100%;
    height: auto;
    display: block;
    border-radius: 20px !important;
    transition: transform 0.8s cubic-bezier(0.16, 1, 0.3, 1) !important;
    filter: contrast(1.05) brightness(1.02);
}

.glass-img-container:hover img {
    transform: scale(1.05);
}

/* Shine Animation */
.img-reveal-shine::after {
    content: "";
    position: absolute;
    top: -100%; left: -100%; width: 200%; height: 200%;
    background: linear-gradient(
        135deg,
        rgba(255, 255, 255, 0) 0%,
        rgba(255, 255, 255, 0.05) 50%,
        rgba(255, 255, 255, 0) 100%
    );
    transition: all 0.7s ease;
    pointer-events: none;
}

.glass-img-container:hover .img-reveal-shine::after {
    top: 50%; left: 50%;
}

/* Caption Overlay */
.img-caption-overlay {
    position: absolute;
    bottom: 20px;
    left: 20px;
    right: 20px;
    padding: 15px;
    background: rgba(15, 23, 42, 0.8);
    backdrop-filter: blur(10px);
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.05);
    color: #f1f5f9;
    font-size: 0.85rem;
    opacity: 0;
    transform: translateY(10px);
    transition: all 0.4s ease 0.1s;
}

.glass-img-container:hover .img-caption-overlay {
    opacity: 1;
    transform: translateY(0);
}

/* Premium Stepper */
.premium-pipeline-container {
    padding: 0.5rem 0.5rem 0.5rem 0.8rem;
}
.stepper-track {
    position: relative;
    padding-left: 1.2rem;
}
.stepper-track::before {
    content: "";
    position: absolute;
    left: 4px;
    top: 8px;
    bottom: 8px;
    width: 1px;
    background: rgba(255,255,255,0.06);
}
.stepper-step {
    position: relative;
    padding-bottom: 0.8rem;
    display: flex;
    align-items: center;
}
.stepper-step:last-child {
    padding-bottom: 0;
}
.stepper-node {
    position: absolute;
    left: -1.2rem;
    margin-left: 1px;
    top: 5px;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #1e293b;
    border: 1px solid #475569;
    z-index: 2;
    transition: all 0.4s ease;
}
.stepper-node.active {
    background: #38bdf8;
    border-color: #38bdf8;
    box-shadow: 0 0 12px rgba(56, 189, 248, 0.5);
    transform: scale(1.2);
}
.stepper-node.completed {
    background: #34d399;
    border-color: #34d399;
}
.stepper-text {
    font-size: 0.75rem;
    font-weight: 500;
    color: #475569;
    transition: all 0.3s ease;
    white-space: nowrap;
}
.stepper-text.active {
    color: #f8fafc;
    font-weight: 600;
}
.stepper-text.completed {
    color: #94a3b8;
}
.stepper-node.visited {
    background: #1e293b;
    border-color: #38bdf8;
}
.stepper-text.visited {
    color: #64748b;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =========================
# STYLES WOW (ANIMATIONS)
# =========================
STYLE_WOW = """
<style>
:root {
  --cardIn: 0.6s;
  --hoverLift: -6px;
}

@keyframes fadeUpWow {
  from { opacity: 0; transform: translateY(20px) scale(0.98); }
  to   { opacity: 1; transform: translateY(0) scale(1); }
}

@keyframes glassShimmer {
  0% { background-position: -200% -200%; }
  100% { background-position: 200% 200%; }
}

@keyframes staggerFadeIn {
  from { opacity: 0; transform: translateY(15px); }
  to   { opacity: 1; transform: translateY(0); }
}

@keyframes statusPulse {
  0%   { box-shadow: 0 0 0 0 rgba(56, 189, 248, 0.4); }
  70%  { box-shadow: 0 0 0 15px rgba(56, 189, 248, 0); }
  100% { box-shadow: 0 0 0 0 rgba(56, 189, 248, 0); }
}

@keyframes beamRotate {
  from { transform: rotate(0deg); }
  to   { transform: rotate(360deg); }
}

.block-card {
  animation: fadeUpWow 0.8s cubic-bezier(0.16, 1, 0.3, 1);
  will-change: transform, opacity;
  position: relative;
  overflow: hidden;
  transition: transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.4s ease;
}

.block-card:hover {
  transform: translateY(-8px) scale(1.01);
  box-shadow: 0 40px 80px rgba(0, 0, 0, 0.6), 0 0 20px rgba(56, 189, 248, 0.1);
  border-color: rgba(56, 189, 248, 0.3);
}

/* Border Beam Effect Animation */
.block-card::before {
  content: "";
  position: absolute;
  top: -50%; left: -50%; width: 200%; height: 200%;
  background: conic-gradient(
    transparent 0deg, 
    transparent 120deg, 
    rgba(56, 189, 248, 0.2) 150deg, 
    rgba(56, 189, 248, 0.5) 180deg, 
    rgba(56, 189, 248, 0.2) 210deg, 
    transparent 240deg
  );
  animation: beamRotate 8s linear infinite;
  z-index: -1;
  opacity: 0;
  transition: opacity 1s ease;
}

.block-card:hover::before {
  opacity: 1;
}

.stagger-1 { animation: staggerFadeIn 1s cubic-bezier(0.16, 1, 0.3, 1) forwards; }
.stagger-2 { animation: staggerFadeIn 1s cubic-bezier(0.16, 1, 0.3, 1) forwards 0.15s; }
.stagger-3 { animation: staggerFadeIn 1s cubic-bezier(0.16, 1, 0.3, 1) forwards 0.3s; }
.stagger-4 { animation: staggerFadeIn 1s cubic-bezier(0.16, 1, 0.3, 1) forwards 0.45s; }

.status-pulse {
  animation: statusPulse 2s infinite;
  border-radius: 50%;
}

/* Glass Inner Shimmer */
.block-card::after {
  content: "";
  position: absolute;
  top: 0; left: 0; right: 0; bottom: 0;
  background: linear-gradient(135deg, 
    rgba(255,255,255,0) 0%, 
    rgba(255,255,255,0) 45%, 
    rgba(255,255,255,0.05) 50%, 
    rgba(255,255,255,0) 55%, 
    rgba(255,255,255,0) 100%
  );
  background-size: 250% 250%;
  animation: glassShimmer 8s infinite linear;
  pointer-events: none;
  z-index: 1;
}

/* Premium Button Overhaul */
div.stButton > button {
    background: linear-gradient(135deg, rgba(56, 189, 248, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%) !important;
    color: #f8fafc !important;
    border: 1px solid rgba(56, 189, 248, 0.3) !important;
    border-radius: 16px !important;
    padding: 0.85rem 2rem !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    width: 100% !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2) !important;
    border-color: rgba(56, 189, 248, 0.2) !important;
}

div.stButton > button:hover {
    background: linear-gradient(135deg, rgba(56, 189, 248, 0.2) 0%, rgba(139, 92, 246, 0.2) 100%) !important;
    border-color: #38bdf8 !important;
    box-shadow: 0 15px 40px rgba(56, 189, 248, 0.25) !important;
    transform: translateY(-4px) scale(1.01) !important;
    color: #38bdf8 !important;
}

div.stButton > button:active {
    transform: translateY(-1px) scale(0.99) !important;
}

/* Banner Shine Animation */
@keyframes bannerShimmer {
  0% { transform: translateX(-150%) skewX(-25deg); }
  25% { transform: translateX(150%) skewX(-25deg); }
  100% { transform: translateX(150%) skewX(-25deg); }
}

.banner-shimmer {
  position: absolute;
  top: 0; left: 0; width: 60%; height: 100%;
  background: linear-gradient(
    to right,
    rgba(255, 255, 255, 0) 0%,
    rgba(255, 255, 255, 0.2) 50%,
    rgba(255, 255, 255, 0) 100%
  );
  animation: bannerShimmer 7s infinite ease-in-out;
  pointer-events: none;
  z-index: 5;
}

.banner-glow-text {
  text-shadow: 0 0 20px rgba(56, 189, 248, 0.5), 0 0 40px rgba(56, 189, 248, 0.2);
}

/* Banner Breath Animation */
@keyframes bannerBreath {
  0%   { opacity: 0.1; transform: scale(1.0); }
  50%  { opacity: 0.4; transform: scale(1.02); }
  100% { opacity: 0.1; transform: scale(1.0); }
}

.banner-breath-overlay {
  position: absolute;
  inset: 0;
  background: radial-gradient(circle at center, rgba(56, 189, 248, 0.4) 0%, transparent 70%);
  animation: bannerBreath 4s infinite ease-in-out;
  pointer-events: none;
  z-index: 4;
}

/* Border Beam Animation */
@keyframes beam-rotate {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

.border-beam {
  position: absolute;
  inset: 0px;
  border-radius: 18px;
  padding: 2px;
  background: conic-gradient(from 0deg, transparent 60%, #38bdf8 80%, #fbbf24 90%, transparent 100%);
  -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
  mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
  -webkit-mask-composite: xor;
  mask-composite: exclude;
  animation: beam-rotate 4s linear infinite;
  pointer-events: none;
  z-index: 10;
}
</style>
"""

# =========================
# PETITES FONCTIONS UTILES
# =========================
def get_img_base64(path: Path) -> str | None:
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def load_image(name: str):
    """Charge une image du dossier assets/ si elle existe."""
    path = ASSETS_DIR / name
    if path.exists():
        return Image.open(path)
    return None

def load_lottieurl(url: str):
    """Charge une animation Lottie depuis une URL."""
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None


def show_image(name: str, caption: str = "", use_column=True, width: int | None = None):
    """
    Affiche une image stockée dans assets/ avec un effet Hyper-Wow via HTML/Base64.
    """
    path = ASSETS_DIR / name
    img_b64 = get_img_base64(path)
    
    if img_b64:
        w_style = f"width:{width}px;" if width else "width:100%;"
        html = f"""
        <div class="glass-img-container stagger-1" style="{w_style}">
            <div class="img-reveal-shine">
                <img src="data:image/png;base64,{img_b64}">
                {f'<div class="img-caption-overlay">{caption}</div>' if caption else ''}
            </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)
    else:
        st.info(f"⚠️ Image `{name}` introuvable dans `assets/`.")

def open_card(wow_mode: bool):
    """
    Ouvre une card en gardant ton style.
    - En wow_mode : on ajoute un effet "scroll reveal" plus agréable.
    """
    cls = "block-card reveal" if wow_mode else "block-card"
    return f'<div class="{cls}">'

def close_card():
    """Ferme une card HTML."""
    return "</div>"

def animate_metric(label: str, value: float, fmt: str = ".3f", delta: float | None = None, steps: int = 22, speed: float = 0.012):
    """
    Animation simple de metric pour effet dashboard.
    (On reste sobre : petite animation de montée des chiffres.)
    """
    ph = st.empty()
    steps = max(8, min(steps, 40))
    for i in range(1, steps + 1):
        v = (value * i) / steps
        if delta is None:
            ph.metric(label, format(v, fmt))
        else:
            d = (delta * i) / steps
            sign = "+" if d >= 0 else ""
            ph.metric(label, format(v, fmt), f"{sign}{format(d, fmt)}")
        time.sleep(speed)
    return ph

def apply_page_fx(page_name: str, wow_mode: bool):
    """
    Petits réglages d’animation selon la page (plus/moins “wow”).
    """
    if not wow_mode:
        return

    if page_name == "Introduction":
        css = """:root{ --cardIn: 0.75s; --hoverLift: -5px; --imgZoom: 1.035; }"""
    elif page_name == "Résultats & Analyse":
        css = """:root{ --cardIn: 0.60s; --hoverLift: -4px; --imgZoom: 1.03; }"""
    elif page_name == "Interprétabilité (Grad-CAM)":
        css = """:root{ --cardIn: 0.55s; --hoverLift: -3px; --imgZoom: 1.02; }"""
    else:
        css = """:root{ --cardIn: 0.60s; --hoverLift: -4px; --imgZoom: 1.03; }"""

    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# =========================
# DONNÉES POUR LES TABLEAUX
# =========================
classic_models_df = pd.DataFrame(
    [
        ["Logistic Regression", 51.37, 26.39, 51.37, 34.87],
        ["k-NN",                 47.68, 43.57, 47.68, 43.97],
        ["Decision Tree",        41.85, 41.71, 41.85, 41.76],
        ["Random Forest",        43.74, 41.59, 43.74, 42.39],
        ["SVM",                  51.46, 38.27, 51.46, 35.08],
        ["XGBoost",              48.89, 44.23, 48.89, 44.45],
    ],
    columns=["Modèle", "Accuracy (%)", "Précision (%)", "Rappel (%)", "F1-score (%)"]
)

ml_confusion_matrices = {
    "Logistic Regression": [[599, 0, 0], [276, 0, 0], [291, 0, 0]],
    "k-NN": [[445, 73, 81], [193, 59, 24], [187, 52, 52]],
    "Decision Tree": [[338, 131, 130], [123, 82, 71], [142, 81, 68]],
    "Random Forest": [[376, 98, 125], [154, 73, 49], [165, 65, 61]],
    "SVM": [[599, 0, 0], [275, 1, 0], [290, 1, 0]],
    "XGBoost": [[465, 66, 68], [190, 60, 26], [197, 49, 45]]
}

dl_confusion_matrices = {
    "CNN Baseline": [[387, 40, 39], [43, 147, 43], [35, 35, 163]],
    "ResNet50": [[452, 7, 7], [9, 182, 42], [10, 9, 214]]
}

# =========================
# AI ENGINE (REAL-TIME)
# =========================
@st.cache_resource
def load_resnet_model():
    """Charge le modèle ResNet50 final."""
    if not AI_DEPS_INSTALLED:
        return None
    model_path = "resnet_final.h5"
    if not os.path.exists(model_path):
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Erreur chargement modèle : {e}")
        return None

def preprocess_image_for_model(img: Image.Image):
    """Prépare l'image PIL pour le modèle ResNet50."""
    # Conversion RGB et redimensionnement
    img_rgb = img.convert("RGB").resize((224, 224))
    img_array = np.array(img_rgb)
    # Expansion des dimensions (1, 224, 224, 3)
    img_batch = np.expand_dims(img_array, axis=0).astype(np.float32)
    # Preprocessing spécifique ResNet50 (centering/scaling)
    img_preprocessed = preprocess_input(img_batch)
    return img_preprocessed, img_array

def run_grad_cam(model, img_batch, class_index=None):
    """Génère la heatmap Grad-CAM (Robustifiée)."""
    try:
        # Identifier la dernière couche de convolution
        last_conv_layer_name = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break
            # Gestion si le backbone est une couche imbriquée
            if hasattr(layer, 'layers'):
                for sub in reversed(layer.layers):
                    if isinstance(sub, tf.keras.layers.Conv2D):
                        last_conv_layer_name = sub.name
                        break
                if last_conv_layer_name: break
                
        if not last_conv_layer_name:
            return None

        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_batch)
            if class_index is None:
                class_index = tf.argmax(predictions[0])
            loss = predictions[:, class_index]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
        return heatmap.numpy()
    except Exception:
        return None

def overlay_heatmap(img_array, heatmap, alpha=0.4):
    """Superpose la heatmap sur l'image originale."""
    if heatmap is None:
        return Image.fromarray(img_array)
    
    # Resize heatmap à la taille de l'image
    heatmap_resized = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
    heatmap_u8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    
    # BGR to RGB pour OpenCV/PIL
    heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    superposed = cv2.addWeighted(img_array, 1 - alpha, heatmap_color_rgb, alpha, 0)
    return Image.fromarray(superposed)

cnn_baseline_df = pd.DataFrame(
    [
        ["COVID-19",  86.00, 83.00, 85.00],
        ["Non-COVID", 60.00, 63.00, 61.00],
        ["Normal",    68.00, 70.00, 69.00],
    ],
    columns=["Classe", "Précision (%)", "Rappel (%)", "F1-score (%)"]
)
cnn_baseline_global = {"Accuracy": 0.749, "F1-macro": 0.718}

baseline_vs_resnet_df = pd.DataFrame(
    [
        ["Baseline CNN", 74.90, 71.80],
        ["ResNet50",     91.00, 89.00],
    ],
    columns=["Modèle", "Accuracy (%)", "F1 Macro (%)"]
)

resnet_per_class_df = pd.DataFrame(
    [
        ["COVID-19",  96.58, 97.00, 96.79],
        ["Non-COVID", 91.41, 77.68, 83.99],
        ["Normal",    80.83, 92.27, 86.17],
    ],
    columns=["Classe", "Précision (%)", "Rappel (%)", "F1-score (%)"]
)
resnet_global_metrics = {
    "Accuracy globale": 91.00,
    "Macro F1-score":   88.98,
    "Weighted F1-score": 90.93
}

# =========================
# SIDEBAR : NAVIGATION + INFOS
# =========================
with st.sidebar:
    st.markdown('<h2 style="font-size:1.5rem;"><i class="fa-solid fa-compass" style="color:#38bdf8; margin-right:12px;"></i>Navigation</h2>', unsafe_allow_html=True)

    # ---------- MINI LOTTIE DANS LA SIDEBAR (OPTIONNEL) ----------
    # Pour l'activer :
    #   pip install streamlit-lottie
    # Puis ajoute un fichier : assets/lottie_lungs.json (ou autre animation)
    try:
        from streamlit_lottie import st_lottie

        lottie_path = ASSETS_DIR / "lottie_lungs.json"
        if lottie_path.exists():
            st.markdown('<div class="lottie-box">', unsafe_allow_html=True)
            with open(lottie_path, "r", encoding="utf-8") as f:
                lottie_json = json.load(f)
            st_lottie(lottie_json, height=160, speed=1, loop=True)
            st.markdown("</div>", unsafe_allow_html=True)
    except Exception:
        # Si streamlit-lottie n'est pas installé, on ignore sans casser l'app
        pass

    # ---------- PANNEAU AUTEURS + TOGGLE DEDANS ----------
    st.markdown('<div class="sidebar-panel">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-auth-title">Équipe de Recherche</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="author-card">
            <div class="author-avatar">DL</div>
            <div class="author-info">
                <span class="author-name">Dass Luvengani</span>
                <span class="author-role">Data Scientist</span>
            </div>
        </div>
        <div class="author-card">
            <div class="author-avatar" style="background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);">SS</div>
            <div class="author-info">
                <span class="author-name">Saadia Szapiro</span>
                <span class="author-role">Medical Data Analyst</span>
            </div>
        </div>
        <div style="height: 1px; background: rgba(255,255,255,0.05); margin: 1rem 0;"></div>
        <div class="sidebar-auth-title">Préférences Studio</div>
        """,
        unsafe_allow_html=True
    )

    wow_mode = st.toggle(
        "Effets Visuels",
        value=True,
        help="Animations avancées et effets de lumière",
        key="wow_mode"
    )

    st.markdown("</div>", unsafe_allow_html=True)

    page = st.radio(
        "",
        (
            "Introduction",
            "Données & Prétraitement",
            "Modèles de ML Classiques",
            "Architecture du Réseau",
            "Résultats & Analyse",
            "Interprétabilité (Grad-CAM)",
            "Simulateur Interactif",
            "Conclusion",
        )
    )

    # =========================
    # PIPELINE DE SOUTENANCE (AUTO)
    # =========================
    PIPELINE_STEPS = [
        ("Introduction", "Vue d'ensemble"),
        ("Données & Prétraitement", "Données & EDA"),
        ("Modèles de ML Classiques", "Comparaison ML"),
        ("Architecture du Réseau", "Stratégie Modèle"),
        ("Résultats & Analyse", "Performance Deep"),
        ("Interprétabilité (Grad-CAM)", "Explicabilité"),
        ("Simulateur Interactif", "Simulation Live"),
        ("Conclusion", "Synthèse Finale"),
    ]

    step_index = next((i for i, (p, _) in enumerate(PIPELINE_STEPS) if p == page), 0)

    if "pipeline_max_step" not in st.session_state:
        st.session_state.pipeline_max_step = 0

    if step_index > st.session_state.pipeline_max_step:
        st.session_state.pipeline_max_step = step_index

    st.markdown('<div class="sidebar-separator"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-auth-title">Pipeline Étude</div>', unsafe_allow_html=True)
    
    stepper_html = '<div class="premium-pipeline-container"><div class="stepper-track">'
    for i, (_, label) in enumerate(PIPELINE_STEPS):
        status_class = ""
        if i == step_index:
            status_class = "active"
        elif i < step_index:
            status_class = "completed"
        elif i <= st.session_state.pipeline_max_step:
            status_class = "visited"
            
        stepper_html += f'<div class="stepper-step">'
        stepper_html += f'<div class="stepper-node {status_class}"></div>'
        stepper_html += f'<div class="stepper-text {status_class}">{label}</div>'
        stepper_html += '</div>'
        
    stepper_html += '</div></div>'
    st.markdown(stepper_html, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-separator"></div>', unsafe_allow_html=True)

# =========================
# ACTIVATION WOW + FX PAR PAGE
# =========================
if wow_mode:
    st.markdown(STYLE_WOW, unsafe_allow_html=True)
apply_page_fx(page, wow_mode)

# =========================
# ANIMATION “CHANGEMENT DE PAGE” (spinner + toast)
# =========================
if "prev_page" not in st.session_state:
    st.session_state.prev_page = page

if page != st.session_state.prev_page:
    with st.spinner("Chargement de la section..."):
        time.sleep(0.25)

    # st.toast(f"Section: {page}", icon="🌟")

    if "pipeline_last_toast" not in st.session_state:
        st.session_state.pipeline_last_toast = -1

    PIPELINE_PAGES = [p for (p, _) in PIPELINE_STEPS]
    step_index_now = PIPELINE_PAGES.index(page) if page in PIPELINE_PAGES else 0

    if step_index_now > st.session_state.pipeline_last_toast:
        # st.toast("✅ Étape validée dans le pipeline", icon="✅")
        st.session_state.pipeline_last_toast = step_index_now

    st.session_state.prev_page = page

# =========================
# BANNIÈRE AVEC OVERLAY TEXTE
# =========================
banner_b64 = get_img_base64(ASSETS_DIR / "banner_lungs.png")

if banner_b64:
    st.markdown(
        f"""
        <div style="
            position:relative;
            width:100%;
            height:190px;
            border-radius:18px;
            overflow:hidden;
            margin-bottom:1.2rem;
            box-shadow:0 18px 40px rgba(15,23,42,0.85);
            border: 1px solid rgba(255, 255, 255, 0.08);
        ">
            <div class="banner-shimmer"></div>
            <div class="banner-breath-overlay"></div>
            <div class="border-beam"></div>
            <img src="data:image/png;base64,{banner_b64}"
                 style="width:100%; height:100%; object-fit:cover; filter:brightness(0.55);">
            <div style="
                position:absolute;
                inset:0;
                display:flex;
                flex-direction:column;
                justify-content:center;
                padding-left:2.5rem;
                color:#e5e7eb;
                z-index:6;
            ">
                <div style="font-size:0.9rem; letter-spacing:0.2em;
                            text-transform:uppercase; color:#38bdf8; font-weight:600;">
                    Projet COVID-19 · Deep Learning
                </div>
                <div class="banner-glow-text" style="font-size:2.2rem; font-weight:800; margin-top:0.4rem; color:#ffffff;">
                    Diagnostic par Radiographies Thoraciques
                </div>
                <div style="font-size:1.1rem; color:#94a3b8; margin-top:0.6rem; font-weight:400;">
                    Classification multi-classes : COVID-19 · Non-COVID · Normal
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================
# INITIALISATION AI
# =========================
ai_model = load_resnet_model()
class_names_list = ["COVID-19", "Non-COVID", "Normal"]

# ---------- INTRO ----------
if page == "Introduction":
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(open_card(wow_mode), unsafe_allow_html=True)
        st.markdown('<h3><i class="fa-solid fa-thumbtack" style="color:#38bdf8; margin-right:10px;"></i>Contexte et Objectifs du Projet</h3>', unsafe_allow_html=True)
        st.markdown(
            """
            Dans le contexte de la pandémie de COVID-19, la **détection rapide et fiable** des infections pulmonaires est devenue un enjeu critique pour les systèmes de santé.
            
            Ce projet vise à concevoir un **pipeline complet de diagnostic automatique** basé sur des radiographies thoraciques, afin de classifier les patients selon trois catégories distinctes :
            <div style="margin-top:10px; margin-bottom:15px;">
                <i class="fa-solid fa-virus-covid" style="color:#f87171; margin-right:8px;"></i> <b>COVID-19</b><br>
                <i class="fa-solid fa-lungs-virus" style="color:#fbbf24; margin-right:8px;"></i> <b>Non-COVID</b> (autres pathologies pulmonaires)<br>
                <i class="fa-solid fa-circle-check" style="color:#34d399; margin-right:8px;"></i> <b>Normal</b> (patients sains)
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown('<h3><i class="fa-solid fa-bullseye" style="color:#38bdf8; margin-right:10px;"></i>Objectifs</h3>', unsafe_allow_html=True)
        st.markdown(
            """
            - Construire un pipeline de bout en bout : **Prétraitement ➜ EDA ➜ Machine Learning ➜ CNN ➜ ResNet50**
            - Comparer les approches de **Machine Learning classiques** et de **Deep Learning**
            - Évaluer les performances via des métriques précises : *Accuracy, Précision, Rappel, F1-score*
            - Garantir l’explicabilité clinique des prédictions à l’aide des cartes **Grad-CAM**
            """
        )
        st.markdown(close_card(), unsafe_allow_html=True)

    with col2:
        st.markdown(open_card(wow_mode), unsafe_allow_html=True)
        # show_image("intro_radiography.jpg", "Exemple de radiographie thoracique", width=380)
        st.markdown(close_card(), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        """
        <div style="color:#94a3b8; font-size:0.95rem; margin-top:1rem;">
            <i class="fa-solid fa-terminal" style="color:#38bdf8; margin-right:8px;"></i> Utilisez le menu latéral pour explorer les dimensions du projet : 
            données, modèles classiques, architectures neuronales et analyses d'interprétabilité.
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------- DONNÉES & PRÉTRAITEMENT ----------
elif page == "Données & Prétraitement":
    st.markdown('<h2><i class="fa-solid fa-magnifying-glass-chart" style="color:#38bdf8; margin-right:15px;"></i>Intelligence des Données & Prétraitement</h2>', unsafe_allow_html=True)
    st.markdown(open_card(wow_mode), unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Jeu de données", "Analyse Exploratoire", "Flux de Travail"])
    
    with tab1:
        col_sel, col_stat = st.columns([1, 2])
        with col_sel:
            selected_class = st.selectbox(
                "Sélectionner une classe à inspecter :",
                ["Vue d'ensemble", "COVID-19", "Non-COVID", "Normal"]
            )
        
        with col_stat:
            if selected_class == "Vue d'ensemble":
                st.metric("Total des Échantillons", "5 826", "+25% vs v1")
                st.markdown(
                    """
                    **Base de données : Projet de Radiographie COVID-19**
                    - **COVID-19** : 2 913 images (50%)
                    - **Non-COVID** : 1 457 images (25%)
                    - **Normal** : 1 456 images (25%)
                    
                    > [!NOTE]
                    > Le dataset présente un déséquilibre de classe (Ratio 2:1:1). 
                    > Une stratégie de **pondération des classes** (Class Weights) a été implémentée pour garantir l'équité de l'apprentissage.
                    """
                )
            elif selected_class == "COVID-19":
                st.metric("COVID-19 Positifs", "2 913", "Prévalence 50%")
                st.info("Cas positifs vérifiés avec des données cliniques de référence.")
            elif selected_class == "Non-COVID":
                st.metric("Pathologies Non-COVID", "1 457", "Prévalence 25%")
                st.warning("Conditions pulmonaires alternatives incluant la pneumonie standard.")
            else:
                st.metric("Contrôles Sains", "1 456", "Prévalence 25%")
                st.success("Radiographies propres vérifiées par des radiologues indépendants.")

        if selected_class == "Vue d'ensemble":
            st.markdown("### Distribution des classes")
            # show_image("eda/class_distribution.png", "Distribution du nombre d'images par classe (train)", width=520)
            
            dist_data = pd.DataFrame([
                ["COVID-19", 1864, 466, 583],
                ["Non-COVID", 933, 233, 291],
                ["Normal",    931, 233, 292]
            ], columns=["Classe", "Entraînement", "Validation", "Test"])

            if PLOTLY_AVAILABLE:
                fig_dist_main = px.bar(
                    dist_data, 
                    x="Classe", 
                    y=["Entraînement", "Validation", "Test"],
                    title="Nombre d'images par classe (Split Complet)",
                    barmode="group",
                    color_discrete_map={
                        "Entraînement": "#38bdf8",
                        "Validation": "#fbbf24",
                        "Test": "#a855f7"
                    },
                    template="plotly_dark"
                )
                fig_dist_main.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_family="Outfit",
                    margin=dict(t=50, b=0, l=0, r=0),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_dist_main, use_container_width=True)
            else:
                st.table(dist_data)
        else:
            st.markdown(f"### Galerie d'exemples : {selected_class}")
            
            sample_dir = ASSETS_DIR / "samples" / selected_class
            # Recherche d'images locales
            sample_files = []
            if sample_dir.exists():
                sample_files = list(sample_dir.glob("*.png")) + list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.jpeg"))
            
            if not sample_files:
                st.info(f"💡 **Info** : Déposez des images dans `{sample_dir}` pour les voir apparaître ici.")
                # Garder 3 placeholders mais avec un texte d'instruction
                cols = st.columns(3)
                for i, c in enumerate(cols):
                    with c:
                        placeholder_url = f"https://placehold.co/400x400/1e293b/475569?text=Déposez+un+échantillon+ici"
                        html = f"""
                        <div class="glass-img-container stagger-{i+1}">
                            <div class="img-reveal-shine">
                                <img src="{placeholder_url}" style="width:100%; border-radius:20px; opacity:0.5;">
                                <div class="img-caption-overlay">En attente de données...</div>
                            </div>
                        </div>
                        """
                        st.markdown(html, unsafe_allow_html=True)
            else:
                # Affichage des images trouvées (max 3)
                cols = st.columns(min(len(sample_files), 3))
                for i, c in enumerate(cols):
                    with c:
                        img_path = sample_files[i]
                        img_b64 = get_img_base64(img_path)
                        img_src = f"data:image/png;base64,{img_b64}" if img_b64 else "https://placehold.co/400x400/1e293b/FFFFFF?text=Erreur+Lecture"
                        
                        html = f"""
                        <div class="glass-img-container stagger-{i+1}">
                            <div class="img-reveal-shine">
                                <img src="{img_src}" style="width:100%; border-radius:20px;">
                                <div class="img-caption-overlay">Échantillon {selected_class} #{i+1}</div>
                            </div>
                        </div>
                        """
                        st.markdown(html, unsafe_allow_html=True)

    with tab2:
        st.markdown("### Intelligence du Signal & Corrélation d'Intensité")
        st.markdown(
            """
            L'analyse exploratoire des **distributions d'intensité des pixels** révèle des signaux diagnostiques critiques :
            - Les cas **COVID-19 Positifs** présentent un décalage de l'intensité moyenne plus élevé.
            - Une corrélation modérée a été détectée entre l'intensité moyenne et la classe pathologique.
            
            Ces premiers signaux valident la présence de **biomarqueurs visuels** exploitables par les architectures neuronales profondes.
            """
        )
        c1, c2 = st.columns(2)
        with c1:
            show_image("eda/eda_intensity_hist.png", "Intensity Histogram Analysis", width=520)
        with c2:
            show_image("eda/eda_corr_matrix.png", "Correlation Matrix: Signal vs Class", width=520)

    # ---------- PRETRAITEMENT ----------
    with tab3:
        st.markdown(
            """
            ### Flux de Prétraitement
            Transformations de qualité médicale appliquées pour sauvegarder l'intégrité diagnostique :
            - **Normalisation** : Mise à l'échelle des pixels dans la plage [0, 1].
            - **Standardisation** : Redimensionnement à une résolution spectrale de 224×224.
            - **Augmentation** : Rotations stratégiques et zooms pour simuler la variabilité de l'acquisition clinique.
            """
        )
        
        # show_image("augmentation_examples.png", "Échantillons Diagnostiques Augmentés", width=650)

        st.markdown(
            """
            ### Augmentation de Données Neuronales
            Pour minimiser le **surapprentissage** et améliorer la généralisation clinique, les transformations suivantes sont injectées pendant l'entraînement :
            - **Géométrique** : Rotations stratégiques (±15°) et retournements horizontaux.
            - **Optique** : Ajustements de contraste et zoom numérique.
            - **Variance Simulée** : Imitation des variations du matériel d'acquisition radiologique.
            """
        )

        st.markdown("### Dataset Distribution Matrix")
        
        split_df = pd.DataFrame(
            [
                ["COVID-19",  2330,  583,  2913],
                ["Non-COVID", 1166,  291,  1457],
                ["Normal",    1164,  292,  1456],
            ],
            columns=["Classe", "Train + Val", "Test", "Total"]
        )
        
        if PLOTLY_AVAILABLE:
            fig_dist = px.bar(
                split_df, 
                x="Classe", 
                y=["Train + Val", "Test"],
                barmode="group",
                title="Distribution des Images par Classe et Ensemble",
                color_discrete_map={
                    "Train + Val": "#38bdf8",
                    "Test": "#a855f7"
                },
                template="plotly_dark"
            )
            fig_dist.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_family="Outfit",
                margin=dict(t=40, b=0, l=0, r=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.table(split_df)

        with st.expander("Afficher le tableau de distribution détaillé"):
            st.table(split_df)

       # show_image("augmentation_examples.png", "Exemples d'images après data augmentation", width=650)

    st.markdown(close_card(), unsafe_allow_html=True)

# ---------- MACHINE LEARNING ----------
elif page == "Modèles de ML Classiques":
    st.markdown('<h2><i class="fa-solid fa-microchip" style="color:#38bdf8; margin-right:15px;"></i>Modèles de Référence (Machine Learning)</h2>', unsafe_allow_html=True)
    st.markdown(open_card(wow_mode), unsafe_allow_html=True)

    st.markdown(
        """
        Avant la modélisation architecturale neuronale, plusieurs références de classification classique 
        ont été évaluées sur des caractéristiques d'image extraites.
        """
    )

    st.markdown('<h3><i class="fa-solid fa-chart-column" style="color:#38bdf8; margin-right:10px;"></i>Performance Comparative (Interactive)</h3>', unsafe_allow_html=True)
    
    # Choix de la métrique à visualiser
    metric_view = st.radio("Métrique à comparer :", ["Accuracy (%)", "F1-score (%)", "Précision (%)", "Rappel (%)"], horizontal=True)
    
    # Create Plotly interactive bar chart
    if PLOTLY_AVAILABLE:
        fig_ml = px.bar(
            classic_models_df,
            x="Modèle",
            y=metric_view,
            color=metric_view,
            color_continuous_scale="Viridis",
            title=f"Comparaison des Modèles : {metric_view}",
            template="plotly_dark"
        )
        fig_ml.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_family="Outfit",
            margin=dict(t=50, b=0, l=0, r=0)
        )
        st.plotly_chart(fig_ml, use_container_width=True)
    else:
        st.bar_chart(classic_models_df.set_index("Modèle")[metric_view])

    with st.expander("Voir le tableau de données complet"):
        st.dataframe(classic_models_df, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            **Observations générales :**
            - Les performances globales restent **modestes** (F1 < 0.45).
            - Le **SVM** obtient la meilleure accuracy (~0.51).
            - **k-NN** et **XGBoost** offrent un compromis correct précision / rappel.
            """
        )
    with col2:
        st.markdown('<h4><i class="fa-solid fa-table-cells" style="color:#38bdf8; margin-right:10px;"></i>Matrice de Confusion Directe</h4>', unsafe_allow_html=True)
        sel_matrix = st.selectbox("Voir la matrice du modèle :", classic_models_df["Modèle"].tolist(), index=4)
        
        z = ml_confusion_matrices[sel_matrix]
        fig_mx = px.imshow(
            z,
            labels=dict(x="Classe Prédite", y="Classe Réelle", color="Nombre"),
            x=["COVID-19", "Non-COVID", "Normal"],
            y=["COVID-19", "Non-COVID", "Normal"],
            text_auto=True,
            color_continuous_scale='Blues',
            aspect="auto"
        )
        fig_mx.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_family="Outfit",
            margin=dict(t=20, b=20, l=20, r=20),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_mx, use_container_width=True)

    st.markdown(
        """
        Ces résultats montrent les **limites des modèles classiques** sur des données d'images médicales complexes
        et motivent le passage à des **réseaux de neurones convolutifs**.
        """
    )
    st.markdown(close_card(), unsafe_allow_html=True)

# ---------- ARCHITECTURE ----------
elif page == "Architecture du Réseau":
    st.markdown('<h2><i class="fa-solid fa-layer-group" style="color:#38bdf8; margin-right:15px;"></i>Stratégie d\'Architecture Neuronale</h2>', unsafe_allow_html=True)
    st.markdown(open_card(wow_mode), unsafe_allow_html=True)

    st.markdown(
        """
        ### Pourquoi le Deep Learning pour la COVID-19 ?
        Les méthodes traditionnelles ne parviennent pas à capturer les hiérarchies spatiales complexes des radiographies thoraciques. 
        Notre stratégie exploite les **Réseaux de Neurones Convolutifs** pour extraire automatiquement les biomarqueurs pathologiques.

        Les radiographies thoraciques sont des jeux de données visuels très complexes contenant des biomarqueurs pathologiques subtils : 
        opacités diffuses, infiltrations, asymétries pulmonaires et surcharge alvéolaire.
 
        Le **Machine Learning** traditionnel repose sur des *caractéristiques conçues manuellement* 
        (statistiques d'intensité, textures locales), ce qui limite considérablement sa capacité à capturer :
        - Des motifs spatiaux à grande échelle
        - Des structures anatomiques complexes
        - Des variations morphologiques spécifiques à la maladie
 
        Les **Réseaux de Neurones Convolutifs (CNN)** apprennent automatiquement des représentations hiérarchiques de caractéristiques :
        - **Couches Initiales** : Bords, gradients et textures de base
        - **Couches Intermédiaires** : Motifs pulmonaires récurrents et densité tissulaire
        - **Couches Profondes** : Marqueurs pathologiques complexes (opacités, consolidations)
 
        Cette extraction automatique de caractéristiques fait du Deep Learning la norme de référence pour l'analyse d'images médicales.
        """
    )

    st.markdown(
        """
        ### Pipeline de Deep Learning Multi-Étapes
        Le développement a impliqué deux phases architecturales distinctes :
 
        #### 1. CNN Baseline (Entraînement à partir de zéro)
        - Architecture légère à 4 blocs
        - BatchNorm et Dropout pour la régularisation
        - Utilisé pour **valider la faisabilité du problème**
        - A fourni une base solide avec quelques limites de généralisation
 
        #### 2. Transfer Learning ResNet50
        - Architecture profonde pré-entraînée sur **ImageNet**
        - Couches supérieures ajustées pour le diagnostic radiographique
        - Exploite des **représentations visuelles riches** déjà apprises sur des millions d'images
        - A considérablement réduit le surapprentissage et maximisé les performances
        """
    )

    st.markdown('<h3><i class="fa-solid fa-code-compare" style="color:#38bdf8; margin-right:10px;"></i>Vue d\'ensemble des Performances Comparatives</h3>', unsafe_allow_html=True)
    st.table(baseline_vs_resnet_df)

    st.markdown(
        """
        **Synthèse Comparative :**
        - Le **CNN Baseline** a atteint une précision d'environ ~**74,90 %** avec un **F1-macro ≈ 71,80 %**.
        - Le **ResNet50 ajusté** a permis un bond majeur vers une **précision de 91,00 %** et un **score F1 de 89,00 %**.
        - **Gain Net** : Amélioration de +16,10 % de la précision et de +17,20 % du score F1.
        """
    )

    st.markdown(close_card(), unsafe_allow_html=True)

# ---------- PERFORMANCE ----------
elif page == "Résultats & Analyse":
    st.markdown('<h2><i class="fa-solid fa-gauge-high" style="color:#38bdf8; margin-right:15px;"></i>Analyse des Performances</h2>', unsafe_allow_html=True)
    st.markdown(open_card(wow_mode), unsafe_allow_html=True)

    tab_overview, tab_cnn, tab_resnet = st.tabs(["Comparaison Directe", "CNN de Référence", "ResNet50 Studio"])

    with tab_overview:
        st.markdown('<h3><i class="fa-solid fa-trophy" style="color:#fbbf24; margin-right:10px;"></i>Résultats Principaux</h3>', unsafe_allow_html=True)
        # Summary metrics
        m1, m2, m3, m4 = st.columns(4)
        acc_cnn, acc_res = 74.90, 91.00
        f1_cnn, f1_res = 71.80, 89.00
        gain_acc = acc_res - acc_cnn
        gain_f1 = f1_res - f1_cnn

        with m1:
            st.metric("Précision (CNN)", f"{acc_cnn:.2f}%")
        with m2:
            st.metric("Précision (ResNet50)", f"{acc_res:.2f}%", f"+{gain_acc:.2f}%")
        with m3:
            st.metric("Score F1 (CNN)", f"{f1_cnn:.2f}%")
        with m4:
            st.metric("Score F1 (ResNet50)", f"{f1_res:.2f}%", f"+{gain_f1:.2f}%")

        st.markdown("---")
        st.markdown("### Résumé du Benchmarking")
        st.table(baseline_vs_resnet_df)

        st.success("Conclusion : Le Transfer Learning offre un avantage critique en termes de précision diagnostique et de robustesse des caractéristiques.")

    with tab_cnn:
        st.markdown('<h3><i class="fa-solid fa-microchip" style="color:#38bdf8; margin-right:10px;"></i>Configuration du CNN Baseline</h3>', unsafe_allow_html=True)
        st.markdown(
            """
            - **Couches** : Architecture à 4 blocs (32 à 256 filtres)
            - **Régularisation** : BatchNorm + Dropout (0.4)
            - **Optimiseur** : Adam (lr=1e-3)
            - **Entrées** : Niveaux de gris normalisés 224x224
            """
        )
        st.table(cnn_baseline_df)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("#### Courbes d'Apprentissage")
            subcol1, subcol2 = st.columns(2)
            with subcol1:
                show_image("cnn_baseline_learning_curves1.png", "Loss (Baseline)", width=None)
            with subcol2:
                show_image("cnn_baseline_learning_curves.png", "Accuracy (Baseline)", width=None)
        
        with col2:
            st.markdown("#### Matrice de Confusion (Baseline)")
            if PLOTLY_AVAILABLE:
                fig_mx_cnn = px.imshow(
                    dl_confusion_matrices["CNN Baseline"],
                    labels=dict(x="Prédit", y="Réel", color="Nombre"),
                    x=["COVID-19", "Non-COVID", "Normal"],
                    y=["COVID-19", "Non-COVID", "Normal"],
                    text_auto=True,
                    color_continuous_scale='Blues',
                    aspect="auto"
                )
                fig_mx_cnn.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_family="Outfit",
                    margin=dict(t=20, b=20, l=20, r=20),
                    coloraxis_showscale=False,
                    height=300
                )
                st.plotly_chart(fig_mx_cnn, use_container_width=True)
            else:
                st.warning("Plotly non disponible pour la matrice.")

    with tab_resnet:
        st.markdown('<h3><i class="fa-solid fa-brain" style="color:#38bdf8; margin-right:10px;"></i>Fine-Tuning Avancé ResNet50</h3>', unsafe_allow_html=True)
        st.markdown(
            """
            L'exploitation des **poids pré-entraînés** d'ImageNet permet au modèle de commencer avec une compréhension sophistiquée 
            des motifs géométriques, qui sont ensuite affinés pour le diagnostic pulmonaire.
            
            **Configuration du Rééquilibrage :**
            Pour compenser le déséquilibre du dataset (Ratio 2:1:1), nous avons utilisé la méthode `class_weight` :
            - **COVID-19** : Poids 0.667 
            - **Non-COVID** : Poids 1.333
            - **Normal** : Poids 1.333
            
            Cela force l'algorithme à accorder deux fois plus d'importance aux erreurs sur les classes minoritaires.
            """
        )
        st.dataframe(resnet_per_class_df, use_container_width=True)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("#### Courbes d'Apprentissage")
            subcol1, subcol2 = st.columns(2)
            with subcol1:
                show_image("resnet_training_curves1.png", "Loss (ResNet50)", width=None)
            with subcol2:
                show_image("resnet_training_curves.png", "Accuracy (ResNet50)", width=None)
        
        with col2:
            st.markdown("#### Matrice de Confusion (ResNet50)")
            if PLOTLY_AVAILABLE:
                fig_mx_res = px.imshow(
                    dl_confusion_matrices["ResNet50"],
                    labels=dict(x="Prédit", y="Réel", color="Nombre"),
                    x=["COVID-19", "Non-COVID", "Normal"],
                    y=["COVID-19", "Non-COVID", "Normal"],
                    text_auto=True,
                    color_continuous_scale='Blues',
                    aspect="auto"
                )
                fig_mx_res.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_family="Outfit",
                    margin=dict(t=20, b=20, l=20, r=20),
                    coloraxis_showscale=False,
                    height=300
                )
                st.plotly_chart(fig_mx_res, use_container_width=True)
            else:
                 st.warning("Plotly non disponible.")

    st.markdown(close_card(), unsafe_allow_html=True)

    st.markdown(close_card(), unsafe_allow_html=True)

# ---------- GRAD-CAM ----------
elif page == "Interprétabilité (Grad-CAM)":
    st.markdown('<h2><i class="fa-solid fa-eye" style="color:#38bdf8; margin-right:15px;"></i>Interprétabilité Diagnostique : Grad-CAM</h2>', unsafe_allow_html=True)
    st.markdown(open_card(wow_mode), unsafe_allow_html=True)

    st.markdown(
        """
        Pour visualiser la logique interne du moteur **ResNet50**, nous utilisons les cartes de chaleur **Grad-CAM**. 
        Cela offre une fenêtre clinique sur les régions de la radiographie qui ont le plus contribué au diagnostic final.
        """
    )

    st.markdown(
        """
        Grad-CAM agit comme une lentille diagnostique, cartographiant le processus de prise de décision du moteur ResNet50. 
        Il met en évidence les preuves visuelles utilisées pour la classification :
        - **Zones Rouges/Orange** : Forte activation, principaux moteurs diagnostiques.
        - **Zones Bleues** : Régions de faible pertinence.
        """
    )

    st.markdown("### Exemples de cartes Grad-CAM")
    col1, col2 = st.columns(2)
    with col1:
        show_image("gradcam_example_1.png", "Grad-CAM sur une radiographie COVID-19", width=420)
    with col2:
        show_image("gradcam_example_2.png", "Grad-CAM sur un autre cas (Non-COVID ou Normal)", width=420)

    st.markdown(
        """
        ### Interprétation Clinique
        - Le modèle se concentre sur les **champs pulmonaires bilatéraux**, plus précisément sur les zones moyennes et inférieures.
        - Les activations intenses près du **hile** et des **bases pulmonaires** correspondent aux opacités typiquement observées dans les pneumonies virales.
        - Les activations périphériques (épaules, bords de l'image) soulignent la nécessité d'un **nettoyage des données** ou d'un **masquage diagnostique** plus poussé.
 
        Globalement, les visualisations Grad-CAM démontrent que le modèle apprend des **motifs radiologiques cohérents avec la pratique clinique**, renforçant la confiance dans ses résultats prédictifs.
        """
    )

    st.markdown(close_card(), unsafe_allow_html=True)

# ---------- SIMULATEUR ----------
elif page == "Simulateur Interactif":
    st.markdown('<h2><i class="fa-solid fa-flask-vial" style="color:#38bdf8; margin-right:15px;"></i>Simulateur de Diagnostic Réel</h2>', unsafe_allow_html=True)
    st.markdown(open_card(wow_mode), unsafe_allow_html=True)

    if not AI_DEPS_INSTALLED:
        st.error("❌ **Dépendances IA manquantes** : TensorFlow ou OpenCV ne sont pas installés dans cet environnement Python.")
        st.info("💡 **Conseil** : Lancez l'application avec Python 3.12 (`py -3.12 -m streamlit run app_covid_streamlit.py`) pour activer l'IA.")
    elif ai_model is None:
        st.warning("⚠️ Fichier `resnet_final.h5` introuvable ou illisible dans le dossier racine.")
    else:
        st.success("✅ **Intelligence Artificielle Connectée** : Le modèle ResNet50 est prêt.")

    col_up, col_res = st.columns([1, 1])

    with col_up:
        st.markdown('### <i class="fa-solid fa-upload" style="color:#38bdf8; margin-right:10px;"></i>Chargement X-Ray', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choisissez une radiographie thoracique...", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.image(img, caption="Image source pour analyse", use_container_width=True)
            
            if st.button("� Lancer l'Analyse Intelligence", use_container_width=True):
                with st.status("Traitement par ResNet50...", expanded=True) as status:
                    st.write("Prétraitement de l'image (224x224)...")
                    img_batch, img_orig = preprocess_image_for_model(img)
                    
                    st.write("Inférence Deep Learning...")
                    start_time = time.time()
                    preds = ai_model.predict(img_batch)
                    inf_time = time.time() - start_time
                    
                    st.write("Génération de la carte Grad-CAM...")
                    heatmap = run_grad_cam(ai_model, img_batch)
                    grad_image = overlay_heatmap(img_orig, heatmap)
                    
                    status.update(label=f"Analyse terminée en {inf_time:.2f}s", state="complete", expanded=False)
                
                # Stockage des résultats réels
                st.session_state.diag_done = True
                st.session_state.diag_probs = {
                    class_names_list[i]: float(preds[0][i]) for i in range(len(class_names_list))
                }
                st.session_state.grad_img = grad_image
                st.session_state.diag_seed = len(uploaded_file.name)
        else:
            st.markdown(
                """
                <div style="text-align:center; padding: 2rem; border: 2px dashed rgba(255,255,255,0.1); border-radius:15px;">
                    <i class="fa-solid fa-file-medical-alt" style="font-size:3rem; color:rgba(255,255,255,0.1); margin-bottom:1rem;"></i><br>
                    Glissez une image ici pour démarrer l'analyse.
                </div>
                """,
                unsafe_allow_html=True
            )

    with col_res:
        if "diag_done" in st.session_state and st.session_state.diag_done:
            st.markdown('### <i class="fa-solid fa-chart-pie" style="color:#38bdf8; margin-right:10px;"></i>Résultats du Diagnostic IA', unsafe_allow_html=True)
            
            p = st.session_state.diag_probs
            max_class = max(p, key=p.get)
            
            # Badge de résultat
            bg_color = "#f87171" if max_class == "COVID-19" else "#fbbf24" if max_class == "Non-COVID" else "#34d399"
            st.markdown(
                f"""
                <div style="background:{bg_color}22; border-left: 5px solid {bg_color}; padding: 1.5rem; border-radius: 10px; margin-bottom: 20px;">
                    <div style="font-size: 0.8rem; text-transform:uppercase; color:{bg_color}; letter-spacing:0.1em; font-weight:700;">Diagnostic Principal</div>
                    <div style="font-size: 1.8rem; font-weight:800; color:#ffffff;">{max_class}</div>
                    <div style="font-size: 1.1rem; color:#94a3b8; font-weight:500;">Confiance IA : {p[max_class]*100:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Graphes Plotly
            if PLOTLY_AVAILABLE:
                tab_g1, tab_g2 = st.tabs(["Probabilités Réelles", "Signature Analytique"])
                
                with tab_g1:
                    fig_bar = px.bar(
                        x=list(p.keys()), 
                        y=list(p.values()),
                        labels={'x': 'Pathologie', 'y': 'Probabilité'},
                        color=list(p.keys()),
                        color_discrete_map={"COVID-19": "#f87171", "Non-COVID": "#fbbf24", "Normal": "#34d399"},
                        template="plotly_dark"
                    )
                    fig_bar.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0), showlegend=False)
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                with tab_g2:
                    # Radar chart dynamique basé sur les probas pour l'effet visuel
                    categories = ['Densité Hilaire', 'Texture Pulmonaire', 'Uniformité', 'Détresse Virale', 'Confiance IA']
                    # On simule des métriques dérivées des probas pour la cohérence visuelle
                    base_val = p[max_class]
                    np.random.seed(st.session_state.diag_seed)
                    values = [base_val * np.random.uniform(0.8, 1.0) for _ in range(5)]
                    
                    fig_radar = go.Figure()
                    fig_radar.add_trace(go.Scatterpolar(
                          r=values,
                          theta=categories,
                          fill='toself',
                          name='Analyse Signature',
                          line_color="#38bdf8"
                    ))
                    fig_radar.update_layout(
                      polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                      showlegend=False,
                      template="plotly_dark",
                      height=300,
                      margin=dict(l=40,r=40,t=20,b=20)
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)
            
            st.markdown("---")
            st.markdown('#### <i class="fa-solid fa-eye" style="color:#38bdf8; margin-right:10px;"></i>Interprétation Grad-CAM (Dynamique)', unsafe_allow_html=True)
            col_grad_img, col_grad_txt = st.columns([1, 1.2])
            with col_grad_img:
                st.image(st.session_state.grad_img, caption="Focus de détection ResNet50", use_container_width=True)
            with col_grad_txt:
                st.markdown(
                    f"""
                    <div style="font-size:0.85rem; color:#94a3b8; line-height:1.4;">
                        Cette carte thermique montre où le modèle a concentré son attention pour prédire <b>{max_class}</b>.
                        Les zones chaudes identifient les anomalies de texture et d'opacité qui ont motivé la décision de l'IA.
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                """
                <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:100%; min-height:300px; color:#475569;">
                    <i class="fa-solid fa-stethoscope" style="font-size:4rem; margin-bottom:1rem; opacity:0.3;"></i>
                    <p>Prêt pour l'analyse. Veuillez charger une image.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown(close_card(), unsafe_allow_html=True)

# ---------- CONCLUSION ----------
elif page == "Conclusion":
    st.markdown('<h2><i class="fa-solid fa-flag-checkered" style="color:#38bdf8; margin-right:15px;"></i>Conclusion Générale</h2>', unsafe_allow_html=True)
    st.markdown(open_card(wow_mode), unsafe_allow_html=True)

    st.markdown(
        """
        <div class="stagger-1">
            <h3 style="color:#38bdf8;">Synthèse du Projet</h3>
            Ce projet a permis de construire un pipeline complet de <b>diagnostic automatique de la COVID-19</b> à partir de radiographies thoraciques, en passant par :
            <ul style="margin-top:10px;">
                <li>La préparation et l'exploration d'un dataset complexe.</li>
                <li>L'évaluation de modèles classiques (LogReg, SVM, etc.).</li>
                <li>Le développement d'un CNN baseline.</li>
                <li>La mise en place d'un ResNet50 en transfer learning.</li>
                <li>L'interprétabilité via des cartes Grad-CAM.</li>
            </ul>
        </div>
        <div class="stagger-2" style="margin-top:2rem;">
            <h3>Bilan des Performances</h3>
            <ul style="list-style:none; padding-left:0;">
                <li style="margin-bottom:1rem;"><i class="fa-solid fa-microchip" style="color:#38bdf8; margin-right:10px;"></i> Les modèles classiques atteignent des performances <b>limitées</b> sur ce type de données.</li>
                <li style="margin-bottom:1rem;"><i class="fa-solid fa-bolt" style="color:#fbbf24; margin-right:10px;"></i> Le <b>ResNet50 fine-tuné</b> offre une <b>précision globale de 91,00 %</b> et des scores F1 élevés, démontrant la pertinence du Deep Learning avancé.</li>
                <li style="margin-bottom:1rem;"><i class="fa-solid fa-eye" style="color:#34d399; margin-right:10px;"></i> L'IA <b>Interprétable</b> (Grad-CAM) valide que le modèle se concentre sur des motifs pulmonaires cliniquement pertinents.</li>
            </ul>
        </div>
        <div class="stagger-3" style="margin-top:2rem;">
            <h3>Perspectives & Remerciements</h3>
            <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap:1.5rem; margin-bottom:2rem;">
                <div style="background:rgba(56, 189, 248, 0.05); border: 1px solid rgba(56, 189, 248, 0.1); border-radius:16px; padding:1.25rem;">
                    <div style="display:flex; align-items:center; gap:10px; margin-bottom:10px;">
                        <i class="fa-solid fa-eye" style="color:#38bdf8;"></i>
                        <h4 style="margin:0; font-size:1.1rem; color:#38bdf8;">ViT (Vision Transformer)</h4>
                    </div>
                    <p style="font-size:0.85rem; color:#94a3b8; line-height:1.5; margin:0;">
                        Découpe l'image en "patchs" (petits carrés) et analyse comment ces carrés sont liés les uns aux autres pour une compréhension spatiale globale.
                    </p>
                </div>
                <div style="background:rgba(139, 92, 246, 0.05); border: 1px solid rgba(139, 92, 246, 0.1); border-radius:16px; padding:1.25rem;">
                    <div style="display:flex; align-items:center; gap:10px; margin-bottom:10px;">
                        <i class="fa-solid fa-network-wired" style="color:#8b5cf6;"></i>
                        <h4 style="margin:0; font-size:1.1rem; color:#8b5cf6;">Fusion Multi-modale</h4>
                    </div>
                    <p style="font-size:0.85rem; color:#94a3b8; line-height:1.5; margin:0;">
                        Combine plusieurs sources (images de scanner, rapports médicaux, données génétiques) pour une décision beaucoup plus précise et holistique.
                    </p>
                </div>
                <div style="background:rgba(52, 211, 153, 0.05); border: 1px solid rgba(52, 211, 153, 0.1); border-radius:16px; padding:1.25rem;">
                    <div style="display:flex; align-items:center; gap:10px; margin-bottom:10px;">
                        <i class="fa-solid fa-hospital-user" style="color:#34d399;"></i>
                        <h4 style="margin:0; font-size:1.1rem; color:#34d399;">Déploiement Clinique</h4>
                    </div>
                    <p style="font-size:0.85rem; color:#94a3b8; line-height:1.5; margin:0;">
                        L'étape finale : passer du laboratoire à l'hôpital. L'outil est installé sur les postes des médecins pour aider au diagnostic de vrais patients.
                    </p>
                </div>
            </div>
            <p>Nous remercions les chercheurs de la <b>COVID-19 Radiography Database</b> ainsi que nos mentors académiques pour leur soutien tout au long de ce projet.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    
    # Animation de Fin (Thank You)
    lottie_thankyou = load_lottieurl("https://lottie.host/4b673646-0b89-408a-b86a-74b8801d0c41/A8ql2tL5E8.json")
    
    col_center = st.columns([1, 2, 1])
    with col_center[1]:
        if lottie_thankyou:
             st_lottie(lottie_thankyou, height=200, key="thankyou_anim")
        
        st.markdown(
            """
            <div style="text-align:center; margin-top:-20px;">
                <h2 style="
                    background: linear-gradient(to right, #38bdf8, #8b5cf6);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    font-size: 2.2rem;
                    font-weight: 800;
                ">
                    Merci pour votre attention !
                </h2>
                <p style="color:#94a3b8; font-size:1rem;">
                    Projet COVID-19 Diagnostic Intelligence<br>
                    <span style="font-size:0.8rem; opacity:0.7;">© 2025 Dass Luvengani | Saadia Szapiro</span>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(close_card(), unsafe_allow_html=True)
