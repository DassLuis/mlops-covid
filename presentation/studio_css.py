"""Feuille de style « studio » : thème sombre, glassmorphism, animations.

Reprend la charte du fichier de référence : dégradé nuit animé, cartes en verre,
accents cyan et violet, police Outfit, iconographie Font Awesome.
"""

CYAN = "#38bdf8"
VIOLET = "#8b5cf6"
AMBRE = "#fbbf24"
VERT = "#34d399"
ROUGE = "#f87171"
ARDOISE = "#94a3b8"
ARDOISE_2 = "#64748b"
ARDOISE_3 = "#475569"
NEIGE = "#f8fafc"
BRUME = "#f1f5f9"

CUSTOM_CSS = """
<style>
/* =========================
   0. ICONOGRAPHY (Font Awesome)
   ========================= */
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

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

[data-testid="stHeader"] { background: transparent; }
#MainMenu, [data-testid="stToolbar"] { visibility: hidden; }
footer { visibility: hidden; }

[data-testid="stMainBlockContainer"] { padding-top: 2.2rem; max-width: 1500px; }

/* =========================
   2. SIDEBAR (Studio-Grade)
   ========================= */
[data-testid="stSidebar"] {
    background: rgba(10, 15, 30, 0.9) !important;
    backdrop-filter: blur(40px) saturate(150%);
    -webkit-backdrop-filter: blur(40px) saturate(150%);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

[data-testid="stSidebar"] h2 {
    font-weight: 800;
    font-size: 1.5rem;
    letter-spacing: -0.03em;
    color: #f8fafc;
    padding: 0.5rem 0 1rem;
}

/* Navigation Overhaul */
.stRadio > div { gap: 0.5rem; }
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
.stRadio [data-testid="stWidgetLabel"] { display: none; }

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
.sidebar-separator {
    height: 1px;
    background: rgba(255, 255, 255, 0.05);
    margin: 1.2rem 0;
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
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #38bdf8 0%, #8b5cf6 100%);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    color: white; font-weight: 700; font-size: 0.8rem;
    box-shadow: 0 4px 10px rgba(56, 189, 248, 0.2);
}
.author-info { display: flex; flex-direction: column; }
.author-name { font-size: 0.85rem; font-weight: 600; color: #f1f5f9; }
.author-role { font-size: 0.65rem; color: #64748b; }

.lottie-box {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 20px;
    margin-bottom: 0.5rem;
    overflow: hidden;
}
[data-testid="stIFrame"], [data-testid="stCustomComponentV1"] {
    background: transparent !important;
    border-radius: 18px;
}

/* Onde de diffusion : l'illustration animée de la barre latérale. */
.onde {
    position: relative;
    height: 150px;
    display: flex; align-items: center; justify-content: center;
}
.onde i {
    position: absolute;
    width: 26px; height: 26px;
    border-radius: 50%;
    border: 2px solid #38bdf8;
    opacity: 0;
    animation: ondePropage 4.5s cubic-bezier(0.2, 0.6, 0.3, 1) infinite;
}
.onde i:nth-child(2) { animation-delay: 1.5s; border-color: #8b5cf6; }
.onde i:nth-child(3) { animation-delay: 3.0s; }
.onde b {
    width: 16px; height: 16px; border-radius: 50%;
    background: #38bdf8;
    box-shadow: 0 0 18px rgba(56, 189, 248, 0.8);
    animation: ondeCoeur 3s ease-in-out infinite;
}
@keyframes ondePropage {
    0%   { transform: scale(0.6); opacity: 0; }
    12%  { opacity: 0.75; }
    100% { transform: scale(4.6); opacity: 0; }
}
@keyframes ondeCoeur {
    0%, 100% { transform: scale(1); }
    50%      { transform: scale(1.28); }
}

/* =========================
   3. CARDS & SECTIONS
   ========================= */
.card-flag { display: none; }

/* La carte enveloppe réellement le bloc Streamlit qui porte le drapeau. */
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .card-flag),
[data-testid="stVerticalBlock"]:has(> div > [data-testid="stMarkdownContainer"] .card-flag) {
    background: rgba(15, 23, 42, 0.4);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-radius: 24px;
    padding: 2rem;
    border: 1px solid rgba(255, 255, 255, 0.05);
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4);
}

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
    color: #f8fafc;
}
h2 { font-size: 2rem !important; }
h3 { font-size: 1.35rem !important; }

p, li, .stMarkdown { color: #cbd5e1; }
strong, b { color: #f1f5f9; }

/* Metrics */
[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 20px;
    padding: 1.25rem;
    transition: all 0.3s ease;
}
[data-testid="stMetric"]:hover {
    background: rgba(255, 255, 255, 0.04);
    border-color: rgba(56, 189, 248, 0.2);
    transform: translateY(-3px);
}
[data-testid="stMetricValue"] {
    font-weight: 800; letter-spacing: -0.02em; color: #f8fafc;
}
[data-testid="stMetricLabel"] p {
    font-size: 0.72rem !important; text-transform: uppercase;
    letter-spacing: 0.1em; color: #64748b !important; font-weight: 600;
}

/* Dataframe & Tables */
.stDataFrame, .stTable {
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.05);
}

/* Onglets */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 0.4rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stTabs"] button p {
    font-size: 0.9rem !important; font-weight: 600;
    letter-spacing: 0.01em; color: #64748b;
}
[data-testid="stTabs"] button[aria-selected="true"] p { color: #38bdf8; }

/* Champs de saisie */
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
    color: #f1f5f9 !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    box-shadow: 0 0 12px rgba(56,189,248,0.45);
}

/* Encarts d'état */
[data-testid="stAlert"] { border-radius: 16px; }

/* Blocs de code */
[data-testid="stCode"], pre {
    border-radius: 14px !important;
    border: 1px solid rgba(255,255,255,0.06);
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
    box-shadow: 0 40px 80px rgba(0, 0, 0, 0.7), 0 0 30px rgba(56, 189, 248, 0.2);
}
.glass-img-container img {
    width: 100%; height: auto; display: block;
    border-radius: 20px !important;
    transition: transform 0.8s cubic-bezier(0.16, 1, 0.3, 1) !important;
    filter: contrast(1.05) brightness(1.02);
}
.glass-img-container:hover img { transform: scale(1.05); }

.img-reveal-shine::after {
    content: "";
    position: absolute;
    top: -100%; left: -100%; width: 200%; height: 200%;
    background: linear-gradient(135deg,
        rgba(255,255,255,0) 0%, rgba(255,255,255,0.05) 50%, rgba(255,255,255,0) 100%);
    transition: all 0.7s ease;
    pointer-events: none;
}
.glass-img-container:hover .img-reveal-shine::after { top: 50%; left: 50%; }

.img-caption-overlay {
    position: absolute; bottom: 20px; left: 20px; right: 20px;
    padding: 15px;
    background: rgba(15, 23, 42, 0.8);
    backdrop-filter: blur(10px);
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.05);
    color: #f1f5f9; font-size: 0.85rem;
    opacity: 0; transform: translateY(10px);
    transition: all 0.4s ease 0.1s;
}
.glass-img-container:hover .img-caption-overlay { opacity: 1; transform: translateY(0); }

/* Premium Stepper */
.premium-pipeline-container { padding: 0.5rem 0.5rem 0.5rem 0.8rem; }
.stepper-track { position: relative; padding-left: 1.2rem; }
.stepper-track::before {
    content: ""; position: absolute; left: 4px; top: 8px; bottom: 8px;
    width: 1px; background: rgba(255,255,255,0.06);
}
.stepper-step { position: relative; padding-bottom: 0.8rem; display: flex; align-items: center; }
.stepper-step:last-child { padding-bottom: 0; }
.stepper-node {
    position: absolute; left: -1.2rem; margin-left: 1px; top: 5px;
    width: 7px; height: 7px; border-radius: 50%;
    background: #1e293b; border: 1px solid #475569; z-index: 2;
    transition: all 0.4s ease;
}
.stepper-node.active {
    background: #38bdf8; border-color: #38bdf8;
    box-shadow: 0 0 12px rgba(56, 189, 248, 0.5); transform: scale(1.2);
}
.stepper-node.completed { background: #34d399; border-color: #34d399; }
.stepper-node.visited { background: #1e293b; border-color: #38bdf8; }
.stepper-text {
    font-size: 0.75rem; font-weight: 500; color: #475569;
    transition: all 0.3s ease; white-space: nowrap;
}
.stepper-text.active { color: #f8fafc; font-weight: 600; }
.stepper-text.completed { color: #94a3b8; }
.stepper-text.visited { color: #64748b; }

/* Pastilles et listes iconographiées */
.pill {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(56,189,248,0.08);
    border: 1px solid rgba(56,189,248,0.18);
    color: #38bdf8;
    border-radius: 999px; padding: 6px 14px;
    font-size: 0.78rem; font-weight: 600;
    letter-spacing: 0.03em;
    margin: 0 8px 8px 0;
}
.pill.amber { background: rgba(251,191,36,0.08); border-color: rgba(251,191,36,0.2); color: #fbbf24; }
.pill.violet { background: rgba(139,92,246,0.08); border-color: rgba(139,92,246,0.2); color: #8b5cf6; }
.pill.green { background: rgba(52,211,153,0.08); border-color: rgba(52,211,153,0.2); color: #34d399; }

.kv { display: flex; flex-direction: column; gap: 0.65rem; margin: 0.4rem 0 0; }
.kv-row {
    display: flex; justify-content: space-between; align-items: baseline;
    gap: 1rem; padding-bottom: 0.55rem;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.kv-row:last-child { border-bottom: none; }
.kv-k { color: #94a3b8; font-size: 0.85rem; }
.kv-v { color: #f1f5f9; font-weight: 600; font-size: 0.9rem; text-align: right; }

.note {
    border-left: 2px solid rgba(56,189,248,0.4);
    padding: 0.15rem 0 0.15rem 0.9rem;
    color: #94a3b8; font-size: 0.88rem; line-height: 1.6;
}
.note.amber { border-left-color: rgba(251,191,36,0.5); }
.note.red { border-left-color: rgba(248,113,113,0.5); }

.finding {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 18px;
    padding: 1.1rem 1.25rem;
    transition: all .3s ease;
    height: 100%;
}
.finding:hover { border-color: rgba(56,189,248,0.22); transform: translateY(-3px); }
.finding-h {
    display: flex; align-items: center; gap: 10px;
    font-weight: 700; color: #f1f5f9; font-size: 0.98rem; margin-bottom: 0.5rem;
}
.finding-b { color: #94a3b8; font-size: 0.87rem; line-height: 1.6; }
</style>
"""

STYLE_WOW = """
<style>
:root { --cardIn: 0.6s; --hoverLift: -6px; }

@keyframes fadeUpWow {
  from { opacity: 0; transform: translateY(20px) scale(0.98); }
  to   { opacity: 1; transform: translateY(0) scale(1); }
}
@keyframes glassShimmer {
  0%   { background-position: -200% -200%; }
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

[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .card-flag.reveal),
.block-card {
  animation: fadeUpWow 0.8s cubic-bezier(0.16, 1, 0.3, 1);
  will-change: transform, opacity;
  position: relative;
  transition: transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.4s ease;
}
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .card-flag.reveal):hover,
.block-card:hover {
  transform: translateY(-8px) scale(1.005);
  box-shadow: 0 40px 80px rgba(0, 0, 0, 0.6), 0 0 20px rgba(56, 189, 248, 0.1);
  border-color: rgba(56, 189, 248, 0.3);
}

.stagger-1 { animation: staggerFadeIn 1s cubic-bezier(0.16, 1, 0.3, 1) forwards; }
.stagger-2 { animation: staggerFadeIn 1s cubic-bezier(0.16, 1, 0.3, 1) forwards 0.15s; }
.stagger-3 { animation: staggerFadeIn 1s cubic-bezier(0.16, 1, 0.3, 1) forwards 0.3s; }
.stagger-4 { animation: staggerFadeIn 1s cubic-bezier(0.16, 1, 0.3, 1) forwards 0.45s; }

.status-pulse { animation: statusPulse 2s infinite; border-radius: 50%; }

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
}
div.stButton > button:hover {
    background: linear-gradient(135deg, rgba(56, 189, 248, 0.2) 0%, rgba(139, 92, 246, 0.2) 100%) !important;
    border-color: #38bdf8 !important;
    box-shadow: 0 15px 40px rgba(56, 189, 248, 0.25) !important;
    transform: translateY(-4px) scale(1.01) !important;
    color: #38bdf8 !important;
}
div.stButton > button:active { transform: translateY(-1px) scale(0.99) !important; }

/* Banner Shine Animation */
@keyframes bannerShimmer {
  0%   { transform: translateX(-150%) skewX(-25deg); }
  25%  { transform: translateX(150%) skewX(-25deg); }
  100% { transform: translateX(150%) skewX(-25deg); }
}
.banner-shimmer {
  position: absolute; top: 0; left: 0; width: 60%; height: 100%;
  background: linear-gradient(to right,
    rgba(255,255,255,0) 0%, rgba(255,255,255,0.2) 50%, rgba(255,255,255,0) 100%);
  animation: bannerShimmer 7s infinite ease-in-out;
  pointer-events: none; z-index: 5;
}
.banner-glow-text {
  text-shadow: 0 0 20px rgba(56, 189, 248, 0.5), 0 0 40px rgba(56, 189, 248, 0.2);
}

@keyframes bannerBreath {
  0%   { opacity: 0.1; transform: scale(1.0); }
  50%  { opacity: 0.4; transform: scale(1.02); }
  100% { opacity: 0.1; transform: scale(1.0); }
}
.banner-breath-overlay {
  position: absolute; inset: 0;
  background: radial-gradient(circle at center, rgba(56, 189, 248, 0.4) 0%, transparent 70%);
  animation: bannerBreath 4s infinite ease-in-out;
  pointer-events: none; z-index: 4;
}

@keyframes beam-rotate {
  from { transform: rotate(0deg); }
  to   { transform: rotate(360deg); }
}
.border-beam {
  position: absolute; inset: 0px; border-radius: 18px; padding: 2px;
  background: conic-gradient(from 0deg, transparent 60%, #38bdf8 80%, #fbbf24 90%, transparent 100%);
  -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
  mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
  -webkit-mask-composite: xor;
  mask-composite: exclude;
  animation: beam-rotate 4s linear infinite;
  pointer-events: none; z-index: 10;
}
</style>
"""
