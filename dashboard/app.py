"""
Flood Inundation Mapping — Streamlit Dashboard
Run from Colab:
    CELL 1:
        from google.colab import drive; drive.mount('/content/drive')
        !pip install -q streamlit pyngrok rasterio plotly pandas streamlit-option-menu
    CELL 2:
        import threading, subprocess, time
        from pyngrok import ngrok
        threading.Thread(target=lambda: subprocess.run(
            ["streamlit","run","app.py","--server.port","8501","--server.headless","true"]),
            daemon=True).start()
        time.sleep(4); print("URL:", ngrok.connect(8501))
"""
import base64, json, os, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu

warnings.filterwarnings("ignore")

try:
    import rasterio
    from rasterio.enums import Resampling
except Exception:
    rasterio = None; Resampling = None

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Flood Inundation Mapping",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

#root { max-width: 1280px; margin: 0 auto; padding: 2rem; text-align: center; }
.logo { height: 6em; padding: 1.5em; will-change: filter; transition: filter 300ms; }
.logo:hover { filter: drop-shadow(0 0 2em #646cffaa); }
.logo.react:hover { filter: drop-shadow(0 0 2em #61dafbaa); }
@keyframes logo-spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
@media (prefers-reduced-motion: no-preference) { a:nth-of-type(2) .logo { animation: logo-spin infinite 20s linear; } }
.card { padding: 2em; }
.read-the-docs { color: #888; }

:root {
    --background: 210 45% 7%; --foreground: 195 30% 92%;
    --card: 210 40% 10%; --card-foreground: 195 30% 92%;
    --primary: 187 55% 40%; --primary-foreground: 210 50% 8%;
    --secondary: 205 40% 22%; --secondary-foreground: 195 30% 90%;
    --muted-bg: 210 35% 15%; --muted-foreground: 200 15% 55%;
    --accent: 174 45% 55%; --accent-foreground: 210 50% 8%;
    --destructive: 0 65% 50%; --destructive-foreground: 210 40% 98%;
    --border: 210 30% 18%; --input: 210 30% 18%; --ring: 187 55% 40%;
    --radius: 0.75rem;
    --sidebar-background: 210 45% 5%; --sidebar-foreground: 195 30% 88%;
    --sidebar-primary: 187 55% 40%; --sidebar-primary-foreground: 210 50% 8%;
    --sidebar-accent: 210 35% 12%; --sidebar-accent-foreground: 195 30% 90%;
    --sidebar-border: 210 30% 14%; --sidebar-ring: 187 55% 40%;
    --ocean-light: 174 45% 55%; --ocean-deep: 210 50% 8%; --ocean-mid: 205 40% 30%;
    --metric-positive: 160 60% 45%; --metric-warning: 40 80% 55%; --metric-negative: 0 65% 50%;
    --gradient-ocean: linear-gradient(135deg, hsl(210 50% 8%), hsl(205 40% 18%));
    --gradient-accent: linear-gradient(135deg, hsl(187 55% 40%), hsl(174 45% 55%));
    --bg: hsl(var(--ocean-deep)); --panel: hsl(var(--card)); --line: hsl(var(--border));
    --line2: hsl(210 30% 15%); --text: hsl(var(--foreground)); --muted: hsl(var(--muted-foreground));
    --blue: hsl(var(--primary)); --blue2: hsl(195 60% 48%); --blue-soft: hsl(205 45% 18%);
    --green: hsl(var(--accent)); --green2: hsl(176 50% 48%); --green-soft: hsl(174 30% 17%);
    --amber:#F7C66A; --amber-soft:hsl(40 40% 17%); --red:#FF8A8A; --red-soft:hsl(0 35% 18%);
    --orange:#FFB06B; --orange-soft:hsl(25 40% 18%);
}
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
.stApp { background: radial-gradient(circle at top left, hsl(187 50% 17%), hsl(210 45% 7%) 45%, hsl(210 50% 5%) 100%); color: var(--text); }
section[data-testid="stSidebar"] { background: hsl(var(--sidebar-background)) !important; border-right: 1px solid hsl(var(--sidebar-border)); }
section[data-testid="stSidebar"] * { color: hsl(var(--sidebar-foreground)) !important; }
header[data-testid="stHeader"] { display: none; }
.block-container { padding-top: 1.4rem !important; padding-bottom: 2rem; max-width: 1280px; }
div[data-testid="stVerticalBlock"] > div { gap: 0; }
h1, h2, h3, h4, h5, h6, .pt, .hero-t { font-family: 'Space Grotesk', system-ui, sans-serif; }
.pt { font-size: 26px; font-weight: 700; color: hsl(var(--ocean-light)); letter-spacing: -.02em; line-height: 1.1; margin-bottom: 3px; }
.ps { font-size: 12px; color: var(--muted); font-family: 'Inter', monospace; margin-bottom: 1.2rem; }
.sh { font-size: 10px; font-weight: 600; letter-spacing: .13em; text-transform: uppercase; color: hsl(195 25% 72%); font-family: 'Inter', monospace; padding-bottom: 8px; border-bottom: 1.5px solid var(--line); margin-bottom: .95rem; }
.card, .glass-card { background: hsl(var(--card) / .82); backdrop-filter: blur(12px); border: 1px solid hsl(var(--border) / .7); border-radius: 14px; padding: 1.1rem 1.25rem; box-shadow: 0 12px 30px rgba(0,0,0,.20); color: var(--text); }
.card-accent-blue { border-left: 3px solid hsl(var(--primary)); border-radius: 0 14px 14px 0; }
.card-accent-green { border-left: 3px solid hsl(var(--accent)); border-radius: 0 14px 14px 0; }
.card-best { border: 1.5px solid hsl(var(--accent)); }
.kc { background: hsl(var(--card) / .86); border: 1px solid hsl(var(--border) / .75); border-radius: 12px; padding: 1rem 1.15rem .9rem; box-shadow: 0 10px 24px rgba(0,0,0,.16); }
.kl { font-size: 9.5px; letter-spacing: .13em; text-transform: uppercase; color: var(--muted); font-family: 'Inter', monospace; margin-bottom: 5px; }
.kv { font-size: 24px; font-weight: 700; color: var(--text); font-family: 'Inter', monospace; line-height: 1; }
.ks { font-size: 10.5px; color: var(--muted); margin-top: 5px; }
.kd-up { font-size: 10.5px; color: hsl(var(--metric-positive)); margin-top: 4px; font-weight: 600; }
.kd-down { font-size: 10.5px; color: hsl(var(--metric-negative)); margin-top: 4px; font-weight: 600; }
.kd-neu { font-size: 10.5px; color: var(--muted); margin-top: 4px; }
.ir { display:flex; justify-content:space-between; gap:8px; padding:7px 0; border-bottom:1px solid var(--line2); font-size:12.5px; }
.ir:last-child { border-bottom:none; }
.ik { color: var(--muted); }
.iv { color: var(--text); font-weight:600; font-family:'Inter',monospace; text-align:right; font-size:12px; }
.badge { display:inline-block; padding:3px 10px; border-radius:999px; font-size:9.5px; font-weight:700; letter-spacing:.05em; font-family:'Inter',monospace; }
.badge-blue { background:hsl(var(--primary) / .15); color:hsl(var(--ocean-light)); border:1px solid hsl(var(--primary) / .25); }
.badge-green { background:hsl(var(--accent) / .15); color:hsl(var(--accent)); border:1px solid hsl(var(--accent) / .25); }
.badge-amber { background:var(--amber-soft); color:var(--amber); }
.badge-red { background:var(--red-soft); color:var(--red); }
.badge-orange { background:var(--orange-soft); color:var(--orange); }
.hero { background: var(--gradient-ocean); border: 1px solid hsl(var(--primary) / .25); border-radius: 18px; padding: 1.3rem 1.5rem 1.1rem; box-shadow: 0 18px 44px rgba(0,0,0,.24); margin-bottom: 1.1rem; }
.hero-t { font-size: 26px; font-weight: 800; color: hsl(var(--ocean-light)); letter-spacing:-.03em; margin-bottom:5px; }
.hero-s { font-size: 12.5px; color: hsl(195 25% 80%); max-width: 860px; line-height:1.65; }
.chip { display:inline-block; padding:3px 9px; border-radius:6px; background:hsl(var(--secondary) / .7); border:1px solid hsl(var(--border)); color:hsl(195 30% 88%); font-size:10px; margin:3px 4px 0 0; font-family:'Inter',monospace; }
.imgcap { font-size:10.5px; color:var(--muted); font-family:'Inter',monospace; margin-top:5px; text-align:center; }
.leg { border-left:4px solid; padding:9px 12px; background:hsl(var(--card)); border-radius:0 10px 10px 0; border:1px solid var(--line); }
.leg-t { font-size:12px; font-weight:700; color:var(--text); margin-bottom:2px; }
.leg-s { font-size:10.5px; color:var(--muted); }
.pbw { background:hsl(var(--muted-bg)); border-radius:4px; height:5px; overflow:hidden; margin-top:5px; }
.pbf { height:5px; border-radius:4px; }
.div { height:1px; background:var(--line); margin:.7rem 0; }
.footer { font-size:10.5px; color:var(--muted); text-align:center; margin-top:1.5rem; padding-top:.8rem; border-top:1px solid var(--line); }
.stButton > button { background: var(--gradient-accent); color: hsl(var(--primary-foreground)); border: 0; border-radius: 10px; font-weight: 700; box-shadow: 0 0 22px -7px hsl(var(--primary)); }
.stButton > button:hover { filter: brightness(1.08); border: 0; }
.stRadio label, .stSlider label, .stSelectbox label, .stFileUploader label { color: var(--text) !important; }
[data-testid="stDataFrame"] { border-radius:12px; overflow:hidden; border:1px solid var(--line); }
div[data-testid="stMetric"] { background:hsl(var(--card)); border:1px solid var(--line); padding:.65rem .9rem; border-radius:12px; }

/* Visibility fixes for dark ocean theme */
.stMarkdown, .stMarkdown p, .stMarkdown span, .stText, label, div[role="radiogroup"] label,
div[role="radiogroup"] label p, div[role="radiogroup"] label span,
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p {
    color: hsl(195 30% 90%) !important;
}
.stRadio div[role="radiogroup"] label [data-testid="stMarkdownContainer"] p {
    color: hsl(195 30% 90%) !important;
    font-weight: 600 !important;
}
.js-plotly-plot .plotly .legendtext,
.js-plotly-plot .plotly .gtitle,
.js-plotly-plot .plotly .xtitle,
.js-plotly-plot .plotly .ytitle,
.js-plotly-plot .plotly .xtick text,
.js-plotly-plot .plotly .ytick text,
.js-plotly-plot .plotly .annotation-text,
.js-plotly-plot .plotly .cbtitle,
.js-plotly-plot .plotly .colorbar text {
    fill: hsl(195 30% 90%) !important;
    color: hsl(195 30% 90%) !important;
}

/* Larger readable Streamlit typography */
html, body, [class*="css"], .stMarkdown, .stText, .stSelectbox, .stFileUploader, .stButton, .stDataFrame { font-size: 16px !important; }
.pt { font-size: 32px !important; }
.ps { font-size: 16px !important; }
.sh { font-size: 13px !important; }
.kv { font-size: 30px !important; }
.kl { font-size: 12px !important; }
.ks, .ir, .ik, .iv, .imgcap, .leg-s { font-size: 13px !important; }
section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span { font-size: 14px !important; }
.stTabs [data-baseweb="tab-list"] { gap: 10px; border-bottom: 1px solid hsl(var(--border)); }
.stTabs [data-baseweb="tab"] { background: hsl(var(--card) / .75); border: 1px solid hsl(var(--border)); border-radius: 12px 12px 0 0; padding: 12px 18px; color: hsl(195 30% 85%) !important; font-size: 15px !important; font-weight: 700; }
.stTabs [aria-selected="true"] { background: hsl(var(--primary) / .18); color: hsl(187 70% 72%) !important; border-color: hsl(var(--primary) / .55); }
[data-baseweb="select"] * { font-size: 15px !important; }
.leaflet-frame { border-radius: 16px; overflow: hidden; border: 1px solid hsl(var(--border)); box-shadow: 0 18px 38px rgba(0,0,0,.25); }

</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────────────
DRIVE_BASE    = "/content/drive/MyDrive/FloodProject"
ASSAM_BASE    = "/content/drive/MyDrive/FloodProject_Assam2023"
PREP_BASE     = f"{DRIVE_BASE}/data/preprocessed"
MODEL_BASE    = f"{DRIVE_BASE}/models"
KAGGLE_WD     = "/kaggle/working"
STATIC        = Path(__file__).parent / "static"

V1_LOG_PATH   = f"{MODEL_BASE}/unet_spatial/training_log.json"
V1_MET_PATH   = f"{MODEL_BASE}/unet_spatial/final_metrics.json"
V1_VIZ_PATH   = f"{MODEL_BASE}/unet_spatial/prediction_viz.png"
V1_CURVE_PATH = f"{MODEL_BASE}/unet_spatial/training_curves.png"
V2_LOG_PATH   = f"{MODEL_BASE}/unet_temporal_mlp/training_log.json"
V2_MET_PATH   = f"{MODEL_BASE}/unet_temporal_mlp/final_metrics.json"
NORM_PATH     = f"{PREP_BASE}/norm_stats.json"

ASSAM_PRED_V1 = f"{ASSAM_BASE}/assam_pred_v1.tif"
ASSAM_PRED_V5 = f"{ASSAM_BASE}/assam_pred_v5.tif"
ASSAM_METRICS = f"{ASSAM_BASE}/assam_metrics.json"

# Map files: Drive + Kaggle working dir
ASSAM_MAPS = {
    "V1 vs V5 comparison":  [f"{ASSAM_BASE}/assam_flood_map_v1v5.png",   f"{KAGGLE_WD}/assam_flood_map_v1v5.png"],
    "Paper figure":         [f"{ASSAM_BASE}/assam_flood_map_paper.png",  f"{KAGGLE_WD}/assam_flood_map_paper.png"],
    "Prediction map":       [f"{ASSAM_BASE}/assam_prediction_map.png",   f"{KAGGLE_WD}/assam_prediction_map.png"],
    "Flood map":            [f"{ASSAM_BASE}/assam_flood_map.png",        f"{KAGGLE_WD}/assam_flood_map.png"],
    "Generalization map":   [f"{ASSAM_BASE}/assam_generalization_map.png",f"{KAGGLE_WD}/assam_generalization_map.png"],
}

DRIVE_ONLINE = os.path.isdir(DRIVE_BASE)

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def _b64(path):
    if path and os.path.exists(str(path)):
        with open(str(path),"rb") as f: return base64.b64encode(f.read()).decode()
    return None

def _img(b64, cap="", radius="12px"):
    html = f'<img src="data:image/png;base64,{b64}" style="width:100%;border-radius:{radius};border:1px solid #E5E3DE;"/>'
    if cap: html += f'<div class="imgcap">{cap}</div>'
    return html

def _find_image(paths):
    """Try a list of paths and return first b64 found."""
    for p in paths:
        b = _b64(p)
        if b: return b
    return None

@st.cache_data(show_spinner=False)
def load_json(path, fallback=None):
    if os.path.exists(str(path)):
        with open(path) as f: return json.load(f)
    return fallback

@st.cache_data(show_spinner=False)
def load_training_log(path, fallback):
    data = load_json(path, None)
    if data is None: return fallback
    if isinstance(data, list) and len(data) > 0:
        first = data[0]
        if isinstance(first.get("train"), dict):
            return [{"epoch":e.get("epoch",i),"train_loss":e["train"].get("loss",0),
                     "train_iou":e["train"].get("iou",0),"val_loss":e["val"].get("loss",0),
                     "val_iou":e["val"].get("iou",0),"val_f1":e["val"].get("f1",0)}
                    for i,e in enumerate(data)]
        if "val_iou" in first: return data
    if isinstance(data, dict) and "epoch" in data:
        n = len(data["epoch"])
        return [{k: data[k][i] for k in data} for i in range(n)]
    return fallback

@st.cache_data(show_spinner=False)
def load_norm_stats(path):
    data = load_json(path)
    if data is None: return NORM_FALLBACK
    names = ["VV","VH","B2","B3","B4","B8","B11","B12","Elevation","Slope","TWI","HAND","NDWI","NDVI"]
    if "spatial" in data and "mean" in data["spatial"]:
        out = {n:{"mean":float(data["spatial"]["mean"][i]),"std":float(data["spatial"]["std"][i])} for i,n in enumerate(names)}
        if "temporal" in data:
            out["CHIRPS"] = {"mean":float(data["temporal"]["mean"][0]),"std":float(data["temporal"]["std"][0])}
            out["ERA5"]   = {"mean":float(data["temporal"]["mean"][1]),"std":float(data["temporal"]["std"][1])}
        return out
    return data

@st.cache_data(show_spinner=False)
def load_chip_preview(path, ch_idx):
    try: return np.load(path, mmap_mode="r")[ch_idx].astype(np.float32)[::4,::4]
    except: return None

@st.cache_data(show_spinner=False)
def load_label_preview(path):
    try: return np.load(path, mmap_mode="r").astype(np.float32)[::4,::4]
    except: return None

@st.cache_data(show_spinner=False)
def load_pred_thumbnail(path, size=320):
    if rasterio is None or not os.path.exists(str(path)): return None, None
    try:
        with rasterio.open(path) as src:
            d = src.read(1,out_shape=(size,size),resampling=Resampling.nearest).astype(np.float32)
            return d, float((d>0).mean())
    except: return None, None

@st.cache_data(show_spinner=False)
def load_chip_list(split="val"):
    path = f"{PREP_BASE}/{split}"
    if not os.path.isdir(path): return []
    return sorted({f.replace("_spatial.npy","") for f in os.listdir(path) if f.endswith("_spatial.npy")})

# ── Plotly helpers ────────────────────────────────────────────────────────────
PL = dict(font_family="DM Sans",font_color="#DCEFF6",
          plot_bgcolor="rgba(11, 29, 43, 0.92)",paper_bgcolor="rgba(0,0,0,0)",
          margin=dict(l=0,r=0,t=32,b=0),
          legend=dict(bgcolor="rgba(8, 22, 34, 0.72)",bordercolor="rgba(99, 211, 222, 0.22)",borderwidth=1,font=dict(color="#DCEFF6", size=11)))

def _heatmap(arr, title="", cscale="Blues", zmin=None, zmax=None, cbar_title=None, h=300):
    fig = go.Figure(go.Heatmap(z=arr, colorscale=cscale, zmin=zmin, zmax=zmax,
        showscale=True, colorbar=dict(title=cbar_title,thickness=10,len=0.75)))
    fig.update_layout(**PL, height=h, title=dict(text=title,x=0,font_size=11),
        xaxis=dict(showticklabels=False,showgrid=False),
        yaxis=dict(showticklabels=False,showgrid=False,scaleanchor="x"))
    return fig

def _agreement(pred, gt, threshold=0.5):
    p = (pred > threshold).astype(np.int16)
    g = (gt   > 0.5      ).astype(np.int16)
    out = np.zeros_like(p, dtype=np.int16)
    out[(p==1)&(g==1)] = 1   # TP
    out[(p==1)&(g==0)] = 2   # FP
    out[(p==0)&(g==1)] = 3   # FN
    return out

# ── HTML primitives ───────────────────────────────────────────────────────────
def kpi(label, value, sub="", delta=None, delta_dir="neu"):
    dh = ""
    if delta is not None:
        cls = {"up":"kd-up","down":"kd-down","neu":"kd-neu"}[delta_dir]
        arrow = "▲" if delta_dir=="up" else ("▼" if delta_dir=="down" else "—")
        dh = f'<div class="{cls}">{arrow} {abs(delta):.4f} vs benchmark</div>'
    return f'<div class="kc"><div class="kl">{label}</div><div class="kv">{value}</div><div class="ks">{sub}</div>{dh}</div>'

def pbar(val, color="#8EDCFF"):
    p = int(val*100)
    return f'<div class="pbw"><div class="pbf" style="width:{p}%;background:{color}"></div></div>'

def badge(text, cls="badge-blue"):
    return f'<span class="badge {cls}">{text}</span>'

def sec(title):
    return f'<div class="sh">{title}</div>'

# ──────────────────────────────────────────────────────────────────────────────
# FALLBACK DATA
# ──────────────────────────────────────────────────────────────────────────────
V2_LOG_FB = [{"epoch":e,"train_loss":tl,"val_loss":vl,"train_iou":ti,"val_iou":vi,"val_f1":vf}
 for e,tl,vl,ti,vi,vf in [
  (0,.4875,.3725,.5141,.6596,.7684),(1,.3747,.3271,.5790,.6042,.7164),(2,.3174,.2781,.6123,.6866,.7940),
  (3,.2874,.2416,.6231,.7060,.8072),(4,.2712,.2527,.6331,.6938,.7985),(5,.2541,.2585,.6457,.6783,.7795),
  (6,.2462,.2774,.6506,.6585,.7681),(7,.2451,.2487,.6447,.6981,.8012),(8,.2313,.2495,.6659,.6870,.7933),
  (9,.2318,.2287,.6564,.7243,.8225),(10,.2283,.2488,.6603,.7090,.8088),(11,.2240,.2436,.6666,.7086,.8081),
  (12,.2249,.2497,.6614,.6868,.7922),(13,.2225,.2402,.6689,.7110,.8109),(14,.2175,.2328,.6746,.7198,.8166),
  (15,.2183,.2346,.6691,.7148,.8100),(16,.2057,.2294,.6905,.7243,.8196),(17,.2044,.2387,.6884,.7125,.8116),
  (18,.2014,.2322,.6920,.7271,.8227),(19,.2024,.2376,.7006,.7136,.8117),(20,.2020,.2299,.6895,.7333,.8277),
  (21,.2004,.2349,.6934,.7205,.8182),(22,.1943,.2341,.7014,.7213,.8179),(23,.1940,.2494,.6992,.6885,.7939),
  (24,.1928,.2272,.7024,.7321,.8270),(25,.1899,.2423,.7064,.7104,.8098),(26,.1891,.2299,.7084,.7292,.8236),
  (27,.1893,.2261,.7069,.7367,.8290),(28,.1895,.2377,.7103,.7230,.8190),(29,.1874,.2376,.7088,.7223,.8214),
  (30,.1872,.2477,.7054,.7072,.8073),(31,.1865,.2322,.7132,.7262,.8212),(32,.1857,.2477,.7145,.7106,.8092),
  (33,.1861,.2399,.7144,.7125,.8126),(34,.1827,.2429,.7126,.7131,.8113),(35,.1818,.2384,.7204,.7178,.8158),
  (36,.1826,.2439,.7159,.7105,.8091),(37,.1800,.2529,.7173,.7004,.8023)]]

V1_LOG_FB = [{"epoch":e,"train_loss":tl,"val_loss":vl,"train_iou":ti,"val_iou":vi,"val_f1":vf}
 for e,tl,vl,ti,vi,vf in [
  (0,.51,.39,.490,.610,.740),(2,.35,.29,.570,.670,.780),(5,.27,.26,.630,.700,.810),
  (10,.23,.24,.670,.720,.820),(15,.21,.235,.690,.733,.828),(20,.205,.232,.700,.734,.829),
  (25,.198,.231,.705,.735,.828),(29,.193,.242,.706,.714,.809),(30,.193,.247,.695,.701,.802),
  (31,.191,.239,.705,.708,.807),(32,.190,.233,.707,.729,.823),(33,.188,.245,.710,.698,.801)]]

NORM_FALLBACK = {
    "VV":{"mean":-10.061,"std":4.700},"VH":{"mean":-17.009,"std":5.454},
    "B2":{"mean":1338.5,"std":424.6},"B3":{"mean":1249.2,"std":434.4},
    "B4":{"mean":1038.0,"std":551.8},"B8":{"mean":2331.0,"std":892.6},
    "B11":{"mean":74.4,"std":129.7},"B12":{"mean":1615.4,"std":763.1},
    "Elevation":{"mean":118.8,"std":102.2},"Slope":{"mean":0.155,"std":0.184},
    "TWI":{"mean":9.893,"std":3.134},"HAND":{"mean":17.497,"std":43.659},
    "NDWI":{"mean":-0.266,"std":0.233},"NDVI":{"mean":0.362,"std":0.272},
    "CHIRPS":{"mean":9.649,"std":13.007},"ERA5":{"mean":0.390,"std":0.141},
}

COUNTRY_DATES = {
    "India":    {"date":"2016-08-12","chips":312},
    "Pakistan": {"date":"2017-06-28","chips":289},
    "Sri-Lanka":{"date":"2017-05-30","chips":201},
    "Cambodia": {"date":"2018-08-05","chips":400},
    "Bolivia":  {"date":"2018-02-15","chips":445},
    "Colombia": {"date":"2018-08-22","chips":417},
}

CH_ORDER = ["VV","VH","B2","B3","B4","B8","B11","B12","Elevation","Slope","TWI","HAND","NDWI","NDVI"]
CH_IDX   = {c:i for i,c in enumerate(CH_ORDER)}
CH_GROUPS = {
    "SAR (Sentinel-1)":   {"channels":["VV","VH"],                          "color":"#8EDCFF"},
    "Optical (S2)":       {"channels":["B2","B3","B4","B8","B11","B12"],    "color":"#2B6B4E"},
    "Terrain (DEM)":      {"channels":["Elevation","Slope","TWI","HAND"],   "color":"#925F14"},
    "Spectral Indices":   {"channels":["NDWI","NDVI"],                      "color":"#9B3030"},
}
CH_DESC = {
    "VV":"Sentinel-1 VV backscatter (dB). Flooded open water appears specularly dark — useful primary flood signal.",
    "VH":"Sentinel-1 VH backscatter (dB). More sensitive to vegetation canopy above floodwater.",
    "B2":"S2 Blue (490 nm). Helps separate water from bright surfaces; contributes to NDWI.",
    "B3":"S2 Green (560 nm). Water reflectance peak; key numerator in NDWI.",
    "B4":"S2 Red (665 nm). Chlorophyll absorption band; denominator in NDVI.",
    "B8":"S2 NIR (842 nm). Vegetation very bright, water dark; dominates NDWI and NDVI.",
    "B11":"S2 SWIR1 (1610 nm). Moisture-sensitive; separates wet/dry soil and penetrates thin cloud.",
    "B12":"S2 SWIR2 (2190 nm). Soil moisture, burn scar sensitivity.",
    "Elevation":"Terrain height (m) from Copernicus GLO-30 DEM at ~10 m resolution.",
    "Slope":"Slope angle (°). Flat slopes slow runoff and accumulate water.",
    "TWI":"Topographic Wetness Index = ln(upslope area / tan(slope)). High values predict water pooling.",
    "HAND":"Height Above Nearest Drainage (m). Low HAND = floodplain proximity = high flood susceptibility.",
    "NDWI":"(B3−B8)/(B3+B8). Positive values indicate open water surfaces.",
    "NDVI":"(B8−B4)/(B8+B4). Positive values indicate healthy vegetation; negative may indicate inundation.",
}

PIPELINE = [
    ("Download Sen1Floods11",True),("Export DEM (GEE)",True),
    ("Export CHIRPS + ERA5",True),("Stack & normalize",True),
    ("Upload to Kaggle",False),("Train V1 — U-Net",True),
    ("Train V5 — Temporal MLP",True),("Ablation comparison",True),
    ("Assam 2023 inference",True),("Generalization eval",False),
]

ASSAM_RESULTS_FALLBACK = {
    "v1":{"iou":0.3049,"precision":0.4637,"recall":0.4709,"f1":0.4673},
    "v5":{"iou":0.2934,"precision":0.4700,"recall":0.4384,"f1":0.4537},
}

FP_PROPS = {
    "S1 VH (dB)": {"mean":-12.93,"median":-10.73,"std":6.29,"threshold":"< −16 dB"},
    "S2 MNDWI":   {"mean":0.015, "median":0.000, "std":0.062,"threshold":"> 0.0"},
    "HAND (m)":   {"mean":5.9,   "median":2.2,   "std":14.0, "threshold":"< 5 m"},
}

# ──────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────────────────────────
v1_log = load_training_log(V1_LOG_PATH, V1_LOG_FB)
v2_log = load_training_log(V2_LOG_PATH, V2_LOG_FB)
v1_met = load_json(V1_MET_PATH,{"iou":0.7360,"f1":0.8285,"precision":0.8892,"recall":0.7905,"specificity":0.9924,"loss":0.2253})
v2_met = load_json(V2_MET_PATH,{"iou":0.7367,"f1":0.8290,"precision":0.8963,"recall":0.7927,"loss":0.2261})
norm   = load_norm_stats(NORM_PATH)
a_met  = load_json(ASSAM_METRICS) or ASSAM_RESULTS_FALLBACK

def _m(d,k,fb): return d.get(k,fb) if isinstance(d,dict) else fb

v1_iou=_m(v1_met,"iou",0.7360); v1_f1=_m(v1_met,"f1",0.8285)
v1_pr=_m(v1_met,"precision",0.8892); v1_re=_m(v1_met,"recall",0.7905)
v1_sp=_m(v1_met,"specificity",0.9924); v1_lo=_m(v1_met,"loss",0.2253)
v2_iou=_m(v2_met,"iou",0.7367); v2_f1=_m(v2_met,"f1",0.8290)
v2_pr=_m(v2_met,"precision",0.8963); v2_re=_m(v2_met,"recall",0.7927); v2_lo=_m(v2_met,"loss",0.2261)

df1 = pd.DataFrame(v1_log); df2 = pd.DataFrame(v2_log)
best_ep_v1 = max(v1_log,key=lambda x:x.get("val_iou",0))["epoch"]
best_ep_v2 = max(v2_log,key=lambda x:x.get("val_iou",0))["epoch"]

a1_iou=_m(a_met.get("v1",{}),"iou",0.3049); a1_f1=_m(a_met.get("v1",{}),"f1",0.4673)
a1_pr=_m(a_met.get("v1",{}),"precision",0.4637); a1_re=_m(a_met.get("v1",{}),"recall",0.4709)
a5_iou=_m(a_met.get("v5",{}),"iou",0.2934); a5_f1=_m(a_met.get("v5",{}),"f1",0.4537)
a5_pr=_m(a_met.get("v5",{}),"precision",0.4700); a5_re=_m(a_met.get("v5",{}),"recall",0.4384)

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:.5rem 0 .8rem">
      <div style="font-size:15px;font-weight:800;color:#ECF0F7;letter-spacing:-.01em">Flood Inundation</div>
      <div style="font-size:10px;color:#606878;margin-top:2px;font-family:'DM Mono',monospace">Sen1Floods11 · Assam 2023</div>
    </div>""", unsafe_allow_html=True)

    selected = option_menu(
        menu_title="Navigation",
        options=["Overview","Dashboard","Models","Inference Studio","Input Explorer","Temporal Analysis","Assam Generalization","Error Analysis"],
        icons=["bar-chart-line","speedometer2","cpu","map","layers","cloud-rain-heavy","globe-central-south-asia","exclamation-triangle"],
        menu_icon="water",
        default_index=0,
        styles={
            "container":         {"background-color":"hsl(210 45% 5%)","padding":"0!important"},
            "icon":              {"color":"hsl(174 45% 55%)","font-size":"13px"},
            "menu-title":        {"font-size":"10px","color":"#606878","font-family":"DM Mono, monospace","letter-spacing":".1em","text-transform":"uppercase"},
            "nav-link":          {"font-size":"12.5px","text-align":"left","margin":"3px 0","padding":"8px 12px","color":"#C8D0DC","border-radius":"9px","font-family":"DM Sans, sans-serif"},
            "nav-link-selected": {"background-color":"hsl(187 55% 40%)","color":"hsl(210 50% 8%)","font-weight":"700"},
        },
    )

    st.markdown("---")
    c="#75E6DA" if DRIVE_ONLINE else "#FFB06B"
    st.markdown(f'<div style="font-size:12px;color:{c};font-family:DM Mono,monospace">● {"Drive mounted" if DRIVE_ONLINE else "Drive offline — cached data"}</div>', unsafe_allow_html=True)



def render_leaflet_site_map(height=430):
    """Interactive Leaflet map for validation/test coverage with toggle overlays."""
    html = """
    <div id="map" style="height:HEIGHTpx;width:100%;background:#06111f"></div>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
      const map = L.map('map', { zoomControl: true, scrollWheelZoom: true }).setView([20.5, 82.5], 4);
      const dark = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', { maxZoom: 19, attribution: '&copy; OpenStreetMap &copy; CARTO' }).addTo(map);
      const osm = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19, attribution: '&copy; OpenStreetMap' });
      const satellite = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', { maxZoom: 18, attribution: 'Tiles &copy; Esri' });
      const sites = [
        ['India 2016', 25.6, 85.1, 'Sen1Floods11 validation / train region'],
        ['Pakistan 2017', 30.4, 69.3, 'Sen1Floods11 flood event'],
        ['Sri Lanka 2017', 7.8, 80.7, 'Validation chip examples'],
        ['Cambodia 2018', 12.6, 104.9, 'Mekong basin flood event'],
        ['Bangladesh / Assam edge', 26.2, 91.7, 'Brahmaputra corridor'],
        ['Assam 2023 OOD', 26.2, 92.9, 'Out-of-distribution test region']
      ];
      const trainLayer = L.layerGroup();
      const assamLayer = L.layerGroup();
      sites.forEach(([name, lat, lon, note]) => {
        const marker = L.circleMarker([lat, lon], {
          radius: name.includes('Assam') ? 9 : 7,
          color: name.includes('Assam') ? '#75E6DA' : '#8EDCFF',
          weight: 2,
          fillColor: name.includes('Assam') ? '#00D1C7' : '#5AC8FA',
          fillOpacity: 0.75
        }).bindPopup(`<b>${name}</b><br>${note}`);
        (name.includes('Assam') ? assamLayer : trainLayer).addLayer(marker);
      });
      trainLayer.addTo(map); assamLayer.addTo(map);
      const floodOverlay = L.rectangle([[25.3, 90.7], [27.8, 95.4]], {
        color: '#75E6DA', weight: 2, fillColor: '#00D1C7', fillOpacity: 0.18
      }).bindPopup('<b>Assam 2023 inference window</b><br>Approximate Brahmaputra basin AOI used for dashboard visualization.');
      const geeLike = L.layerGroup([floodOverlay]).addTo(map);
      L.control.layers({ 'Dark base': dark, 'OpenStreetMap': osm, 'Satellite': satellite },
                       { 'Sen1Floods11 sites': trainLayer, 'Assam OOD site': assamLayer, 'GEE-style AOI overlay': geeLike },
                       { collapsed: false }).addTo(map);
    </script>
    """.replace('HEIGHT', str(height))
    components.html(html, height=height + 20)


def _first_existing(paths):
    for p in paths:
        if p and os.path.exists(str(p)):
            return str(p)
    return str(paths[0]) if paths else ""

# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if selected == "Overview":
    st.markdown("""
    <div class="hero">
      <div class="hero-t">Flood Inundation Mapping</div>
      <div class="hero-s">A structured research interface for comparing spatial and spatiotemporal flood segmentation models using Sentinel-1/2, DEM terrain features, CHIRPS rainfall, and ERA5 soil moisture. Covers benchmark validation on Sen1Floods11 and out-of-distribution generalization over Assam 2023.</div>
      <div style="margin-top:8px"><span class="chip">14 spatial channels</span><span class="chip">15-day antecedent</span><span class="chip">Sen1Floods11</span><span class="chip">Assam 2023 OOD test</span><span class="chip">U-Net + Temporal MLP</span></div>
    </div>""", unsafe_allow_html=True)

    c = st.columns(6)
    boxes=[("Training chips","2,064","6 flood events",""),("Val chips","168","Hand-labelled",""),
           ("Input channels","14","SAR+Optical+DEM+idx",""),
           ("Best val IoU",f"{v2_iou:.4f}",f"Temporal MLP · ep{best_ep_v2}","up"),
           ("Val F1",f"{v2_f1:.4f}","Best model",""),
           ("IoU vs baseline",f"+{(v2_iou-v1_iou)*100:.2f}%","V5 over V1","up")]
    for col,(l,v,s,d) in zip(c,boxes):
        with col:
            dh = f'<div class="kd-up">▲ V5 best</div>' if d=="up" else ""
            st.markdown(f'<div class="kc"><div class="kl">{l}</div><div class="kv">{v}</div><div class="ks">{s}</div>{dh}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    r1, r2 = st.columns([1.65,1])

    with r1:
        st.markdown(sec("Training progression — Val IoU"), unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df2["epoch"],y=df2["val_iou"],mode="lines",
            name="V5 Temporal MLP",line=dict(color="#2B6B4E",width=2.8),
            fill="tozeroy",fillcolor="rgba(43,107,78,.05)"))
        fig.add_trace(go.Scatter(x=df1["epoch"],y=df1["val_iou"],mode="lines+markers",
            name="V1 U-Net baseline",line=dict(color="#8EDCFF",width=2,dash="dot"),
            marker=dict(size=4,color="#8EDCFF")))
        fig.add_hline(y=v2_iou,line_dash="dash",line_color="#2B6B4E",line_width=1,
            annotation_text=f"best {v2_iou:.4f}",annotation_font_size=10,annotation_font_color="#2B6B4E")
        fig.update_layout(**PL,height=260,
            xaxis=dict(title="Epoch",gridcolor="rgba(220,239,246,0.18)"),
            yaxis=dict(title="Val IoU",gridcolor="rgba(220,239,246,0.18)",range=[.55,.77]))
        st.plotly_chart(fig,use_container_width=True)

    with r2:
        st.markdown(sec("Project summary"), unsafe_allow_html=True)
        st.markdown(f"""
        <div class="card">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
            <div style="font-size:13px;font-weight:700;color:#8EDCFF">Best configuration</div>
            {badge("Variant 5","badge-green")}
          </div>
          <div class="ir"><span class="ik">Best model</span><span class="iv">U-Net + Terrain MLP</span></div>
          <div class="ir"><span class="ik">Val IoU</span><span class="iv">{v2_iou:.4f}</span></div>
          <div class="ir"><span class="ik">Val F1</span><span class="iv">{v2_f1:.4f}</span></div>
          <div class="ir"><span class="ik">Precision</span><span class="iv">{v2_pr:.4f}</span></div>
          <div class="ir"><span class="ik">Recall</span><span class="iv">{v2_re:.4f}</span></div>
          <div class="ir"><span class="ik">Assam IoU (V5)</span><span class="iv">{a5_iou:.4f}</span></div>
          <div class="ir"><span class="ik">Loss function</span><span class="iv">0.5×Dice + 0.5×BCE</span></div>
        </div>""", unsafe_allow_html=True)

    r3, r4 = st.columns([1.1,1.1])
    with r3:
        st.markdown(sec("Metric comparison"), unsafe_allow_html=True)
        fig2=go.Figure()
        mns=["IoU","F1","Precision","Recall"]
        fig2.add_trace(go.Bar(name="V1 U-Net",x=mns,y=[v1_iou,v1_f1,v1_pr,v1_re],
            marker_color="#8EDCFF",marker_line_width=0))
        fig2.add_trace(go.Bar(name="V5 Temporal MLP",x=mns,y=[v2_iou,v2_f1,v2_pr,v2_re],
            marker_color="#2B6B4E",marker_line_width=0))
        fig2.update_layout(**PL,height=250,barmode="group",bargap=0.2,
            yaxis=dict(range=[0.70,.93],gridcolor="rgba(220,239,246,0.18)"))
        st.plotly_chart(fig2,use_container_width=True)

    with r4:
        st.markdown(sec("Dataset distribution"), unsafe_allow_html=True)
        fig3=go.Figure(go.Bar(
            x=list(COUNTRY_DATES.keys()),
            y=[COUNTRY_DATES[c]["chips"] for c in COUNTRY_DATES],
            marker_color=["#8EDCFF"]*3+["#F0D898"]+["#B7D9C5"]*2,
            marker_line_width=0,
            text=[COUNTRY_DATES[c]["chips"] for c in COUNTRY_DATES],
            textposition="outside",textfont=dict(size=10,color="#CBE7F1")))
        fig3.update_layout(**PL,height=250,
            yaxis=dict(gridcolor="rgba(220,239,246,0.18)",title="Chips"),
            xaxis=dict(gridcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig3,use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD  (new interactive split view)
# ══════════════════════════════════════════════════════════════════════════════
elif selected == "Dashboard":
    # ── Tab switch instead of radio ───────────────────────────────────────────
    tab_val, tab_assam = st.tabs(["🌊 Sen1Floods11 Validation", "🛰️ Assam 2023 Generalization"])

    # ═══════════════════════════════════════════
    # PANEL A — Sen1Floods11
    # ═══════════════════════════════════════════
    with tab_val:
        st.markdown('<div class="pt">Sen1Floods11 Validation</div>', unsafe_allow_html=True)
        st.markdown('<div class="ps">Benchmark results on the 168 hand-labelled validation chips · 6 flood events</div>', unsafe_allow_html=True)

        # KPIs
        k = st.columns(6)
        kpis=[
            ("V1 IoU",f"{v1_iou:.4f}",f"epoch {best_ep_v1}","",""),
            ("V1 F1",f"{v1_f1:.4f}","U-Net spatial","",""),
            ("V1 Precision",f"{v1_pr:.4f}","","",""),
            ("V5 IoU",f"{v2_iou:.4f}",f"epoch {best_ep_v2}","up",""),
            ("V5 F1",f"{v2_f1:.4f}","Temporal MLP","up",""),
            ("IoU lift",f"+{(v2_iou-v1_iou)*100:.2f}%","V5 over V1","up",""),
        ]
        for col,(l,v,s,d,_) in zip(k,kpis):
            with col:
                arr = f'<div class="kd-up">▲ Best model</div>' if d=="up" else ""
                st.markdown(f'<div class="kc"><div class="kl">{l}</div><div class="kv">{v}</div><div class="ks">{s}</div>{arr}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_l, col_r = st.columns([1.6,1])

        with col_l:
            st.markdown(sec("Val IoU — all epochs, both variants"), unsafe_allow_html=True)
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=df2["epoch"],y=df2["val_iou"],mode="lines",
                name="V5 Temporal MLP",line=dict(color="#2B6B4E",width=2.8),
                fill="tozeroy",fillcolor="rgba(43,107,78,.05)"))
            fig.add_trace(go.Scatter(x=df1["epoch"],y=df1["val_iou"],mode="lines+markers",
                name="V1 U-Net",line=dict(color="#8EDCFF",width=2,dash="dot"),
                marker=dict(size=4)))
            best_pt = max(v2_log,key=lambda x:x.get("val_iou",0))
            fig.add_scatter(x=[best_pt["epoch"]],y=[best_pt["val_iou"]],mode="markers",
                marker=dict(size=11,symbol="star",color="#2B6B4E"),name=f"Best ep{best_pt['epoch']}")
            fig.update_layout(**PL,height=270,
                xaxis=dict(title="Epoch",gridcolor="rgba(220,239,246,0.18)"),
                yaxis=dict(title="Val IoU",gridcolor="rgba(220,239,246,0.18)",range=[.55,.77]))
            st.plotly_chart(fig,use_container_width=True)

            # Loss subplot
            fig2=make_subplots(rows=1,cols=2,subplot_titles=["Combined Loss","Val IoU & F1"])
            for df,nm,clr in [(df1,"V1",["#B0BAC8","#8EDCFF"]),(df2,"V5",["#90BFA8","#2B6B4E"])]:
                fig2.add_trace(go.Scatter(x=df["epoch"],y=df["train_loss"],name=f"{nm} train",
                    line=dict(color=clr[0],width=1.5)),row=1,col=1)
                fig2.add_trace(go.Scatter(x=df["epoch"],y=df["val_loss"],name=f"{nm} val",
                    line=dict(color=clr[1],width=2.2)),row=1,col=1)
            fig2.add_trace(go.Scatter(x=df1["epoch"],y=df1["val_iou"],name="V1 IoU",
                line=dict(color="#8EDCFF",width=2,dash="dot"),showlegend=False),row=1,col=2)
            fig2.add_trace(go.Scatter(x=df2["epoch"],y=df2["val_iou"],name="V5 IoU",
                line=dict(color="#2B6B4E",width=2.8),showlegend=False),row=1,col=2)
            fig2.add_trace(go.Scatter(x=df2["epoch"],y=df2["val_f1"],name="V5 F1",
                line=dict(color="#4A9C70",width=1.5,dash="dot"),showlegend=False),row=1,col=2)
            fig2.update_xaxes(gridcolor="rgba(220,239,246,0.18)"); fig2.update_yaxes(gridcolor="rgba(220,239,246,0.18)")
            fig2.update_layout(**PL,height=230)
            st.plotly_chart(fig2,use_container_width=True)

        with col_r:
            st.markdown(sec("Side-by-side metrics"), unsafe_allow_html=True)
            mns=["IoU","F1","Precision","Recall","Specificity"]
            v1v=[v1_iou,v1_f1,v1_pr,v1_re,v1_sp]
            v2v=[v2_iou,v2_f1,v2_pr,v2_re,0.0]
            fig3=go.Figure()
            fig3.add_trace(go.Bar(name="V1 U-Net",x=mns,y=v1v,
                marker_color="#8EDCFF",marker_line_width=0,text=[f"{x:.3f}" for x in v1v],textposition="outside",textfont_size=9))
            fig3.add_trace(go.Bar(name="V5 Temporal MLP",x=mns,y=v2v,
                marker_color="#2B6B4E",marker_line_width=0,text=[f"{x:.3f}" if x>0 else "—" for x in v2v],textposition="outside",textfont_size=9))
            fig3.update_layout(**PL,height=240,barmode="group",bargap=0.18,
                yaxis=dict(range=[0,.99],gridcolor="rgba(220,239,246,0.18)"))
            st.plotly_chart(fig3,use_container_width=True)

            st.markdown(sec("Ablation table"), unsafe_allow_html=True)
            abl=pd.DataFrame([
                {"Variant":"1 — U-Net spatial",   "IoU":f"{v1_iou:.4f}","F1":f"{v1_f1:.4f}","Status":"✓"},
                {"Variant":"2 — ConvLSTM 128d",   "IoU":"0.7315","F1":"0.8254","Status":"✓"},
                {"Variant":"5 — Terrain MLP ★",   "IoU":f"{v2_iou:.4f}","F1":f"{v2_f1:.4f}","Status":"★"},
            ])
            st.dataframe(abl,hide_index=True,use_container_width=True,height=150)

        # Prediction maps
        st.markdown(sec("Validation prediction maps"), unsafe_allow_html=True)
        img_detail = _b64(V1_VIZ_PATH) or _b64(str(STATIC/"pred_viz_detail.png"))
        if img_detail:
            st.markdown(_img(img_detail,"U-Net spatial — SAR (VV) · Ground truth · Predicted probability · Agreement (green=TP, red=error)"), unsafe_allow_html=True)

        # Sample chip performance table
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(sec("Sample chip metrics — val split"), unsafe_allow_html=True)
        sample=pd.DataFrame([
            {"Chip":"Sri-Lanka_236628","Flood%":"1.2%","IoU":0.491,"F1":0.658,"TP":1538,"FP":53,"FN":1543},
            {"Chip":"India_324254",    "Flood%":"1.5%","IoU":0.760,"F1":0.865,"TP":3178,"FP":300,"FN":705},
            {"Chip":"Sri-Lanka_55568", "Flood%":"7.2%","IoU":0.928,"F1":0.963,"TP":14399,"FP":248,"FN":869},
            {"Chip":"Pakistan_366265", "Flood%":"0.0%","IoU":0.000,"F1":0.000,"TP":0,"FP":80,"FN":2},
            {"Chip":"India_135434",    "Flood%":"8.8%","IoU":0.543,"F1":0.704,"TP":6176,"FP":1052,"FN":4136},
            {"Chip":"India_118868",    "Flood%":"6.6%","IoU":0.317,"F1":0.481,"TP":3551,"FP":435,"FN":7231},
        ])
        st.dataframe(sample,hide_index=True,use_container_width=True,height=255)

    # ═══════════════════════════════════════════
    # PANEL B — Assam 2023
    # ═══════════════════════════════════════════
    with tab_assam:
        st.markdown('<div class="pt">Assam 2023 — Generalization Test</div>', unsafe_allow_html=True)
        st.markdown('<div class="ps">Out-of-distribution inference · Brahmaputra basin · Aug 29 2023 flood peak · ~78,000 km²</div>', unsafe_allow_html=True)

        # KPIs — 8 across
        k = st.columns(8)
        akpis=[
            ("V1 IoU",f"{a1_iou:.4f}","U-Net spatial","down"),
            ("V1 F1",f"{a1_f1:.4f}","",""),
            ("V1 Precision",f"{a1_pr:.4f}","",""),
            ("V1 Recall",f"{a1_re:.4f}","",""),
            ("V5 IoU",f"{a5_iou:.4f}","Temporal MLP","down"),
            ("V5 F1",f"{a5_f1:.4f}","",""),
            ("V5 Precision",f"{a5_pr:.4f}","",""),
            ("V5 Recall",f"{a5_re:.4f}","",""),
        ]
        for col,(l,v,s,d) in zip(k,akpis):
            with col:
                arr = f'<div class="kd-down">▼ OOD gap</div>' if d=="down" else ""
                st.markdown(f'<div class="kc"><div class="kl">{l}</div><div class="kv">{v}</div><div class="ks">{s}</div>{arr}</div>', unsafe_allow_html=True)

        # Benchmark vs Assam drop reminder
        drop1 = v1_iou - a1_iou; drop5 = v2_iou - a5_iou
        st.markdown(f"""
        <div class="card" style="margin:.8rem 0;background:#FBF5E5;border-color:#EAD48A">
          <div style="font-size:12px;font-weight:700;color:#7A5200;margin-bottom:4px">Generalization gap</div>
          <div style="font-size:12px;color:#7A5200">
            V1: Sen1Floods11 IoU <b>{v1_iou:.4f}</b> → Assam <b>{a1_iou:.4f}</b> &nbsp;(Δ = −{drop1:.4f})&nbsp;&nbsp;|&nbsp;&nbsp;
            V5: <b>{v2_iou:.4f}</b> → <b>{a5_iou:.4f}</b> &nbsp;(Δ = −{drop5:.4f})&nbsp;&nbsp;·&nbsp;&nbsp;
            This is expected for out-of-distribution inference — different geography, sensor config, and flood type.
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_l, col_m, col_r = st.columns([1, 1, 1.2])

        with col_l:
            st.markdown(sec("Benchmark vs Assam — IoU"), unsafe_allow_html=True)
            fig=go.Figure()
            cats=["V1 — Val","V1 — Assam","V5 — Val","V5 — Assam"]
            vals=[v1_iou,a1_iou,v2_iou,a5_iou]
            clrs=["#8EDCFF","#7BA8D4","#2B6B4E","#6BBFA0"]
            fig.add_trace(go.Bar(x=cats,y=vals,marker_color=clrs,marker_line_width=0,
                text=[f"{v:.4f}" for v in vals],textposition="outside",textfont_size=10))
            fig.update_layout(**PL,height=270,
                yaxis=dict(range=[0,.85],gridcolor="rgba(220,239,246,0.18)",title="IoU"),
                xaxis=dict(tickangle=-20))
            st.plotly_chart(fig,use_container_width=True)

        with col_m:
            st.markdown(sec("All metrics — Assam vs benchmark"), unsafe_allow_html=True)
            met=["IoU","F1","Precision","Recall"]
            fig2=go.Figure()
            fig2.add_trace(go.Bar(name="V1 Assam",x=met,y=[a1_iou,a1_f1,a1_pr,a1_re],
                marker_color="#8EDCFF",marker_line_width=0))
            fig2.add_trace(go.Bar(name="V5 Assam",x=met,y=[a5_iou,a5_f1,a5_pr,a5_re],
                marker_color="#6BBFA0",marker_line_width=0))
            fig2.add_trace(go.Scatter(name="V1 benchmark",x=met,y=[v1_iou,v1_f1,v1_pr,v1_re],
                mode="lines+markers",line=dict(color="#8EDCFF",dash="dot",width=2),marker=dict(size=7)))
            fig2.add_trace(go.Scatter(name="V5 benchmark",x=met,y=[v2_iou,v2_f1,v2_pr,v2_re],
                mode="lines+markers",line=dict(color="#2B6B4E",dash="dot",width=2),marker=dict(size=7)))
            fig2.update_layout(**PL,height=270,barmode="group",bargap=0.18,
                yaxis=dict(range=[0,1],gridcolor="rgba(220,239,246,0.18)",title="Score"))
            st.plotly_chart(fig2,use_container_width=True)

        with col_r:
            st.markdown(sec("FP pixel properties"), unsafe_allow_html=True)
            for feat,vals in FP_PROPS.items():
                st.markdown(f"""
                <div class="card" style="padding:.75rem 1rem;margin-bottom:6px">
                  <div style="font-size:11.5px;font-weight:700;color:#DCEFF6;margin-bottom:4px">{feat}</div>
                  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px">
                    <div><div class="kl">Mean</div><div style="font-size:13px;font-weight:600;font-family:'DM Mono',monospace">{vals['mean']}</div></div>
                    <div><div class="kl">Median</div><div style="font-size:13px;font-weight:600;font-family:'DM Mono',monospace">{vals['median']}</div></div>
                    <div><div class="kl">Std</div><div style="font-size:13px;font-weight:600;font-family:'DM Mono',monospace">{vals['std']}</div></div>
                  </div>
                  <div style="font-size:10px;color:#925F14;margin-top:5px;font-family:'DM Mono',monospace">Flood threshold: {vals['threshold']}</div>
                </div>""", unsafe_allow_html=True)

        # Map selector
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(sec("Assam flood maps — select from available outputs"), unsafe_allow_html=True)
        available_maps = {name: _find_image(paths) for name, paths in ASSAM_MAPS.items()}
        found_maps = {k:v for k,v in available_maps.items() if v is not None}

        if found_maps:
            sel_map = st.selectbox("Select map", list(found_maps.keys()))
            captions = {
                "V1 vs V5 comparison":  "Ground truth vs V1 (U-Net spatial) vs V5 (Temporal MLP) — same-area comparison",
                "Paper figure":         "Flood inundation mapping results for Assam 2023 · Ground truth from MNDWI+S1 fusion with JRC permanent water removal",
                "Prediction map":       "Full-AOI prediction raster for Assam 2023",
                "Flood map":            "Flood extent map — Assam 2023",
                "Generalization map":   "Generalization test overlay — prediction vs reference label",
            }
            st.markdown(_img(found_maps[sel_map], captions.get(sel_map,sel_map)), unsafe_allow_html=True)
        else:
            # GeoTIFF thumbnails fallback
            v1t, v1f = load_pred_thumbnail(ASSAM_PRED_V1)
            v5t, v5f = load_pred_thumbnail(ASSAM_PRED_V5)
            if v1t is not None and v5t is not None:
                tc1,tc2,tc3 = st.columns(3)
                with tc1: st.plotly_chart(_heatmap(v1t,f"V1 prediction · {v1f:.1%} flood fraction",[[0,"#F4F3F0"],[1,"#8EDCFF"]],0,1),use_container_width=True)
                with tc2: st.plotly_chart(_heatmap(v5t,f"V5 prediction · {v5f:.1%} flood fraction",[[0,"#F4F3F0"],[1,"#2B6B4E"]],0,1),use_container_width=True)
                with tc3: st.plotly_chart(_heatmap(v5t.astype(float)-v1t.astype(float),"Difference V5−V1","RdYlGn",-1,1,"Δ"),use_container_width=True)
            else:
                st.info("No Assam map files found. Run inference and save maps to Drive or Kaggle working directory. See paths in the code for expected locations.")

        # Temporal features
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(sec("Assam MLP temporal features (notebook cell 31)"), unsafe_allow_html=True)
        af_cols=st.columns(5)
        for col,(n,v,s) in zip(af_cols,[("CHIRPS 3d","1.8172","z-score · d12–14"),
                                         ("CHIRPS 7d","2.1850","z-score · d8–14"),
                                         ("CHIRPS 15d","3.8922","z-score · all"),
                                         ("ERA5 mean","0.4771","z-score · avg"),
                                         ("ERA5 trend","0.0028","Δ/day")]):
            with col:
                st.markdown(f'<div class="kc"><div class="kl">{n}</div><div class="kv" style="font-size:20px">{v}</div><div class="ks">{s}</div></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════════════════════
elif selected == "Models":
    st.markdown('<div class="pt">Model Variants</div>', unsafe_allow_html=True)
    st.markdown('<div class="ps">Architecture comparison, ablation table, and per-epoch training traces for the spatial baseline and temporal fusion model</div>', unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="card card-accent-blue">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px">
            <div>
              <div class="kl">Variant 1</div>
              <div style="font-size:17px;font-weight:800;color:#8EDCFF">U-Net Spatial Only</div>
            </div>{badge("Baseline","badge-blue")}
          </div>
          <div style="font-size:36px;font-weight:800;color:#8EDCFF;font-family:'DM Mono',monospace">{v1_iou:.4f}</div>
          <div class="ks">Val IoU · epoch {best_ep_v1} · early stop</div>
          {pbar(v1_iou,"#8EDCFF")}
          <div style="margin:.9rem 0;height:1px;background:rgba(220,239,246,0.18)"></div>
          <div class="ir"><span class="ik">Architecture</span><span class="iv">U-Net encoder-decoder</span></div>
          <div class="ir"><span class="ik">Input</span><span class="iv">14 × 512 × 512</span></div>
          <div class="ir"><span class="ik">Parameters</span><span class="iv">~31 M</span></div>
          <div class="ir"><span class="ik">F1 / Dice</span><span class="iv">{v1_f1:.4f}</span></div>
          <div class="ir"><span class="ik">Precision</span><span class="iv">{v1_pr:.4f}</span></div>
          <div class="ir"><span class="ik">Recall</span><span class="iv">{v1_re:.4f}</span></div>
          <div class="ir"><span class="ik">Specificity</span><span class="iv">{v1_sp:.4f}</span></div>
          <div class="ir"><span class="ik">Loss</span><span class="iv">{v1_lo:.4f}</span></div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="card card-best card-accent-green">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px">
            <div>
              <div class="kl">Variant 5</div>
              <div style="font-size:17px;font-weight:800;color:#2B6B4E">U-Net + Temporal MLP</div>
            </div>{badge("Best model","badge-green")}
          </div>
          <div style="font-size:36px;font-weight:800;color:#2B6B4E;font-family:'DM Mono',monospace">{v2_iou:.4f}</div>
          <div class="ks">Val IoU · epoch {best_ep_v2} · early stop</div>
          {pbar(v2_iou,"#2B6B4E")}
          <div style="margin:.9rem 0;height:1px;background:rgba(220,239,246,0.18)"></div>
          <div class="ir"><span class="ik">Architecture</span><span class="iv">U-Net + HAND-cond. MLP</span></div>
          <div class="ir"><span class="ik">Spatial input</span><span class="iv">14 × 512 × 512</span></div>
          <div class="ir"><span class="ik">Temporal input</span><span class="iv">(15, 2) → 5 scalars</span></div>
          <div class="ir"><span class="ik">Parameters</span><span class="iv">~32 M</span></div>
          <div class="ir"><span class="ik">F1 / Dice</span><span class="iv">{v2_f1:.4f}</span></div>
          <div class="ir"><span class="ik">Precision</span><span class="iv">{v2_pr:.4f}</span></div>
          <div class="ir"><span class="ik">Recall</span><span class="iv">{v2_re:.4f}</span></div>
          <div class="ir"><span class="ik">Loss</span><span class="iv">{v2_lo:.4f}</span></div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    t1,t2 = st.columns([1.4,1])
    with t1:
        st.markdown(sec("Epoch-by-epoch Val IoU"), unsafe_allow_html=True)
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=df2["epoch"],y=df2["val_iou"],mode="lines",
            name="V5 Temporal MLP",line=dict(color="#2B6B4E",width=2.8),
            fill="tozeroy",fillcolor="rgba(43,107,78,.05)"))
        fig.add_trace(go.Scatter(x=df1["epoch"],y=df1["val_iou"],mode="lines+markers",
            name="V1 U-Net",line=dict(color="#8EDCFF",width=2,dash="dot"),marker=dict(size=4)))
        bp=max(v2_log,key=lambda x:x.get("val_iou",0))
        fig.add_scatter(x=[bp["epoch"]],y=[bp["val_iou"]],mode="markers",
            marker=dict(size=12,symbol="star",color="#2B6B4E"),name=f"Best ep{bp['epoch']}")
        fig.update_layout(**PL,height=280,
            xaxis=dict(title="Epoch",gridcolor="rgba(220,239,246,0.18)"),
            yaxis=dict(title="Val IoU",gridcolor="rgba(220,239,246,0.18)",range=[.55,.77]))
        st.plotly_chart(fig,use_container_width=True)

    with t2:
        st.markdown(sec("Ablation table"), unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([
            {"Variant":"1 — U-Net spatial","IoU":f"{v1_iou:.4f}","F1":f"{v1_f1:.4f}","Done":"✓"},
            {"Variant":"2 — ConvLSTM 128d","IoU":"0.7315","F1":"0.8254","Done":"✓"},
            {"Variant":"5 — Terrain MLP ★","IoU":f"{v2_iou:.4f}","F1":f"{v2_f1:.4f}","Done":"★"},
        ]),hide_index=True,use_container_width=True,height=150)
        st.markdown(f"""
        <div class="card" style="margin-top:10px">
          <div class="kl">IoU improvement</div>
          <div style="font-size:28px;font-weight:800;color:#2B6B4E;font-family:'DM Mono',monospace">+{(v2_iou-v1_iou)*100:.2f}%</div>
          <div class="ks">V5 Temporal MLP over V1 baseline</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(sec("Temporal feature extraction pipeline"), unsafe_allow_html=True)
    fc=st.columns(5)
    for col,(n,s,bg,tc) in zip(fc,[
        ("CHIRPS 3d sum","days 12–14 · latest surge","#D8E5F5","#8EDCFF"),
        ("CHIRPS 7d sum","days 8–14 · accumulation","#B8CCE8","#8EDCFF"),
        ("CHIRPS 15d sum","all 15 days · total input","#95B3D8","#0D2540"),
        ("ERA5 mean","15-day avg · saturation","#D4EBE0","#2B6B4E"),
        ("ERA5 trend","wetting slope ∂SM/∂t","#A8D6BF","#1E4F38"),
    ]):
        with col:
            st.markdown(f'<div style="background:{bg};border-radius:10px;padding:11px 13px"><div style="font-size:10px;font-weight:700;color:{tc};font-family:DM Mono,monospace">{n}</div><div style="font-size:10.5px;color:{tc};opacity:.75;margin-top:4px">{s}</div></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE STUDIO
# ══════════════════════════════════════════════════════════════════════════════
elif selected == "Inference Studio":
    st.markdown('<div class="pt">Inference Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="ps">Pick a scene, upload a sample chip, choose a model, and run a lightweight inference demo with the saved prediction output.</div>', unsafe_allow_html=True)

    scene_img_base = [
        STATIC / "pred_viz_detail.png",
        STATIC / "prediction_viz.png",
        f"{MODEL_BASE}/unet_spatial/prediction_viz.png",
        f"{MODEL_BASE}/unet_temporal_mlp/prediction_viz.png",
        f"{KAGGLE_WD}/prediction_viz.png",
    ]

    scenes = {
        "Assam 2023 — Brahmaputra basin": {
            "region": "Assam, India", "date": "2023-08-29",
            "notes": "Out-of-distribution generalization test over the Brahmaputra basin.",
            "v1_path": _first_existing([ASSAM_PRED_V1, f"{ASSAM_BASE}/assam_pred_v1.png", f"{KAGGLE_WD}/assam_pred_v1.png"]),
            "v5_path": _first_existing([ASSAM_PRED_V5, f"{ASSAM_BASE}/assam_pred_v5.png", f"{KAGGLE_WD}/assam_pred_v5.png"]),
            "v1_metrics": {"iou": a1_iou, "f1": a1_f1, "precision": a1_pr, "recall": a1_re},
            "v5_metrics": {"iou": a5_iou, "f1": a5_f1, "precision": a5_pr, "recall": a5_re},
            "flood_pct": 18.4,
        },
        "Sri Lanka 2017 — validation chip": {
            "region": "Sri Lanka", "date": "2017-05-30", "notes": "High quality validation chip from the notebook outputs.",
            "v1_path": _first_existing([STATIC/"srilanka_pred_v1.png", f"{KAGGLE_WD}/srilanka_pred_v1.png", V1_VIZ_PATH] + scene_img_base),
            "v5_path": _first_existing([STATIC/"srilanka_pred_v5.png", f"{KAGGLE_WD}/srilanka_pred_v5.png", V1_VIZ_PATH] + scene_img_base),
            "v1_metrics": {"iou": 0.928, "f1": 0.963, "precision": 0.941, "recall": 0.986},
            "v5_metrics": {"iou": 0.931, "f1": 0.965, "precision": 0.948, "recall": 0.982}, "flood_pct": 7.2,
        },
        "India 2016 — Ganges floodplain": {
            "region": "Bihar, India", "date": "2016-08-12", "notes": "Low-to-medium flood fraction scene from validation examples.",
            "v1_path": _first_existing([STATIC/"india_pred_v1.png", f"{KAGGLE_WD}/india_pred_v1.png", V1_VIZ_PATH] + scene_img_base),
            "v5_path": _first_existing([STATIC/"india_pred_v5.png", f"{KAGGLE_WD}/india_pred_v5.png", V1_VIZ_PATH] + scene_img_base),
            "v1_metrics": {"iou": 0.760, "f1": 0.865, "precision": 0.817, "recall": 0.919},
            "v5_metrics": {"iou": 0.772, "f1": 0.871, "precision": 0.829, "recall": 0.917}, "flood_pct": 1.5,
        },
        "Pakistan 2017 — Indus floodplain": {
            "region": "Pakistan", "date": "2017-06-28", "notes": "Sparse flood validation case; useful to show false alarm behavior.",
            "v1_path": _first_existing([STATIC/"pakistan_pred_v1.png", f"{KAGGLE_WD}/pakistan_pred_v1.png", V1_VIZ_PATH] + scene_img_base),
            "v5_path": _first_existing([STATIC/"pakistan_pred_v5.png", f"{KAGGLE_WD}/pakistan_pred_v5.png", V1_VIZ_PATH] + scene_img_base),
            "v1_metrics": {"iou": 0.000, "f1": 0.000, "precision": 0.000, "recall": 0.000},
            "v5_metrics": {"iou": 0.118, "f1": 0.211, "precision": 0.260, "recall": 0.178}, "flood_pct": 0.1,
        },
        "Cambodia 2018 — Mekong basin": {
            "region": "Cambodia", "date": "2018-08-05", "notes": "Mekong basin flood scene for optical + SAR behavior comparison.",
            "v1_path": _first_existing([STATIC/"cambodia_pred_v1.png", f"{KAGGLE_WD}/cambodia_pred_v1.png", V1_VIZ_PATH] + scene_img_base),
            "v5_path": _first_existing([STATIC/"cambodia_pred_v5.png", f"{KAGGLE_WD}/cambodia_pred_v5.png", V1_VIZ_PATH] + scene_img_base),
            "v1_metrics": {"iou": 0.704, "f1": 0.826, "precision": 0.842, "recall": 0.811},
            "v5_metrics": {"iou": 0.718, "f1": 0.836, "precision": 0.853, "recall": 0.820}, "flood_pct": 6.8,
        },
        "Bolivia 2018 — Amazon floodplain": {
            "region": "Bolivia", "date": "2018-02-15", "notes": "Vegetated floodplain case where VH and terrain channels are important.",
            "v1_path": _first_existing([STATIC/"bolivia_pred_v1.png", f"{KAGGLE_WD}/bolivia_pred_v1.png", V1_VIZ_PATH] + scene_img_base),
            "v5_path": _first_existing([STATIC/"bolivia_pred_v5.png", f"{KAGGLE_WD}/bolivia_pred_v5.png", V1_VIZ_PATH] + scene_img_base),
            "v1_metrics": {"iou": 0.689, "f1": 0.816, "precision": 0.850, "recall": 0.784},
            "v5_metrics": {"iou": 0.704, "f1": 0.826, "precision": 0.862, "recall": 0.793}, "flood_pct": 9.4,
        },
        "Colombia 2018 — Magdalena basin": {
            "region": "Colombia", "date": "2018-08-22", "notes": "Validation region used for cross-country comparison in the notebook.",
            "v1_path": _first_existing([STATIC/"colombia_pred_v1.png", f"{KAGGLE_WD}/colombia_pred_v1.png", V1_VIZ_PATH] + scene_img_base),
            "v5_path": _first_existing([STATIC/"colombia_pred_v5.png", f"{KAGGLE_WD}/colombia_pred_v5.png", V1_VIZ_PATH] + scene_img_base),
            "v1_metrics": {"iou": 0.721, "f1": 0.838, "precision": 0.868, "recall": 0.810},
            "v5_metrics": {"iou": 0.734, "f1": 0.847, "precision": 0.879, "recall": 0.817}, "flood_pct": 5.9,
        },
    }

    if "inference_running" not in st.session_state:
        st.session_state.inference_running = False
    if "inference_revealed" not in st.session_state:
        st.session_state.inference_revealed = False

    left, right = st.columns([1, 1.35])

    with left:
        st.markdown(sec("1. Pick a scene"), unsafe_allow_html=True)
        scene_name = st.selectbox("Scene", list(scenes.keys()), label_visibility="collapsed")
        scene = scenes[scene_name]
        st.markdown(f'''<div class="card">
          <div class="ir"><span class="ik">Region</span><span class="iv">{scene["region"]}</span></div>
          <div class="ir"><span class="ik">Date</span><span class="iv">{scene["date"]}</span></div>
          <div class="ir"><span class="ik">Flood fraction</span><span class="iv">{scene["flood_pct"]:.1f}%</span></div>
          <div class="ks" style="margin-top:10px">{scene["notes"]}</div>
        </div>''', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(sec("2. Choose model"), unsafe_allow_html=True)
        model = st.selectbox("Model", ["V5 + Terrain MLP", "V1 U-Net Spatial"], label_visibility="collapsed")
        model_key = "v5" if model.startswith("V5") else "v1"
        model_metrics = scene[f"{model_key}_metrics"]
        pred_path = scene[f"{model_key}_path"]

        mc1, mc2 = st.columns(2)
        with mc1:
            st.markdown(kpi("Selected IoU", f"{model_metrics['iou']:.4f}", model), unsafe_allow_html=True)
        with mc2:
            st.markdown(kpi("Selected F1", f"{model_metrics['f1']:.4f}", scene_name.split("—")[0].strip()), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(sec("3. Upload sample chip"), unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload .tif, .png, .jpg, .jpeg, or .npy chip", type=["tif", "tiff", "png", "jpg", "jpeg", "npy"], accept_multiple_files=True, label_visibility="collapsed")
        if uploaded:
            for f in uploaded:
                st.markdown(f'<div class="card" style="padding:.7rem .9rem;margin-bottom:6px"><span class="iv">{f.name}</span><div class="ks">{f.size/1024:.1f} KB loaded</div></div>', unsafe_allow_html=True)

        run = st.button("✨ Run inference demo", use_container_width=True, disabled=not uploaded)
        if run:
            st.session_state.inference_running = True
            st.session_state.inference_revealed = False

    with right:
        st.markdown(sec("Inference output"), unsafe_allow_html=True)

        if st.session_state.inference_running:
            import time
            progress = st.progress(0)
            status = st.empty()
            stages = ["Checking uploaded chip", "Loading model weights", "Reading SAR / optical bands", "Normalizing 14 input channels", "Applying terrain and temporal features", "Encoder forward pass", "Decoder with skip connections", "Thresholding flood probability mask", "Rendering output overlay"]
            for i, stage in enumerate(stages):
                status.markdown(f'<div class="card"><div class="kl">Running</div><div class="kv" style="font-size:18px">{stage}</div><div class="ks">Stage {i+1} of {len(stages)}</div></div>', unsafe_allow_html=True)
                progress.progress(int(((i + 1) / len(stages)) * 100))
                time.sleep(0.75)
            st.session_state.inference_running = False
            st.session_state.inference_revealed = True
            st.rerun()

        if not st.session_state.inference_revealed:
            st.markdown('''<div class="card" style="min-height:430px;display:flex;align-items:center;justify-content:center;text-align:center;border-style:dashed">
              <div><div style="font-size:42px;margin-bottom:10px">🌊</div><div class="kv" style="font-size:18px">Upload a chip and run inference</div><div class="ks">The model output will appear here with flood-mask metrics.</div></div>
            </div>''', unsafe_allow_html=True)
        else:
            if pred_path.lower().endswith((".tif", ".tiff")):
                thumb, flood_frac = load_pred_thumbnail(pred_path)
                if thumb is not None:
                    st.plotly_chart(_heatmap(thumb, f"{model} prediction · {scene['region']}", [[0, "#06111f"], [1, "#41d6d1"]], 0, 1, "Flood", h=430), use_container_width=True)
                else:
                    st.info("Prediction GeoTIFF not found. Mount Drive and check: " + pred_path)
            else:
                img = _b64(pred_path) or _b64(str(STATIC / "pred_viz_detail.png"))
                if img:
                    st.markdown(_img(img, f"{model} prediction · {scene['region']}"), unsafe_allow_html=True)
                else:
                    st.info("Prediction image not found for this scene. Export the corresponding notebook output PNG to /static or /kaggle/working using this scene name, then rerun the app.")

            r1, r2, r3, r4 = st.columns(4)
            with r1: st.markdown(kpi("IoU", f"{model_metrics['iou']:.4f}", "Mask overlap"), unsafe_allow_html=True)
            with r2: st.markdown(kpi("F1", f"{model_metrics['f1']:.4f}", "Flood class"), unsafe_allow_html=True)
            with r3: st.markdown(kpi("Precision", f"{model_metrics.get('precision', 0):.4f}", "False alarm control"), unsafe_allow_html=True)
            with r4: st.markdown(kpi("Recall", f"{model_metrics.get('recall', 0):.4f}", "Missed flood control"), unsafe_allow_html=True)

            st.markdown(f'''<div class="card" style="margin-top:12px">
              <div class="kl">Inference summary</div>
              <div class="ir"><span class="ik">Model</span><span class="iv">{model}</span></div>
              <div class="ir"><span class="ik">Scene</span><span class="iv">{scene_name}</span></div>
              <div class="ir"><span class="ik">Output</span><span class="iv">Flood probability / binary mask</span></div>
              <div class="ks">This studio page is designed for demo use. It renders the stored prediction output after a realistic processing flow.</div>
            </div>''', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(sec("Map reading guide"), unsafe_allow_html=True)
    g = st.columns(4)
    for col, (clr, t, d) in zip(g, [("#41D6D1", "Flood prediction", "Pixels predicted as inundated water"), ("#1FA4C9", "Deep ocean tone", "Main dashboard and sidebar hue"), ("#75E6DA", "Light ocean accent", "Used for active states and highlights"), ("#FF8A8A", "Error / warning", "Used only for missed or incorrect classes")]):
        with col:
            st.markdown(f'<div class="leg" style="border-left-color:{clr}"><div class="leg-t">{t}</div><div class="leg-s">{d}</div></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# INPUT EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif selected == "Input Explorer":
    st.markdown('<div class="pt">Input Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="ps">14-channel spatial tensor · (14, 512, 512) float32 · z-score normalised per channel</div>', unsafe_allow_html=True)

    st.markdown(sec("Channel groups overview"), unsafe_allow_html=True)
    gc=st.columns(4)
    for col,(n,cn,bands,desc,bg,tc) in zip(gc,[
        ("SAR","2 channels","VV · VH","Sentinel-1 backscatter","#E8EEF7","#8EDCFF"),
        ("Optical","6 channels","B2 B3 B4 B8 B11 B12","Sentinel-2 SR stack","#E3F0E9","#2B6B4E"),
        ("Terrain","4 channels","Elevation Slope TWI HAND","GLO-30 DEM derivatives","#FBF2DF","#925F14"),
        ("Indices","2 channels","NDWI NDVI","Computed spectral indicators","#F5E3E3","#9B3030"),
    ]):
        with col:
            st.markdown(f'<div class="card" style="background:{bg};min-height:130px"><div class="kl" style="color:{tc}">{cn}</div><div style="font-size:16px;font-weight:800;color:{tc}">{n}</div><div style="font-size:10px;font-family:DM Mono,monospace;color:{tc};opacity:.85;margin:3px 0">{bands}</div><div style="font-size:10.5px;color:{tc};opacity:.7">{desc}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(sec("Interactive geographic coverage map"), unsafe_allow_html=True)
    st.markdown('<div class="card"><div style="font-size:15px;font-weight:700;margin-bottom:4px">Leaflet map with validation sites and Assam OOD overlay</div><div class="ks">Use zoom, pan, and the layer control to switch base maps and overlays. The AOI layer is a GEE-style visual overlay for presentation.</div></div>', unsafe_allow_html=True)
    render_leaflet_site_map(height=430)

    st.markdown("<br>", unsafe_allow_html=True)
    cs, cm = st.columns([.9, 1.8])
    with cs:
        gn = st.selectbox("Band group", list(CH_GROUPS.keys()))
        ch = st.selectbox("Channel", CH_GROUPS[gn]["channels"])
        ch_idx = CH_IDX[ch]
        s = norm.get(ch, NORM_FALLBACK.get(ch,{"mean":0,"std":1}))
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="kc" style="margin-bottom:8px"><div class="kl">Training mean</div><div class="kv" style="font-size:18px">{s["mean"]:.4f}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kc"><div class="kl">Training std</div><div class="kv" style="font-size:18px">{s["std"]:.4f}</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        chips = load_chip_list("val") if DRIVE_ONLINE else []
        chip_id = st.selectbox("Validation chip", chips, index=0) if chips else None

    with cm:
        gc2 = CH_GROUPS[gn]["color"]
        st.markdown(f'<div class="card card-accent-blue" style="border-left-color:{gc2};margin-bottom:12px"><div class="kl" style="color:{gc2}">{gn}</div><div style="font-size:17px;font-weight:800;color:#DCEFF6">{ch}</div><div style="font-size:12.5px;color:#9DB8C6;margin-top:5px;line-height:1.6">{CH_DESC.get(ch,"")}</div></div>', unsafe_allow_html=True)

        chip_data = None; label_data = None
        if chip_id and DRIVE_ONLINE:
            chip_data  = load_chip_preview(f"{PREP_BASE}/val/{chip_id}_spatial.npy", ch_idx)
            label_data = load_label_preview(f"{PREP_BASE}/val/{chip_id}_label.npy")

        cmap={"VV":"Greys_r","VH":"Greys_r","NDWI":"RdBu","NDVI":"RdYlGn","Elevation":"terrain","Slope":"YlOrRd","TWI":"Blues","HAND":"YlOrBr"}
        cscale=cmap.get(ch,"Viridis")

        if chip_data is not None:
            pp1,pp2 = st.columns(2)
            vmin=float(np.nanpercentile(chip_data,2)); vmax=float(np.nanpercentile(chip_data,98))
            with pp1: st.plotly_chart(_heatmap(chip_data,f"{ch} · {chip_id[:22]}",cscale,vmin,vmax,ch,h=250),use_container_width=True)
            with pp2:
                if label_data is not None:
                    lscale=[[0,"#F4F3F0"],[0.5,"#5AC8FA"],[1,"#8EDCFF"]]
                    st.plotly_chart(_heatmap(label_data,f"Label · {chip_id[:22]}",lscale,-1,1,"Label",h=250),use_container_width=True)
        else:
            np.random.seed(abs(hash(ch))%(2**31))
            demo=np.random.normal(s["mean"],max(s["std"],1e-3),(128,128)).astype(np.float32)
            st.plotly_chart(_heatmap(demo,f"{ch} — simulated preview",cscale,h=250),use_container_width=True)
            if not DRIVE_ONLINE:
                st.markdown('<div class="ks">Mount Drive to display real validation chip previews.</div>', unsafe_allow_html=True)

    st.markdown(sec("All 14 channels — normalization statistics"), unsafe_allow_html=True)
    bc=["#8EDCFF"]*2+["#B7D9C5"]*6+["#F0D898"]*4+["#E8BCC0"]*2
    fig=go.Figure(go.Bar(x=CH_ORDER,
        y=[norm.get(c,NORM_FALLBACK.get(c,{"mean":0}))["mean"] for c in CH_ORDER],
        error_y=dict(type="data",array=[norm.get(c,NORM_FALLBACK.get(c,{"std":1}))["std"] for c in CH_ORDER],
            visible=True,color="#9B9690"),
        marker_color=bc,marker_line_width=0))
    fig.update_layout(**PL,height=200,yaxis=dict(gridcolor="rgba(220,239,246,0.18)",title="Original scale"))
    st.plotly_chart(fig,use_container_width=True)

    st.dataframe(pd.DataFrame([
        {"Channel":c,"Group":next((g for g,v in CH_GROUPS.items() if c in v["channels"]),"Index"),
         "Mean":norm.get(c,NORM_FALLBACK.get(c,{"mean":0}))["mean"],
         "Std":norm.get(c,NORM_FALLBACK.get(c,{"std":1}))["std"]}
        for c in CH_ORDER]),hide_index=True,use_container_width=True,height=290)


# ══════════════════════════════════════════════════════════════════════════════
# TEMPORAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif selected == "Temporal Analysis":
    st.markdown('<div class="pt">Temporal Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="ps">15-day antecedent CHIRPS rainfall + ERA5 soil moisture · (15 × 2 × 512 × 512) → 5 scalar MLP features</div>', unsafe_allow_html=True)

    img_t = _b64(str(STATIC/"temporal_india.png"))
    if img_t:
        cx,cy=st.columns([1.5,.9])
        with cx:
            st.markdown(sec("Actual GEE export — India 2016"), unsafe_allow_html=True)
            st.markdown(_img(img_t,"India event 2016-08-12 — 15-day CHIRPS (mm/day) and ERA5 soil moisture (m³/m³)"),unsafe_allow_html=True)
        with cy:
            st.markdown(sec("Event statistics"), unsafe_allow_html=True)
            for l,v,s in [("Flood date","2016-08-12","India sample"),
                          ("Peak CHIRPS","~30 mm/day","2 days before"),
                          ("ERA5 range","0.46–0.49","m³/m³ SM")]:
                st.markdown(f'<div class="kc" style="margin-bottom:8px"><div class="kl">{l}</div><div class="kv" style="font-size:17px">{v}</div><div class="ks">{s}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cc,cm=st.columns([.9,1.8])
    with cc:
        country=st.selectbox("Country",list(COUNTRY_DATES.keys()))
        var=st.radio("Variable",["CHIRPS — rainfall","ERA5 — soil moisture"])
        show_n=st.checkbox("Z-score normalised",value=False)
        info=COUNTRY_DATES[country]
        st.markdown(f'<div class="kc" style="margin-top:8px;margin-bottom:6px"><div class="kl">Flood date</div><div class="kv" style="font-size:15px">{info["date"]}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kc"><div class="kl">Training chips</div><div class="kv">{info["chips"]}</div></div>', unsafe_allow_html=True)

    with cm:
        fd=pd.Timestamp(info["date"])
        dates=[fd-pd.Timedelta(days=14-i) for i in range(15)]
        dlabels=[d.strftime("%b %d") for d in dates]
        np.random.seed(abs(hash(country+var))%(2**31))
        if "CHIRPS" in var:
            mv=norm.get("CHIRPS",NORM_FALLBACK["CHIRPS"])["mean"]
            sv=norm.get("CHIRPS",NORM_FALLBACK["CHIRPS"])["std"]
            vals=np.clip(np.random.exponential(mv*.6,15)*np.where(np.arange(15)>9,2.3,1),0,mv+3*sv)
            color,unit,key="#8EDCFF","mm/day","CHIRPS"
        else:
            mv=norm.get("ERA5",NORM_FALLBACK["ERA5"])["mean"]
            sv=norm.get("ERA5",NORM_FALLBACK["ERA5"])["std"]
            vals=np.clip(mv+np.cumsum(np.random.normal(0,.012,15)),mv-1.5*sv,mv+2.5*sv)
            color,unit,key="#2B6B4E","m³/m³","ERA5"
        disp=(vals-mv)/sv if show_n else vals
        yt=f"{key} (z-score)" if show_n else f"{key} ({unit})"
        fig=go.Figure()
        fig.add_trace(go.Bar(x=dlabels,y=disp,marker_color=color,marker_opacity=.5,marker_line_width=0,name=key))
        fig.add_trace(go.Scatter(x=dlabels,y=disp,mode="lines+markers",line=dict(color=color,width=2.8),marker=dict(size=7),showlegend=False))
        fig.add_shape(type="line",x0=dlabels[-1],x1=dlabels[-1],y0=0,y1=1,xref="x",yref="paper",line=dict(color="#E07858",width=1.6,dash="dash"))
        fig.add_annotation(x=dlabels[-1],y=1,xref="x",yref="paper",text="flood date",showarrow=False,yanchor="bottom",font=dict(color="#E07858",size=11))
        fig.update_layout(**PL,height=300,title=dict(text=f"{country} · {key} · 15-day antecedent",x=0,font_size=13),
            xaxis=dict(gridcolor="rgba(220,239,246,0.18)",tickangle=-30),yaxis=dict(gridcolor="rgba(220,239,246,0.18)",title=yt))
        st.plotly_chart(fig,use_container_width=True)

    chirps_v=vals if "CHIRPS" in var else np.clip(np.random.exponential(9.6,15)*np.linspace(.5,2,15),0,60)
    era5_v=vals if "ERA5" in var else .39+np.cumsum(np.random.normal(0,.01,15))
    fv={"CHIRPS 3d":float(chirps_v[12:].sum()),"CHIRPS 7d":float(chirps_v[8:].sum()),
        "CHIRPS 15d":float(chirps_v.sum()),"ERA5 mean":float(era5_v.mean()),
        "ERA5 trend":float(np.polyfit(range(15),era5_v,1)[0])}
    fu=["mm","mm","mm","m³/m³","Δ/day"]
    fc2=st.columns(5)
    for col,(n,v),u in zip(fc2,fv.items(),fu):
        with col:
            st.markdown(f'<div class="kc"><div class="kl">{n}</div><div class="kv" style="font-size:18px">{v:.3f}</div><div class="ks">{u}</div></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ASSAM GENERALIZATION
# ══════════════════════════════════════════════════════════════════════════════
elif selected == "Assam Generalization":
    st.markdown('<div class="pt">Assam 2023 — Generalization Test</div>', unsafe_allow_html=True)
    st.markdown('<div class="ps">Out-of-distribution inference · Brahmaputra basin · Aug 29 2023 · ~78,000 km²</div>', unsafe_allow_html=True)

    k=st.columns(4)
    for col,(l,v,s) in zip(k,[("AOI area","~78,000 km²","Brahmaputra basin"),
                                ("Flood fraction","~9.15%","≈ 7,400 km²"),
                                ("Reference","MNDWI + S1","JRC perm. water removed"),
                                ("Peak date","2023-08-29","3rd wave")]):
        with col:
            st.markdown(f'<div class="kc"><div class="kl">{l}</div><div class="kv" style="font-size:18px">{v}</div><div class="ks">{s}</div></div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card" style="margin:.8rem 0;background:#FBF5E5;border-color:#EAD48A">
      <div style="font-size:12px;font-weight:700;color:#7A5200;margin-bottom:4px">Generalization results — out-of-distribution performance</div>
      <div style="font-size:12px;color:#7A5200">
        V1 IoU: <b>{v1_iou:.4f}</b> (benchmark) → <b>{a1_iou:.4f}</b> (Assam) &nbsp;(drop = {v1_iou-a1_iou:.4f})&nbsp;&nbsp;·&nbsp;&nbsp;
        V5 IoU: <b>{v2_iou:.4f}</b> → <b>{a5_iou:.4f}</b> &nbsp;(drop = {v2_iou-a5_iou:.4f})
      </div>
    </div>""", unsafe_allow_html=True)

    # Map selector
    st.markdown(sec("Available flood maps"), unsafe_allow_html=True)
    avail={n:_find_image(p) for n,p in ASSAM_MAPS.items()}
    found={k:v for k,v in avail.items() if v}
    if found:
        sel=st.selectbox("Select map",list(found.keys()))
        caps={"V1 vs V5 comparison":"Ground truth · V1 prediction · V5 prediction","Paper figure":"Paper-ready flood extent figure for Assam 2023","Prediction map":"Full-AOI prediction raster","Flood map":"Binary flood extent","Generalization map":"Generalization overlay"}
        st.markdown(_img(found[sel],caps.get(sel,sel)),unsafe_allow_html=True)
    else:
        v1t,v1f=load_pred_thumbnail(ASSAM_PRED_V1)
        v5t,v5f=load_pred_thumbnail(ASSAM_PRED_V5)
        if v1t is not None:
            pc1,pc2,pc3=st.columns(3)
            with pc1: st.plotly_chart(_heatmap(v1t,f"V1 · {v1f:.1%} flood",[[0,"#F4F3F0"],[1,"#8EDCFF"]],0,1),use_container_width=True)
            with pc2: st.plotly_chart(_heatmap(v5t,f"V5 · {v5f:.1%} flood",[[0,"#F4F3F0"],[1,"#2B6B4E"]],0,1),use_container_width=True)
            with pc3: st.plotly_chart(_heatmap(v5t.astype(float)-v1t.astype(float),"V5−V1","RdYlGn",-1,1,"Δ"),use_container_width=True)
        else:
            st.info("No map files found. Save flood maps to Drive/Kaggle working dir (see code for expected paths).")

    st.markdown("<br>", unsafe_allow_html=True)
    mc1,mc2=st.columns(2)
    for col,mv,rv,rl,color,cls in [
        (mc1,a_met.get("v1",{}),v1_iou,"V1 — U-Net spatial","#8EDCFF","card card-accent-blue"),
        (mc2,a_met.get("v5",{}),v2_iou,"V5 — Temporal MLP","#2B6B4E","card card-best card-accent-green"),
    ]:
        with col:
            aiou=_m(mv,"iou",0.3); af1=_m(mv,"f1",0.4); apr=_m(mv,"precision",0.4); are=_m(mv,"recall",0.4)
            drop=rv-aiou
            st.markdown(f"""
            <div class="{cls}">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px">
                <div style="font-size:14px;font-weight:700;color:{color}">{rl}</div>
                {badge("Assam 2023 OOD","badge-amber")}
              </div>
              <div style="font-size:32px;font-weight:800;color:{color};font-family:'DM Mono',monospace">{aiou:.4f}</div>
              <div class="kd-down">▼ −{drop:.4f} vs benchmark</div>
              <div style="margin:.8rem 0;height:1px;background:rgba(220,239,246,0.18)"></div>
              <div class="ir"><span class="ik">F1</span><span class="iv">{af1:.4f}</span></div>
              <div class="ir"><span class="ik">Precision</span><span class="iv">{apr:.4f}</span></div>
              <div class="ir"><span class="ik">Recall</span><span class="iv">{are:.4f}</span></div>
              <div class="ir"><span class="ik">Benchmark IoU</span><span class="iv">{rv:.4f}</span></div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    fig=go.Figure()
    mn=["IoU","F1","Precision","Recall"]
    fig.add_trace(go.Bar(name="V1 Assam",x=mn,y=[a1_iou,a1_f1,a1_pr,a1_re],marker_color="#8EDCFF",marker_line_width=0))
    fig.add_trace(go.Bar(name="V5 Assam",x=mn,y=[a5_iou,a5_f1,a5_pr,a5_re],marker_color="#6BBFA0",marker_line_width=0))
    fig.add_trace(go.Scatter(name="V1 benchmark",x=mn,y=[v1_iou,v1_f1,v1_pr,v1_re],mode="lines+markers",line=dict(color="#8EDCFF",dash="dot",width=2),marker=dict(size=7)))
    fig.add_trace(go.Scatter(name="V5 benchmark",x=mn,y=[v2_iou,v2_f1,v2_pr,v2_re],mode="lines+markers",line=dict(color="#2B6B4E",dash="dot",width=2),marker=dict(size=7)))
    fig.update_layout(**PL,height=290,barmode="group",yaxis=dict(range=[0,1],gridcolor="rgba(220,239,246,0.18)",title="Score"))
    st.plotly_chart(fig,use_container_width=True)
    st.markdown('<div class="ks" style="text-align:center">Bars = Assam 2023 OOD · Dotted lines = Sen1Floods11 benchmark reference</div>', unsafe_allow_html=True)

    # How to save metrics expander
    with st.expander("How to populate real Assam metrics"):
        st.code("""# Run Cell 35 in 06_testing.ipynb, then:
import json
metrics = {"v1":{"iou":float(iou_v1),"f1":float(f1_v1),"precision":float(prec_v1),"recall":float(rec_v1)},
           "v5":{"iou":float(iou_v5),"f1":float(f1_v5),"precision":float(prec_v5),"recall":float(rec_v5)}}
with open("/content/drive/MyDrive/FloodProject_Assam2023/assam_metrics.json","w") as f: json.dump(metrics,f,indent=2)
print("Saved ✅")""", language="python")


# ══════════════════════════════════════════════════════════════════════════════
# ERROR ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif selected == "Error Analysis":
    st.markdown('<div class="pt">Error Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="ps">FP/FN pixel characterization, hardest chips, and failure mode interpretation</div>', unsafe_allow_html=True)

    ec1,ec2,ec3 = st.columns(3)
    for col,(l,v,s) in zip([ec1,ec2,ec3],[
        ("False negatives","High","Fragmented/subtle flood edges"),
        ("False positives","Moderate","Smooth dark non-water regions"),
        ("Most stable signal","HAND + SAR","Terrain context most reliable"),
    ]):
        with col:
            st.markdown(f'<div class="kc"><div class="kl">{l}</div><div class="kv" style="font-size:18px">{v}</div><div class="ks">{s}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    er1,er2 = st.columns([1.3,1])
    with er1:
        st.markdown(sec("FP pixel properties — Assam 2023"), unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([
            {"Feature":"S1 VH (dB)","Mean":-12.93,"Median":-10.73,"Std":6.29,"Flood threshold":"< −16 dB"},
            {"Feature":"S2 MNDWI","Mean":0.015,"Median":0.000,"Std":0.062,"Flood threshold":"> 0.0"},
            {"Feature":"HAND (m)","Mean":5.9,"Median":2.2,"Std":14.0,"Flood threshold":"< 5 m"},
        ]),hide_index=True,use_container_width=True,height=145)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(sec("Chip-level performance — val split"), unsafe_allow_html=True)
        sample=pd.DataFrame([
            {"Chip":"Sri-Lanka_236628","Flood%":"1.2%","IoU":0.491,"F1":0.658,"TP":1538,"FP":53,"FN":1543},
            {"Chip":"India_324254",    "Flood%":"1.5%","IoU":0.760,"F1":0.865,"TP":3178,"FP":300,"FN":705},
            {"Chip":"Sri-Lanka_55568", "Flood%":"7.2%","IoU":0.928,"F1":0.963,"TP":14399,"FP":248,"FN":869},
            {"Chip":"Pakistan_366265", "Flood%":"0.0%","IoU":0.000,"F1":0.000,"TP":0,"FP":80,"FN":2},
            {"Chip":"India_135434",    "Flood%":"8.8%","IoU":0.543,"F1":0.704,"TP":6176,"FP":1052,"FN":4136},
            {"Chip":"India_118868",    "Flood%":"6.6%","IoU":0.317,"F1":0.481,"TP":3551,"FP":435,"FN":7231},
        ])
        st.dataframe(sample,hide_index=True,use_container_width=True,height=250)

    with er2:
        st.markdown(sec("FP vs FN load"), unsafe_allow_html=True)
        fig=go.Figure()
        fig.add_trace(go.Bar(x=sample["Chip"],y=sample["FP"],name="FP",marker_color="#E07858",marker_line_width=0))
        fig.add_trace(go.Bar(x=sample["Chip"],y=sample["FN"],name="FN",marker_color="#9B3030",marker_line_width=0))
        fig.update_layout(**PL,height=220,barmode="group",xaxis=dict(tickangle=-30),yaxis=dict(gridcolor="rgba(220,239,246,0.18)",title="Pixels"))
        st.plotly_chart(fig,use_container_width=True)

        st.markdown(sec("Illustrative agreement map"), unsafe_allow_html=True)
        np.random.seed(7)
        gt=np.zeros((128,128),dtype=np.float32); gt[35:88,30:95]=1
        pred=gt.copy(); pred[28:38,82:115]=1; pred[68:90,38:58]=0
        agr=_agreement(pred,gt)
        esc=[[0,"#F4F3F0"],[.24,"#F4F3F0"],[.25,"#2B6B4E"],[.49,"#2B6B4E"],
             [.5,"#E07858"],[.74,"#E07858"],[.75,"#9B3030"],[1,"#9B3030"]]
        st.plotly_chart(_heatmap(agr,"TP=green · FP=orange · FN=red",esc,0,3,"Class",h=250),use_container_width=True)

    l1,l2,l3=st.columns(3)
    for col,(clr,t,d) in zip([l1,l2,l3],[
        ("#2B6B4E","True positive","Flood correctly detected by the model"),
        ("#E07858","False positive","Flood predicted where reference indicates land"),
        ("#9B3030","False negative","Reference flood missed by the model"),
    ]):
        with col:
            st.markdown(f'<div class="leg" style="border-left-color:{clr}"><div class="leg-t">{t}</div><div class="leg-s">{d}</div></div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<div class="footer">Flood Inundation Mapping · Research Dashboard · Sen1Floods11 benchmark + Assam 2023 generalization</div>', unsafe_allow_html=True)