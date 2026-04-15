# ---------------------------
# FIX IMPORT PATH
# ---------------------------
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ---------------------------
# IMPORTS
# ---------------------------
import streamlit as st
import pandas as pd
import numpy as np
import time

from src.predict import load_model, predict
from src.alert import generate_alert

# ---------------------------
# CONFIG
# ---------------------------
st.set_page_config(page_title="AI Predictive Maintenance", layout="wide", initial_sidebar_state="expanded")

# ---------------------------
# PREMIUM GLASSMORPHISM + AMBIENT BACKLIGHT + NEON BORDERS (OVERFLOW FIXED)
# ---------------------------
st.markdown("""
<style>
/* Import Premium Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global Reset & Typography */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* Fix CSS Overflow Issues */
.glass-card, [data-testid="stVegaLiteChart"], [data-testid="metric-container"], [data-testid="stFileUploader"] {
    box-sizing: border-box !important;
    max-width: 100% !important;
}

/* Multi-Color Ambient Mesh Background */
.stApp {
    background: 
        radial-gradient(circle at 15% 0%, rgba(138, 43, 226, 0.10) 0%, transparent 40%),  /* Purple Top Left */
        radial-gradient(circle at 85% 100%, rgba(0, 198, 255, 0.08) 0%, transparent 40%), /* Blue Bottom Right */
        radial-gradient(circle at 50% 50%, rgba(0, 255, 159, 0.04) 0%, transparent 50%),  /* Green Center */
        #07080b; /* Deep Base to make backlights pop */
    color: #ffffff;
    overflow-x: hidden;
}

/* Hide Streamlit Default Header */
header[data-testid="stHeader"] {
    background-color: transparent !important;
}

/* Glassmorphic Sidebar */
[data-testid="stSidebar"] {
    background-color: rgba(11, 13, 20, 0.5) !important;
    backdrop-filter: blur(24px) !important;
    -webkit-backdrop-filter: blur(24px) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.04) !important;
}

[data-testid="stSidebar"] p, [data-testid="stSidebar"] div {
    color: #94a3b8 !important;
}

/* ---------------------------------------------------
   PROFESSIONAL NEON BORDERS & AMBIENT BACKLIGHTS
------------------------------------------------------ */

/* Generic Base for Custom HTML Cards */
.glass-card {
    background: rgba(18, 20, 28, 0.65);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    color: #ffffff;
    font-size: 0.95rem;
    line-height: 1.6;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    overflow: hidden;
    word-wrap: break-word;
}

/* Color-coded Neon Borders and Backlights for Cards */
.bg-glow-blue { 
    border: 1px solid rgba(0, 198, 255, 0.4);
    box-shadow: 0 10px 45px -10px rgba(0, 198, 255, 0.18), inset 0 0 15px rgba(0, 198, 255, 0.05); 
}
.bg-glow-purple { 
    border: 1px solid rgba(179, 102, 255, 0.4);
    box-shadow: 0 10px 45px -10px rgba(179, 102, 255, 0.18), inset 0 0 15px rgba(179, 102, 255, 0.05); 
}
.bg-glow-red { 
    border: 1px solid rgba(255, 75, 75, 0.4);
    box-shadow: 0 10px 45px -10px rgba(255, 75, 75, 0.15), inset 0 0 15px rgba(255, 75, 75, 0.05); 
}
.bg-glow-orange { 
    border: 1px solid rgba(255, 171, 0, 0.4);
    box-shadow: 0 10px 45px -10px rgba(255, 171, 0, 0.15), inset 0 0 15px rgba(255, 171, 0, 0.05); 
}

/* Hover effect intensifies the Neon Border */
.bg-glow-blue:hover { border-color: rgba(0, 198, 255, 0.8); transform: translateY(-2px); }
.bg-glow-purple:hover { border-color: rgba(179, 102, 255, 0.8); transform: translateY(-2px); }
.bg-glow-red:hover { border-color: rgba(255, 75, 75, 0.8); transform: translateY(-2px); }
.bg-glow-orange:hover { border-color: rgba(255, 171, 0, 0.8); transform: translateY(-2px); }

/* Streamlit Native File Uploader */
[data-testid="stFileUploader"] {
    background: rgba(18, 20, 28, 0.65);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(0, 198, 255, 0.4);
    border-radius: 16px;
    padding: 16px;
    box-shadow: 0 10px 45px -10px rgba(0, 198, 255, 0.15), inset 0 0 15px rgba(0, 198, 255, 0.05);
    transition: all 0.3s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(0, 198, 255, 0.8);
    transform: translateY(-2px);
}

/* Streamlit Native Metric Cards (KPIs) */
div[data-testid="metric-container"] {
    background: rgba(18, 20, 28, 0.65);
    backdrop-filter: blur(20px);
    border-radius: 16px;
    padding: 20px 24px;
    transition: all 0.4s ease;
}
/* Assigning specific Neon Borders to the 3 KPI boxes */
[data-testid="column"]:nth-child(1) div[data-testid="metric-container"] {
    border: 1px solid rgba(179, 102, 255, 0.4); /* Status: Purple */
    box-shadow: 0 10px 45px -10px rgba(179, 102, 255, 0.18), inset 0 0 15px rgba(179, 102, 255, 0.05);
}
[data-testid="column"]:nth-child(2) div[data-testid="metric-container"] {
    border: 1px solid rgba(255, 75, 75, 0.4); /* Probability: Red */
    box-shadow: 0 10px 45px -10px rgba(255, 75, 75, 0.15), inset 0 0 15px rgba(255, 75, 75, 0.05);
}
[data-testid="column"]:nth-child(3) div[data-testid="metric-container"] {
    border: 1px solid rgba(0, 255, 159, 0.4); /* Health: Green */
    box-shadow: 0 10px 45px -10px rgba(0, 255, 159, 0.15), inset 0 0 15px rgba(0, 255, 159, 0.05);
}
[data-testid="column"]:nth-child(1) div[data-testid="metric-container"]:hover { border-color: rgba(179, 102, 255, 0.8); transform: translateY(-2px); }
[data-testid="column"]:nth-child(2) div[data-testid="metric-container"]:hover { border-color: rgba(255, 75, 75, 0.8); transform: translateY(-2px); }
[data-testid="column"]:nth-child(3) div[data-testid="metric-container"]:hover { border-color: rgba(0, 255, 159, 0.8); transform: translateY(-2px); }

/* Streamlit Native Charts Backlight & Neon Border */
[data-testid="stVegaLiteChart"] {
    background: rgba(255, 255, 255, 0.02); 
    backdrop-filter: blur(20px);
    border: 1px solid rgba(0, 198, 255, 0.4); /* Blue Neon Border for Graphs */
    border-radius: 16px;
    padding: 10px !important; /* Reduced padding to avoid overflow */
    box-shadow: 0 10px 45px -10px rgba(0, 198, 255, 0.15), inset 0 0 15px rgba(0, 198, 255, 0.05);
    transition: all 0.3s ease;
    overflow: hidden !important;
}
[data-testid="stVegaLiteChart"]:hover {
    border-color: rgba(0, 198, 255, 0.8);
}

/* ---------------------------------------------------
   CLEAN HEADINGS (SEPARATED FROM BACKLIGHTS)
------------------------------------------------------ */
.block-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 16px;
    margin-top: 12px;
    display: flex;
    align-items: center;
    padding-left: 12px;
    border-left: 3px solid transparent;
    border-image: linear-gradient(to bottom, #00c6ff, #b366ff) 1;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* Metric Text Adjustments */
[data-testid="stMetricValue"] {
    color: #ffffff;
    font-size: 2.4rem;
    font-weight: 700;
    margin-top: 5px;
    letter-spacing: -1px;
}
[data-testid="stMetricLabel"] {
    color: #94a3b8;
    font-weight: 500;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Glassmorphic Scroll Box for Logs */
.scroll-box {
    height: 250px;
    overflow-y: auto;
    background: rgba(10, 12, 18, 0.6);
    padding: 16px;
    border-radius: 12px;
    border: 1px solid transparent;
    font-family: 'Courier New', Courier, monospace;
    font-size: 0.85rem;
    color: #cbd5e1;
}
.scroll-box::-webkit-scrollbar { width: 6px; }
.scroll-box::-webkit-scrollbar-track { background: transparent; }
.scroll-box::-webkit-scrollbar-thumb {
    background: rgba(179, 102, 255, 0.3);
    border-radius: 10px;
}
.scroll-box::-webkit-scrollbar-thumb:hover { background: rgba(179, 102, 255, 0.6); }

/* Alert Text Colors */
.alert-high { color: #ff4b4b; font-weight: 600; }
.alert-warn { color: #ffab00; font-weight: 600; }
.alert-good { color: #00ff9f; font-weight: 500; }
.alert-info { color: #00c6ff; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# TITLE / HERO SECTION
# ---------------------------
st.markdown("""
<div style="padding: 30px 0 40px 0; text-align: center;">
    <h1 style="font-size: 3.2rem; font-weight: 700; line-height: 1.2; margin-top:0; color: #ffffff;">
        AI Predictive Maintenance
    </h1>
    <p style="color:#94a3b8; font-size:1.1rem; margin-top: 10px; font-weight: 300; letter-spacing: 0.5px;">
        Real-time telemetry, multi-sensor anomaly detection, & failure simulation
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# SIDEBAR
# ---------------------------
# Sleek SVG Gear Icon & Header
st.sidebar.markdown("""
<div style="display:flex; align-items:center; margin-bottom: 24px;">
    <div style="margin-right: 12px; display: flex; align-items: center; justify-content: center; width: 34px; height: 34px; border-radius: 8px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#00c6ff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="3"></circle>
            <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
        </svg>
    </div>
    <h2 style='margin:0; font-size: 1.15rem; font-weight: 600; color: #f8fafc !important;'>System Settings</h2>
</div>
""", unsafe_allow_html=True)

# Sidebar active state mockup
st.sidebar.markdown("""
<div style="background: rgba(255, 255, 255, 0.03); color: #e2e8f0; padding: 10px 16px; border-radius: 8px; font-weight: 500; border-left: 3px solid #b366ff; margin-bottom: 12px; box-shadow: inset 0 0 20px rgba(179, 102, 255, 0.05);">
    📊 Live Dashboard
</div>
<div style="color: #94a3b8; padding: 10px 16px; font-weight: 500; margin-bottom: 12px;">
    ⚙️ Simulation Controls
</div>
""", unsafe_allow_html=True)

speed = st.sidebar.slider("Simulation Speed", 0.5, 3.0, 1.0)
rows = st.sidebar.slider("Simulation Rows", 20, 200, 50)

st.sidebar.divider()
uploaded_file = st.sidebar.file_uploader("📂 Upload CSV Data")

st.sidebar.markdown("<br><br><br>", unsafe_allow_html=True)
st.sidebar.markdown(
    """
    <div style='color:#64748b; font-size: 0.85rem; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 15px; font-weight: 300;'>
        Platform Version 2.3.1<br>
        <span style='color: #94a3b8;'>Designed by <b style='color:#ffffff; font-weight: 500;'>Aniket satpathy</b></span>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# DATA
# ---------------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.DataFrame(
        np.random.randn(rows, 25),
        columns=[f"sensor_{i}" for i in range(25)]
    )

# ---------------------------
# LOAD MODEL
# ---------------------------
model = load_model()

# ---------------------------
# FUNCTIONS
# ---------------------------
def analyze_sensors(data):
    issues = []
    for col in data.columns:
        val = data[col].values[0]
        if abs(val) > 2:
            issues.append((col, "High"))
        elif abs(val) > 1:
            issues.append((col, "Moderate"))
    return issues

def health_score(prob):
    return int((1 - prob) * 100)

def generate_recommendation(issues, prob):
    if prob > 0.8:
        return "<span class='alert-high'>🔴 Immediate shutdown & full inspection required.</span>"
    elif prob > 0.6:
        return "<span class='alert-warn'>🟠 Schedule maintenance within 24 hours.</span>"
    elif prob > 0.4:
        return "<span style='color:#ffab00; font-weight: 500;'>🟡 Monitor system closely.</span>"
    
    if issues:
        return f"<span class='alert-info'>⚙️ Check {issues[0][0]} for anomalies.</span>"
    
    return "<span class='alert-good'>🟢 System operating normally. All parameters optimal.</span>"

# ---------------------------
# LAYOUT
# ---------------------------
kpi_container = st.container()
st.markdown("<br>", unsafe_allow_html=True)

col_left, col_right = st.columns([2, 1])

with col_left:
    chart_container = st.container()
    trend_container = st.container()
    log_container = st.container()

with col_right:
    sensor_container = st.container()
    critical_container = st.container()
    recommend_container = st.container()

# KPI
with kpi_container:
    c1, c2, c3 = st.columns(3)
    status_box = c1.empty()
    prob_box = c2.empty()
    health_box = c3.empty()

# Graph
with chart_container:
    st.markdown('<div class="block-title">📡 Live Sensor Telemetry</div>', unsafe_allow_html=True)
    chart_placeholder = st.empty()

# Failure Trend
with trend_container:
    st.markdown('<div class="block-title">📉 Failure Probability Trend</div>', unsafe_allow_html=True)
    trend_placeholder = st.empty()

# Sensor Panel
with sensor_container:
    st.markdown('<div class="block-title">🧠 Sensor Health Array</div>', unsafe_allow_html=True)
    sensor_placeholder = st.empty()

# Critical Sensors
with critical_container:
    st.markdown('<div class="block-title">⚠️ Critical Alerts</div>', unsafe_allow_html=True)
    critical_placeholder = st.empty()

# Recommendation
with recommend_container:
    st.markdown('<div class="block-title">✨ AI Recommendation</div>', unsafe_allow_html=True)
    recommendation_placeholder = st.empty()

# Logs
with log_container:
    st.markdown('<div class="block-title">📜 System Event Logs</div>', unsafe_allow_html=True)
    log_placeholder = st.empty()

logs = []
failure_history = []

# ---------------------------
# LOOP
# ---------------------------
for i in range(len(df)):

    row = df.iloc[[i]]

    try:
        pred, prob = predict(model, row)
        prob_val = prob[0][1]
        alert = generate_alert(prob)
    except:
        prob_val = np.random.rand()
        alert = "Simulation Mode"
        pred = [0]

    failure_history.append(prob_val)

    # KPI Updates
    status_box.metric("Machine Status", alert)
    prob_box.metric("Failure Probability", f"{prob_val:.2f}", delta=f"{'+' if i%2==0 else '-'}{np.random.uniform(0.01, 0.05):.2f} Variance", delta_color="inverse")
    health_box.metric("Health Score", f"{health_score(prob_val)}%", delta="Live Sync", delta_color="normal")

    # Graph (Explicitly use container width to prevent overflow)
    chart_placeholder.line_chart(df.iloc[:i+1], height=280, use_container_width=True)

    # Failure Trend 
    trend_df = pd.DataFrame(failure_history, columns=["Failure Probability"])
    trend_placeholder.line_chart(trend_df, height=200, use_container_width=True)

    # Sensor Analysis (Neon Orange Border + Glow)
    issues = analyze_sensors(row)

    if issues:
        sensor_text = "<br>".join([f"<span style='color:#cbd5e1;'>{s}</span> <span style='float:right;' class='{'alert-high' if lvl=='High' else 'alert-warn'}'>{lvl}</span>" for s, lvl in issues[:6]])
    else:
        sensor_text = "<span class='alert-good'>✓ All sensors nominal</span>"

    sensor_placeholder.markdown(f'<div class="glass-card bg-glow-orange" style="min-height: 180px;">{sensor_text}</div>', unsafe_allow_html=True)

    # Critical Sensors (Neon Red Border + Glow)
    critical = [x for x in issues if x[1] == "High"]

    if critical:
        critical_text = "<br>".join([f"🔴 <b class='alert-high'>{s}</b> immediate maintenance required" for s, _ in critical[:3]])
    else:
        critical_text = "<span style='color:#94a3b8;'>No critical issues detected in current tick.</span>"

    critical_placeholder.markdown(f'<div class="glass-card bg-glow-red" style="min-height: 140px;">{critical_text}</div>', unsafe_allow_html=True)

    # Recommendation (Neon Purple Border + Glow)
    recommendation = generate_recommendation(issues, prob_val)

    recommendation_placeholder.markdown(f"""
    <div class="glass-card bg-glow-purple" style="min-height: 120px;">
    {recommendation}
    </div>
    """, unsafe_allow_html=True)

    # Logs (Neon Blue Border + Glow)
    timestamp = time.strftime("%H:%M:%S")
    if prob_val > 0.7:
        logs.append(f"<span class='alert-high'>[{timestamp}] CRITICAL: High Risk detected at sequence {i}</span>")
    elif prob_val > 0.4:
        logs.append(f"<span class='alert-warn'>[{timestamp}] WARN: Threshold variance at sequence {i}</span>")
    else:
        logs.append(f"<span class='alert-good'>[{timestamp}] INFO: Telemetry nominal at sequence {i}</span>")

    logs = logs[-25:]

    log_placeholder.markdown(f"""
    <div class="glass-card bg-glow-blue" style="padding: 10px; margin-bottom: 0;">
        <div class="scroll-box">
            {'<br>'.join(logs[::-1])} 
        </div>
    </div>
    """, unsafe_allow_html=True)

    time.sleep(1 / speed)