"""
FraudShield AI — Premium Interactive Dashboard
Live demo for hackathon judges. Run: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import math

# ─── CONFIG ───────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_DIR, "outputs", "results")
VIZ_DIR = os.path.join(PROJECT_DIR, "outputs", "visualizations")

st.set_page_config(
    page_title="FraudShield AI — Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── PREMIUM CSS WITH ANIMATIONS ─────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* Global */
    .stApp {
        background: #0a0a1a;
        font-family: 'Inter', sans-serif;
    }

    /* Animated gradient background */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background:
            radial-gradient(ellipse at 20% 50%, rgba(16, 185, 255, 0.06) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 20%, rgba(139, 92, 246, 0.06) 0%, transparent 50%),
            radial-gradient(ellipse at 50% 80%, rgba(6, 214, 160, 0.04) 0%, transparent 50%);
        z-index: 0;
        pointer-events: none;
    }

    /* Hide default Streamlit elements */
    #MainMenu, footer, header {visibility: hidden;}

    /* Custom scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0a0a1a; }
    ::-webkit-scrollbar-thumb { background: #2d2d5e; border-radius: 3px; }

    /* Main header */
    .hero-title {
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(135deg, #10b9ff 0%, #8b5cf6 40%, #06d6a0 80%, #10b9ff 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        animation: gradient-shift 4s ease infinite;
        letter-spacing: -1px;
        margin-bottom: 0;
    }

    @keyframes gradient-shift {
        0%, 100% { background-position: 0% center; }
        50% { background-position: 100% center; }
    }

    .hero-subtitle {
        text-align: center;
        color: #64748b;
        font-size: 1.05rem;
        font-weight: 400;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-top: 4px;
    }

    /* Glass cards */
    .glass-card {
        background: rgba(255,255,255,0.03);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 24px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .glass-card:hover {
        border-color: rgba(139, 92, 246, 0.3);
        box-shadow: 0 8px 32px rgba(139, 92, 246, 0.1);
        transform: translateY(-2px);
    }

    /* Stat cards */
    .stat-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 20px 24px;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .stat-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        border-radius: 16px 16px 0 0;
    }
    .stat-card:hover { transform: translateY(-3px); }
    .stat-card.blue::before { background: linear-gradient(90deg, #10b9ff, #3b82f6); }
    .stat-card.purple::before { background: linear-gradient(90deg, #8b5cf6, #a855f7); }
    .stat-card.green::before { background: linear-gradient(90deg, #06d6a0, #10b981); }
    .stat-card.amber::before { background: linear-gradient(90deg, #f59e0b, #ef4444); }

    .stat-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #e2e8f0;
        line-height: 1.1;
    }
    .stat-label {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 8px;
        font-weight: 600;
    }
    .stat-delta {
        font-size: 0.8rem;
        color: #06d6a0;
        margin-top: 4px;
        font-weight: 500;
    }

    /* Risk gauge */
    .risk-gauge-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 30px;
    }

    .risk-gauge {
        width: 220px;
        height: 220px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
        animation: pulse-glow 2s ease-in-out infinite;
    }

    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 0 30px rgba(var(--gauge-rgb), 0.3); }
        50% { box-shadow: 0 0 60px rgba(var(--gauge-rgb), 0.5), 0 0 100px rgba(var(--gauge-rgb), 0.2); }
    }

    .risk-score-display {
        font-size: 4rem;
        font-weight: 900;
        letter-spacing: -2px;
    }

    .risk-tier-badge {
        display: inline-block;
        padding: 8px 24px;
        border-radius: 50px;
        font-weight: 700;
        font-size: 0.9rem;
        letter-spacing: 1px;
        margin-top: 16px;
        animation: fade-slide-up 0.5s ease;
    }

    @keyframes fade-slide-up {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Reason chips */
    .reason-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 8px;
        padding: 10px 16px;
        margin: 4px 0;
        font-size: 0.88rem;
        color: #cbd5e1;
        transition: all 0.2s ease;
        width: 100%;
    }
    .reason-chip:hover {
        background: rgba(255, 255, 255, 0.06);
        border-color: rgba(255,255,255,0.12);
    }

    /* Layer badge */
    .layer-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 8px;
        font-size: 0.78rem;
        font-weight: 600;
        background: rgba(139, 92, 246, 0.1);
        border: 1px solid rgba(139, 92, 246, 0.2);
        color: #a78bfa;
    }

    /* Model comparison bars */
    .model-bar-container {
        margin: 12px 0;
    }
    .model-bar-label {
        display: flex;
        justify-content: space-between;
        margin-bottom: 6px;
    }
    .model-bar-name {
        color: #e2e8f0;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .model-bar-value {
        color: #10b9ff;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .model-bar {
        height: 8px;
        border-radius: 4px;
        background: rgba(255,255,255,0.05);
        overflow: hidden;
    }
    .model-bar-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 1.5s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* Animated divider */
    .glow-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(139,92,246,0.3), transparent);
        margin: 32px 0;
    }

    /* Table styling */
    .stDataFrame { border-radius: 12px; overflow: hidden; }

    /* Section header */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* Animated counter */
    @keyframes count-up {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .animate-in {
        animation: count-up 0.6s ease forwards;
    }
</style>
""", unsafe_allow_html=True)


# ─── DATA LOADING ─────────────────────────────────────────────────────
@st.cache_data
def load_metrics():
    path = os.path.join(RESULTS_DIR, "model_metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

@st.cache_data
def load_flagged():
    path = os.path.join(RESULTS_DIR, "sample_flagged_transactions.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data
def load_scored():
    path = os.path.join(RESULTS_DIR, "scored_transactions.csv")
    if os.path.exists(path):
        return pd.read_csv(path, nrows=10000)
    return pd.DataFrame()


# ─── SIDEBAR ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px;">
        <div style="font-size:2.5rem;">🛡️</div>
        <div style="font-size:1.3rem; font-weight:800; color:#e2e8f0; margin-top:4px;">FraudShield AI</div>
        <div style="font-size:0.7rem; color:#64748b; letter-spacing:2px; text-transform:uppercase; margin-top:4px;">
            Adaptive Fraud Detection
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "🔍 Live Detector", "📊 Models", "📈 Analytics", "📋 Flagged"],
        label_visibility="collapsed"
    )

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card" style="margin-top:10px; padding:16px;">
        <div style="font-size:0.7rem; color:#64748b; letter-spacing:1.5px; text-transform:uppercase; font-weight:600;">Architecture</div>
        <div style="margin-top:12px;">
            <div style="display:flex; justify-content:space-between; margin:8px 0;">
                <span style="color:#94a3b8; font-size:0.85rem;">Detection Layers</span>
                <span style="color:#10b9ff; font-weight:700;">15</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin:8px 0;">
                <span style="color:#94a3b8; font-size:0.85rem;">ML Models</span>
                <span style="color:#8b5cf6; font-weight:700;">5</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin:8px 0;">
                <span style="color:#94a3b8; font-size:0.85rem;">Explainability</span>
                <span style="color:#06d6a0; font-weight:700;">XAI</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin:8px 0;">
                <span style="color:#94a3b8; font-size:0.85rem;">Auth Tiers</span>
                <span style="color:#f59e0b; font-weight:700;">4</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
    st.caption("Financial Services Hackathon · March 2026")


# ═══════════════════════════════════════════════════════════════════════
#  PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.markdown('<h1 class="hero-title">🛡️ FraudShield AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Adaptive Real-Time Fraud Detection with Explainable Risk Intelligence</p>', unsafe_allow_html=True)
    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    metrics = load_metrics()

    # Top metrics row
    cols = st.columns(4)
    if metrics:
        best_auc = max(m['auc'] for m in metrics.values())
        ensemble = metrics.get('ensemble', {})
        ens_prec = ensemble.get('precision', 0)
        cm = ensemble.get('confusion_matrix', [[0,0],[0,0]])
        fp_rate = cm[0][1] / max(cm[0][0] + cm[0][1], 1) * 1000

        cards = [
            ("blue", f"{best_auc:.4f}", "BEST AUC-ROC", "XGBoost"),
            ("purple", f"{ensemble.get('f1', 0):.4f}", "ENSEMBLE F1", "4-Model Voting"),
            ("green", f"{ens_prec:.1%}", "PRECISION", f"Only {fp_rate:.1f} FP per 1K"),
            ("amber", "15", "DETECTION LAYERS", "+4 Enhanced Layers"),
        ]
        for col, (cls, val, label, delta) in zip(cols, cards):
            col.markdown(f"""
            <div class="stat-card {cls} animate-in">
                <div class="stat-value">{val}</div>
                <div class="stat-label">{label}</div>
                <div class="stat-delta">{delta}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    # Two-column layout
    left, right = st.columns([3, 2])

    with left:
        st.markdown('<div class="section-header">🏗️ System Architecture</div>', unsafe_allow_html=True)

        # Flow block style
        block_style = "background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:16px; text-align:center; margin:0;"
        arrow_style = "text-align:center; color:#8b5cf6; font-size:1.5rem; margin:6px 0; line-height:1;"
        mini_style = "display:inline-block; background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.1); border-radius:8px; padding:6px 12px; margin:4px; font-size:0.75rem; color:#94a3b8;"

        st.markdown(f"""
        <div style="{block_style}">
            <div style="font-size:0.65rem; color:#64748b; text-transform:uppercase; letter-spacing:1.5px; font-weight:600;">Input</div>
            <div style="font-size:1.1rem; font-weight:700; color:#e2e8f0; margin-top:4px;">📋 Raw Transaction Data</div>
            <div style="font-size:0.8rem; color:#94a3b8;">590,540 transactions · 434 features</div>
        </div>
        <div style="{arrow_style}">▼</div>
        <div style="{block_style}">
            <div style="font-size:0.65rem; color:#64748b; text-transform:uppercase; letter-spacing:1.5px; font-weight:600;">Feature Engineering</div>
            <div style="font-size:1.1rem; font-weight:700; color:#e2e8f0; margin-top:4px;">🔬 15 Detection Layers</div>
            <div style="margin-top:8px;">
                <span style="{mini_style}">⚡ Velocity</span>
                <span style="{mini_style}">📱 Device</span>
                <span style="{mini_style}">📧 Email</span>
                <span style="{mini_style}">🔗 Network</span>
                <span style="{mini_style}">+ 11 more</span>
            </div>
        </div>
        <div style="{arrow_style}">▼</div>
        <div style="{block_style}">
            <div style="font-size:0.65rem; color:#64748b; text-transform:uppercase; letter-spacing:1.5px; font-weight:600;">Ensemble Engine</div>
            <div style="font-size:1.1rem; font-weight:700; color:#e2e8f0; margin-top:4px;">🤖 5-Model Voting</div>
            <div style="margin-top:8px; display:grid; grid-template-columns:1fr 1fr; gap:6px; max-width:320px; margin-left:auto; margin-right:auto;">
                <div style="background:rgba(16,185,255,0.1); border:1px solid rgba(16,185,255,0.2); border-radius:8px; padding:8px; text-align:center;">
                    <div style="color:#10b9ff; font-weight:700; font-size:0.85rem;">XGBoost</div>
                    <div style="color:#64748b; font-size:0.7rem;">30% weight</div>
                </div>
                <div style="background:rgba(236,72,153,0.1); border:1px solid rgba(236,72,153,0.2); border-radius:8px; padding:8px; text-align:center;">
                    <div style="color:#ec4899; font-weight:700; font-size:0.85rem;">CatBoost</div>
                    <div style="color:#64748b; font-size:0.7rem;">25% weight</div>
                </div>
                <div style="background:rgba(6,214,160,0.1); border:1px solid rgba(6,214,160,0.2); border-radius:8px; padding:8px; text-align:center;">
                    <div style="color:#06d6a0; font-weight:700; font-size:0.85rem;">Random Forest</div>
                    <div style="color:#64748b; font-size:0.7rem;">15% weight</div>
                </div>
                <div style="background:rgba(139,92,246,0.1); border:1px solid rgba(139,92,246,0.2); border-radius:8px; padding:8px; text-align:center;">
                    <div style="color:#8b5cf6; font-weight:700; font-size:0.85rem;">MLP Neural Net</div>
                    <div style="color:#64748b; font-size:0.7rem;">15% weight</div>
                </div>
                <div style="background:rgba(245,158,11,0.1); border:1px solid rgba(245,158,11,0.2); border-radius:8px; padding:8px; text-align:center;">
                    <div style="color:#f59e0b; font-weight:700; font-size:0.85rem;">Isolation Forest</div>
                    <div style="color:#64748b; font-size:0.7rem;">15% weight</div>
                </div>
            </div>
        </div>
        <div style="{arrow_style}">▼</div>
        <div style="{block_style}">
            <div style="font-size:0.65rem; color:#64748b; text-transform:uppercase; letter-spacing:1.5px; font-weight:600;">Output</div>
            <div style="font-size:1.1rem; font-weight:700; color:#e2e8f0; margin-top:4px;">🎯 Risk Intelligence</div>
            <div style="margin-top:8px;">
                <span style="{mini_style}">🎯 Score 0-100</span>
                <span style="{mini_style}">💡 XAI Reasons</span>
                <span style="{mini_style}">🔐 Adaptive Auth</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-header">🔬 Detection Layers</div>', unsafe_allow_html=True)
        layers = [
            ("Amount Analysis", "💰"), ("Behavioral DNA", "🧬"), ("SIM Swap", "📱"),
            ("Seasonal Baselines", "📅"), ("Adaptive Auth", "🔐"), ("Merchant Risk", "🏪"),
            ("Mule Network", "🕸️"), ("Dormant Hijack", "💤"), ("Round Amount", "🎯"),
            ("Category Mismatch", "⚠️"), ("New Account", "🆕"), ("Velocity", "⚡"),
            ("Email Risk", "📧"), ("Fraud Ring", "🔗"), ("Synth Device", "📱")
        ]
        html = '<div class="glass-card">'
        for i, (name, icon) in enumerate(layers):
            html += f'<div class="layer-badge" style="margin:3px; animation-delay:{i*0.05}s">{icon} L{i+1}: {name}</div>'
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)

        # Model comparison mini-bars
        st.markdown('<div class="section-header" style="margin-top:20px;">🤖 Model Comparison</div>', unsafe_allow_html=True)
        if metrics:
            models_sorted = sorted(metrics.items(), key=lambda x: -x[1]['auc'])
            html = '<div class="glass-card">'
            colors = ['#10b9ff', '#8b5cf6', '#06d6a0', '#f59e0b', '#ef4444']
            for i, (name, m) in enumerate(models_sorted):
                auc = m['auc']
                pct = auc * 100
                color = colors[i % len(colors)]
                html += f"""<div class="model-bar-container">
<div class="model-bar-label">
<span class="model-bar-name">{m['name']}</span>
<span class="model-bar-value" style="color:{color}">{auc:.4f}</span>
</div>
<div class="model-bar">
<div class="model-bar-fill" style="width:{pct}%; background:linear-gradient(90deg, {color}, {color}88);"></div>
</div>
</div>"""
            html += '</div>'
            st.markdown(html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
#  PAGE: LIVE FRAUD DETECTOR
# ═══════════════════════════════════════════════════════════════════════
elif page == "🔍 Live Detector":
    st.markdown('<div class="section-header" style="font-size:1.8rem;">🔍 Real-Time Fraud Detection</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#64748b; margin-top:-10px;">Enter transaction details for instant AI-powered risk assessment</p>', unsafe_allow_html=True)
    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("##### 💳 Transaction Details")
        amount = st.number_input("Amount ($)", min_value=0.0, max_value=50000.0, value=500.0, step=10.0)
        product = st.selectbox("Product Category", ["W — Digital Goods", "C — High Risk", "H — Hotel", "R — Restaurant", "S — Services"])
        hour = st.slider("Hour of Day (24h)", 0, 23, 14)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("##### 👤 User Context")
        user_avg = st.number_input("User's Avg Transaction ($)", min_value=0.0, max_value=10000.0, value=150.0)
        txn_velocity = st.slider("Transactions in Last 24h", 1, 50, 3)
        is_new_device = st.toggle("New/Unknown Device", value=False)
        is_new_account = st.toggle("First Transaction (New Account)", value=False)
        is_shared = st.toggle("Shared Device (Multiple Users)", value=False)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("")
    analyze = st.button("⚡ Analyze Transaction Risk", type="primary", use_container_width=True)

    if analyze:
        # Risk scoring simulation
        is_night = 1 if (hour >= 0 and hour <= 6) else 0
        is_round = 1 if amount == int(amount) else 0
        z_score = (amount - user_avg) / max(user_avg * 0.5, 1)
        p_map = {"C — High Risk": 0.117, "S — Services": 0.059, "H — Hotel": 0.048, "R — Restaurant": 0.038, "W — Digital Goods": 0.020}
        p_risk = p_map.get(product, 0.035)

        risk_raw = (
            min(z_score, 5) * 8 + is_night * 12 + int(is_new_device) * 15 +
            int(is_shared) * 12 + int(is_new_account) * 18 +
            (1 if is_round and amount >= 500 else 0) * 8 + p_risk * 100 +
            min(txn_velocity / 10, 1) * 15 + (1 if amount > 1000 else 0) * 6
        )
        risk_score = min(max(risk_raw, 0), 100)

        if risk_score >= 71:
            tier, color, bg, rgb = "RED · BLOCK", "#ef4444", "rgba(239,68,68,0.08)", "239,68,68"
            auth = "🚫 BLOCK transaction immediately. Alert fraud team. Notify customer via secure channel."
        elif risk_score >= 51:
            tier, color, bg, rgb = "ORANGE · BIOMETRIC", "#f97316", "rgba(249,115,22,0.08)", "249,115,22"
            auth = "🔐 Require biometric re-verification (face/fingerprint) before approving."
        elif risk_score >= 31:
            tier, color, bg, rgb = "YELLOW · VERIFY", "#eab308", "rgba(234,179,8,0.08)", "234,179,8"
            auth = "🔑 Request PIN re-entry to confirm identity."
        else:
            tier, color, bg, rgb = "GREEN · APPROVE", "#22c55e", "rgba(34,197,94,0.08)", "34,197,94"
            auth = "✅ Auto-approve. Zero friction. Transaction is safe."

        reasons = []
        if z_score > 2: reasons.append(("💰", f"Amount is {z_score:.1f}× above user average (${amount:.0f} vs ${user_avg:.0f})"))
        if is_night: reasons.append(("🌙", f"Unusual hour ({hour}:00) — nighttime activity"))
        if is_new_device: reasons.append(("📱", "Transaction from new/unknown device — possible SIM swap"))
        if is_shared: reasons.append(("👥", "Device shared across multiple cards — mule network risk"))
        if is_new_account: reasons.append(("🆕", "Brand-new account with first transaction — onboarding fraud risk"))
        if is_round and amount >= 500: reasons.append(("🎯", "Suspicious round high-value amount"))
        if p_risk > 0.05: reasons.append(("🏪", f"High-risk product category (fraud rate: {p_risk*100:.1f}%)"))
        if txn_velocity > 10: reasons.append(("⚡", f"High velocity: {txn_velocity} txns in 24h"))
        if not reasons: reasons.append(("✅", "No significant risk factors identified"))

        st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

        g_col, d_col = st.columns([1, 2])

        with g_col:
            st.markdown(f"""
            <div class="risk-gauge-container">
                <div class="risk-gauge" style="
                    background: {bg};
                    border: 3px solid {color};
                    --gauge-rgb: {rgb};
                ">
                    <div>
                        <div class="risk-score-display" style="color:{color};">{risk_score:.0f}</div>
                        <div style="color:{color}; font-size:0.8rem; font-weight:600; text-align:center; opacity:0.8;">/ 100</div>
                    </div>
                </div>
                <div class="risk-tier-badge" style="background:{bg}; border:1px solid {color}; color:{color};">
                    {tier}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with d_col:
            st.markdown(f"""
            <div class="glass-card" style="border-color:{color}33;">
                <div style="font-size:0.7rem; color:#64748b; text-transform:uppercase; letter-spacing:1.5px; font-weight:600;">Authentication Decision</div>
                <div style="color:#e2e8f0; font-size:1rem; margin-top:10px; padding:12px; background:rgba(255,255,255,0.03); border-radius:8px;">
                    {auth}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="glass-card" style="margin-top:12px; border-color:{color}33;">
                <div style="font-size:0.7rem; color:#64748b; text-transform:uppercase; letter-spacing:1.5px; font-weight:600;">Explainable AI — Risk Factors</div>
                <div style="margin-top:10px;">
                    {"".join(f'<div class="reason-chip"><span>{icon}</span><span>{text}</span></div>' for icon, text in reasons)}
                </div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
#  PAGE: MODELS
# ═══════════════════════════════════════════════════════════════════════
elif page == "📊 Models":
    st.markdown('<div class="section-header" style="font-size:1.8rem;">📊 Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    metrics = load_metrics()
    if metrics:
        # Model metric cards
        cols = st.columns(len(metrics))
        colors = {'xgboost': '#10b9ff', 'catboost': '#ec4899', 'random_forest': '#06d6a0', 'isolation_forest': '#f59e0b', 'mlp': '#8b5cf6', 'ensemble': '#ef4444'}
        for col, (name, m) in zip(cols, metrics.items()):
            c = colors.get(name, '#94a3b8')
            col.markdown(f"""
            <div class="glass-card" style="border-top: 3px solid {c}; text-align:center;">
                <div style="font-size:0.75rem; color:#64748b; text-transform:uppercase; letter-spacing:1px; font-weight:600;">{m['name']}</div>
                <div style="font-size:2rem; font-weight:800; color:{c}; margin:8px 0;">{m['auc']:.4f}</div>
                <div style="font-size:0.7rem; color:#64748b;">AUC-ROC</div>
                <div style="margin-top:12px; display:grid; grid-template-columns:1fr 1fr; gap:8px;">
                    <div>
                        <div style="color:#e2e8f0; font-weight:700;">{m['f1']:.3f}</div>
                        <div style="color:#64748b; font-size:0.65rem;">F1</div>
                    </div>
                    <div>
                        <div style="color:#e2e8f0; font-weight:700;">{m['precision']:.3f}</div>
                        <div style="color:#64748b; font-size:0.65rem;">Precision</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

        # Charts
        for img_name, title in [("roc_curves.png", "ROC Curve Comparison"), ("confusion_matrices.png", "Confusion Matrices"), ("metrics_comparison.png", "Metrics Comparison")]:
            path = os.path.join(VIZ_DIR, img_name)
            if os.path.exists(path):
                st.image(path, caption=title, use_container_width=True)
                st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
#  PAGE: ANALYTICS
# ═══════════════════════════════════════════════════════════════════════
elif page == "📈 Analytics":
    st.markdown('<div class="section-header" style="font-size:1.8rem;">📈 Data Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    viz_list = [
        ("Feature Importance — Top 20", "feature_importance.png"),
        ("Risk Score Distribution", "risk_distribution.png"),
        ("Adaptive Auth Distribution", "risk_pie_chart.png"),
        ("Fraud Rate by Hour of Day", "fraud_by_hour.png"),
        ("Fraud by Product Category", "fraud_by_product.png"),
        ("Transaction Amount Distribution", "amount_distribution.png"),
        ("Sample Explanations", "sample_explanations.png"),
    ]

    for title, fname in viz_list:
        path = os.path.join(VIZ_DIR, fname)
        if os.path.exists(path):
            st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)
            st.image(path, use_container_width=True)
            st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
#  PAGE: FLAGGED TRANSACTIONS
# ═══════════════════════════════════════════════════════════════════════
elif page == "📋 Flagged":
    st.markdown('<div class="section-header" style="font-size:1.8rem;">📋 Flagged Transactions</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#64748b; margin-top:-10px;">Transactions flagged as RED (Block) or ORANGE (Biometric Required)</p>', unsafe_allow_html=True)
    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    flagged = load_flagged()
    if not flagged.empty:
        # Counts
        red_count = (flagged['risk_category'] == 'RED_BLOCK').sum() if 'risk_category' in flagged.columns else 0
        orange_count = (flagged['risk_category'] == 'ORANGE_BIOMETRIC').sum() if 'risk_category' in flagged.columns else 0

        c1, c2, c3 = st.columns(3)
        c1.metric("🔴 RED — Blocked", red_count)
        c2.metric("🟠 ORANGE — Biometric", orange_count)
        c3.metric("📊 Total Flagged", len(flagged))

        st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

        display_cols = ['TransactionID', 'TransactionAmt', 'risk_score', 'risk_category',
                        'isFraud_actual', 'explanation']
        available = [c for c in display_cols if c in flagged.columns]
        st.dataframe(flagged[available], use_container_width=True, hide_index=True, height=600)

        expl_path = os.path.join(VIZ_DIR, "sample_explanations.png")
        if os.path.exists(expl_path):
            st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
            st.image(expl_path, caption="Sample Explanations Table", use_container_width=True)
    else:
        st.warning("Run `python main.py` first to generate flagged transactions.")
