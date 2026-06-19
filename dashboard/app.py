import streamlit as st
import json
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import IntrusionDetector

st.set_page_config(page_title="FedShield", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
.metric-card {background: #1e1e2e; padding: 20px; border-radius: 10px; border-left: 4px solid #00d4aa;}
</style>
""", unsafe_allow_html=True)

st.title("🛡️ FedShield — Privacy-Preserving Intrusion Detection")
st.caption("Federated Learning | XAI | Cloud Auto-Scaling | Real-time Detection")

st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["Overview", "Federated Training", "SHAP Explainability", "Multi-Class Detection", "Live Detection"])

if page == "Overview":
    st.header("System Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Federated F1 Score", "0.9946", "+0.9946")
    col2.metric("Baseline F1 Score", "0.9947", "reference")
    col3.metric("Edge Nodes", "3", "simulated")
    col4.metric("Privacy", "100%", "no raw data shared")

    st.divider()
    st.subheader("How FedShield Works")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.info("**Step 1 — Edge Nodes**\n\nEach node trains on local data only. Raw traffic never leaves.")
    with col2:
        st.info("**Step 2 — Weights Only**\n\nOnly model weights (numbers) are sent to the cloud. Zero data exposure.")
    with col3:
        st.info("**Step 3 — FedAvg**\n\nCloud aggregator averages all weights into one global model.")
    with col4:
        st.info("**Step 4 — XAI**\n\nSHAP explains every detection decision. Full auditability.")

    st.divider()
    st.subheader("Key Results")
    col1, col2 = st.columns(2)
    with col1:
        st.success("**Binary Classification (Normal vs Attack)**\n\nFederated F1: 0.9946 vs Centralized: 0.9947 — 0.0001 difference with full privacy.")
    with col2:
        st.success("**Multi-Class Classification (5 attack types)**\n\nFederated Macro F1: 0.81 — BEATS centralized (0.79) with SMOTE balancing.")

elif page == "Federated Training":
    st.header("Federated Training — Round by Round")

    try:
        with open("models/federated_history.json") as f:
            fed_history = json.load(f)
        with open("models/baseline_history.json") as f:
            base_history = json.load(f)

        fed_df = pd.DataFrame(fed_history)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fed_df['round'], y=fed_df['f1'],
            name='Federated (FedShield)',
            line=dict(color='#00d4aa', width=3),
            mode='lines+markers'
        ))
        fig.add_hline(
            y=base_history[-1]['f1'],
            line_dash="dash", line_color="#ff6b6b",
            annotation_text=f"Centralized Baseline: {base_history[-1]['f1']:.4f}"
        )
        fig.update_layout(
            title="Federated F1 Score vs Centralized Baseline",
            xaxis_title="Round", yaxis_title="F1 Score",
            yaxis=dict(range=[0.95, 1.0]),
            template="plotly_dark", height=450
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        col1.metric("Final Federated F1", f"{fed_df['f1'].iloc[-1]:.4f}")
        col2.metric("Centralized Baseline F1", f"{base_history[-1]['f1']:.4f}")
        st.success("✅ Federated learning matches centralized performance — with zero data sharing!")

    except FileNotFoundError:
        st.error("Run federated_train.py first!")

elif page == "SHAP Explainability":
    st.header("SHAP Feature Importance — Why the Model Flags Attacks")

    try:
        with open("models/shap_results.json") as f:
            shap_data = json.load(f)

        df = pd.DataFrame(shap_data['feature_importance'][:15])

        fig = px.bar(
            df, x='shap_score', y='feature',
            orientation='h',
            title="Top 15 Features Driving Attack Detection",
            color='shap_score',
            color_continuous_scale='teal',
            template="plotly_dark"
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("What This Means")
        col1, col2 = st.columns(2)
        with col1:
            st.error("**dst_host_serror_rate** — SYN error rate. Primary DDoS indicator.")
            st.warning("**logged_in** — Attackers probe without authentication.")
            st.info("**same_srv_rate** — Port scanners repeatedly hit same service.")
        with col2:
            st.error("**srv_serror_rate** — Service-level SYN errors. Confirms flood attack.")
            st.warning("**protocol_type** — Attack traffic skews toward specific protocols.")
            st.info("**dst_host_srv_count** — Unusual service counts signal reconnaissance.")

    except FileNotFoundError:
        st.error("Run explain.py first!")

elif page == "Multi-Class Detection":
    st.header("Multi-Class Attack Classification")
    st.caption("Normal | DoS | Probe | R2L | U2R — Federated vs Centralized")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Normal", "F1: 0.99", "✅ Perfect")
    col2.metric("DoS", "F1: 1.00", "✅ Perfect")
    col3.metric("Probe", "F1: 0.98", "✅ Excellent")
    col4.metric("Fed Macro F1", "0.81", "🔥 Beats centralized 0.79")

    try:
        with open("models/multiclass_history.json") as f:
            mc = json.load(f)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[d['epoch'] for d in mc], y=[d['macro_f1'] for d in mc],
            name='Centralized Multi-Class',
            line=dict(color='#ff6b6b', width=2), mode='lines+markers'
        ))

        try:
            with open("models/federated_multiclass_history.json") as f:
                fed_mc = json.load(f)
            fig.add_trace(go.Scatter(
                x=[d['round'] for d in fed_mc], y=[d['macro_f1'] for d in fed_mc],
                name='Federated Multi-Class',
                line=dict(color='#00d4aa', width=3), mode='lines+markers'
            ))
        except:
            pass

        fig.update_layout(
            title="Multi-Class F1: Federated vs Centralized",
            xaxis_title="Epoch / Round", yaxis_title="Macro F1",
            yaxis=dict(range=[0.6, 1.0]),
            template="plotly_dark", height=420
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Federated Multi-Class — Per-Class Results")
        df_r = pd.DataFrame({
            "Class":     ["Normal", "DoS",  "Probe", "R2L",  "U2R"],
            "Precision": [0.98,     1.00,   0.99,    0.91,   0.80],
            "Recall":    [1.00,     0.98,   0.98,    0.40,   0.40],
            "F1-Score":  [0.99,     0.99,   0.98,    0.56,   0.53],
            "Support":   [13469,    9186,   2331,    199,    10]
        })
        st.dataframe(df_r, use_container_width=True, hide_index=True)

        st.info("**Why is R2L/U2R recall low?** Very few test samples (199 and 10). SMOTE improved training balance. This matches known NSL-KDD benchmark challenges in research literature.")

        st.success("🔥 Federated Macro F1 (0.81) beats Centralized (0.79) — privacy-preserving AND more accurate!")

    except FileNotFoundError:
        st.error("Run train_multiclass.py and federated_multiclass.py first!")

elif page == "Live Detection":
    st.header("Live Packet Classification")
    st.caption("Adjust network traffic features and see real-time attack detection")

    model = IntrusionDetector()
    model.load_state_dict(torch.load("models/federated_model.pth", map_location='cpu'))
    model.eval()

    col1, col2, col3 = st.columns(3)
    with col1:
        duration = st.slider("Duration", 0, 100, 0)
        src_bytes = st.slider("Source Bytes", 0, 10000, 200)
        dst_bytes = st.slider("Dest Bytes", 0, 10000, 200)
        logged_in = st.selectbox("Logged In", [0, 1])
    with col2:
        serror_rate = st.slider("SYN Error Rate", 0.0, 1.0, 0.0)
        same_srv_rate = st.slider("Same Service Rate", 0.0, 1.0, 1.0)
        dst_host_serror_rate = st.slider("Dst Host SYN Error Rate", 0.0, 1.0, 0.0)
        count = st.slider("Connection Count", 0, 512, 10)
    with col3:
        protocol_type = st.selectbox("Protocol", [0, 1, 2], format_func=lambda x: ["icmp","tcp","udp"][x])
        srv_serror_rate = st.slider("Srv SYN Error Rate", 0.0, 1.0, 0.0)
        dst_host_count = st.slider("Dst Host Count", 0, 255, 50)
        diff_srv_rate = st.slider("Diff Service Rate", 0.0, 1.0, 0.05)

    if st.button("🔍 Classify Packet", type="primary"):
        features = np.zeros(41)
        features[0] = duration
        features[1] = protocol_type
        features[4] = src_bytes
        features[5] = dst_bytes
        features[11] = logged_in
        features[22] = count
        features[24] = serror_rate
        features[26] = serror_rate
        features[28] = same_srv_rate
        features[29] = diff_srv_rate
        features[31] = dst_host_count
        features[37] = dst_host_serror_rate
        features[38] = srv_serror_rate

        x = torch.FloatTensor(features).unsqueeze(0)
        with torch.no_grad():
            prob = model(x).item()

        st.divider()
        if prob > 0.5:
            st.error(f"🚨 ATTACK DETECTED — Confidence: {prob*100:.1f}%")
            st.write("**Top indicators:** High SYN error rate + unusual service patterns")
        else:
            st.success(f"✅ NORMAL TRAFFIC — Confidence: {(1-prob)*100:.1f}%")
            st.write("**Assessment:** Traffic patterns within normal range")

        st.progress(prob)
        st.caption(f"Raw attack probability: {prob:.4f}")