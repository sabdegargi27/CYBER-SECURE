import streamlit as st
import pandas as pd
import joblib
import time
import datetime
import os
import hashlib
import random
import subprocess
import sys
import matplotlib.pyplot as plt

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="CyberSecure: AI Triage",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

# --- 2. CUSTOM CSS (FIXED BUTTON SELECTORS) ---
st.markdown("""
<style>
    /* FORCE DARK MODE BACKGROUND */
    .stApp {
        background-color: #0E1117 !important;
        color: #FAFAFA !important;
    }
    
    /* GLOBAL TEXT COLOR */
    h1, h2, h3, h4, h5, h6, p, div, span, label, li {
        color: #FAFAFA !important;
    }
    
    /* --- BUTTON STYLING (Fixed) --- */
    
    /* 1. Primary Buttons (The Red Ones) */
    /* Streamlit uses data-testid="baseButton-primary" for type="primary" */
    button[data-testid="baseButton-primary"] {
        background-color: #FF4B4B !important;
        border: none !important;
    }
    button[data-testid="baseButton-primary"] * {
        color: white !important;
        font-weight: bold !important;
    }

    /* 2. Secondary Buttons (The White Ones) */
    /* Streamlit uses data-testid="baseButton-secondary" for type="secondary" (default) */
    button[data-testid="baseButton-secondary"] {
        background-color: #FFFFFF !important;
        border: 1px solid #ccc !important;
    }
    /* Force Text Black inside Secondary Buttons */
    button[data-testid="baseButton-secondary"] * {
        color: #000000 !important;
        font-weight: 600 !important;
    }

    /* METRICS CARDS */
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        color: #00FFCC !important; /* Cyber Green */
        font-weight: 700;
    }
    div[data-testid="stMetricLabel"] p {
        color: #A0A0A0 !important;
        font-size: 14px !important;
    }
    .stMetric {
        background-color: #1A1C24 !important;
        border: 1px solid #444 !important;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.5);
    }
    
    /* TABLE STYLING */
    td {
        color: #FAFAFA !important;
    }
    /* Attack Type Column (Red) */
    td:nth-child(4) {
        font-weight: bold;
        color: #FF4B4B !important;
    }
    
    /* EXPANDER HEADER */
    .streamlit-expanderHeader {
        background-color: #262730 !important;
        border-radius: 5px;
    }
    .streamlit-expanderHeader p {
        color: white !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. CORE CLASSES & LOGIC ---
class Block:
    def __init__(self, index, flow_index, timestamp, details, prediction, attack_type, confidence, action, reason, prev_hash):
        self.index = index
        self.flow_index = flow_index
        self.timestamp = timestamp
        self.details = details
        self.prediction = prediction
        self.attack_type = attack_type
        self.confidence = confidence
        self.action = action
        self.reason = reason
        self.prev_hash = prev_hash
        self.hash = self.calculate_hash()
    
    def calculate_hash(self):
        s = f"{self.index}{self.flow_index}{self.timestamp}{self.details}{self.prediction}{self.attack_type}{self.confidence}{self.action}{self.reason}{self.prev_hash}"
        return hashlib.sha256(s.encode()).hexdigest()

def create_genesis_block():
    return Block(0, 0, str(datetime.datetime.now()), "GENESIS", "INIT", "System Start", 1.0, "START", "Initialization", "0")

def get_security_action(attack_type, confidence):
    attack_type = str(attack_type).lower()
    if "normal" in attack_type or "benign" in attack_type:
        return "✅ Log & Allow", "Traffic matches benign profile."
    
    if "dos" in attack_type:
        return "⛔ HIGH: BLOCK SOURCE IP", f"DoS pattern detected ({confidence:.1%})."
    elif "probe" in attack_type or "port" in attack_type:
        return "⚠️ WARNING: DROP PACKETS", "Scanning activity detected."
    elif "ransomware" in attack_type:
        return "🔥 CRITICAL: ISOLATE HOST", "Ransomware signature match."
    else:
        if confidence > 0.9: return "⛔ BLOCK IP", "High confidence anomaly."
        else: return "👀 FLAG FOR DPI", "Suspicious pattern."

# --- 4. ASSET LOADING ---
@st.cache_resource
def load_assets():
    try:
        if not os.path.exists("models/rf_model.pkl"): return None, None, None, None
        model = joblib.load("models/rf_model.pkl")
        le = joblib.load("models/label_encoder.pkl") if os.path.exists("models/label_encoder.pkl") else None
        model_cols = joblib.load("models/columns.pkl") if os.path.exists("models/columns.pkl") else None
        
        if os.path.exists("data/test_samples.csv"):
            data = pd.read_csv("data/test_samples.csv").sample(frac=1).reset_index(drop=True)
        elif os.path.exists("cleaned_data.csv"):
            data = pd.read_csv("cleaned_data.csv", nrows=500).sample(frac=1).reset_index(drop=True)
        else: return None, None, None, None
            
        return model, le, model_cols, data
    except Exception: return None, None, None, None

model, label_encoder, model_cols, simulation_data = load_assets()

# --- 5. STATE MANAGEMENT ---
if 'blockchain' not in st.session_state: st.session_state['blockchain'] = [create_genesis_block()]
if 'row_index' not in st.session_state: st.session_state['row_index'] = 0
if 'is_running' not in st.session_state: st.session_state['is_running'] = False
if 'chart_data' not in st.session_state: st.session_state['chart_data'] = []
if 'last_confidence' not in st.session_state: st.session_state['last_confidence'] = 0.99

# --- 6. NAVIGATION SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/shield.png", width=60)
    st.title("CyberSecure")
    st.caption("v5.1 | Enterprise Edition")
    
    menu = st.radio(
        "Main Navigation", 
        ["🏠 Home", "🛡️ Live Monitor", "☠️ Threat Vault", "📂 Data Lab", "🧠 Intelligence"],
        index=0
    )
    
    st.divider()
    st.info("System Status: ONLINE")

# --- 7. PAGE VIEWS ---

# === PAGE 1: HOME ===
if menu == "🏠 Home":
    st.title("CyberSecure AI: Network Triage System")
    st.markdown("### The First Line of Defense Against Modern Threats")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **Problem:** SOC Analysts are drowning in alerts.  
        **Solution:** AI-driven Triage with Blockchain Audit Trails.
        
        #### Key Capabilities:
        * 🧠 **AI Core:** 99.7% Recall Random Forest Model
        * 🔗 **Trust:** SHA-256 Blockchain Ledger
        * ⚡ **Speed:** <50ms Decision Time
        """)
        if st.button("🚀 Launch Mission Control", type="primary"):
            st.write("Select 'Live Monitor' in the sidebar.")
            
    with col2:
        st.metric("Total Packets Analyzed", len(st.session_state['blockchain']))
        st.metric("Active Threats Blocked", sum(1 for b in st.session_state['blockchain'] if b.prediction == "MALICIOUS"))

# === PAGE 2: LIVE MONITOR (THE DASHBOARD) ===
elif menu == "🛡️ Live Monitor":
    st.title("🛡️ Security Operations Center")
    
    # --- CONTROL PANEL ---
    with st.expander("🎛️ Mission Control (Remote)", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            # FIX: Changed kind="primary" to type="primary"
            if st.button("▶️ START FEED", type="primary", use_container_width=True):
                st.session_state['is_running'] = True
        with c2:
            # FIX: Changed kind="secondary" to type="secondary"
            if st.button("⏸️ PAUSE FEED", type="secondary", use_container_width=True):
                st.session_state['is_running'] = False
        with c3:
            if st.button("🔴 INJECT ATTACK", type="secondary", use_container_width=True):
                prev = st.session_state['blockchain'][-1]
                bad_block = Block(len(st.session_state['blockchain']), st.session_state['row_index'] + 1000, str(datetime.datetime.now()), "Src: 10.10.99.5", "MALICIOUS", "Ransomware", 0.99, "🔥 ISOLATE", "Manual Injection", prev.hash)
                st.session_state['blockchain'].append(bad_block)
                st.session_state['chart_data'].append(0.99)
                st.session_state['last_confidence'] = 0.99
                st.rerun()
        with c4:
            if st.button("🗑️ RESET LOGS", type="secondary", use_container_width=True):
                st.session_state['blockchain'] = [create_genesis_block()]
                st.session_state['chart_data'] = []
                st.rerun()
        
        speed = st.slider("Scan Speed", 0.0, 1.0, 0.2, label_visibility="collapsed")

    # --- LIVE METRICS ---
    threats = sum(1 for b in st.session_state['blockchain'] if b.prediction == "MALICIOUS")
    current_idx = st.session_state['row_index'] + 1
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Flow Index", current_idx)
    m2.metric("Threats Blocked", threats, delta_color="inverse")
    m3.metric("Network Status", "CRITICAL" if threats > 5 else "SECURE", delta_color="inverse" if threats > 5 else "normal")
    m4.metric("Live AI Confidence", f"{st.session_state['last_confidence']:.1%}")
    
    # Live Graph
    if st.session_state['chart_data']:
        st.area_chart(st.session_state['chart_data'][-50:], height=120, color="#FF4B4B")
    
    # Live Processing Loop
    if st.session_state['is_running'] and model is not None:
        row = simulation_data.iloc[st.session_state['row_index']]
        input_df = pd.DataFrame([row.drop('Attack Type', errors='ignore')])
        if model_cols: input_df = input_df.reindex(columns=model_cols, fill_value=0)
        else: input_df = input_df.values.reshape(1, -1)
        
        pred_idx = model.predict(input_df)[0]
        probs = model.predict_proba(input_df)[0]
        confidence = probs.max()
        st.session_state['last_confidence'] = confidence
        
        if label_encoder: attack_name = label_encoder.inverse_transform([pred_idx])[0]
        else: attack_name = "Malicious" if pred_idx == 1 else "Normal"
        
        status = "BENIGN" if "normal" in str(attack_name).lower() else "MALICIOUS"
        action, reason = get_security_action(attack_name, confidence)
        prev_hash = st.session_state['blockchain'][-1].hash
        info_val = input_df.iloc[0, 1] if hasattr(input_df, 'iloc') else "N/A"
        
        new_block = Block(len(st.session_state['blockchain']), current_idx, str(datetime.datetime.now()), f"Info: {info_val}", status, attack_name, confidence, action, reason, prev_hash)
        st.session_state['blockchain'].append(new_block)
        
        chart_val = confidence if status == "MALICIOUS" else 0
        st.session_state['chart_data'].append(chart_val)
        
        st.session_state['row_index'] = (st.session_state['row_index'] + 1) % len(simulation_data)
        time.sleep(speed)
        st.rerun()

    # Ledger
# Ledger
    st.subheader("🔒 Immutable Blockchain Ledger")
    chain_data = [{
        "Index": b.flow_index, 
        "Time": b.timestamp.split()[1][:8], 
        "Status": b.prediction,
        "Type": b.attack_type, 
        "Conf": f"{b.confidence:.0%}", 
        "Action": b.action,
        "Reason": b.reason, 
        "Hash": b.hash,
        "Prev Hash": b.prev_hash  # <--- ADD THIS LINE
    } for b in st.session_state['blockchain'][::-1]]
    
    def highlight(row):
        return ['background-color: rgba(255, 75, 75, 0.2)'] * len(row) if row['Status'] == 'MALICIOUS' else ['background-color: rgba(76, 175, 80, 0.2)'] * len(row)
    
    st.dataframe(pd.DataFrame(chain_data).style.apply(highlight, axis=1), use_container_width=True)

# === PAGE 3: THREAT VAULT ===
elif menu == "☠️ Threat Vault":
    st.title("☠️ Threat Vault")
    st.markdown("### 🔴 Critical Incident Log (Malicious Only)")
    
    threat_blocks = [b for b in st.session_state['blockchain'] if b.prediction == "MALICIOUS"]
    
    if not threat_blocks:
        st.success("✅ No Active Threats Recorded in the Vault.")
    else:
        st.error(f"⚠️ {len(threat_blocks)} Critical Threats Archived")
        vault_data = [{
            "Index": b.flow_index, "Time": b.timestamp, "Type": b.attack_type,
            "Confidence": f"{b.confidence:.1%}", "Action": b.action, "Hash": b.hash
        } for b in threat_blocks[::-1]]
        st.dataframe(pd.DataFrame(vault_data), use_container_width=True)
        
        csv = pd.DataFrame(vault_data).to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Threat Report (CSV)", csv, "threat_report.csv", "text/csv", key='download-csv', type='primary')

# === PAGE 4: DATA LAB ===
elif menu == "📂 Data Lab":
    st.title("📂 Forensic Data Lab")
    
    tab_up, tab_nb = st.tabs(["📤 Upload & Scan", "📓 Jupyter Notebook"])
    
    with tab_up:
        uploaded_file = st.file_uploader("Upload Network Logs (.csv)", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write(f"**Scanning {len(df)} rows...**")
            
            if model_cols:
                scan_input = df.reindex(columns=model_cols, fill_value=0)
            else:
                scan_input = df.select_dtypes(include=['number'])
            
            preds = model.predict(scan_input)
            probs = model.predict_proba(scan_input)
            
            results = df.copy()
            if label_encoder:
                results['Attack Type'] = label_encoder.inverse_transform(preds)
            else:
                results['Attack Type'] = ["MALICIOUS" if p == 1 else "BENIGN" for p in preds]
            
            results['Confidence'] = [probs[i].max() for i in range(len(probs))]
            results['Status'] = results['Attack Type'].apply(lambda x: "BENIGN" if "normal" in str(x).lower() else "MALICIOUS")
            results['Suggested Action'] = results.apply(lambda x: get_security_action(x['Attack Type'], x['Confidence'])[0], axis=1)
            
            st.success("Scan Complete.")
            
            def highlight_scan(row):
                return ['background-color: rgba(255, 75, 75, 0.2)'] * len(row) if row['Status'] == 'MALICIOUS' else [''] * len(row)

            st.dataframe(results.style.apply(highlight_scan, axis=1), use_container_width=True)
            
    with tab_nb:
        st.markdown("### 📓 Research Notebook")
        st.code("import pandas as pd\nmodel.fit(X, y)", language="python")

# === PAGE 5: INTELLIGENCE ===
elif menu == "🧠 Intelligence":
    st.title("🧠 Model Explainability")
    col_x1, col_x2 = st.columns(2)
    
    with col_x1:
        st.markdown("### 1. Global Feature Importance")
        if st.button("⚡ Refresh Brain Graph", type="secondary"):
            subprocess.run([sys.executable, "generate_xai.py"], check=True)
            st.rerun()
        if os.path.exists("assets/feature_importance.png"):
            st.image("assets/feature_importance.png")
            
    with col_x2:
        st.markdown("### 2. Session Threat Pulse")
        if len(st.session_state['blockchain']) > 1:
            df_sess = pd.DataFrame([b.attack_type for b in st.session_state['blockchain']], columns=["Attack Type"])
            df_attacks = df_sess[~df_sess["Attack Type"].str.contains("Normal|GENESIS", case=False, na=False)]
            if not df_attacks.empty:
                st.bar_chart(df_attacks["Attack Type"].value_counts())
            else:
                st.info("No attacks in this session.")
        else:
            st.info("Start feed to gather data.")