import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
import numpy as np

# --- CONFIGURATION ---
MODEL_FILE = "models/rf_model.pkl"
OUTPUT_FILE = "assets/feature_importance.png"

# Hardcoded Fallback Names (Standard NSL-KDD names)
# This ensures we NEVER show "Feature 0" even if files are missing
FALLBACK_NAMES = [
    "Duration", "Protocol_Type", "Service", "Flag", "Src_Bytes", "Dst_Bytes",
    "Land", "Wrong_Fragment", "Urgent", "Hot", "Failed_Logins", "Logged_In",
    "Compromised", "Root_Shell", "Su_Attempted", "Num_Root", "File_Creations",
    "Shells", "Access_Files", "Outbound_Cmds", "Host_Login", "Guest_Login",
    "Count", "Srv_Count", "Serror_Rate", "Srv_Serror_Rate", "Rerror_Rate",
    "Srv_Rerror_Rate", "Same_Srv_Rate", "Diff_Srv_Rate", "Srv_Diff_Host_Rate",
    "Dst_Host_Count", "Dst_Host_Srv_Count", "Dst_Host_Same_Srv_Rate",
    "Dst_Host_Diff_Srv_Rate", "Dst_Host_Same_Src_Port_Rate",
    "Dst_Host_Srv_Diff_Host_Rate", "Dst_Host_Serror_Rate",
    "Dst_Host_Srv_Serror_Rate", "Dst_Host_Rerror_Rate", "Dst_Host_Srv_Rerror_Rate"
]

def generate_xai():
    print("🚀 Generaring Graph with Real Names...")
    
    if not os.path.exists(MODEL_FILE):
        print("❌ Error: Model not found.")
        return

    model = joblib.load(MODEL_FILE)
    importances = model.feature_importances_
    
    # 1. Try to find real names from the saved columns file
    feature_names = []
    if os.path.exists("models/columns.pkl"):
        feature_names = joblib.load("models/columns.pkl")
    elif os.path.exists("cleaned_data.csv"):
        df = pd.read_csv("cleaned_data.csv", nrows=1)
        feature_names = [c for c in df.columns if c.lower() not in ['label', 'attack type']]
    
    # 2. If that fails (or lengths don't match), use the Fallback list
    if len(feature_names) != len(importances):
        print("⚠️ Name mismatch. Using Standard KDD Feature Names.")
        # Slice the fallback list to match the model's feature count
        feature_names = FALLBACK_NAMES[:len(importances)]
        # If still not enough, fill with "Extra_Feat_X"
        if len(feature_names) < len(importances):
            feature_names += [f"Extra_Feat_{i}" for i in range(len(importances) - len(feature_names))]

    # 3. Sort and Plot Top 10
    indices = np.argsort(importances)[-10:]
    
    plt.figure(figsize=(10, 6))
    plt.title("Top 10 Factors Driving Intrusion Detection", fontsize=14, fontweight='bold')
    plt.barh(range(len(indices)), importances[indices], color='#FF4B4B')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices], fontsize=10)
    plt.xlabel("Importance Score (Impact on AI Decision)", fontweight='bold')
    plt.tight_layout()
    
    os.makedirs("assets", exist_ok=True)
    plt.savefig(OUTPUT_FILE)
    plt.close()
    print("✅ Graph Saved!")

if __name__ == "__main__":
    generate_xai()