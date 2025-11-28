import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score
from sklearn.preprocessing import LabelEncoder
import joblib

# --- CONFIGURATION ---
INPUT_FILE = "data/cleaned_data.csv" 
MODEL_DIR = "models"
DATA_DIR = "data"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def train():
    print(f"🚀 Looking for data at: {os.path.abspath(INPUT_FILE)}")
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ ERROR: Could not find '{INPUT_FILE}'.")
        return

    # 1. LOAD DATA
    print("✅ File found! Loading...")
    df = pd.read_csv(INPUT_FILE)

    # Sample for memory (Safety limit)
    df = df.sample(frac=0.2, random_state=42).reset_index(drop=True)
    print(f"⚡ Training on {len(df)} rows.")

    # 2. PREPARE MULTI-CLASS DATA
    print("⚙️  Encoding Labels (Multi-Class)...")
    
    # Initialize Encoder
    le = LabelEncoder()
    
    # Encode Target
    y = le.fit_transform(df['Attack Type'])
    X = df.drop(['Attack Type'], axis=1)

    # Convert columns to float32 (Memory optimization)
    for col in X.columns:
        if X[col].dtype == 'object':
            try:
                X[col] = X[col].astype(float)
            except:
                X = X.drop(col, axis=1)
        else:
            X[col] = X[col].astype(np.float32)

    # 3. SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 4. TRAIN
    print("🧠 Training Multi-Class Random Forest...")
    model = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    # 5. EVALUATE
    preds = model.predict(X_test)
    
    # --- THE FIX: DISPLAY AS PERCENTAGE ---
    acc = accuracy_score(y_test, preds)
    rec = recall_score(y_test, preds, average='weighted')
    
    print("-" * 30)
    print(f"✅ Overall Accuracy: {acc:.2%}")  # <--- Shows 99.83%
    print(f"✅ Weighted Recall:  {rec:.2%}")  # <--- Shows 99.83%
    print("-" * 30)

    # 6. SAVE ASSETS
    joblib.dump(model, os.path.join(MODEL_DIR, "rf_model.pkl"))
    joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.pkl"))
    joblib.dump(X.columns.tolist(), os.path.join(MODEL_DIR, "columns.pkl"))
    
    # 7. CREATE DEMO DATA
    demo_df = X_test.copy()
    demo_df['Attack Type'] = le.inverse_transform(y_test)
    demo_df.sample(min(500, len(demo_df))).to_csv(os.path.join(DATA_DIR, "test_samples.csv"), index=False)
    
    print(f"🎉 SUCCESS! Model saved to '{MODEL_DIR}/rf_model.pkl'")

if __name__ == "__main__":
    train()