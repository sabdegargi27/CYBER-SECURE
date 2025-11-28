import joblib
import os

# Path to your trained model
MODEL_PATH = "models/rf_model.pkl"
COLUMNS_PATH = "models/columns.pkl"

if not os.path.exists(MODEL_PATH):
    print(f"❌ Model not found at '{MODEL_PATH}'. Run your training first.")
    exit()

# Load your model (just to ensure it exists)
model = joblib.load(MODEL_PATH)
print("✅ Model loaded successfully.")

# Load your training data columns manually from the CSV used for training
import pandas as pd
TRAIN_DATA_PATH = "data/cleaned_data.csv"  # same file you used to train
if not os.path.exists(TRAIN_DATA_PATH):
    print(f"❌ Training data not found at '{TRAIN_DATA_PATH}'.")
    exit()

df = pd.read_csv(TRAIN_DATA_PATH)
# Drop label and target columns
columns = df.drop(['Attack Type'], axis=1).columns.tolist()

# Save columns
os.makedirs("models", exist_ok=True)
joblib.dump(columns, COLUMNS_PATH)
print(f"💾 columns.pkl created at '{COLUMNS_PATH}' with {len(columns)} columns.")
