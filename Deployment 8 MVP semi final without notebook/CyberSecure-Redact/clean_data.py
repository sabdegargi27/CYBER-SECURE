import pandas as pd
import os

def clean_data():
    # Define the input and output file names
    raw_file = "nsl_kdd_dataset.csv"
    output_file = "cleaned_data.csv"

    # Check if raw file exists before trying to read it
    if not os.path.exists(raw_file):
        print(f"❌ Error: The file '{raw_file}' was not found in this folder.")
        print("   Please make sure you downloaded 'nsl_kdd_dataset.csv' and put it here.")
        return

    print(f"🧹 Reading raw data from '{raw_file}'...")
    df = pd.read_csv(raw_file)

    # 1. CLEAN LABELS (Text -> Numbers)
    # Convert 'normal' to 0, and everything else (attacks) to 1
    if 'label' in df.columns:
        print("⚙️  Processing labels...")
        df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)
    
    # 2. SAVE THE CLEAN FILE
    df.to_csv(output_file, index=False)
    print(f"✅ Success! Created '{output_file}' with {len(df)} rows.")

if __name__ == "__main__":
    clean_data()