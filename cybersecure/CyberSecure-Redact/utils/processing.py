"""
utils/processing.py
Simple data loading and preprocessing helper functions.
Designed to be straightforward for beginners.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
from typing import Tuple, Dict

def load_csv(path: str) -> pd.DataFrame:
    """Load CSV or txt with comma or whitespace separator (tries common formats)."""
    try:
        df = pd.read_csv(path)
    except Exception:
        # fallback: whitespace / tab separated
        df = pd.read_csv(path, sep=r'\s+', engine='python', header=None)
    return df

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Make column names simple (lowercase, no spaces)."""
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]
    return df

def prepare_labels(df: pd.DataFrame, label_col: str = 'label') -> pd.DataFrame:
    """
    Convert label column to binary:
    - 0 -> benign/normal
    - 1 -> intrusion/attack
    Accepts common label names in NSL-KDD or similar datasets.
    """
    df = df.copy()
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataframe.")
    def to_binary(x):
        s = str(x).strip().lower()
        if s in ['normal', 'normal.', 'benign', '0', 'none']:
            return 0
        # treat everything else as intrusion (attack name like 'neptune', 'smurf', etc.)
        return 1
    df['label'] = df[label_col].apply(to_binary)
    return df

def encode_categorical(df: pd.DataFrame, fit_encoders: Dict[str, LabelEncoder] = None
                      ) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Encode object / categorical columns using LabelEncoder.
    If fit_encoders provided, use them (for inference). Otherwise fit new encoders and return them.
    """
    df = df.copy()
    encoders = {} if fit_encoders is None else fit_encoders
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    for c in cat_cols:
        if fit_encoders is None:
            le = LabelEncoder()
            df[c] = le.fit_transform(df[c].astype(str))
            encoders[c] = le
        else:
            le = encoders.get(c)
            if le is None:
                # if encoder missing, create and fit on current values (fallback)
                le = LabelEncoder()
                df[c] = le.fit_transform(df[c].astype(str))
                encoders[c] = le
            else:
                df[c] = le.transform(df[c].astype(str))
    return df, encoders

def basic_clean_and_split(path: str, label_col: str = 'label'):
    """
    Full simple pipeline to:
    - load file
    - standardize columns
    - prepare binary label
    - separate X, y
    - return X, y, and encoders (fitted)
    """
    df = load_csv(path)
    df = standardize_column_names(df)
    df = prepare_labels(df, label_col=label_col)
    # drop columns that are obviously non-feature (if present)
    drop_cols = [c for c in ['id', 'timestamp'] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    y = df['label']
    X = df.drop(columns=['label'])
    X, encoders = encode_categorical(X)  # fit encoders here
    return X, y, encoders

def preprocess_for_inference(df_input: pd.DataFrame, encoders: Dict[str, LabelEncoder]):
    """
    Given a new dataframe and encoders, standardize and encode it for model inference.
    """
    df = df_input.copy()
    df = standardize_column_names(df)
    # if label exists remove it for inference
    if 'label' in df.columns:
        df = df.drop(columns=['label'])
    df, _ = encode_categorical(df, fit_encoders=encoders)
    # ensure numeric columns are numeric
    df = df.apply(pd.to_numeric, errors='ignore')
    return df

def save_encoders(encoders: Dict[str, LabelEncoder], path: str):
    joblib.dump(encoders, path)

def load_encoders(path: str):
    return joblib.load(path)
