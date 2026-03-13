"""
FraudShield AI — Data Loading & Cleaning Module
Handles loading IEEE-CIS dataset, merging, and initial cleanup.
"""

import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load_and_merge(data_dir=None):
    """Load transaction + identity CSVs and merge on TransactionID."""
    if data_dir is None:
        data_dir = DATA_DIR

    print("[DATA] Loading train_transaction.csv...")
    txn = pd.read_csv(os.path.join(data_dir, "train_transaction.csv"))
    print(f"  → {txn.shape[0]:,} rows × {txn.shape[1]} columns")

    print("[DATA] Loading train_identity.csv...")
    ident = pd.read_csv(os.path.join(data_dir, "train_identity.csv"))
    print(f"  → {ident.shape[0]:,} rows × {ident.shape[1]} columns")

    print("[DATA] Merging on TransactionID (LEFT JOIN)...")
    df = txn.merge(ident, on="TransactionID", how="left")
    print(f"  → Merged: {df.shape[0]:,} rows × {df.shape[1]} columns")

    return df


def drop_high_missing(df, threshold=0.70):
    """Drop columns with missing percentage above threshold."""
    missing_pct = df.isnull().mean()
    
    # ALWAYS keep these essential columns even if they have high missing
    essential = {'DeviceInfo', 'DeviceType', 'P_emaildomain', 'R_emaildomain',
                 'TransactionID', 'isFraud', 'TransactionAmt', 'TransactionDT',
                 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
                 'ProductCD', 'addr1', 'addr2', 'dist1'}
    
    drop_cols = [col for col in df.columns
                 if missing_pct[col] > threshold and col not in essential]

    print(f"[DATA] Dropping {len(drop_cols)} columns with >{threshold*100:.0f}% missing")
    df = df.drop(columns=drop_cols)
    print(f"  → Remaining: {df.shape[1]} columns")
    return df


def initial_clean(df):
    """Basic cleaning: fix types, handle obvious issues."""
    # Convert card4, card6 to string for proper encoding later
    for col in ['card4', 'card6', 'ProductCD', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9']:
        if col in df.columns:
            df[col] = df[col].astype(str).replace('nan', np.nan)

    print(f"[DATA] Initial cleaning complete. Shape: {df.shape}")
    return df


def load_data(data_dir=None):
    """Full data loading pipeline."""
    df = load_and_merge(data_dir)
    df = drop_high_missing(df, threshold=0.70)
    df = initial_clean(df)
    return df
