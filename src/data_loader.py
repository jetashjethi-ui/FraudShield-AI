"""
FraudShield AI — Data Loading & Cleaning Module
Handles loading IEEE-CIS dataset, merging, and initial cleanup.
"""

import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

CATEGORICAL_COLUMNS = {
    "ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
    "DeviceType", "DeviceInfo", "M1", "M2", "M3", "M4", "M5", "M6",
    "M7", "M8", "M9", "id_12", "id_15", "id_16", "id_23", "id_27",
    "id_28", "id_29", "id_30", "id_31", "id_33", "id_34", "id_35",
    "id_36", "id_37", "id_38",
}


def _compact_csv_dtypes(path):
    """Read numeric IEEE-CIS columns as float32 and strings as categories."""
    probe = pd.read_csv(path, nrows=10000, low_memory=False)
    columns = probe.columns
    observed_text = set(probe.select_dtypes(include=["object", "string"]).columns)
    del probe
    categorical = CATEGORICAL_COLUMNS | observed_text
    dtype = {col: ("category" if col in categorical else "float32") for col in columns}
    if "TransactionID" in dtype:
        dtype["TransactionID"] = "int32"
    if "isFraud" in dtype:
        dtype["isFraud"] = "int8"
    return dtype


def load_and_merge(data_dir=None):
    """Load transaction + identity CSVs and merge on TransactionID."""
    if data_dir is None:
        data_dir = DATA_DIR

    transaction_path = os.path.join(data_dir, "train_transaction.csv")
    identity_path = os.path.join(data_dir, "train_identity.csv")
    missing = [path for path in (transaction_path, identity_path) if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError("Required IEEE-CIS input file(s) not found: " + ", ".join(missing))

    print("[DATA] Loading train_transaction.csv...")
    txn = pd.read_csv(
        transaction_path, dtype=_compact_csv_dtypes(transaction_path),
        low_memory=False, memory_map=True,
    )
    print(f"  → {txn.shape[0]:,} rows × {txn.shape[1]} columns")

    print("[DATA] Loading train_identity.csv...")
    ident = pd.read_csv(
        identity_path, dtype=_compact_csv_dtypes(identity_path),
        low_memory=False, memory_map=True,
    )
    print(f"  → {ident.shape[0]:,} rows × {ident.shape[1]} columns")

    print("[DATA] Merging on TransactionID (LEFT JOIN)...")
    if "TransactionID" not in txn or "TransactionID" not in ident:
        raise ValueError("Both IEEE-CIS tables must contain TransactionID.")
    if txn["TransactionID"].duplicated().any() or ident["TransactionID"].duplicated().any():
        raise ValueError("TransactionID must be unique in both source tables.")
    df = txn.merge(ident, on="TransactionID", how="left", validate="one_to_one", copy=False)
    if "isFraud" not in df:
        raise ValueError("Transaction table is missing the isFraud target column.")
    print(f"  → Merged: {df.shape[0]:,} rows × {df.shape[1]} columns")

    return df


def high_missing_columns(df, threshold=0.70):
    """Return high-missing columns using only the supplied reference period."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    missing_pct = df.isnull().mean()
    
    # ALWAYS keep these essential columns even if they have high missing
    essential = {'DeviceInfo', 'DeviceType', 'P_emaildomain', 'R_emaildomain',
                 'TransactionID', 'isFraud', 'TransactionAmt', 'TransactionDT',
                 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
                 'ProductCD', 'addr1', 'addr2', 'dist1'}
    
    return [col for col in df.columns if missing_pct[col] > threshold and col not in essential]


def drop_high_missing(df, threshold=0.70):
    """Drop columns with missing percentage above threshold."""
    drop_cols = high_missing_columns(df, threshold)

    print(f"[DATA] Dropping {len(drop_cols)} columns with >{threshold*100:.0f}% missing")
    df = df.drop(columns=drop_cols)
    print(f"  → Remaining: {df.shape[1]} columns")
    return df


def initial_clean(df):
    """Basic cleaning: fix types, handle obvious issues."""
    # Keep low-cardinality values compact and preserve nulls for the shared
    # train/serve pipeline's missing-value indicators.
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            if not isinstance(df[col].dtype, pd.CategoricalDtype):
                df[col] = df[col].astype("category")

    print(f"[DATA] Initial cleaning complete. Shape: {df.shape}")
    return df


def load_data(data_dir=None):
    """Full data loading pipeline."""
    df = load_and_merge(data_dir)
    df = initial_clean(df)
    return df
