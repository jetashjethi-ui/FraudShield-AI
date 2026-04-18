"""
FraudShield AI — Feature Engineering Module
Implements all 15 Detection Layers as engineered features.
"""

import pandas as pd
import numpy as np
from src.graph_engine import build_graph_features


def build_all_features(df):
    """Master function: build all 25 layers of features."""
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING -- 25 DETECTION LAYERS")
    print("=" * 70)

    df = amount_features(df)          # Layer 1, 9
    df = time_features(df)            # Layer 2, 4
    df = behavioral_features(df)      # Layer 2
    df = augment_device_data(df)      # Synthetic device augmentation
    df = sim_swap_features(df)        # Layer 3
    df = seasonal_features(df)        # Layer 4
    df = merchant_risk_features(df)   # Layer 6
    df = shared_device_features(df)   # Layer 7
    df = dormant_account_features(df) # Layer 8
    df = round_amount_features(df)    # Layer 9
    df = category_mismatch(df)        # Layer 10
    df = new_account_features(df)     # Layer 11
    df = velocity_features(df)        # Layer 12
    df = email_risk_features(df)      # Layer 13
    df = network_features(df)         # Layer 14
    df = build_graph_features(df)     # Layer 16 -- GRAPH ANALYSIS
    df = target_encoding(df)          # Layer 17 -- TARGET ENCODING
    df = uid_features(df)             # Layer 18 -- USER IDENTITY PROFILING
    df = uid_time_windows(df)         # Layer 19 -- TIME-WINDOW VELOCITY
    df = v_feature_aggregations(df)   # Layer 20 -- V-FEATURE INTELLIGENCE
    df = frequency_encoding(df)       # Layer 21 -- FREQUENCY ENCODING
    df = peer_group_deviation(df)     # Layer 22 -- PEER GROUP ANOMALY
    df = transaction_entropy(df)      # Layer 23 -- ENTROPY ANALYSIS
    df = lag_features(df)             # Layer 24 -- LAG / SEQUENTIAL PATTERNS
    df = cross_feature_fraud_rates(df)# Layer 25 -- CROSS-FEATURE COMBINATIONS
    df = interaction_features(df)     # Cross-layer combos (expanded)
    df = missing_indicators(df)       # Missingness as signal

    print(f"\n[FEATURES] Total features after engineering: {df.shape[1]}")
    return df


# ─── LAYER 1 & 9: AMOUNT FEATURES ────────────────────────────────────
def amount_features(df):
    """Amount-based features including log transform and round detection."""
    print("\n  [Layer 1+9] Amount Features...")

    df['log_amount'] = np.log1p(df['TransactionAmt'])
    df['amount_sqrt'] = np.sqrt(df['TransactionAmt'])
    df['amount_decimal'] = df['TransactionAmt'] - np.floor(df['TransactionAmt'])
    df['is_round_amount'] = (df['amount_decimal'] == 0).astype(int)

    print(f"    → log_amount, amount_sqrt, amount_decimal, is_round_amount")
    return df


# ─── LAYER 2: TIME FEATURES ──────────────────────────────────────────
def time_features(df):
    """Time-based features from TransactionDT."""
    print("  [Layer 2] Time Features...")

    df['hour_of_day'] = ((df['TransactionDT'] % 86400) / 3600).astype(int)
    df['is_night'] = ((df['hour_of_day'] >= 0) & (df['hour_of_day'] <= 6)).astype(int)
    df['day_of_week'] = ((df['TransactionDT'] / 86400).astype(int)) % 7
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['month_approx'] = ((df['TransactionDT'] / 2592000).astype(int)) % 12

    print(f"    → hour_of_day, is_night, day_of_week, is_weekend, month_approx")
    return df


# ─── LAYER 2: BEHAVIORAL FINGERPRINTING ──────────────────────────────
def behavioral_features(df):
    """Per-user spending DNA: mean, std, z-score by card1."""
    print("  [Layer 2] Behavioral Fingerprinting (Spending DNA)...")

    card_stats = df.groupby('card1')['TransactionAmt'].agg(
        avg_amt='mean',
        std_amt='std',
        user_txn_count='count'
    ).reset_index()

    # Count distinct ProductCD per user
    product_diversity = df.groupby('card1')['ProductCD'].nunique().reset_index()
    product_diversity.columns = ['card1', 'user_merchant_diversity']

    card_stats = card_stats.merge(product_diversity, on='card1', how='left')
    card_stats['std_amt'] = card_stats['std_amt'].fillna(1)  # avoid div by 0

    df = df.merge(card_stats, on='card1', how='left')

    # Derived features
    df['amount_zscore'] = (df['TransactionAmt'] - df['avg_amt']) / df['std_amt'].clip(lower=1)
    df['amount_to_mean_ratio'] = df['TransactionAmt'] / df['avg_amt'].clip(lower=0.01)

    print(f"    → avg_amt, std_amt, user_txn_count, user_merchant_diversity")
    print(f"    → amount_zscore, amount_to_mean_ratio")
    return df


# ─── SYNTHETIC DEVICE DATA AUGMENTATION ──────────────────────────────
def augment_device_data(df):
    """Fill missing DeviceInfo with synthetic data based on existing patterns."""
    print("  [Augmentation] Synthetic Device Data...")

    if 'DeviceInfo' not in df.columns:
        print("    → DeviceInfo not found, skipping")
        return df

    rng = np.random.RandomState(42)
    missing_mask = df['DeviceInfo'].isna()
    n_missing = missing_mask.sum()
    print(f"    → {n_missing:,} rows missing DeviceInfo ({n_missing/len(df)*100:.1f}%)")

    # Get existing device distribution per card
    known = df[~missing_mask]
    card_devices = known.groupby('card1')['DeviceInfo'].agg(list).to_dict()

    # Global device distribution (fallback)
    all_devices = known['DeviceInfo'].value_counts()
    top_devices = all_devices.head(100).index.tolist()
    top_probs = (all_devices.head(100).values / all_devices.head(100).values.sum())

    synth_devices = []
    for idx in df[missing_mask].index:
        card = df.loc[idx, 'card1']
        if card in card_devices and rng.random() < 0.92:
            # 92% of the time: assign user's known device (normal)
            synth_devices.append(rng.choice(card_devices[card]))
        else:
            # 8% of the time: assign random device (simulate device change)
            synth_devices.append(rng.choice(top_devices, p=top_probs))

    df.loc[missing_mask, 'DeviceInfo'] = synth_devices
    print(f"    → Filled {n_missing:,} rows with synthetic device info")
    print(f"    → ~8% assigned unusual devices to simulate account takeover")
    return df


# ─── LAYER 3: SIM SWAP / DEVICE CHANGE DETECTION ─────────────────────
def sim_swap_features(df):
    """Detect unusual device changes per card (account takeover signal)."""
    print("  [Layer 3] SIM Swap / Device Change Detection...")

    if 'DeviceInfo' not in df.columns:
        df['devices_per_card'] = 0
        df['is_unusual_device'] = 0
        print("    → DeviceInfo not found, skipping")
        return df

    # Count distinct devices per card
    device_count = df.groupby('card1')['DeviceInfo'].nunique().reset_index()
    device_count.columns = ['card1', 'devices_per_card']
    df = df.merge(device_count, on='card1', how='left')
    df['devices_per_card'] = df['devices_per_card'].fillna(0)

    # Find most common device per card
    mode_device = df.groupby('card1')['DeviceInfo'].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan
    ).reset_index()
    mode_device.columns = ['card1', 'usual_device']
    df = df.merge(mode_device, on='card1', how='left')

    df['is_unusual_device'] = (
        (df['DeviceInfo'].notna()) &
        (df['usual_device'].notna()) &
        (df['DeviceInfo'] != df['usual_device'])
    ).astype(int)

    df.drop(columns=['usual_device'], inplace=True)

    print(f"    → devices_per_card, is_unusual_device")
    return df


# ─── LAYER 4: SEASONAL BASELINES ─────────────────────────────────────
def seasonal_features(df):
    """Per-user per-month spending averages for contextual anomaly detection."""
    print("  [Layer 4] Seasonal & Contextual Baselines...")

    monthly_avg = df.groupby(['card1', 'month_approx'])['TransactionAmt'].mean().reset_index()
    monthly_avg.columns = ['card1', 'month_approx', 'monthly_avg_amt']

    df = df.merge(monthly_avg, on=['card1', 'month_approx'], how='left')
    df['amt_vs_seasonal'] = df['TransactionAmt'] / df['monthly_avg_amt'].clip(lower=0.01)

    print(f"    → monthly_avg_amt, amt_vs_seasonal")
    return df


# ─── LAYER 6: MERCHANT RISK PROFILING ────────────────────────────────
def merchant_risk_features(df):
    """Compute fraud rates per ProductCD and addr1 region."""
    print("  [Layer 6] Merchant Risk Profiling...")

    # Fraud rate by ProductCD
    product_risk = df.groupby('ProductCD')['isFraud'].mean().reset_index()
    product_risk.columns = ['ProductCD', 'product_fraud_rate']
    df = df.merge(product_risk, on='ProductCD', how='left')

    # Fraud rate by region (addr1)
    if 'addr1' in df.columns:
        region_risk = df.groupby('addr1')['isFraud'].mean().reset_index()
        region_risk.columns = ['addr1', 'region_fraud_rate']
        df = df.merge(region_risk, on='addr1', how='left')
        df['region_fraud_rate'] = df['region_fraud_rate'].fillna(df['isFraud'].mean())
    else:
        df['region_fraud_rate'] = df['isFraud'].mean()

    print(f"    → product_fraud_rate, region_fraud_rate")
    return df


# ─── LAYER 7: SHARED DEVICE / MULE NETWORK ───────────────────────────
def shared_device_features(df):
    """Detect devices shared across multiple identities (mule networks)."""
    print("  [Layer 7] Shared Device / Mule Network Detection...")

    if 'DeviceInfo' not in df.columns:
        df['cards_per_device'] = 0
        df['emails_per_device'] = 0
        df['is_shared_device'] = 0
        print("    → DeviceInfo not found, skipping")
        return df

    # Cards per device
    cards_per_dev = df.groupby('DeviceInfo')['card1'].nunique().reset_index()
    cards_per_dev.columns = ['DeviceInfo', 'cards_per_device']
    df = df.merge(cards_per_dev, on='DeviceInfo', how='left')
    df['cards_per_device'] = df['cards_per_device'].fillna(0)

    # Emails per device
    if 'P_emaildomain' in df.columns:
        emails_per_dev = df.groupby('DeviceInfo')['P_emaildomain'].nunique().reset_index()
        emails_per_dev.columns = ['DeviceInfo', 'emails_per_device']
        df = df.merge(emails_per_dev, on='DeviceInfo', how='left')
        df['emails_per_device'] = df['emails_per_device'].fillna(0)
    else:
        df['emails_per_device'] = 0

    df['is_shared_device'] = (df['cards_per_device'] > 2).astype(int)

    print(f"    → cards_per_device, emails_per_device, is_shared_device")
    return df


# ─── LAYER 8: DORMANT ACCOUNT HIJACK ─────────────────────────────────
def dormant_account_features(df):
    """Flag low-activity accounts with sudden high-value transactions."""
    print("  [Layer 8] Dormant Account Hijack Detection...")

    df['is_low_activity_user'] = (df['user_txn_count'] < 5).astype(int)
    df['dormant_high_value'] = (
        (df['is_low_activity_user'] == 1) &
        (df['TransactionAmt'] > df['avg_amt'] * 2)
    ).astype(int)

    print(f"    → is_low_activity_user, dormant_high_value")
    return df


# ─── LAYER 9: ROUND AMOUNT SUSPICION ─────────────────────────────────
def round_amount_features(df):
    """Score suspicion for perfectly round transaction amounts."""
    print("  [Layer 9] Round Amount Suspicion Scoring...")

    df['is_suspicious_round'] = (
        (df['is_round_amount'] == 1) &
        (df['TransactionAmt'] >= 500)
    ).astype(int)

    print(f"    → is_suspicious_round")
    return df


# ─── LAYER 10: MERCHANT CATEGORY MISMATCH ────────────────────────────
def category_mismatch(df):
    """Detect when a user shops in an unusual product category."""
    print("  [Layer 10] Merchant Category Mismatch...")

    df['is_low_diversity_user'] = (df['user_merchant_diversity'] <= 2).astype(int)
    df['category_mismatch_risk'] = (
        (df['is_low_diversity_user'] == 1) &
        (df['product_fraud_rate'] > 0.05)
    ).astype(int)

    print(f"    → is_low_diversity_user, category_mismatch_risk")
    return df


# ─── LAYER 11: NEW ACCOUNT FIRST TRANSACTION ABUSE ───────────────────
def new_account_features(df):
    """Flag brand-new accounts making high-value first transactions."""
    print("  [Layer 11] New Account First Transaction Abuse...")

    df['is_first_txn'] = (df['user_txn_count'] == 1).astype(int)
    df['first_txn_high_value'] = (
        (df['is_first_txn'] == 1) &
        (df['TransactionAmt'] > 500)
    ).astype(int)

    print(f"    → is_first_txn, first_txn_high_value")
    return df


# ─── LAYER 12: TRANSACTION VELOCITY ──────────────────────────────────
def velocity_features(df):
    """Transaction speed/frequency features — one of the strongest fraud signals."""
    print("  [Layer 12] Transaction Velocity Features...")

    # Sort by card and time for sequential analysis
    df = df.sort_values(['card1', 'TransactionDT']).reset_index(drop=True)

    # Time since last transaction for this card
    df['time_since_last_txn'] = df.groupby('card1')['TransactionDT'].diff().fillna(999999)
    df['time_since_last_txn_hrs'] = df['time_since_last_txn'] / 3600.0

    # Rapid-fire flag: less than 5 minutes since last txn
    df['is_rapid_fire'] = (df['time_since_last_txn'] < 300).astype(int)

    # Rolling transaction count (last N transactions' time window)
    df['txn_count_30min'] = (
        df.groupby('card1')['TransactionDT']
        .transform(lambda x: x.rolling(window=3, min_periods=1).apply(
            lambda w: ((w.iloc[-1] - w) < 1800).sum() if len(w) > 0 else 0, raw=False
        ))
    ).fillna(0)

    # Amount velocity: cumulative amount in recent txns
    df['amt_cumsum_last3'] = df.groupby('card1')['TransactionAmt'].transform(
        lambda x: x.rolling(window=3, min_periods=1).sum()
    ).fillna(0)

    # Amount acceleration: current amount vs previous amount
    df['prev_amount'] = df.groupby('card1')['TransactionAmt'].shift(1).fillna(0)
    df['amount_acceleration'] = (df['TransactionAmt'] / df['prev_amount'].clip(lower=0.01))
    df['amount_acceleration'] = df['amount_acceleration'].clip(upper=100)

    print(f"    → time_since_last_txn, is_rapid_fire, txn_count_30min")
    print(f"    → amt_cumsum_last3, amount_acceleration")
    return df


# ─── LAYER 13: EMAIL DOMAIN RISK SCORING ─────────────────────────────
def email_risk_features(df):
    """Free email vs corporate email fraud profiles."""
    print("  [Layer 13] Email Domain Risk Scoring...")

    FREE_EMAILS = {'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'aol.com',
                   'icloud.com', 'mail.com', 'ymail.com', 'live.com', 'msn.com',
                   'protonmail.com', 'gmx.com', 'yahoo.co.jp', 'yahoo.co.uk',
                   'comcast.net', 'cox.net', 'sbcglobal.net', 'verizon.net',
                   'att.net', 'bellsouth.net', 'charter.net', 'earthlink.net'}

    if 'P_emaildomain' in df.columns:
        df['is_free_email'] = df['P_emaildomain'].isin(FREE_EMAILS).astype(int)
        df['is_corporate_email'] = (
            df['P_emaildomain'].notna() &
            ~df['P_emaildomain'].isin(FREE_EMAILS)
        ).astype(int)

        # Fraud rate per email domain
        email_fraud = df.groupby('P_emaildomain')['isFraud'].mean().reset_index()
        email_fraud.columns = ['P_emaildomain', 'email_fraud_rate']
        df = df.merge(email_fraud, on='P_emaildomain', how='left')
        df['email_fraud_rate'] = df['email_fraud_rate'].fillna(df['isFraud'].mean())

        # Email-card mismatch: how many different cards use this email domain
        email_cards = df.groupby('P_emaildomain')['card1'].nunique().reset_index()
        email_cards.columns = ['P_emaildomain', 'cards_per_email_domain']
        df = df.merge(email_cards, on='P_emaildomain', how='left')
        df['cards_per_email_domain'] = df['cards_per_email_domain'].fillna(0)
    else:
        df['is_free_email'] = 0
        df['is_corporate_email'] = 0
        df['email_fraud_rate'] = 0
        df['cards_per_email_domain'] = 0

    print(f"    → is_free_email, is_corporate_email, email_fraud_rate, cards_per_email_domain")
    return df


# ─── LAYER 14: GRAPH-LITE NETWORK / FRAUD RING DETECTION ─────────────
def network_features(df):
    """Detect fraud rings by finding shared attributes across users."""
    print("  [Layer 14] Graph-Lite Network (Fraud Ring Detection)...")

    # Users per address (same billing = possible fraud ring)
    if 'addr1' in df.columns:
        addr_users = df.groupby('addr1')['card1'].nunique().reset_index()
        addr_users.columns = ['addr1', 'users_per_address']
        df = df.merge(addr_users, on='addr1', how='left')
        df['users_per_address'] = df['users_per_address'].fillna(1)
    else:
        df['users_per_address'] = 1

    # Shared card property combo (card4 + card5 + card6 = same bank/type)
    df['card_combo'] = (
        df['card4'].astype(str) + '_' +
        df['card5'].astype(str) + '_' +
        df['card6'].astype(str)
    )
    combo_users = df.groupby('card_combo')['card1'].nunique().reset_index()
    combo_users.columns = ['card_combo', 'users_same_card_type']
    df = df.merge(combo_users, on='card_combo', how='left')
    df.drop(columns=['card_combo'], inplace=True)

    # Fraud neighbors: does this user share addr1 with any known fraud?
    if 'addr1' in df.columns:
        fraud_addrs = set(df[df['isFraud'] == 1]['addr1'].dropna().unique())
        df['has_fraud_neighbor'] = df['addr1'].isin(fraud_addrs).astype(int)
    else:
        df['has_fraud_neighbor'] = 0

    # Transaction clustering: how many txns happen at the same hour from same addr
    if 'addr1' in df.columns:
        df['addr_hour_combo'] = df['addr1'].astype(str) + '_' + df['hour_of_day'].astype(str)
        addr_hour_count = df.groupby('addr_hour_combo')['TransactionID'].count().reset_index()
        addr_hour_count.columns = ['addr_hour_combo', 'txns_same_addr_hour']
        df = df.merge(addr_hour_count, on='addr_hour_combo', how='left')
        df.drop(columns=['addr_hour_combo'], inplace=True)
    else:
        df['txns_same_addr_hour'] = 0

    print(f"    → users_per_address, users_same_card_type, has_fraud_neighbor, txns_same_addr_hour")
    return df


# ─── INTERACTION FEATURES (Cross-layer combos) ───────────────────────
def interaction_features(df):
    """Create powerful interaction features across all layers."""
    print("  [Cross-Layer] Interaction Features (Expanded)...")

    # Original interactions
    df['amount_x_night'] = df['log_amount'] * df['is_night']
    df['amount_x_weekend'] = df['log_amount'] * df['is_weekend']
    df['round_x_night'] = df['is_suspicious_round'] * df['is_night']
    df['round_x_new_device'] = df['is_suspicious_round'] * df['is_unusual_device']
    df['night_x_unusual_device'] = df['is_night'] * df['is_unusual_device']
    df['dormant_x_night'] = df['is_low_activity_user'] * df['is_night']
    df['new_acct_x_round'] = df['first_txn_high_value'] * df['is_suspicious_round']

    # NEW interactions with velocity
    df['rapid_fire_x_night'] = df.get('is_rapid_fire', pd.Series(0, index=df.index)) * df['is_night']
    df['rapid_fire_x_high_amt'] = df.get('is_rapid_fire', pd.Series(0, index=df.index)) * (df['TransactionAmt'] > 500).astype(int)
    df['velocity_x_new_device'] = df.get('is_rapid_fire', pd.Series(0, index=df.index)) * df['is_unusual_device']

    # NEW interactions with email/network
    df['free_email_x_high_amt'] = df.get('is_free_email', pd.Series(0, index=df.index)) * (df['TransactionAmt'] > 500).astype(int)
    df['fraud_neighbor_x_night'] = df.get('has_fraud_neighbor', pd.Series(0, index=df.index)) * df['is_night']
    df['shared_device_x_network'] = df['is_shared_device'] * (df.get('users_per_address', pd.Series(1, index=df.index)) > 5).astype(int)

    # Graph-layer interactions (Layer 16 combos)
    df['graph_fraud_comm_x_night'] = df.get('graph_community_fraud_rate', pd.Series(0, index=df.index)) * df['is_night']
    df['graph_fraud_comm_x_high_amt'] = df.get('graph_community_fraud_rate', pd.Series(0, index=df.index)) * (df['TransactionAmt'] > 500).astype(int)
    df['graph_bridge_x_unusual_device'] = df.get('graph_is_bridge', pd.Series(0, index=df.index)) * df['is_unusual_device']
    df['graph_fraud_neighbor_x_round'] = df.get('graph_fraud_neighbor_ratio', pd.Series(0, index=df.index)) * df['is_suspicious_round']

    print(f"    -> 17 interaction features created")
    return df


# --- UID FEATURES (Layer 18) -- USER IDENTITY PROFILING ------------------
def uid_features(df):
    """Create pseudo user IDs and compute per-user aggregations.
    Top Kaggle solutions used UID features to boost AUC significantly.
    Combines card1 + addr1 to fingerprint individual users."""
    print("  [Layer 18] UID Features (User Identity Profiling)...")

    # Build UID from card1 + addr1 (primary user fingerprint)
    df['uid'] = df['card1'].astype(str) + '_' + df['addr1'].astype(str)

    # Also build a secondary UID with card1 + addr1 + D1 for finer granularity
    if 'D1' in df.columns:
        df['uid2'] = df['uid'] + '_' + df['D1'].fillna(-1).astype(int).astype(str)
    else:
        df['uid2'] = df['uid']

    # --- Per-UID aggregations on primary UID ---
    uid_amt = df.groupby('uid')['TransactionAmt'].agg(
        uid_amt_mean='mean',
        uid_amt_std='std',
        uid_amt_max='max',
        uid_amt_min='min',
        uid_txn_count='count'
    ).reset_index()
    uid_amt['uid_amt_std'] = uid_amt['uid_amt_std'].fillna(0)
    uid_amt['uid_amt_range'] = uid_amt['uid_amt_max'] - uid_amt['uid_amt_min']
    df = df.merge(uid_amt, on='uid', how='left')

    # Amount deviation from user's own pattern
    df['uid_amt_zscore'] = (df['TransactionAmt'] - df['uid_amt_mean']) / df['uid_amt_std'].clip(lower=1)
    df['uid_amt_ratio'] = df['TransactionAmt'] / df['uid_amt_mean'].clip(lower=0.01)

    # Is this the user's largest transaction ever?
    df['uid_is_max_amt'] = (df['TransactionAmt'] >= df['uid_amt_max'] * 0.99).astype(int)

    # --- Per-UID fraud history (only works when isFraud is available) ---
    if 'isFraud' in df.columns:
        global_fraud_rate = df['isFraud'].mean()
        smooth = 50

        uid_fraud = df.groupby('uid')['isFraud'].agg(['mean', 'sum', 'count']).reset_index()
        uid_fraud.columns = ['uid', 'uid_fraud_rate_raw', 'uid_fraud_count', 'uid_total_count']
        # Smoothed fraud rate to avoid overfitting on rare UIDs
        uid_fraud['uid_fraud_rate'] = (
            uid_fraud['uid_total_count'] * uid_fraud['uid_fraud_rate_raw'] + smooth * global_fraud_rate
        ) / (uid_fraud['uid_total_count'] + smooth)
        df = df.merge(uid_fraud[['uid', 'uid_fraud_rate', 'uid_fraud_count']], on='uid', how='left')
    else:
        df['uid_fraud_rate'] = 0
        df['uid_fraud_count'] = 0

    # --- Per-UID device diversity ---
    if 'DeviceInfo' in df.columns:
        uid_devices = df.groupby('uid')['DeviceInfo'].nunique().reset_index()
        uid_devices.columns = ['uid', 'uid_device_count']
        df = df.merge(uid_devices, on='uid', how='left')
    else:
        df['uid_device_count'] = 1

    # --- Per-UID email diversity ---
    if 'P_emaildomain' in df.columns:
        uid_email = df.groupby('uid')['P_emaildomain'].nunique().reset_index()
        uid_email.columns = ['uid', 'uid_email_count']
        df = df.merge(uid_email, on='uid', how='left')
    else:
        df['uid_email_count'] = 1

    # --- Per-UID product diversity ---
    uid_products = df.groupby('uid')['ProductCD'].nunique().reset_index()
    uid_products.columns = ['uid', 'uid_product_count']
    df = df.merge(uid_products, on='uid', how='left')

    # --- Per-UID time gap analysis ---
    df_sorted = df.sort_values(['uid', 'TransactionDT'])
    df_sorted['uid_time_diff'] = df_sorted.groupby('uid')['TransactionDT'].diff()
    df['uid_time_diff'] = df_sorted['uid_time_diff']
    df['uid_time_diff'] = df['uid_time_diff'].fillna(-1)

    # Very fast repeat (< 60 seconds between transactions)
    df['uid_rapid_repeat'] = (df['uid_time_diff'].between(0, 60)).astype(int)
    # Burst activity (< 5 minutes)
    df['uid_burst'] = (df['uid_time_diff'].between(0, 300)).astype(int)

    # --- UID2 (finer) aggregations ---
    uid2_count = df.groupby('uid2')['TransactionAmt'].agg(
        uid2_txn_count='count',
        uid2_amt_mean='mean'
    ).reset_index()
    df = df.merge(uid2_count, on='uid2', how='left')

    # --- Clean up: drop string UID columns (can't feed to models) ---
    df = df.drop(columns=['uid', 'uid2'], errors='ignore')

    features_added = [
        'uid_amt_mean', 'uid_amt_std', 'uid_amt_max', 'uid_amt_min',
        'uid_txn_count', 'uid_amt_range', 'uid_amt_zscore', 'uid_amt_ratio',
        'uid_is_max_amt', 'uid_fraud_rate', 'uid_fraud_count',
        'uid_device_count', 'uid_email_count', 'uid_product_count',
        'uid_time_diff', 'uid_rapid_repeat', 'uid_burst',
        'uid2_txn_count', 'uid2_amt_mean'
    ]
    print(f"    -> {len(features_added)} UID features created")
    return df


# --- LAYER 19: TIME-WINDOW VELOCITY PER UID --------------------------------
def uid_time_windows(df):
    """Transaction velocity per user in 1hr, 6hr, 24hr windows.
    Uses time-binning for speed (O(n) instead of O(n²))."""
    print("  [Layer 19] Time-Window Velocity (per UID)...")

    # Rebuild UID temporarily for grouping
    df['_uid'] = df['card1'].astype(str) + '_' + df['addr1'].astype(str)

    # Create time bins
    df['_hour_bin'] = df['TransactionDT'] // 3600     # 1-hour bins
    df['_6h_bin'] = df['TransactionDT'] // 21600      # 6-hour bins
    df['_day_bin'] = df['TransactionDT'] // 86400      # 24-hour bins

    for bin_col, label in [('_hour_bin', '1h'), ('_6h_bin', '6h'), ('_day_bin', '24h')]:
        # Count transactions per uid per time bin
        bin_stats = df.groupby(['_uid', bin_col])['TransactionAmt'].agg(
            **{f'uid_txn_{label}': 'count', f'uid_amt_sum_{label}': 'sum'}
        ).reset_index()
        df = df.merge(bin_stats, on=['_uid', bin_col], how='left', suffixes=('', '_drop'))
        # Drop any duplicate columns from merge
        df = df.drop(columns=[c for c in df.columns if c.endswith('_drop')], errors='ignore')

    # Derived: is this a velocity spike?
    df['uid_velocity_spike_1h'] = (df['uid_txn_1h'] >= 3).astype(int)
    df['uid_velocity_spike_24h'] = (df['uid_txn_24h'] >= 10).astype(int)
    if 'uid_amt_mean' in df.columns:
        df['uid_amt_spike_24h'] = (df['uid_amt_sum_24h'] > df['uid_amt_mean'] * 3).astype(int)
    else:
        df['uid_amt_spike_24h'] = 0

    # Per-UID max velocity (peak hour activity)
    max_hourly = df.groupby('_uid')['uid_txn_1h'].max().reset_index()
    max_hourly.columns = ['_uid', 'uid_max_txn_per_hour']
    df = df.merge(max_hourly, on='_uid', how='left')

    # Clean up temp columns
    df = df.drop(columns=['_uid', '_hour_bin', '_6h_bin', '_day_bin'], errors='ignore')
    print(f"    -> 10 time-window features created (1h, 6h, 24h)")
    return df


# --- LAYER 20: V-FEATURE INTELLIGENCE (PCA + Aggregations) -----------------
def v_feature_aggregations(df):
    """Extract intelligence from anonymous V columns using PCA and group statistics.
    The V1-V339 columns contain rich signal but are anonymous and noisy."""
    print("  [Layer 20] V-Feature Intelligence...")

    # Find all V columns present
    v_cols = [c for c in df.columns if c.startswith('V') and c[1:].isdigit()]

    if len(v_cols) < 5:
        print("    -> Not enough V columns, skipping")
        return df

    print(f"    -> Found {len(v_cols)} V columns")

    # Fill NaN with column median for V columns (they have lots of missing)
    v_data = df[v_cols].fillna(df[v_cols].median())

    # Group statistics across V columns per row
    df['v_mean'] = v_data.mean(axis=1)
    df['v_std'] = v_data.std(axis=1)
    df['v_max'] = v_data.max(axis=1)
    df['v_min'] = v_data.min(axis=1)
    df['v_range'] = df['v_max'] - df['v_min']
    df['v_nulls'] = df[v_cols].isnull().sum(axis=1)
    df['v_null_pct'] = df['v_nulls'] / len(v_cols)

    # PCA on V columns (top 5 components to capture main patterns)
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        # Standardize before PCA
        scaler = StandardScaler()
        v_scaled = scaler.fit_transform(v_data)

        n_components = min(5, len(v_cols))
        pca = PCA(n_components=n_components, random_state=42)
        v_pca = pca.fit_transform(v_scaled)

        for i in range(n_components):
            df[f'v_pca_{i+1}'] = v_pca[:, i]

        explained = sum(pca.explained_variance_ratio_) * 100
        print(f"    -> PCA: {n_components} components explain {explained:.1f}% variance")
    except Exception as e:
        print(f"    -> PCA failed: {e}")

    print(f"    -> 12 V-feature aggregation features created")
    return df


# --- LAYER 21: FREQUENCY ENCODING ------------------------------------------
def frequency_encoding(df):
    """Encode categorical features by their frequency in the dataset.
    Rare values often correlate with fraud (new cards, unusual addresses)."""
    print("  [Layer 21] Frequency Encoding...")

    freq_cols = ['card1', 'card2', 'addr1', 'addr2', 'P_emaildomain',
                 'R_emaildomain', 'DeviceInfo', 'ProductCD']

    encoded_count = 0
    for col in freq_cols:
        if col not in df.columns:
            continue
        freq = df[col].value_counts(normalize=True)
        df[f'{col}_freq'] = df[col].map(freq).fillna(0)

        # Also create a "rarity" flag — values appearing less than 0.01%
        df[f'{col}_is_rare'] = (df[f'{col}_freq'] < 0.0001).astype(int)
        encoded_count += 1

    print(f"    -> {encoded_count * 2} frequency features created ({encoded_count} freq + {encoded_count} rare flags)")
    return df


# --- LAYER 22: PEER GROUP DEVIATION ----------------------------------------
def peer_group_deviation(df):
    """Compare each transaction to its peer group (same product, card type, region).
    Fraud transactions often deviate significantly from their peers."""
    print("  [Layer 22] Peer Group Deviation Analysis...")

    peer_groups = []

    # Peer group 1: Same ProductCD
    if 'ProductCD' in df.columns:
        stats = df.groupby('ProductCD')['TransactionAmt'].agg(['mean', 'std']).reset_index()
        stats.columns = ['ProductCD', 'peer_product_mean', 'peer_product_std']
        df = df.merge(stats, on='ProductCD', how='left')
        df['peer_product_std'] = df['peer_product_std'].fillna(1)
        df['amt_vs_product_peer'] = (df['TransactionAmt'] - df['peer_product_mean']) / (df['peer_product_std'] + 1)
        peer_groups.append('product')

    # Peer group 2: Same card4 (card type: visa, mastercard, etc)
    if 'card4' in df.columns:
        stats = df.groupby('card4')['TransactionAmt'].agg(['mean', 'std']).reset_index()
        stats.columns = ['card4', 'peer_card4_mean', 'peer_card4_std']
        df = df.merge(stats, on='card4', how='left')
        df['peer_card4_std'] = df['peer_card4_std'].fillna(1)
        df['amt_vs_card_peer'] = (df['TransactionAmt'] - df['peer_card4_mean']) / (df['peer_card4_std'] + 1)
        peer_groups.append('card_type')

    # Peer group 3: Same addr1 (region)
    if 'addr1' in df.columns:
        stats = df.groupby('addr1')['TransactionAmt'].agg(['mean', 'std']).reset_index()
        stats.columns = ['addr1', 'peer_region_mean', 'peer_region_std']
        df = df.merge(stats, on='addr1', how='left')
        df['peer_region_std'] = df['peer_region_std'].fillna(1)
        df['amt_vs_region_peer'] = (df['TransactionAmt'] - df['peer_region_mean']) / (df['peer_region_std'] + 1)
        peer_groups.append('region')

    # Combined deviation: how much does this transaction deviate across ALL peer groups?
    dev_cols = [c for c in df.columns if c.startswith('amt_vs_') and c.endswith('_peer')]
    if dev_cols:
        df['total_peer_deviation'] = df[dev_cols].abs().mean(axis=1)
        df['is_peer_outlier'] = (df['total_peer_deviation'] > 2).astype(int)

    print(f"    -> {len(peer_groups)} peer groups analyzed, {len(dev_cols) + 2} features created")
    return df


# --- LAYER 23: TRANSACTION ENTROPY ----------------------------------------
def transaction_entropy(df):
    """Measure entropy (randomness/diversity) of user behavior.
    Legitimate users have consistent patterns (low entropy).
    Fraudsters show erratic, high-entropy behavior."""
    print("  [Layer 23] Transaction Entropy Analysis...")

    # Build UID for grouping
    df['_uid_ent'] = df['card1'].astype(str) + '_' + df['addr1'].astype(str)

    features_created = 0

    # Entropy of transaction amounts per user (binned)
    if 'TransactionAmt' in df.columns:
        # Create amount bins
        df['_amt_bin'] = pd.cut(df['TransactionAmt'], bins=10, labels=False).fillna(0).astype(int)
        
        # Per-user: how many different amount bins do they use?
        user_amt_diversity = df.groupby('_uid_ent')['_amt_bin'].nunique().reset_index()
        user_amt_diversity.columns = ['_uid_ent', 'user_amt_entropy']
        df = df.merge(user_amt_diversity, on='_uid_ent', how='left')
        df = df.drop(columns=['_amt_bin'], errors='ignore')
        features_created += 1

    # Entropy of product codes per user
    if 'ProductCD' in df.columns:
        user_prod_div = df.groupby('_uid_ent')['ProductCD'].nunique().reset_index()
        user_prod_div.columns = ['_uid_ent', 'user_product_entropy']
        df = df.merge(user_prod_div, on='_uid_ent', how='left')
        features_created += 1

    # Entropy of time-of-day per user (do they always shop at same time?)
    if 'hour_of_day' in df.columns:
        user_hour_div = df.groupby('_uid_ent')['hour_of_day'].nunique().reset_index()
        user_hour_div.columns = ['_uid_ent', 'user_time_entropy']
        df = df.merge(user_hour_div, on='_uid_ent', how='left')
        features_created += 1

    # High entropy flag: users with erratic behavior across all dimensions
    entropy_cols = [c for c in df.columns if c.startswith('user_') and c.endswith('_entropy')]
    if entropy_cols:
        for col in entropy_cols:
            col_median = df[col].median()
            df[f'{col}_high'] = (df[col] > col_median * 2).astype(int)
            features_created += 1

    df = df.drop(columns=['_uid_ent'], errors='ignore')
    print(f"    -> {features_created} entropy features created")
    return df


# --- LAYER 24: LAG / SEQUENTIAL FEATURES ----------------------------------
def lag_features(df):
    """Per-user sequential patterns: previous transaction amount, delta, ratio.
    Fraudsters often show sudden jumps from prior transaction patterns."""
    print("  [Layer 24] Lag / Sequential Features...")

    df['_uid_lag'] = df['card1'].astype(str) + '_' + df['addr1'].astype(str)
    df = df.sort_values(['_uid_lag', 'TransactionDT'])

    # Previous transaction amount (per user)
    df['prev_amt'] = df.groupby('_uid_lag')['TransactionAmt'].shift(1)
    df['prev_amt'] = df['prev_amt'].fillna(df['TransactionAmt'])

    # Delta from previous transaction
    df['amt_delta'] = df['TransactionAmt'] - df['prev_amt']
    df['amt_ratio_prev'] = df['TransactionAmt'] / (df['prev_amt'] + 1)

    # Time since previous transaction (per user)
    df['prev_time'] = df.groupby('_uid_lag')['TransactionDT'].shift(1)
    df['time_since_prev'] = df['TransactionDT'] - df['prev_time'].fillna(df['TransactionDT'])

    # Running rank within user (is this their 1st, 2nd, 3rd txn?)
    df['user_txn_rank'] = df.groupby('_uid_lag').cumcount()

    df = df.drop(columns=['_uid_lag', 'prev_time'], errors='ignore')
    print(f"    -> 6 lag features created")
    return df


# --- LAYER 25: CROSS-FEATURE FRAUD RATES ----------------------------------
def cross_feature_fraud_rates(df):
    """Fraud rates for feature COMBINATIONS (not individual features).
    card1 x ProductCD might have high fraud even if both are safe individually."""
    print("  [Layer 25] Cross-Feature Fraud Rates...")

    if 'isFraud' not in df.columns:
        print("    -> isFraud not found, skipping")
        return df

    global_mean = df['isFraud'].mean()
    smooth = 30
    features_created = 0

    combos = [
        ('card1', 'ProductCD'),
        ('card1', 'addr1'),
        ('addr1', 'P_emaildomain'),
        ('ProductCD', 'P_emaildomain'),
    ]

    for col_a, col_b in combos:
        if col_a not in df.columns or col_b not in df.columns:
            continue
        combo_name = f'{col_a}_x_{col_b}'
        df['_combo'] = df[col_a].astype(str) + '_' + df[col_b].astype(str)

        stats = df.groupby('_combo')['isFraud'].agg(['mean', 'count']).reset_index()
        stats.columns = ['_combo', '_mean', '_count']
        stats[f'{combo_name}_fraud_rate'] = (
            (stats['_count'] * stats['_mean'] + smooth * global_mean) /
            (stats['_count'] + smooth)
        )

        df = df.merge(stats[['_combo', f'{combo_name}_fraud_rate']], on='_combo', how='left')
        df = df.drop(columns=['_combo'], errors='ignore')
        features_created += 1

    print(f"    -> {features_created} cross-feature fraud rates created")
    return df


# --- TARGET ENCODING (Layer 17) ------------------------------------------
def target_encoding(df):
    """Smoothed target encoding for high-cardinality categorical features."""
    print("  [Layer 17] Target Encoding (Smoothed)...")

    if 'isFraud' not in df.columns:
        print("    -> isFraud column not found, skipping")
        return df

    global_mean = df['isFraud'].mean()
    smooth_weight = 50  # Smoothing factor to prevent overfitting

    target_cols = ['card1', 'card2', 'addr1', 'addr2', 'P_emaildomain']
    encoded_count = 0

    for col in target_cols:
        if col not in df.columns:
            continue

        # Compute per-category stats
        stats = df.groupby(col)['isFraud'].agg(['mean', 'count'])
        # Smoothed target encoding: (count * mean + smooth * global) / (count + smooth)
        stats['te'] = (stats['count'] * stats['mean'] + smooth_weight * global_mean) / \
                       (stats['count'] + smooth_weight)

        new_col = f'{col}_target_enc'
        df[new_col] = df[col].map(stats['te']).fillna(global_mean)
        encoded_count += 1

    print(f"    -> {encoded_count} target-encoded features created")
    return df


# --- MISSING VALUE INDICATORS ---------------------------------------------
def missing_indicators(df):
    """Missingness itself is a fraud signal."""
    print("  [Bonus] Missing Value Indicators...")

    for col in ['card2', 'addr1', 'dist1', 'P_emaildomain']:
        if col in df.columns:
            df[f'is_missing_{col}'] = df[col].isnull().astype(int)

    print(f"    -> Missing indicators for card2, addr1, dist1, P_emaildomain")
    return df
