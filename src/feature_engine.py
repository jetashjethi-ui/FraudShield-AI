"""
FraudShield AI — Feature Engineering Module
Implements all 15 Detection Layers as engineered features.
"""

import pandas as pd
import numpy as np


def build_all_features(df):
    """Master function: build all 15 layers of features."""
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING — 15 DETECTION LAYERS")
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
    df = velocity_features(df)        # Layer 12 — NEW
    df = email_risk_features(df)      # Layer 13 — NEW
    df = network_features(df)         # Layer 14 — NEW
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

    print(f"    → 13 interaction features created")
    return df


# ─── MISSING VALUE INDICATORS ────────────────────────────────────────
def missing_indicators(df):
    """Missingness itself is a fraud signal."""
    print("  [Bonus] Missing Value Indicators...")

    for col in ['card2', 'addr1', 'dist1', 'P_emaildomain']:
        if col in df.columns:
            df[f'is_missing_{col}'] = df[col].isnull().astype(int)

    print(f"    → Missing indicators for card2, addr1, dist1, P_emaildomain")
    return df
