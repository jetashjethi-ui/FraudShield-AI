"""
FraudShield AI — Risk Scoring & Explainability Engine
Layers 1 (XAI) and 5 (Adaptive Auth) implementation.
"""

import pandas as pd
import numpy as np


def compute_risk_scores(df, predictions):
    """
    Compute weighted risk scores (0-100) using all 11 layer signals.
    Layer 5: Adaptive Step-Up Authentication.
    """
    print("\n" + "=" * 70)
    print("RISK SCORING ENGINE")
    print("=" * 70)

    # Extract signals (ensure they exist with defaults)
    ml_proba = predictions.get('ensemble_proba', np.zeros(len(df)))
    iso_scores = predictions.get('iso_scores', np.zeros(len(df)))

    def safe_col(col, default=0):
        return df[col].fillna(default).values if col in df.columns else np.full(len(df), default)

    # Weighted risk formula
    risk_raw = (
        ml_proba * 35 +                                        # ML ensemble probability
        iso_scores * 20 +                                       # Isolation Forest anomaly
        safe_col('is_unusual_device') * 10 +                    # SIM swap (Layer 3)
        safe_col('is_shared_device') * 7 +                      # Mule network (Layer 7)
        safe_col('is_low_activity_user') * 5 +                  # Dormant account (Layer 8)
        safe_col('first_txn_high_value') * 8 +                  # New account abuse (Layer 11)
        safe_col('is_suspicious_round') * 5 +                   # Round amount (Layer 9)
        safe_col('is_night') * np.clip(safe_col('amount_zscore'), 0, 5) * 2 +  # Night + unusual
        safe_col('product_fraud_rate') * 3 +                    # Risky merchant (Layer 6)
        safe_col('round_x_new_device') * 3 +                    # Combo feature
        safe_col('category_mismatch_risk') * 2                  # Category mismatch (Layer 10)
    )

    # Normalize to 0-100
    risk_max = risk_raw.max() if risk_raw.max() > 0 else 1
    risk_score = np.clip((risk_raw / risk_max) * 100, 0, 100)

    # Risk categories (Layer 5: Adaptive Step-Up Auth)
    risk_category = np.where(
        risk_score >= 71, 'RED_BLOCK',
        np.where(risk_score >= 51, 'ORANGE_BIOMETRIC',
                 np.where(risk_score >= 31, 'YELLOW_PIN_VERIFY',
                          'GREEN_APPROVE'))
    )

    auth_recommendation = np.where(
        risk_category == 'RED_BLOCK',
        'BLOCK transaction. Notify customer. Explain reason. Flag for investigation.',
        np.where(risk_category == 'ORANGE_BIOMETRIC',
                 'Request biometric re-verification before proceeding.',
                 np.where(risk_category == 'YELLOW_PIN_VERIFY',
                          'Request PIN re-entry for confirmation.',
                          'Auto-approve. No additional authentication needed.'))
    )

    print(f"  Risk score range: {risk_score.min():.1f} to {risk_score.max():.1f}")
    print(f"  Risk distribution:")
    for cat in ['GREEN_APPROVE', 'YELLOW_PIN_VERIFY', 'ORANGE_BIOMETRIC', 'RED_BLOCK']:
        count = (risk_category == cat).sum()
        pct = count / len(risk_category) * 100
        print(f"    {cat:20s}: {count:>8,} ({pct:5.1f}%)")

    return risk_score, risk_category, auth_recommendation


def generate_explanations(df, risk_score):
    """
    Layer 1: Explainable AI — generate human-readable reasons for each flagged transaction.
    """
    print("\n[XAI] Generating explanations for flagged transactions...")

    def safe_val(row, col, default=0):
        return row.get(col, default) if pd.notna(row.get(col, None)) else default

    explanations = []
    for idx, row in df.iterrows():
        reasons = []

        # Amount anomaly
        zscore = safe_val(row, 'amount_zscore')
        ratio = safe_val(row, 'amount_to_mean_ratio', 1)
        if zscore > 3:
            reasons.append(f"Amount is {ratio:.1f}x above user average")

        # Night transaction
        if safe_val(row, 'is_night') == 1:
            hour = safe_val(row, 'hour_of_day')
            reasons.append(f"Transaction at unusual hour ({int(hour)}:00)")

        # Unusual device (SIM swap)
        if safe_val(row, 'is_unusual_device') == 1:
            reasons.append("New/unusual device detected - possible SIM swap or account takeover")

        # Shared device (mule network)
        if safe_val(row, 'is_shared_device') == 1:
            reasons.append("Device shared across multiple identities - possible mule network")

        # New account high value
        if safe_val(row, 'first_txn_high_value') == 1:
            reasons.append("Brand-new account with unusually high first transaction")

        # Round amount
        if safe_val(row, 'is_suspicious_round') == 1:
            reasons.append("Perfectly round high-value amount (common fraud pattern)")

        # Dormant account
        if safe_val(row, 'is_low_activity_user') == 1 and safe_val(row, 'TransactionAmt', 0) > safe_val(row, 'avg_amt', 999999) * 2:
            reasons.append("Dormant account suddenly active with high-value transaction")

        # High-risk merchant
        pf_rate = safe_val(row, 'product_fraud_rate')
        if pf_rate > 0.05:
            reasons.append(f"High-risk merchant category (fraud rate: {pf_rate*100:.1f}%)")

        # Category mismatch
        if safe_val(row, 'category_mismatch_risk') == 1:
            reasons.append("Transaction in unusual merchant category for this user")

        # Seasonal anomaly
        seasonal = safe_val(row, 'amt_vs_seasonal', 1)
        if seasonal > 3:
            reasons.append(f"Amount is {seasonal:.1f}x above seasonal baseline for this user")

        if not reasons:
            reasons.append("No specific risk factors identified")

        explanations.append(" | ".join(reasons))

    return explanations


def build_output_table(df, risk_score, risk_category, auth_recommendation, explanations, predictions):
    """Create the final output table with all risk intelligence."""
    print("[OUTPUT] Building scored output table...")

    output = pd.DataFrame({
        'TransactionID': df['TransactionID'].values,
        'TransactionAmt': df['TransactionAmt'].values,
        'isFraud_actual': df['isFraud'].values,
        'ml_fraud_probability': predictions['ensemble_proba'],
        'risk_score': risk_score,
        'risk_category': risk_category,
        'auth_recommendation': auth_recommendation,
        'explanation': explanations,
    })

    return output
