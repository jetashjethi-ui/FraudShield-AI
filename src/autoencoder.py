"""
FraudShield AI -- Deep Learning Autoencoder for Anomaly Detection
Trains on normal (non-fraud) transactions only, then uses reconstruction
error to score how 'abnormal' each transaction is. High reconstruction error
= likely fraud. This is a fundamentally different approach from the
supervised models — it learns the 'shape' of legitimate transactions.

Complements supervised models by detecting anomalies without labels.
"""

import numpy as np
import pandas as pd
import os
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import roc_auc_score


class FraudAutoencoder:
    """
    Autoencoder-based anomaly detector for fraud detection.
    
    Architecture:
        Input(N) -> Encoder(64) -> Bottleneck(16) -> Decoder(64) -> Output(N)
    
    The bottleneck forces the network to learn a compressed representation
    of normal transactions. Fraudulent transactions produce high
    reconstruction error because they don't 'fit' the learned patterns.
    """

    def __init__(self, encoding_dim=16, hidden_dim=64):
        self.encoding_dim = encoding_dim
        self.hidden_dim = hidden_dim
        self.scaler = StandardScaler()
        self.encoder = None
        self.autoencoder = None
        self.threshold = None
        self.train_errors = None

    def fit(self, X_train, y_train=None):
        """
        Train autoencoder on NORMAL transactions only.
        If y_train is provided, filters to only non-fraud samples.
        """
        # Filter to normal transactions only
        if y_train is not None:
            if hasattr(y_train, 'values'):
                mask = y_train.values == 0
            else:
                mask = y_train == 0
            if hasattr(X_train, 'iloc'):
                X_normal = X_train.iloc[mask]
            else:
                X_normal = X_train[mask]
        else:
            X_normal = X_train

        print(f"    -> Training on {len(X_normal):,} normal transactions")

        # Scale features
        X_scaled = self.scaler.fit_transform(X_normal)

        # Replace any NaN/Inf after scaling
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        n_features = X_scaled.shape[1]

        # Build autoencoder as MLPRegressor (input -> hidden -> bottleneck -> hidden -> input)
        # The "autoencoder" learns to reconstruct its input through a bottleneck
        self.autoencoder = MLPRegressor(
            hidden_layer_sizes=(self.hidden_dim, self.encoding_dim, self.hidden_dim),
            activation='relu',
            solver='adam',
            max_iter=100,
            batch_size=256,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42,
            verbose=False,
        )

        # Train: input = output (reconstruct itself)
        self.autoencoder.fit(X_scaled, X_scaled)

        # Compute reconstruction errors on training data to set threshold
        reconstructed = self.autoencoder.predict(X_scaled)
        self.train_errors = np.mean((X_scaled - reconstructed) ** 2, axis=1)

        # Set threshold at 95th percentile of normal errors
        self.threshold = np.percentile(self.train_errors, 95)
        print(f"    -> Autoencoder trained ({self.autoencoder.n_iter_} epochs)")
        print(f"    -> Mean reconstruction error: {self.train_errors.mean():.6f}")
        print(f"    -> Anomaly threshold (95th pct): {self.threshold:.6f}")

        return self

    def predict_anomaly_scores(self, X):
        """
        Compute anomaly scores for new data.
        Returns normalized scores between 0 (normal) and 1 (highly anomalous).
        """
        X_scaled = self.scaler.transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        reconstructed = self.autoencoder.predict(X_scaled)
        errors = np.mean((X_scaled - reconstructed) ** 2, axis=1)

        # Normalize: 0 = no anomaly, 1 = very anomalous
        # Use training error stats for normalization
        max_error = max(np.percentile(self.train_errors, 99.9), self.threshold * 3)
        scores = np.clip(errors / max_error, 0, 1)

        return scores

    def evaluate(self, X_test, y_test):
        """Evaluate autoencoder's fraud detection ability."""
        scores = self.predict_anomaly_scores(X_test)
        auc = roc_auc_score(y_test, scores)

        # Binary predictions using threshold
        X_scaled = self.scaler.transform(X_test)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        reconstructed = self.autoencoder.predict(X_scaled)
        errors = np.mean((X_scaled - reconstructed) ** 2, axis=1)
        predictions = (errors > self.threshold).astype(int)

        return {
            'auc': auc,
            'scores': scores,
            'predictions': predictions,
            'mean_error_normal': float(errors[y_test == 0].mean()) if (y_test == 0).any() else 0,
            'mean_error_fraud': float(errors[y_test == 1].mean()) if (y_test == 1).any() else 0,
        }


def run_autoencoder_detection(X_train_original, y_train_original, X_test, y_test, output_dir):
    """
    Train and evaluate the autoencoder anomaly detector.
    Uses ORIGINAL (pre-SMOTE) training data for realistic normal patterns.
    """
    print("\n" + "=" * 70)
    print("DEEP LEARNING AUTOENCODER (Anomaly Detection)")
    print("=" * 70)
    print("  [INFO] Training on normal transactions only (unsupervised)")
    print("  [INFO] Architecture: Input -> 64 -> 16 (bottleneck) -> 64 -> Output")

    ae = FraudAutoencoder(encoding_dim=16, hidden_dim=64)
    ae.fit(X_train_original, y_train_original)

    # Evaluate
    eval_results = ae.evaluate(X_test, y_test)
    print(f"\n  [RESULT] Autoencoder AUC: {eval_results['auc']:.4f}")
    print(f"  [RESULT] Mean error (normal): {eval_results['mean_error_normal']:.6f}")
    print(f"  [RESULT] Mean error (fraud):  {eval_results['mean_error_fraud']:.6f}")
    print(f"  [RESULT] Error ratio (fraud/normal): {eval_results['mean_error_fraud'] / max(eval_results['mean_error_normal'], 1e-8):.2f}x")

    # Save results
    ae_results = {
        'name': 'Deep Autoencoder',
        'type': 'unsupervised_anomaly_detection',
        'architecture': 'Input -> 64 -> 16 (bottleneck) -> 64 -> Output',
        'auc': eval_results['auc'],
        'mean_error_normal': eval_results['mean_error_normal'],
        'mean_error_fraud': eval_results['mean_error_fraud'],
        'error_ratio': eval_results['mean_error_fraud'] / max(eval_results['mean_error_normal'], 1e-8),
        'threshold': float(ae.threshold),
        'training_epochs': ae.autoencoder.n_iter_,
    }

    results_path = os.path.join(output_dir, "results", "autoencoder_results.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(ae_results, f, indent=2, default=str)

    print(f"  [SAVE] Results saved to {results_path}")

    return ae, eval_results
