"""Fast preflight checks for the training/serving backend contract."""

import asyncio
import copy
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def training_frame(rows=240):
    rng = np.random.RandomState(17)
    amount = rng.lognormal(mean=4.2, sigma=0.7, size=rows).astype(np.float32)
    return pd.DataFrame({
        "TransactionID": np.arange(rows, dtype=np.int32),
        "TransactionDT": np.arange(rows, dtype=np.int32) * 3600,
        "TransactionAmt": amount,
        "card1": rng.randint(100, 900, size=rows).astype(np.float32),
        "ProductCD": pd.Categorical(rng.choice(["W", "C", "R"], size=rows)),
        "DeviceInfo": pd.Categorical(rng.choice(["phone-a", "phone-b", None], size=rows)),
        "isFraud": (rng.rand(rows) < 0.18).astype(np.int8),
    })


def test_feature_contract_is_stable_for_unseen_and_missing_categories():
    from src.inference_pipeline import FraudFeaturePipeline

    train = training_frame()
    original = train.copy(deep=True)
    pipeline = FraudFeaturePipeline().fit(train)
    holdout = pd.DataFrame({
        "TransactionAmt": [50.0, np.nan],
        "TransactionDT": [2_000_000, 2_100_000],
        "card1": [220.0, 330.0],
        "ProductCD": ["unseen-product", None],
        "DeviceInfo": ["brand-new-device", None],
    })

    batch = pipeline.transform(holdout, update_state=False)
    sequential_pipeline = copy.deepcopy(pipeline)
    one_by_one = pd.concat(
        [sequential_pipeline.transform(holdout.iloc[[index]]) for index in range(len(holdout))],
        ignore_index=True,
    )

    assert list(batch.columns) == pipeline.get_feature_names_out()
    assert batch.columns.equals(one_by_one.columns)
    np.testing.assert_allclose(batch.to_numpy(), one_by_one.to_numpy(), rtol=0, atol=1e-6)
    assert batch.dtypes.eq(np.dtype("float32")).all()
    assert np.isfinite(batch.to_numpy()).all()
    assert "isFraud" not in batch.columns
    pd.testing.assert_frame_equal(train, original)


def test_training_pipeline_does_not_learn_from_holdout_labels_or_categories():
    from src.inference_pipeline import FraudFeaturePipeline

    train = training_frame()
    pipeline = FraudFeaturePipeline().fit(train.drop(columns="isFraud"))
    assert "holdout-only" not in pipeline.category_levels_.get("ProductCD", [])
    assert "isFraud" not in pipeline.transform(train).columns


def test_entity_history_features_are_prior_only_and_holdout_state_is_not_committed():
    from src.inference_pipeline import FraudFeaturePipeline

    pipeline = FraudFeaturePipeline().fit(training_frame().drop(columns="isFraud"))
    events = pd.DataFrame({
        "TransactionAmt": [100.0, 200.0, 80.0],
        "card1": [44, 44, 44],
        "addr1": [12, 12, 13],
        "DeviceInfo": ["same-device", "same-device", "same-device"],
        "P_emaildomain": ["mail.test", "mail.test", "mail.test"],
        "TransactionDT": [1, 2, 3],
    })

    scored = pipeline.transform(events, update_state=False)
    assert scored["card1_prior_7d_count_log1p"].iloc[0] == 0
    assert scored["card1_prior_7d_count_log1p"].iloc[1] == pytest.approx(np.log1p(1))
    assert scored["card1_prior_7d_mean_amount"].iloc[1] == pytest.approx(100.0)
    assert scored["card1_amount_vs_prior_7d_mean"].iloc[1] == pytest.approx(2.0)
    assert scored["card1_addr1_prior_7d_count_log1p"].iloc[2] == 0
    assert not pipeline.history_state_["card1"]

    live = pipeline.transform(events.iloc[:2])
    assert live["card1_prior_7d_mean_amount"].iloc[1] == pytest.approx(100.0)
    events_seen, amount_sum = pipeline.history_state_["card1"]["44"]
    assert len(events_seen) == 2
    assert amount_sum == pytest.approx(300.0)


def test_api_fails_closed_without_a_complete_bundle(tmp_path, monkeypatch):
    import api
    from fastapi import Response

    monkeypatch.setattr(api, "ARTIFACTS_DIR", tmp_path)
    assert api.load_models() is False
    response = Response()
    result = asyncio.run(api.readiness(response))
    assert result["ready"] is False
    assert response.status_code == 503


def test_single_and_batch_scoring_use_identical_probability(monkeypatch):
    import api
    from src.inference_pipeline import FraudFeaturePipeline

    class FixedClassifier:
        def __init__(self, probability):
            self.probability = probability

        def predict_proba(self, X):
            p = np.full(len(X), self.probability)
            return np.column_stack([1 - p, p])

    class FixedBooster:
        def predict(self, X):
            return np.full(len(X), 0.25)

    class IdentityScaler:
        def transform(self, X):
            return X

    class FixedCalibrator:
        def predict(self, scores):
            return np.asarray(scores)

    pipeline = FraudFeaturePipeline().fit(training_frame().drop(columns="isFraud"))
    features = pipeline.get_feature_names_out()
    monkeypatch.setattr(api, "feature_pipeline", pipeline)
    monkeypatch.setattr(api, "feature_names", features)
    monkeypatch.setattr(api, "decision_threshold", 0.4)
    monkeypatch.setattr(api, "models", {
        "xgboost": FixedClassifier(0.2),
        "lightgbm": FixedBooster(),
        "catboost": FixedClassifier(0.25),
        "random_forest": FixedClassifier(0.3),
        "scaler": IdentityScaler(),
        "mlp": FixedClassifier(0.4),
        "meta_learner": FixedClassifier(0.61),
        "probability_calibrator": FixedCalibrator(),
    })
    txn = api.TransactionRequest(TransactionAmt=430.0, ProductCD="W", card1=250)
    single = asyncio.run(api.predict(txn))
    batch = asyncio.run(api.batch_score(api.BatchRequest(transactions=[txn])))

    assert single["fraud_probability"] == batch["results"][0].fraud_probability
    assert single["risk_score"] == batch["results"][0].risk_score


def test_failed_prediction_does_not_commit_history(monkeypatch):
    import api
    from fastapi import HTTPException
    from src.inference_pipeline import FraudFeaturePipeline

    class FailingClassifier:
        def predict_proba(self, X):
            raise RuntimeError("simulated model failure")

    pipeline = FraudFeaturePipeline().fit(training_frame().drop(columns="isFraud"))
    monkeypatch.setattr(api, "feature_pipeline", pipeline)
    monkeypatch.setattr(api, "feature_names", pipeline.get_feature_names_out())
    monkeypatch.setattr(api, "models", {"xgboost": FailingClassifier()})
    txn = api.TransactionRequest(TransactionAmt=430.0, ProductCD="W", card1=250)

    with pytest.raises(HTTPException) as error:
        api.score_transaction(txn)

    assert error.value.status_code == 500
    assert not pipeline.history_state_["card1"]
    assert pipeline.history_last_processed_time_ is None


def test_batch_rejects_partial_timestamps_before_scoring(monkeypatch):
    import api
    from datetime import datetime, timezone
    from fastapi import HTTPException

    scored = []
    monkeypatch.setattr(api, "score_transaction", lambda txn: scored.append(txn))
    request = api.BatchRequest(transactions=[
        api.TransactionRequest(
            TransactionAmt=20.0,
            transaction_timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
        ),
        api.TransactionRequest(TransactionAmt=30.0),
    ])

    with pytest.raises(HTTPException) as error:
        asyncio.run(api.batch_score(request))

    assert error.value.status_code == 422
    assert scored == []


def test_temporal_stack_smoke_uses_oof_training_and_returns_api_models(monkeypatch):
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    import src.models as model_module

    profile = {
        "available_ram_gb": 8.0,
        "gpu_free_gb": 0.0,
        "use_gpu": False,
        "xgb_gpu": False,
        "catboost_gpu": False,
        "n_jobs": 1,
    }
    monkeypatch.setattr(model_module, "training_profile", lambda: profile)
    monkeypatch.setattr(model_module, "_build_base_models", lambda _profile: {
        "xgboost": {"name": "XGBoost smoke", "model": LogisticRegression(max_iter=40), "needs_scaling": False, "gpu_enabled": False},
        "lightgbm": {"name": "LightGBM smoke", "model": LogisticRegression(max_iter=40), "needs_scaling": False, "gpu_enabled": False},
        "catboost": {"name": "CatBoost smoke", "model": HistGradientBoostingClassifier(max_iter=5), "needs_scaling": False, "gpu_enabled": False},
        "random_forest": {"name": "RF smoke", "model": RandomForestClassifier(n_estimators=5, max_depth=3, random_state=2), "needs_scaling": False, "gpu_enabled": False},
        "mlp": {"name": "MLP smoke", "model": LogisticRegression(max_iter=40), "needs_scaling": True, "gpu_enabled": False},
    })
    rng = np.random.RandomState(6)
    rows = 480
    X = pd.DataFrame(rng.normal(size=(rows, 6)).astype(np.float32), columns=[f"f{i}" for i in range(6)])
    y = pd.Series((rng.rand(rows) < 0.2).astype(np.int8))

    trained, results, probabilities = model_module.train_models(
        X.iloc[:380], y.iloc[:380], X.iloc[380:], y.iloc[380:]
    )

    assert {"xgboost", "lightgbm", "random_forest", "mlp", "meta_learner", "probability_calibrator"} <= set(trained)
    assert "ensemble" in results
    assert len(probabilities["ensemble_proba"]) == len(y.iloc[380:])
    assert np.isfinite(probabilities["ensemble_proba"]).all()
