"""FraudShield scoring API.

The API loads the same fitted feature pipeline and model stack produced by
``main.py``. It fails closed when artifacts are missing rather than returning
an invented neutral probability.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import hmac
import json
import os
import pickle
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from src.inference_pipeline import FraudFeaturePipeline

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "outputs" / "model_artifacts"
RESULTS_DIR = BASE_DIR / "outputs" / "results"


@asynccontextmanager
async def lifespan(_app):
    load_models()
    yield

app = FastAPI(
    title="FraudShield AI Scoring API",
    description="Transaction risk scoring using the fitted preprocessing and model stack.",
    version="5.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

origins = [item.strip() for item in os.getenv(
    "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
).split(",") if item.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)
models: dict = {}
_feature_state_lock = threading.RLock()
feature_names: list[str] = []
feature_pipeline: Optional[FraudFeaturePipeline] = None
decision_threshold = 0.5
model_error: Optional[str] = None
started_at = time.monotonic()


class TransactionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    TransactionAmt: float = Field(..., ge=0, le=1_000_000_000, allow_inf_nan=False)
    ProductCD: str = "W"
    card1: Optional[int] = None
    card2: Optional[float] = None
    card3: Optional[float] = None
    card4: Optional[str] = None
    card5: Optional[float] = None
    card6: Optional[str] = None
    addr1: Optional[float] = None
    addr2: Optional[float] = None
    dist1: Optional[float] = None
    P_emaildomain: Optional[str] = None
    DeviceInfo: Optional[str] = None
    DeviceType: Optional[str] = None
    hour_of_day: int = Field(default=12, ge=0, le=23)
    transaction_timestamp: Optional[datetime] = None
    extra_features: dict[str, str | int | float | None] = Field(default_factory=dict, max_length=500)


class BatchRequest(BaseModel):
    transactions: list[TransactionRequest] = Field(..., min_length=1, max_length=500)


class ROIRequest(BaseModel):
    annual_transactions: int = Field(..., gt=0)
    fraud_rate: float = Field(..., gt=0, lt=1, allow_inf_nan=False)
    average_fraud_loss: float = Field(..., ge=0, allow_inf_nan=False)
    false_positive_cost: float = Field(..., ge=0, allow_inf_nan=False)
    manual_review_cost: float = Field(..., ge=0, allow_inf_nan=False)
    baseline_recall: float = Field(..., ge=0, le=1, allow_inf_nan=False)
    baseline_false_positive_rate: float = Field(..., ge=0, le=1, allow_inf_nan=False)


class RiskResponse(BaseModel):
    transaction_id: str
    risk_score: float
    risk_category: str
    auth_recommendation: str
    fraud_probability: float
    base_model_probabilities: dict[str, float]
    context_signals: list[str]
    processing_time_ms: float
    timestamp: str


def _check_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    expected = os.getenv("FRAUDSHIELD_API_KEY")
    if expected and (not x_api_key or not hmac.compare_digest(x_api_key, expected)):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def load_models() -> bool:
    """Load a complete, mutually compatible serving bundle."""
    global feature_pipeline, feature_names, decision_threshold, model_error
    models.clear()
    model_error = None
    try:
        with open(ARTIFACTS_DIR / "bundle.json", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("schema_version") != 4:
            raise ValueError("Incompatible model bundle schema")
        with open(ARTIFACTS_DIR / "feature_pipeline.pkl", "rb") as handle:
            feature_pipeline = pickle.load(handle)
        with open(ARTIFACTS_DIR / "feature_names.json", encoding="utf-8") as handle:
            feature_names = json.load(handle)
        with open(ARTIFACTS_DIR / "stacking_config.json", encoding="utf-8") as handle:
            config = json.load(handle)

        import xgboost as xgb
        import lightgbm as lgb
        from catboost import CatBoostClassifier

        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(str(ARTIFACTS_DIR / "xgboost_model.json"))
        models["xgboost"] = xgb_model
        models["lightgbm"] = lgb.Booster(model_file=str(ARTIFACTS_DIR / "lightgbm_model.txt"))
        catboost_model = CatBoostClassifier()
        catboost_model.load_model(str(ARTIFACTS_DIR / "catboost_model.cbm"))
        models["catboost"] = catboost_model

        for name in ("random_forest", "mlp", "scaler", "meta_learner", "probability_calibrator"):
            with open(ARTIFACTS_DIR / f"{name}.pkl", "rb") as handle:
                models[name] = pickle.load(handle)

        expected_models = ["xgboost", "lightgbm", "catboost", "random_forest", "mlp"]
        if config.get("base_models") != expected_models:
            raise ValueError("Saved stack model order is incompatible with this API version")
        if manifest.get("base_models") != expected_models:
            raise ValueError("Bundle manifest model order is incompatible with this API version")
        if not feature_names or not isinstance(feature_pipeline, FraudFeaturePipeline):
            raise ValueError("Feature preprocessing metadata is missing or invalid")
        if manifest.get("feature_count") != len(feature_names):
            raise ValueError("Bundle feature count does not match the saved feature list")
        decision_threshold = float(config.get("decision_threshold", 0.5))
        if not 0 <= decision_threshold <= 1:
            raise ValueError("Saved decision threshold is outside [0, 1]")
        print(f"[API] Loaded complete model bundle with {len(feature_names)} features")
        return True
    except Exception as exc:
        models.clear()
        feature_pipeline = None
        feature_names = []
        model_error = str(exc)
        print(f"[API] Model bundle unavailable: {exc}")
        return False


def _input_frame(txn: TransactionRequest) -> pd.DataFrame:
    row = txn.model_dump(exclude={"transaction_timestamp", "extra_features"})
    allowed = set(feature_pipeline.numeric_columns_) | set(feature_pipeline.categorical_columns_)
    unknown = set(txn.extra_features) - allowed
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown model input field(s): {', '.join(sorted(unknown)[:10])}")
    overlap = set(txn.extra_features) & set(row)
    if overlap:
        raise HTTPException(status_code=422, detail="extra_features cannot override declared transaction fields")
    row.update(txn.extra_features)
    timestamp = txn.transaction_timestamp
    if timestamp is not None:
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        dt = timestamp.astimezone(timezone.utc)
        row["hour_of_day"] = dt.hour
        row["TransactionDT"] = dt.timestamp()
    else:
        # Anchor timestamp-free demo calls to live arrival time so rolling
        # history features can expire events instead of drifting forever.
        row["TransactionDT"] = time.time()
    return pd.DataFrame([row])


def score_transaction(txn: TransactionRequest) -> RiskResponse:
    if feature_pipeline is None or not models:
        raise HTTPException(status_code=503, detail="Scoring model is not ready")
    start = time.perf_counter()
    try:
        # Preview against a local history copy, then commit only if all model
        # and calibration steps succeed. Keep the lock through commit so
        # concurrent requests cannot score against the same history state.
        with _feature_state_lock:
            model_input = _input_frame(txn)
            X = feature_pipeline.transform(model_input, update_state=False).reindex(
                columns=feature_names, fill_value=0
            )
            base_scores: dict[str, float] = {}
            base_scores["xgboost"] = float(models["xgboost"].predict_proba(X)[0, 1])
            # LightGBM binary Booster.predict returns probabilities by default.
            base_scores["lightgbm"] = float(models["lightgbm"].predict(X)[0])
            base_scores["catboost"] = float(models["catboost"].predict_proba(X)[0, 1])
            base_scores["random_forest"] = float(models["random_forest"].predict_proba(X)[0, 1])
            scaled = models["scaler"].transform(X)
            base_scores["mlp"] = float(models["mlp"].predict_proba(scaled)[0, 1])
            ordered = [base_scores[name] for name in (
                "xgboost", "lightgbm", "catboost", "random_forest", "mlp"
            )]
            raw_probability = float(models["meta_learner"].predict_proba(np.asarray([ordered]))[0, 1])
            fraud_probability = float(models["probability_calibrator"].predict([raw_probability])[0])
            if not np.isfinite(fraud_probability):
                raise ValueError("Model returned a non-finite probability")

            p = min(1.0, max(0.0, fraud_probability))
            risk_score = round(p * 100.0, 1)
            if p >= 0.75:
                category, action = "RED_BLOCK", "Block and send for human review."
            elif p >= 0.35:
                category, action = "ORANGE_BIOMETRIC", "Request step-up verification and review."
            elif p >= decision_threshold:
                category, action = "YELLOW_PIN_VERIFY", "Request additional verification."
            else:
                category, action = "GREEN_APPROVE", "Approve under the configured decision threshold."

            reasons = []
            if txn.TransactionAmt > 1000:
                reasons.append(f"High transaction amount ({txn.TransactionAmt:,.2f})")
            if txn.hour_of_day <= 6 or txn.hour_of_day >= 23:
                reasons.append(f"Transaction occurred at an unusual hour ({txn.hour_of_day}:00)")
            if p >= decision_threshold:
                reasons.append(f"Stacked model probability exceeds the decision threshold ({decision_threshold:.3f})")
            if not reasons:
                reasons.append("No additional rule-based risk factors identified")

            response = RiskResponse(
                transaction_id=f"TXN-{uuid.uuid4().hex}",
                risk_score=risk_score,
                risk_category=category,
                auth_recommendation=action,
                fraud_probability=round(p, 6),
                base_model_probabilities={key: round(value, 6) for key, value in base_scores.items()},
                context_signals=reasons,
                processing_time_ms=0.0,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

            # Only successful predictions become part of future rolling history.
            feature_pipeline.transform(model_input, update_state=True)
            response.processing_time_ms = round((time.perf_counter() - start) * 1000, 2)
            return response
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Transaction scoring failed") from exc


@app.get("/", tags=["Info"])
async def root():
    return {
        "service": "FraudShield AI Scoring API",
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
        "readiness": "/ready",
    }


@app.post("/predict", tags=["Scoring"], dependencies=[Depends(_check_api_key)])
async def predict(txn: TransactionRequest):
    result = score_transaction(txn)
    return {
        "risk_score": result.risk_score,
        "risk_category": result.risk_category,
        "fraud_probability": result.fraud_probability,
        "reasons": [{"text": signal} for signal in result.context_signals],
    }


@app.post("/score", response_model=RiskResponse, tags=["Scoring"], dependencies=[Depends(_check_api_key)])
async def score(txn: TransactionRequest):
    return score_transaction(txn)


@app.post("/batch", tags=["Scoring"], dependencies=[Depends(_check_api_key)])
async def batch_score(request: BatchRequest):
    transactions = request.transactions
    has_timestamp = [txn.transaction_timestamp is not None for txn in transactions]
    if any(has_timestamp) and not all(has_timestamp):
        raise HTTPException(
            status_code=422,
            detail="Provide transaction_timestamp for every transaction in a batch, or omit it for all.",
        )
    if all(has_timestamp):
        def timestamp_key(index):
            value = transactions[index].transaction_timestamp
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc).timestamp()

        order = sorted(range(len(transactions)), key=timestamp_key)
    else:
        # Without timestamps, the client-supplied order defines the sequence
        # for prior-transaction features.
        order = range(len(transactions))
    ordered_results = [None] * len(transactions)
    for index in order:
        ordered_results[index] = score_transaction(transactions[index])
    return {"results": ordered_results, "count": len(transactions)}


@app.get("/health", tags=["System"])
async def health():
    """Liveness endpoint: the process is running even if artifacts are absent."""
    ready = feature_pipeline is not None and bool(models)
    return {
        "status": "healthy" if ready else "degraded",
        "ready": ready,
        "models_loaded": len(models),
        "features_count": len(feature_names),
        "uptime_seconds": round(time.monotonic() - started_at, 2),
    }


@app.get("/ready", tags=["System"])
async def readiness(response: Response):
    """Readiness endpoint for load balancers and deployment health checks."""
    ready = feature_pipeline is not None and bool(models)
    if not ready:
        response.status_code = 503
    return {"ready": ready, "models_loaded": len(models)}


@app.get("/model-info", tags=["System"])
async def model_info():
    return {
        "ready": feature_pipeline is not None and bool(models),
        "models_loaded": [name for name in models if name not in {"scaler"}],
        "feature_count": len(feature_names),
        "stacking_order": ["xgboost", "lightgbm", "catboost", "random_forest", "mlp"],
        "decision_threshold": decision_threshold,
    }


@app.get("/metrics", tags=["System"])
async def model_metrics():
    """Return the saved, chronological holdout evaluation metrics for the UI."""
    metrics_path = RESULTS_DIR / "model_metrics.json"
    if not metrics_path.is_file():
        raise HTTPException(status_code=503, detail="Holdout model metrics are unavailable")
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=503, detail="Holdout model metrics are invalid") from exc
    if not isinstance(metrics, dict) or not metrics:
        raise HTTPException(status_code=503, detail="Holdout model metrics are invalid")
    return metrics


@app.get("/roi", tags=["Business Intelligence"])
async def roi_calculator():
    return {
        "status": "requires_business_inputs",
        "message": "POST transaction volume, fraud prevalence, and cost assumptions to /roi.",
    }


@app.post("/roi", tags=["Business Intelligence"])
async def calculate_roi(inputs: ROIRequest):
    """Estimate costs from explicit assumptions and saved holdout metrics."""
    if feature_pipeline is None or not models:
        raise HTTPException(status_code=503, detail="A current model bundle is not ready")
    metrics_path = RESULTS_DIR / "model_metrics.json"
    if not metrics_path.is_file():
        raise HTTPException(status_code=503, detail="Holdout model metrics are unavailable")
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        ensemble = metrics["ensemble"]
        recall = float(ensemble["recall"])
        precision = float(ensemble["precision"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=503, detail="Holdout model metrics are invalid") from exc
    if precision <= 0:
        raise HTTPException(status_code=503, detail="The measured model precision cannot support an ROI estimate")

    fraud_count = inputs.annual_transactions * inputs.fraud_rate
    legitimate_count = inputs.annual_transactions - fraud_count

    def total_cost(detected_fraud: float, detection_recall: float, false_positives: float) -> float:
        caught = detected_fraud * detection_recall
        missed = detected_fraud - caught
        reviews = caught + false_positives
        return (
            missed * inputs.average_fraud_loss
            + false_positives * inputs.false_positive_cost
            + reviews * inputs.manual_review_cost
        )

    baseline_caught = fraud_count * inputs.baseline_recall
    baseline_missed = fraud_count - baseline_caught
    baseline_false_positives = legitimate_count * inputs.baseline_false_positive_rate
    baseline_cost = (
        baseline_missed * inputs.average_fraud_loss
        + baseline_false_positives * inputs.false_positive_cost
        + (baseline_caught + baseline_false_positives) * inputs.manual_review_cost
    )
    model_false_positives = min(
        legitimate_count,
        max(0.0, fraud_count * recall / precision - fraud_count * recall),
    )
    model_cost = total_cost(fraud_count, recall, model_false_positives)
    return {
        "status": "estimate",
        "annual_savings": round(baseline_cost - model_cost, 2),
        "baseline_cost": round(baseline_cost, 2),
        "model_cost": round(model_cost, 2),
        "model_metrics": {"precision": precision, "recall": recall},
        "assumptions": inputs.model_dump(),
    }
