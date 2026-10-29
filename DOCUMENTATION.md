# Backend technical notes

## Data and evaluation

`src/data_loader.py` reads the IEEE-CIS transaction and identity tables and joins on `TransactionID`. `main.py` sorts by `TransactionDT`, reserves the latest 20% as a final holdout, and fits preprocessing only against the earlier period. The holdout is not used for model selection or decision-threshold tuning.

`src/inference_pipeline.py` defines the shared model input contract. Numeric fields use fixed zero fills plus missing-value indicators; low-cardinality categories use one-hot encoding, and high-cardinality categories use deterministic hash buckets. It excludes target and transaction identifiers, handles missing and unseen categories, and emits a stable feature order for batch and API inputs. Prior-seven-day counts and mean amounts are computed causally for card and card/address keys: each transaction is featurized before it is added to history. The fitted pipeline is saved as `outputs/model_artifacts/feature_pipeline.pkl`.

## Modeling

`src/models.py` uses expanding forward-in-time folds with equal-time boundaries to create out-of-time base-model probabilities. Resampling is confined to training folds and used only for XGBoost and MLP; LightGBM, CatBoost, and Random Forest use native class weighting. The served stack uses XGBoost, LightGBM, CatBoost, Random Forest, and MLP probabilities. The XGBoost meta-model is trained on out-of-time base predictions; Isolation Forest and the autoencoder are auxiliary research baselines, not served models.

An isotonic calibrator is fit on a chronological segment of meta-level out-of-time predictions. A later segment selects a decision threshold using configurable illustrative cost weights. The newest-period holdout is reserved for final evaluation. Cost outputs are relative units unless a user explicitly supplies a projection horizon; they are not measured business savings.

## Serving

`api.py` loads the persisted feature pipeline and schema-versioned complete model bundle. It generates all five base-model probabilities in the same order as training, applies the persisted calibration model, and returns a calibrated fraud probability with a threshold-derived recommendation. It returns `503` if the bundle is absent, incompatible, or scoring is unavailable rather than fabricating a probability. The current bundle schema is version 4.

- `/health`: process liveness.
- `/ready`: model-bundle readiness for deployment health checks.
- `/score`: detailed transaction response.
- `/predict`: compact response.
- `/batch`: up to 500 requests per call.
- `/roi`: scenario calculation from supplied assumptions and saved metrics.
- `/model-info`: stack metadata and threshold.
- `/metrics`: saved holdout metrics for the dashboard.

Set `FRAUDSHIELD_API_KEY` to require `X-API-Key` on scoring routes. Set `CORS_ORIGINS` to trusted browser origins. Use TLS at the deployment edge and keep credentials out of client-side code.

## Repository scope and artifacts

The maintained source tree contains the active training, evaluation, and serving pipeline. The schema-versioned inference bundle and dashboard metrics are included so a clean checkout can run the scoring demo. Raw training data, generated plots, PDFs, and transaction-level CSV exports are excluded from Git. Training a replacement bundle requires the IEEE-CIS tables in `data/` and the full dependencies in `requirements.txt`.

## Rebuild

```powershell
pip install -r requirements.txt
python main.py
uvicorn api:app --host 127.0.0.1 --port 8000
```

For API deployment, build `Dockerfile.deploy` after the current bundle is present. See [DEPLOYMENT.md](DEPLOYMENT.md).
