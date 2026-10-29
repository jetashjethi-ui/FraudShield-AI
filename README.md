# FraudShield AI

FraudShield AI is a research and demo project for exploring transaction-risk scoring. It combines a FastAPI scoring service with a dark, motion-led web dashboard. It is built around the IEEE-CIS Fraud Detection dataset and includes a versioned model bundle so the demo can score transactions after installation.

> **Research/demo only.** This project is not production banking software. Scores are for demonstration and analysis; they never block real payments.

## Live demo

- **Dashboard:** [fraud-shield-ai-two.vercel.app](https://fraud-shield-ai-two.vercel.app/)
- **API service:** [fraudshield-ai-ocjx.onrender.com](https://fraudshield-ai-ocjx.onrender.com/)
- **Interactive API docs:** [Render `/docs`](https://fraudshield-ai-ocjx.onrender.com/docs)
- **API health:** [Render `/health`](https://fraudshield-ai-ocjx.onrender.com/health) · [Render `/ready`](https://fraudshield-ai-ocjx.onrender.com/ready)
- **Dashboard-to-API check:** [Vercel backend health](https://fraud-shield-ai-two.vercel.app/api/backend/health)

The dashboard sends API requests through its server-side proxy. Render uses a free instance that can sleep when idle; its first request after inactivity may take 50 seconds or more. If the dashboard reports a timeout during wake-up, wait briefly and refresh.

## What is included

- **Overview:** live service and model status, plus saved evaluation summaries.
- **Single scoring:** submit a transaction and inspect the risk score, model signals, and response details.
- **Batch scoring:** validate and score CSV transactions, then download the results.
- **Performance and ROI:** inspect saved evaluation results and explore user-supplied cost scenarios.
- **System status:** view API readiness and service information.
- **FastAPI backend:** single, batch, and ROI routes backed by the included five-model ensemble.
- **Training pipeline:** chronological evaluation, shared train/serve feature processing, probability calibration, and threshold selection.

## Architecture

```text
Browser
  ├─ Static HTML/CSS/JS dashboard served by Next.js on Vercel
  └─ API requests → same-origin Next.js proxy on Vercel
       └─ FastAPI scoring service on Render
            ├─ fitted feature pipeline
            ├─ XGBoost, LightGBM, CatBoost, Random Forest, and MLP
            └─ stack, calibrator, and saved evaluation results

Optional model training:
IEEE-CIS transaction + identity CSVs → chronological training/evaluation → model bundle
```

The dashboard proxies requests through its own server routes. The browser does not call FastAPI directly, and an optional API key stays on the server. The scoring pipeline uses the same fitted preprocessing contract as training. Its prior-seven-day entity features are calculated causally as transactions are processed.

## Run the demo on Windows

Requirements: **Python 3.12**, **Node.js 22 LTS**, and **pnpm** (available through Corepack).

Clone the repository, then start the API in one PowerShell window:

```powershell
git clone https://github.com/jetashjethi-ui/FraudShield-AI.git
Set-Location FraudShield-AI
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.deploy.txt
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Start the dashboard in a second PowerShell window:

```powershell
Set-Location FraudShield-AI\frontend
Copy-Item .env.example .env.local
corepack pnpm install --frozen-lockfile
corepack pnpm dev
```

Open [http://localhost:3000](http://localhost:3000). The API is available at [http://127.0.0.1:8000](http://127.0.0.1:8000), with interactive documentation at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). The included bundle supports the demo without the training dataset.

## Train a new model bundle

Training is optional. Download the IEEE-CIS transaction and identity tables according to the dataset's terms and place these files in `data/`:

```text
data/train_transaction.csv
data/train_identity.csv
```

Install the full training dependencies and run the pipeline:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

Training replaces the serving bundle in `outputs/model_artifacts/` and writes evaluation reports to `outputs/results/`. It can take a while and needs substantial system memory. Review the resource guidance in [DOCUMENTATION.md](DOCUMENTATION.md) before starting. The raw dataset is not included in this repository.

## API routes

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/` | Service name, version, and status-route links |
| `POST` | `/score` | Detailed single-transaction score |
| `POST` | `/predict` | Compact single-transaction score |
| `POST` | `/batch` | Score 1–500 transactions |
| `GET` | `/roi` | Describe the inputs required by the ROI estimator |
| `POST` | `/roi` | Calculate a scenario from supplied assumptions and saved model metrics |
| `GET` | `/health` | Process liveness |
| `GET` | `/ready` | Model-bundle readiness |
| `GET` | `/model-info` | Loaded model and feature information |
| `GET` | `/metrics` | Saved evaluation metrics for the dashboard |
| `GET` | `/docs` | Interactive API documentation |
| `GET` | `/redoc` | Alternative API documentation |
| `GET` | `/openapi.json` | OpenAPI schema |

For request schemas and scoring behavior, see [DOCUMENTATION.md](DOCUMENTATION.md). The API processes timestamped batches chronologically and returns results in the original order. Without timestamps, supplied order determines the history features. Entity history is held in memory by one API process, so multi-worker deployment needs shared history storage.

## Configuration

The dashboard reads server-only settings from `frontend/.env.local`; start from the committed `frontend/.env.example`:

```dotenv
FRAUDSHIELD_API_BASE_URL=http://127.0.0.1:8000
FRAUDSHIELD_API_KEY=
```

Set `FRAUDSHIELD_API_KEY` in both the API environment and the dashboard environment when API-key protection is enabled. Do not use a `NEXT_PUBLIC_` prefix for secrets. Set `CORS_ORIGINS` to the trusted browser origins when accessing the API from other clients.

## Repository layout

```text
api.py                      FastAPI service
main.py                     training and evaluation entry point
src/                        feature, model, inference, and evaluation code
outputs/model_artifacts/    included schema-versioned scoring bundle
outputs/results/            saved model and evaluation summaries
frontend/                   static dashboard and Next.js server-side API proxy
tests/                      backend preflight tests
```

Raw data, local environments, secrets, generated plots, PDFs, and transaction-level CSV exports are excluded from Git. The prebuilt model bundle and dashboard metrics are included because the scoring demo needs them.

## Notes and limitations

- The bundle contains Python pickle files. Only load artifacts from a source you trust, and use the pinned serving dependencies in `requirements.deploy.txt`.
- The dataset is historical and can differ from live payment traffic. Saved metrics do not guarantee future performance.
- Default threshold and ROI costs are illustrative, not institution-specific financial advice or savings claims.
- This demo is not connected to a payment processor and does not authorize, decline, or block payments.

See [MODEL_CARD.md](MODEL_CARD.md) for evaluation context and limitations, [DEPLOYMENT.md](DEPLOYMENT.md) for the API container, and [DOCUMENTATION.md](DOCUMENTATION.md) for implementation details.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
