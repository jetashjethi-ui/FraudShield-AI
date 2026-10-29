# Deployment guide

## Hosted demo

- Dashboard: [https://fraud-shield-ai-two.vercel.app/](https://fraud-shield-ai-two.vercel.app/)
- FastAPI service: [https://fraudshield-ai-ocjx.onrender.com/](https://fraudshield-ai-ocjx.onrender.com/)
- API docs: [https://fraudshield-ai-ocjx.onrender.com/docs](https://fraudshield-ai-ocjx.onrender.com/docs)
- Readiness: [https://fraudshield-ai-ocjx.onrender.com/ready](https://fraudshield-ai-ocjx.onrender.com/ready)

The frontend and API are separate deployments. Vercel hosts the dashboard from the `frontend/` directory. Render hosts the Dockerized FastAPI service from the repository's `render.yaml` blueprint and `Dockerfile.deploy`.

### Vercel environment variables

Set `FRAUDSHIELD_API_BASE_URL` to `https://fraudshield-ai-ocjx.onrender.com` in the Vercel project's **Production** environment. Also set it for **Preview** if preview deployments should use the hosted API. These are server-only variables; do not add a `NEXT_PUBLIC_` prefix. Redeploy Vercel after changing environment variables.

`FRAUDSHIELD_API_KEY` is optional. If API-key protection is enabled, set the same value in Vercel and Render. The frontend proxy sends it only with protected scoring requests.

Render's free service may spin down when idle. Its first request after inactivity can take 50 seconds or more, so the dashboard may show one timeout while Render wakes. Refresh after the service has had time to start.

## API deployment

The repository includes a schema-versioned inference bundle in `outputs/model_artifacts/` and the dashboard metrics file in `outputs/results/model_metrics.json`. A clean checkout can build and run the demo API without the IEEE-CIS training data.

## Build and run with Docker

From the repository root:

```powershell
docker build -f Dockerfile.deploy -t fraudshield-ai-api .
docker run --rm -p 8000:8000 fraudshield-ai-api
```

Open `/health` for process liveness, `/ready` for model readiness, and `/docs` for the interactive API reference. The container health check uses `/ready` and reports unhealthy if the included model bundle cannot be loaded.

The image is CPU-only and serves the committed bundle; it does not train models. To deploy a newly trained bundle, run the pipeline in `main.py` with the required IEEE-CIS files in `data/`, then rebuild the image. Do not mix artifacts from different schema versions.

## Configuration and limits

- Set `CORS_ORIGINS` to exact trusted origins when browser clients need direct API access.
- Set `FRAUDSHIELD_API_KEY` to require `X-API-Key` on protected scoring routes.
- Keep secrets out of the dashboard's client-side code. Use TLS at any public deployment edge.
- Request history is held in memory per API process. Use one process for the demo; multi-worker or multi-replica scoring needs shared history storage.
- This is a research/demo tool, not production banking software. Scores never block real payments.

The included pickled artifacts should only be loaded from a trusted source. Use the pinned serving dependencies in `requirements.deploy.txt` and Python 3.12 for compatibility.

`render.yaml` contains the optional Render Docker service configuration. Set deployment secrets in the hosting provider's environment settings; do not commit them.
