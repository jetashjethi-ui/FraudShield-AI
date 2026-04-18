# FraudShield AI — Deployment Guide

## 🚀 Quick Deploy

### Option 1: Render.com (API + Landing + Mobile)

1. Push code to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Render auto-detects `render.yaml` — click **Deploy**
5. Your live URL: `https://fraudshield-ai.onrender.com`

**What you get:**
- Landing page at `/`
- Mobile PWA at `/mobile/`
- Swagger API docs at `/docs`
- WebSocket feed at `/ws/feed`
- Health check at `/health`

---

### Option 2: Streamlit Cloud (Dashboard)

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **New app** → Connect GitHub repo
3. Set main file path: `dashboard.py`
4. Click **Deploy**

**What you get:**
- Full 17-page dashboard at `https://your-app.streamlit.app`

---

### Option 3: Docker (Local / VPS)

```bash
# Full stack (dashboard + API)
docker-compose up

# API only (lightweight)
docker build -f Dockerfile.deploy -t fraudshield-api .
docker run -p 8000:8000 fraudshield-api
```

---

## 🔗 URLs After Deployment

| Service | Local | Render | Streamlit Cloud |
|---------|-------|--------|-----------------|
| Landing | localhost:8000 | fraudshield-ai.onrender.com | — |
| Mobile | localhost:8000/mobile/ | fraudshield-ai.onrender.com/mobile/ | — |
| API Docs | localhost:8000/docs | fraudshield-ai.onrender.com/docs | — |
| Dashboard | localhost:8501 | — | your-app.streamlit.app |
| WebSocket | ws://localhost:8000/ws/feed | wss://fraudshield-ai.onrender.com/ws/feed | — |

---

## ⚠️ Notes

- **Render free tier** spins down after 15 min of inactivity. First request after sleep takes ~30s.
- **Streamlit Cloud** is always-on for public repos.
- Model artifacts (`.pkl`, `.json`) in `outputs/` are loaded if present, otherwise mock scoring is used.
- GPU is NOT needed for deployment — only for training (`python main.py`).
