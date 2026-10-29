# FraudShield AI dashboard

The dashboard is the web interface for the FraudShield AI FastAPI service. Its static HTML/CSS/JavaScript page is served at `/`; Next.js provides the same-origin server-side API proxy. It includes an animated Three.js landing experience, responsive overview, single and CSV batch scoring, performance and ROI views, and a system-status page. Reduced-motion preferences are respected.

## Start locally

Follow the backend setup in the [root README](../README.md) first. In a second terminal, from this folder:

```powershell
Copy-Item .env.example .env.local
corepack pnpm install --frozen-lockfile
corepack pnpm dev
```

Open [http://localhost:3000](http://localhost:3000). The server-side routes proxy requests to FastAPI using `FRAUDSHIELD_API_BASE_URL` and, optionally, `FRAUDSHIELD_API_KEY`. Keep credentials in `.env.local`; never expose them through a `NEXT_PUBLIC_` variable.

## Production build

```powershell
corepack pnpm install --frozen-lockfile
corepack pnpm build
corepack pnpm start
```
