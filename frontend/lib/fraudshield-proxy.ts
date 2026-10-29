// Server-only backend-for-frontend proxy for the FraudShield FastAPI service.
// Each route handler passes a fixed upstream path; no client input ever selects the URL.

type ProxyErrorKind = 'not_configured' | 'timeout' | 'unreachable' | 'malformed' | 'bad_request' | 'too_large'

interface ProxyOptions {
  path: '/health' | '/ready' | '/model-info' | '/metrics' | '/score' | '/batch' | '/roi'
  method: 'GET' | 'POST'
  request?: Request
  protectedRoute?: boolean
  timeoutMs?: number
  maxBodyBytes?: number
}

const NO_STORE = { 'Cache-Control': 'no-store' }

function proxyError(status: number, kind: ProxyErrorKind, detail: string) {
  return Response.json({ detail, proxy_error: kind }, { status, headers: NO_STORE })
}

function resolveBaseUrl(): URL | null {
  const raw = process.env.FRAUDSHIELD_API_BASE_URL?.trim()
  if (!raw) return null
  try {
    const url = new URL(raw)
    if (url.protocol !== 'http:' && url.protocol !== 'https:') return null
    return url
  } catch {
    return null
  }
}

// FastAPI errors are { detail: string } or { detail: [{ loc, msg, type }] } for 422.
// Forward only those safe fields; never forward stack traces or headers.
function sanitizeError(payload: unknown): unknown {
  if (!payload || typeof payload !== 'object') return { detail: 'The backend returned an error.' }
  const detail = (payload as { detail?: unknown }).detail
  if (typeof detail === 'string') return { detail }
  if (Array.isArray(detail)) {
    return {
      detail: detail.slice(0, 50).map((item) => {
        const entry = item as { loc?: unknown; msg?: unknown; type?: unknown }
        return {
          loc: Array.isArray(entry.loc) ? entry.loc.map(String) : [],
          msg: typeof entry.msg === 'string' ? entry.msg : 'Invalid value',
          type: typeof entry.type === 'string' ? entry.type : undefined,
        }
      }),
    }
  }
  return { detail: 'The backend returned an error.' }
}

export async function proxyToFastAPI({
  path,
  method,
  request,
  protectedRoute = false,
  timeoutMs = 10_000,
  maxBodyBytes = 64 * 1024,
}: ProxyOptions): Promise<Response> {
  const base = resolveBaseUrl()
  if (!base) {
    return proxyError(
      503,
      'not_configured',
      'FRAUDSHIELD_API_BASE_URL is not set (or is not a valid http/https URL) on the Next.js server.',
    )
  }

  const headers: Record<string, string> = { Accept: 'application/json' }
  let body: string | undefined

  if (method === 'POST') {
    if (!request) return proxyError(400, 'bad_request', 'Missing request body.')
    const text = await request.text()
    if (new TextEncoder().encode(text).length > maxBodyBytes) {
      return proxyError(413, 'too_large', `Request body exceeds ${Math.round(maxBodyBytes / 1024)} KB.`)
    }
    try {
      JSON.parse(text)
    } catch {
      return proxyError(400, 'bad_request', 'Request body must be valid JSON.')
    }
    body = text
    headers['Content-Type'] = 'application/json'
    const key = process.env.FRAUDSHIELD_API_KEY
    if (protectedRoute && key) headers['X-API-Key'] = key
  }

  const target = new URL(path, base.href.endsWith('/') ? base.href : `${base.href}/`)
  // new URL('/score', base) drops any base path; keep the configured prefix if present.
  target.pathname = `${base.pathname.replace(/\/$/, '')}${path}`

  let upstream: Response
  try {
    upstream = await fetch(target, {
      method,
      headers,
      body,
      cache: 'no-store',
      signal: AbortSignal.timeout(timeoutMs),
    })
  } catch (error) {
    const name = (error as { name?: string })?.name
    if (name === 'TimeoutError' || name === 'AbortError') {
      return proxyError(504, 'timeout', `FastAPI did not respond within ${Math.round(timeoutMs / 1000)} s.`)
    }
    return proxyError(502, 'unreachable', 'The Next.js server could not reach the configured FastAPI service.')
  }

  let payload: unknown
  try {
    payload = await upstream.json()
  } catch {
    return proxyError(502, 'malformed', `FastAPI returned a non-JSON response (HTTP ${upstream.status}).`)
  }

  if (!upstream.ok) {
    return Response.json(sanitizeError(payload), { status: upstream.status, headers: NO_STORE })
  }
  return Response.json(payload, { status: upstream.status, headers: NO_STORE })
}
