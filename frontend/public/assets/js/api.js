// Browser client. Every request goes to the same-origin Next.js proxy at
// /api/backend/*; the browser never talks to FastAPI and never holds an API key.

const BASE = '/api/backend';

export class ApiError extends Error {
  constructor(message, { kind = 'http', status = 0, detail = null } = {}) {
    super(message);
    this.name = 'ApiError';
    this.kind = kind;
    this.status = status;
    this.detail = detail;
  }
}

export function describeDetail(detail) {
  if (!detail) return '';
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        const loc = Array.isArray(item.loc) ? item.loc.filter((part) => part !== 'body') : [];
        const where = loc.length ? `${loc.join(' › ')}: ` : '';
        return `${where}${item.msg || 'Invalid value'}`;
      })
      .join('; ');
  }
  return 'The backend returned an error.';
}

const PROXY_KIND = {
  not_configured: 'config',
  timeout: 'unreachable',
  unreachable: 'unreachable',
  malformed: 'malformed',
  bad_request: 'http',
  too_large: 'http',
};

function messageFor(status, data) {
  const detail = describeDetail(data && data.detail);
  if (status === 401) return 'FastAPI rejected the request: invalid or missing API key. Set FRAUDSHIELD_API_KEY on the Next.js server to match the backend.';
  if (status === 422) return detail ? `Validation failed — ${detail}` : 'Validation failed.';
  if (status === 503 && !data?.proxy_error) return detail || 'The scoring service is not ready (HTTP 503).';
  return detail || `HTTP ${status}`;
}

async function request(name, { method = 'GET', body, timeout = 10000, allowStatus = [] } = {}) {
  const path = `${BASE}/${name}`;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  try {
    const response = await fetch(path, {
      method,
      headers: body ? { 'Content-Type': 'application/json', Accept: 'application/json' } : { Accept: 'application/json' },
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
      cache: 'no-store',
    });
    const type = response.headers.get('content-type') || '';
    if (!type.includes('application/json')) {
      throw new ApiError(`The Next.js proxy did not return JSON for ${path} (HTTP ${response.status}).`, {
        kind: 'malformed',
        status: response.status,
      });
    }
    let data;
    try {
      data = await response.json();
    } catch {
      throw new ApiError(`Malformed JSON from ${path}.`, { kind: 'malformed', status: response.status });
    }
    if (!response.ok && !allowStatus.includes(response.status)) {
      const kind = data && data.proxy_error ? PROXY_KIND[data.proxy_error] || 'http' : 'http';
      throw new ApiError(messageFor(response.status, data), { kind, status: response.status, detail: data && data.detail });
    }
    if (data === null || typeof data !== 'object') {
      throw new ApiError(`Unexpected response shape from ${path}.`, { kind: 'malformed', status: response.status });
    }
    return data;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if (error && error.name === 'AbortError') {
      throw new ApiError(`Request to ${path} timed out after ${Math.round(timeout / 1000)} s.`, { kind: 'unreachable' });
    }
    throw new ApiError(`Could not reach the Next.js proxy at ${path}.`, { kind: 'unreachable' });
  } finally {
    clearTimeout(timer);
  }
}

export const api = {
  health: () => request('health'),
  // /ready answers 503 with a valid body when models are not loaded.
  ready: () => request('ready', { allowStatus: [503] }),
  modelInfo: () => request('model-info'),
  metrics: () => request('metrics'),
  roiInfo: () => request('roi'),
  roi: (assumptions) => request('roi', { method: 'POST', body: assumptions, timeout: 16000 }),
  score: (transaction) => request('score', { method: 'POST', body: transaction, timeout: 22000 }),
  batch: (transactions) => request('batch', { method: 'POST', body: { transactions }, timeout: 115000 }),
};
