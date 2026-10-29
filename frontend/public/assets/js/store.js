import { api } from './api.js';

const listeners = new Set();
const POLL_MS = 30000;

// Session results live in page memory only; nothing is written to storage.
export const store = {
  health: null,
  healthError: null,
  ready: null,
  readyError: null,
  modelInfo: null,
  modelInfoError: null,
  metrics: null,
  metricsError: null,
  metricsCheckedAt: null,
  checkedAt: null,
  loading: false,
  autoRefresh: true,
  session: [],
};

export function subscribe(listener) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function emit() {
  for (const listener of listeners) listener(store);
}

let inflight = null;

// Polling covers only read-only status routes. Metrics are static evaluation
// files, so they load on first check and on manual refresh only.
export function refreshSystem({ includeMetrics = true } = {}) {
  if (inflight) return inflight;
  store.loading = true;
  emit();
  const calls = [api.health(), api.ready(), api.modelInfo(), includeMetrics ? api.metrics() : null];
  inflight = Promise.allSettled(calls).then(([health, ready, info, metrics]) => {
    const settle = (result, key) => {
      store[key] = result.status === 'fulfilled' ? result.value : null;
      store[`${key}Error`] = result.status === 'rejected' ? result.reason : null;
    };
    settle(health, 'health');
    settle(ready, 'ready');
    settle(info, 'modelInfo');
    if (includeMetrics) {
      settle(metrics, 'metrics');
      store.metricsCheckedAt = new Date();
    }
    store.checkedAt = new Date();
    store.loading = false;
    inflight = null;
    emit();
  });
  return inflight;
}

export function serviceState() {
  if (!store.checkedAt) return 'checking';
  if (!store.health) return store.healthError && store.healthError.kind === 'config' ? 'unconfigured' : 'offline';
  return store.health.status === 'healthy' && store.health.ready === true ? 'healthy' : 'degraded';
}

export const SERVICE_LABEL = {
  checking: 'Checking',
  healthy: 'Healthy',
  degraded: 'Degraded',
  offline: 'Unreachable',
  unconfigured: 'Not configured',
};

export function decisionThreshold() {
  const t = store.modelInfo && store.modelInfo.decision_threshold;
  return typeof t === 'number' && Number.isFinite(t) && t >= 0 && t <= 1 ? t : null;
}

let timer = null;

export function setAutoRefresh(enabled) {
  store.autoRefresh = enabled;
  clearInterval(timer);
  timer = null;
  if (enabled) {
    timer = setInterval(() => {
      if (!document.hidden) refreshSystem({ includeMetrics: false });
    }, POLL_MS);
  }
  emit();
}

export function addSessionResult(entry) {
  store.session.unshift(entry);
  if (store.session.length > 200) store.session.length = 200;
  emit();
}

export function clearSession() {
  store.session = [];
  emit();
}
