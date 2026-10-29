import { h, fmt, isNum, titleCase } from '../util.js';
import { store, subscribe, refreshSystem, setAutoRefresh, serviceState, SERVICE_LABEL } from '../store.js';
import { offlineNotice, row } from '../components.js';

const ENDPOINTS = [
  ['GET', '/health', 'Service health', 'Polled every 30 s while this tab is visible'],
  ['GET', '/ready', 'Model readiness', 'Checked with system status'],
  ['GET', '/model-info', 'Loaded models, stacking order, threshold', 'Polled with /health'],
  ['GET', '/metrics', 'Saved holdout evaluation metrics', 'Loaded on system check and manual refresh'],
  ['POST', '/score', 'Score one transaction', 'Only on submit'],
  ['POST', '/batch', 'Score up to 500 transactions', 'Only on submit'],
  ['POST', '/roi', 'Calculate an ROI scenario from your assumptions', 'Only on submit'],
  ['GET', '/docs', 'OpenAPI documentation', 'Available from the backend'],
];

export function initStatus(root) {
  const slot = root.querySelector('[data-slot="status-body"]');
  const checked = root.querySelector('[data-slot="status-checked"]');
  const refreshBtn = root.querySelector('[data-action="status-refresh"]');
  const auto = root.querySelector('#status-auto');
  const endpoints = root.querySelector('[data-slot="status-endpoints"]');

  refreshBtn.addEventListener('click', () => refreshSystem());
  auto.addEventListener('change', () => setAutoRefresh(auto.checked));

  endpoints.replaceChildren(
    ...ENDPOINTS.map(([method, path, purpose, usage]) =>
      row({ label: method, value: path, small: true, compact: true, note: purpose, tags: [usage] }),
    ),
  );

  function render() {
    const state = serviceState();
    const health = store.health;
    const info = store.modelInfo;
    const ready = store.ready;
    const ensemble = store.metrics && store.metrics.ensemble;
    auto.checked = store.autoRefresh;
    refreshBtn.disabled = store.loading;
    refreshBtn.textContent = store.loading ? 'Checking…' : 'Refresh now';
    checked.textContent = store.checkedAt ? `Last checked ${fmt.time(store.checkedAt)}` : 'Not checked yet';

    const tone = { degraded: 'orange', offline: 'red' }[state];
    const parts = [];
    if (state === 'offline') parts.push(offlineNotice(store.healthError));
    parts.push(
      h(
        'ul',
        { class: 'rows' },
        row({ label: 'Derived', value: SERVICE_LABEL[state], tone, note: 'Healthy requires status "healthy" and at least one loaded model.' }),
        row({ label: 'status', value: health ? health.status : '—', small: true, compact: true, muted: !health, tags: ['/health'] }),
        row({ label: 'ready', value: ready ? (ready.ready ? 'true' : 'false') : '—', small: true, compact: true, muted: !ready, tags: ['/ready'] }),
        row({ label: 'models_loaded', value: health ? fmt.int(health.models_loaded) : '—', small: true, compact: true, muted: !health, tags: ['/health'] }),
        row({ label: 'features_count', value: health ? fmt.int(health.features_count) : '—', small: true, compact: true, muted: !health, tags: ['/health'] }),
        row({
          label: 'ensemble holdout AUC',
          value: ensemble && isNum(ensemble.auc) ? fmt.dec(ensemble.auc) : '—',
          small: true,
          compact: true,
          muted: !ensemble || !isNum(ensemble.auc),
          note: store.metricsError ? 'Saved evaluation metrics could not be loaded.' : 'Loaded from outputs/results/model_metrics.json.',
          tags: ['/metrics'],
        }),
        row({ label: 'uptime', value: health ? `${fmt.int(health.uptime_seconds)} s` : '—', small: true, compact: true, muted: !health, tags: ['/health'] }),
        row({
          label: 'Loaded keys',
          value: info && info.models_loaded ? `${info.models_loaded.length} artifacts` : '—',
          small: true,
          compact: true,
          muted: !info,
          tags: info && info.models_loaded ? info.models_loaded.map(titleCase) : undefined,
        }),
        row({ label: 'feature_count', value: info ? fmt.int(info.feature_count) : '—', small: true, compact: true, muted: !info, tags: ['/model-info'] }),
      ),
    );
    slot.replaceChildren(...parts);
  }

  subscribe(render);
  render();
  return { enter: render };
}
