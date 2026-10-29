import { fmt, isNum, titleCase } from '../util.js';
import { store, subscribe, serviceState, SERVICE_LABEL } from '../store.js';
import { offlineNotice, row, tierRows } from '../components.js';

export function initOverview(root) {
  const alertSlot = root.querySelector('[data-slot="overview-alert"]');
  const statusSlot = root.querySelector('[data-slot="overview-status"]');
  const archSlot = root.querySelector('[data-slot="overview-arch"]');
  const tiersSlot = root.querySelector('[data-slot="overview-tiers"]');
  const sessionSlot = root.querySelector('[data-slot="overview-session"]');

  function renderStatus() {
    const state = serviceState();
    const health = store.health;
    const info = store.modelInfo;
    const aucModels = Object.entries(store.metrics || {})
      .filter(([, metrics]) => metrics && isNum(metrics.auc))
      .sort((a, b) => b[1].auc - a[1].auc);
    const bestAuc = aucModels[0];

    alertSlot.replaceChildren(state === 'offline' ? offlineNotice(store.healthError) : '');

    let serviceNote = 'Derived from GET /health.';
    if (state === 'degraded') {
      serviceNote =
        'The API is reachable but reports no loaded models. Scoring is unavailable until the model bundle loads; /score returns HTTP 503 in this state.';
    } else if (state === 'healthy' && health.features_count === 0) {
      serviceNote = 'Models are loaded but the feature list is empty, so scores may not reflect trained features.';
    } else if (state === 'checking') {
      serviceNote = 'Contacting GET /health…';
    }

    const tone = { healthy: undefined, degraded: 'orange', offline: 'red', checking: undefined }[state];

    statusSlot.replaceChildren(
      row({ label: 'Service', value: SERVICE_LABEL[state], tone, note: serviceNote, tags: ['GET /health'] }),
      row({
        label: 'Models',
        value: health ? fmt.int(health.models_loaded) : '—',
        muted: !health,
        note: info && info.models_loaded && info.models_loaded.length ? 'Artifacts loaded by the API process.' : undefined,
        tags: info && Array.isArray(info.models_loaded) ? info.models_loaded.map(titleCase) : undefined,
      }),
      row({ label: 'Features', value: health ? fmt.int(health.features_count) : '—', muted: !health }),
      row({
        label: 'Best holdout AUC',
        value: bestAuc ? fmt.dec(bestAuc[1].auc) : store.metricsCheckedAt ? 'Unavailable' : '—',
        muted: !bestAuc,
        note: bestAuc
          ? `${titleCase(bestAuc[0])} · saved evaluation metrics.`
          : store.metricsError
            ? 'The saved evaluation metrics could not be loaded.'
            : undefined,
      }),
      row({ label: 'Uptime', value: health ? `${fmt.int(health.uptime_seconds)} s` : '—', muted: !health, small: true }),
    );

    const stack = info && Array.isArray(info.stacking_order) ? info.stacking_order : null;
    if (stack) {
      archSlot.replaceChildren(
        row({ label: 'Ensemble stacking order', value: `${stack.length} base models`, small: true, compact: true, tags: stack.map(titleCase) }),
        row({ label: 'Decision threshold', value: fmt.dec(info.decision_threshold), small: true, compact: true, tags: ['/model-info'] }),
        row({ label: 'Feature count', value: fmt.int(info.feature_count), small: true, compact: true, tags: ['/model-info'] }),
      );
    } else {
      archSlot.replaceChildren(
        row({ label: 'Model information', value: 'Unavailable', muted: true, small: true, compact: true, note: 'Requires GET /model-info.' }),
      );
    }

    tiersSlot.replaceChildren(...tierRows(info && info.decision_threshold));
  }

  function renderSession() {
    const scored = store.session.length;
    sessionSlot.replaceChildren(
      row({
        label: 'This session',
        value: scored ? `${scored} scored` : 'Nothing scored yet',
        muted: !scored,
        small: true,
        compact: true,
        note: 'Kept in memory only. Reloading the page clears it.',
      }),
    );
  }

  const render = () => {
    renderStatus();
    renderSession();
  };

  subscribe(render);
  render();

  return { enter: render };
}
